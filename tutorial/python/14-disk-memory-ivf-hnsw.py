import numpy as np
import faiss
import time
import os
import platform
import resource
import tempfile
import shutil

# ==============================================================================
# 0. 设置参数与环境
# ==============================================================================
d = 128                          # 向量维度
nb = 1000000                     # 数据集大小 (100万)
nq = 10000                       # 查询集大小
cell_size = 128
nlist = nb // cell_size  
ntrain = 10000                   # 训练集大小
chunk_size = 100000              # 数据块大小
k = 10                           # 查找最近的k个邻居
np.random.seed(1234)

# 创建临时目录
dataset_dir = tempfile.mkdtemp()
data_filename = os.path.join(dataset_dir, "full_dataset.dat")
index_filename = os.path.join(dataset_dir, "large_ivf_hnsw_on_disk.index")

print("="*60)
print("Phase 0: 环境设置")
print(f"数据集大小 (nb): {nb}, 训练集大小 (ntrain): {ntrain}, 分块大小 (chunk_size): {chunk_size}")
print(f"数据集文件: {data_filename}")
print(f"索引文件: {index_filename}")
print("="*60)

# ==============================================================================
# 1. 生成完整数据集并存储到单个文件
# ==============================================================================
print("\nPhase 1: 生成完整数据集并存储到单个文件")

# 创建空文件并设置大小
total_bytes = nb * d * 4  # float32 = 4字节
with open(data_filename, 'wb') as f:
    f.seek(total_bytes - 1)
    f.write(b'\0')  # 创建稀疏文件

print(f"已创建数据集文件 ({total_bytes/(1024**2):.2f} MB)")

# 使用内存映射写入数据
start_time = time.time()
with open(data_filename, 'r+b') as f:
    # 创建内存映射
    mmap = np.memmap(f, dtype='float32', mode='r+', shape=(nb, d))
    
    # 分块生成并写入数据
    for i in range(0, nb, chunk_size):
        end_i = min(i + chunk_size, nb)
        chunk_size_actual = end_i - i
        
        # 生成数据块
        chunk_data = np.random.random((chunk_size_actual, d)).astype('float32')
        chunk_data[:, 0] += np.arange(i, end_i) / 1000.
        
        # 写入到内存映射位置
        mmap[i:end_i] = chunk_data
        
        # 保存前5个向量用于完整性检查
        if i == 0:
            sanity_vectors = chunk_data[:5].copy()
        
        print(f"  -> 已生成并写入数据块 {i//chunk_size + 1}/{(nb-1)//chunk_size + 1} (索引 {i} 到 {end_i-1})")
        
        # 确保数据写入磁盘
        mmap.flush()

print(f"数据集生成完成，总耗时: {time.time() - start_time:.2f} 秒")

# ==============================================================================
# 2. 从数据集中采样训练数据
# ==============================================================================
print("\nPhase 2: 从数据集采样训练数据")

# 创建采样索引 - 从整个数据集中均匀采样
train_indices = np.random.choice(nb, ntrain, replace=False)
train_indices.sort()  # 排序以提高读取效率

# 收集训练样本
xt = np.zeros((ntrain, d), dtype='float32')
start_time = time.time()

# 使用内存映射读取数据
with open(data_filename, 'rb') as f:
    # 创建只读内存映射
    mmap = np.memmap(f, dtype='float32', mode='r', shape=(nb, d))
    
    # 读取训练样本
    for i, idx in enumerate(train_indices):
        xt[i] = mmap[idx]
        
        # 每1000个样本打印一次进度
        if i % 1000 == 0:
            print(f"  -> 已采样 {i+1}/{ntrain} 个训练样本")

print(f"训练数据采样完成，耗时: {time.time() - start_time:.2f} 秒")

# ==============================================================================
# 3. 训练量化器
# ==============================================================================
print("\nPhase 3: 训练 HNSW 粗量化器")
coarse_quantizer = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_L2)
index_for_training = faiss.IndexIVFFlat(coarse_quantizer, d, nlist, faiss.METRIC_L2)
index_for_training.verbose = True

print("开始训练聚类中心...")
start_time = time.time()
index_for_training.train(xt)
end_time = time.time()

print(f"量化器训练完成，耗时: {end_time - start_time:.2f} 秒")
print(f"粗量化器中的质心数量: {coarse_quantizer.ntotal}")
del xt, index_for_training

# ==============================================================================
# 4. 创建磁盘索引并分块添加数据
# ==============================================================================
print("\nPhase 4: 创建磁盘索引并分块添加数据")

# 创建空的索引框架
index_shell = faiss.IndexIVFFlat(coarse_quantizer, d, nlist, faiss.METRIC_L2)
print("将空的索引框架写入磁盘...")
faiss.write_index(index_shell, index_filename)
del index_shell

# 获取IO标志 (兼容不同Faiss版本)
try:
    IO_FLAG_READ_WRITE = faiss.IO_FLAG_READ_WRITE
except AttributeError:
    try:
        IO_FLAG_READ_WRITE = faiss.index_io.IO_FLAG_READ_WRITE
    except AttributeError:
        IO_FLAG_READ_WRITE = 0

print(f"使用IO标志: {IO_FLAG_READ_WRITE} (读写模式)")

# 打开磁盘索引准备添加数据
index_ondisk = faiss.read_index(index_filename, IO_FLAG_READ_WRITE)

start_time = time.time()

# 使用内存映射分块读取数据
with open(data_filename, 'rb') as f:
    # 创建只读内存映射
    mmap = np.memmap(f, dtype='float32', mode='r', shape=(nb, d))
    
    # 分块添加数据
    for i in range(0, nb, chunk_size):
        end_i = min(i + chunk_size, nb)
        chunk_size_actual = end_i - i
        
        # 获取数据块 (不复制数据)
        xb_chunk = mmap[i:end_i]
        
        # 添加到索引
        index_ondisk.add(xb_chunk)
        
        print(f"  -> 已添加数据块 {i//chunk_size + 1}/{(nb-1)//chunk_size + 1} (索引 {i} 到 {end_i-1}, {chunk_size_actual} 向量)")
        
        # 释放内存引用
        del xb_chunk

print(f"所有数据块添加完成，总耗时: {time.time() - start_time:.2f} 秒")
print(f"磁盘索引中的向量总数: {index_ondisk.ntotal}")

# ==============================================================================
# 5. 完整性检查
# ==============================================================================
if sanity_vectors is not None:
    print("\n进行完整性检查...")
    D_check, I_check = index_ondisk.search(sanity_vectors, k)
    
    passed = True
    for j in range(5):
        if I_check[j, 0] != j:
            print(f"错误: 第{j}个向量的最近邻居索引是{I_check[j,0]}而不是{j}")
            passed = False
    
    if passed:
        print("完整性检查通过: 所有向量的最近邻居都是自身")
    else:
        print("完整性检查失败: 某些向量的最近邻居不是自身")

# 保存最终索引
print("正在将最终索引写回磁盘...")
faiss.write_index(index_ondisk, index_filename)
del index_ondisk

# ==============================================================================
# 6. 使用内存映射进行搜索
# ==============================================================================
print("\nPhase 5: 使用内存映射模式进行搜索")

# 获取MMAP标志 (兼容不同Faiss版本)
try:
    IO_FLAG_MMAP = faiss.IO_FLAG_MMAP
except AttributeError:
    try:
        IO_FLAG_MMAP = faiss.index_io.IO_FLAG_MMAP
    except AttributeError:
        IO_FLAG_MMAP = 4

print(f"使用IO标志: {IO_FLAG_MMAP} (内存映射模式)")

# 以mmap模式打开索引
index_final = faiss.read_index(index_filename, IO_FLAG_MMAP)
index_final.nprobe = 32
print(f"索引已准备好搜索 (nprobe={index_final.nprobe})")

# 生成查询向量
print("生成查询向量并执行搜索...")
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

# 执行搜索
start_time = time.time()
D, I = index_final.search(xq, k)
end_time = time.time()
print(f"搜索完成，耗时: {end_time - start_time:.2f} 秒")

# 打印部分结果
print("\n查询结果示例 (最后5个查询):")
print("索引 (I):")
print(I[-5:])
print("距离 (D):")
print(D[-5:])

# ==============================================================================
# 7. 清理与资源报告
# ==============================================================================
print("\n" + "="*60)
if platform.system() in ["Linux", "Darwin"]:
    peak_memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Linux":
        peak_memory_bytes *= 1024
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    print(f"峰值内存占用: {peak_memory_mb:.2f} MB")
else:
    print("当前操作系统非 Linux/macOS，无法自动报告峰值内存。")

print(f"清理临时目录: {dataset_dir}")
shutil.rmtree(dataset_dir)
print("="*60)
