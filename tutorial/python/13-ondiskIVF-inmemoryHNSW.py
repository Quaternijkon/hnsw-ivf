# -*- coding: UTF-8 -*-

import numpy as np
import faiss
import time
import os
import platform
import resource # 使用 resource 来监控峰值内存

# ==============================================================================
# 0. 设置参数与环境
# ==============================================================================
d = 128                          # 向量维度
nb = 1000000                     # ★★★ 假设有一个非常大的数据集（100万）
nq = 10000                       # 查询集大小
cell_size = 128
nlist = nb // cell_size  
ntrain = 10000                  # 用于训练量化器的样本数
chunk_size = 100000              # ★★★ 每次处理的数据块大小，需确保 chunk_size * d * 4 < RAM
k = 10                           # 查找最近的10个邻居
np.random.seed(1234)

# 索引文件名
index_filename = "large_ivf_hnsw_on_disk.index"

# 确保旧文件被清理
if os.path.exists(index_filename):
    os.remove(index_filename)

print("="*60)
print("Phase 0: 环境设置")
print(f"数据集大小 (nb): {nb}, 训练集大小 (ntrain): {ntrain}, 分块大小 (chunk_size): {chunk_size}")
print(f"索引将保存在磁盘文件: {index_filename}")
print("="*60)


# ==============================================================================
# 1. 训练量化器 (在内存中完成)
# ==============================================================================
print("\nPhase 1: 训练 HNSW 粗量化器 (in-memory)")
coarse_quantizer = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_L2)
index_for_training = faiss.IndexIVFFlat(coarse_quantizer, d, nlist, faiss.METRIC_L2)
index_for_training.verbose = True

print("生成训练数据...")
xt = np.random.random((ntrain, d)).astype('float32')
xt[:, 0] += np.arange(ntrain) / 1000.

print("训练聚类中心并构建 HNSW 量化器...")
start_time = time.time()
index_for_training.train(xt)
end_time = time.time()

print(f"量化器训练完成，耗时: {end_time - start_time:.2f} 秒")
print(f"粗量化器中的质心数量: {coarse_quantizer.ntotal}")
del xt
del index_for_training


# ==============================================================================
# 2. 创建一个空的、基于磁盘的索引框架
# ==============================================================================
print("\nPhase 2: 创建空的磁盘索引框架")
index_shell = faiss.IndexIVFFlat(coarse_quantizer, d, nlist, faiss.METRIC_L2)
print("将空的索引框架写入磁盘...")
faiss.write_index(index_shell, index_filename)
del index_shell


# ==============================================================================
# 3. 分块向磁盘索引中添加数据
# ==============================================================================
print("\nPhase 3: 分块添加数据到磁盘索引")

# 兼容不同Faiss版本的IO标志处理
try:
    # 尝试使用新版本的常量
    IO_FLAG_READ_WRITE = faiss.IO_FLAG_READ_WRITE
except AttributeError:
    try:
        # 尝试旧版本的常量
        IO_FLAG_READ_WRITE = faiss.index_io.IO_FLAG_READ_WRITE
    except AttributeError:
        # 如果都不存在，使用默认值0（读写模式）
        IO_FLAG_READ_WRITE = 0

print(f"使用IO标志: {IO_FLAG_READ_WRITE} (读写模式)")

index_ondisk = faiss.read_index(index_filename, IO_FLAG_READ_WRITE)
start_time = time.time()

# 保存前5个向量用于Sanity Check
sanity_vectors = None

for i in range(0, nb, chunk_size):
    end_i = min(i + chunk_size, nb)
    print(f"  -> 正在处理块 {i // chunk_size + 1}/ {nb // chunk_size}: 向量 {i} 到 {end_i-1}")
    
    xb_chunk = np.random.random((end_i - i, d)).astype('float32')
    xb_chunk[:, 0] += np.arange(i, end_i) / 1000.
    
    # 如果是第一个块，保存前5个向量用于Sanity Check
    if i == 0 and sanity_vectors is None:
        sanity_vectors = xb_chunk[:5].copy()
    
    index_ondisk.add(xb_chunk)
    del xb_chunk

print(f"\n所有数据块添加完成，总耗时: {time.time() - start_time:.2f} 秒")
print(f"磁盘索引中的向量总数 (ntotal): {index_ondisk.ntotal}")

# ===========================================================
# 添加Sanity Check - 检查索引是否正常工作
# ===========================================================
if sanity_vectors is not None:
    print("\n进行Sanity Check...")
    print("在索引中搜索前5个向量本身:")
    D_check, I_check = index_ondisk.search(sanity_vectors, k)
    
    print("Sanity Check - 索引结果 (I):")
    print(I_check)
    print("Sanity Check - 距离结果 (D):")
    print(D_check)
    
    # 检查结果
    passed = True
    for j in range(5):
        if I_check[j, 0] != j:
            print(f"错误: 第{j}个向量的最近邻居索引是{I_check[j,0]}而不是{j}")
            passed = False
        if not np.isclose(D_check[j, 0], 0.0, atol=1e-6):
            print(f"警告: 第{j}个向量的最近邻居距离是{D_check[j,0]}而不是0")
            # 这可能是浮点精度问题，不一定是错误
    
    if passed:
        print("Sanity Check 通过: 所有向量的最近邻居都是自身")
    else:
        print("Sanity Check 失败: 某些向量的最近邻居不是自身")
else:
    print("无法进行Sanity Check: 未保存前5个向量")

print("正在将最终索引写回磁盘...")
faiss.write_index(index_ondisk, index_filename)
del index_ondisk


# ==============================================================================
# 4. 使用内存映射 (mmap) 进行搜索
# ==============================================================================
print("\nPhase 4: 使用内存映射模式进行搜索")
print("以 mmap 模式打开磁盘索引...")

# 兼容不同Faiss版本的IO标志处理
try:
    # 尝试使用新版本的常量
    IO_FLAG_MMAP = faiss.IO_FLAG_MMAP
except AttributeError:
    try:
        # 尝试旧版本的常量
        IO_FLAG_MMAP = faiss.index_io.IO_FLAG_MMAP
    except AttributeError:
        # 如果都不存在，使用默认值4（内存映射模式）
        IO_FLAG_MMAP = 4

print(f"使用IO标志: {IO_FLAG_MMAP} (内存映射模式)")

index_final = faiss.read_index(index_filename, IO_FLAG_MMAP)

index_final.nprobe = 32
print(f"索引已准备好搜索 (nprobe={index_final.nprobe})")

print("生成查询向量并执行搜索...")
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

start_time = time.time()
D, I = index_final.search(xq, k)
end_time = time.time()
print(f"搜索完成，耗时: {end_time - start_time:.2f} 秒")

print("\n查询结果的索引 (I[-5:])和距离 (D[-5:]):")
print("索引 (I):")
print(I[-5:])
print("距离 (D):")
print(D[-5:])

# 添加额外的Sanity Check：检查搜索结果的合理性
print("\n额外Sanity Check: 检查搜索结果的距离是否合理...")
min_dist = D.min()
max_dist = D.max()
mean_dist = D.mean()
print(f"最小距离: {min_dist:.6f}, 最大距离: {max_dist:.6f}, 平均距离: {mean_dist:.6f}")

if min_dist < 0:
    print("警告: 发现负距离值，这可能是索引配置问题")
else:
    print("距离值均为非负，符合预期")


# ==============================================================================
# 5. 报告峰值内存并清理
# ==============================================================================
print("\n" + "="*60)
if platform.system() in ["Linux", "Darwin"]:
    peak_memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Linux":
        peak_memory_bytes *= 1024
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    print(f"整个程序运行期间的峰值内存占用: {peak_memory_mb:.2f} MB")
else:
    print("当前操作系统非 Linux/macOS，无法自动报告峰值内存。")
print("="*60)

print(f"\n清理临时索引文件: {index_filename}")
os.remove(index_filename)