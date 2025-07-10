# -*- coding: UTF-8 -*-

import numpy as np
import faiss
import time
import os
import platform
import struct # 用于解析二进制文件头

# 在非 Linux/macOS 系统上，resource 模块可能不可用
try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False
    print("警告: 'resource' 模块在此操作系统上不可用，无法监控峰值内存。")

# ==============================================================================
# 0. 设置参数、文件路径与辅助函数
# ==============================================================================
print("="*60)
print("Phase 0: 环境设置")

# --- 文件路径 ---
learn_file = "./data/sift_learn.fbin"   # 训练数据集
base_file = "./data/sift_base.fbin"     # 基础向量集 (将被索引)
query_file = "./data/sift_query.fbin"   # 查询向量集

# --- 索引参数 ---
chunk_size = 100000         # ★★★ 每次处理的数据块大小, 确保 chunk_size * d * 4 < 可用 RAM
cell_size = 128             # 建议每个 cell 包含的向量数，用于计算 nlist
k = 10                      # 查找最近的10个邻居
np.random.seed(1234)

# --- 索引文件名 ---
index_filename = "large_ivf_hnsw_on_disk.index" # 保留你原来的索引名

# --- 清理旧文件 ---
if os.path.exists(index_filename):
    os.remove(index_filename)
    print(f"已删除旧索引文件: {index_filename}")

# --- 辅助函数: 读取 .fbin 文件 ---
def read_fbin(filename, start_idx=0, count=-1):
    """
    读取 .fbin 格式的文件。

    文件格式假定为:
    - 第一个 4 字节整数 (int32): 向量总数 (num_vectors)
    - 第二个 4 字节整数 (int32): 向量维度 (dim)
    - 之后是 num_vectors * dim 个 4 字节浮点数 (float32)

    :param filename: 文件路径
    :param start_idx: 开始读取的向量索引, 默认为 0
    :param count: 要读取的向量数量。-1 表示读取从 start_idx 到末尾的所有向量。
    :return: numpy.ndarray
    """
    with open(filename, "rb") as f:
        num_vectors_total, dim = struct.unpack('ii', f.read(8))
        
        # 计算要读取的向量数量
        if count == -1:
            # 读取从 start_idx 到末尾的所有数据
            num_to_read = num_vectors_total - start_idx
        else:
            # 读取指定数量的数据，但不能超过文件末尾
            num_to_read = min(count, num_vectors_total - start_idx)

        if num_to_read <= 0:
            return np.empty((0, dim), dtype='float32')

        # 定位到开始读取的位置
        # 8 字节文件头 + 每个向量 (dim * 4 字节)
        f.seek(8 + start_idx * dim * 4)
        
        # 读取数据
        data = np.fromfile(f, dtype='float32', count=num_to_read * dim)
        return data.reshape(num_to_read, dim)

def read_fbin_header(filename):
    """只读取 .fbin 文件的头部信息 (总数, 维度)"""
    with open(filename, "rb") as f:
        num_vectors, dim = struct.unpack('ii', f.read(8))
    return num_vectors, dim

def fbin_chunk_generator(filename, chunk_size):
    """
    一个生成器，用于分块读取 .fbin 文件，以节省内存。
    """
    num_vectors_total, dim = read_fbin_header(filename)
    for i in range(0, num_vectors_total, chunk_size):
        chunk = read_fbin(filename, start_idx=i, count=chunk_size)
        yield chunk

# --- 从文件头获取核心参数 ---
print("正在从数据文件头部读取基本信息...")
nb, d = read_fbin_header(base_file)
ntrain, d_train = read_fbin_header(learn_file)
nq, d_query = read_fbin_header(query_file)

# --- 校验维度一致性 ---
assert d == d_train, f"错误: 基础数据集维度 ({d}) 与训练集维度 ({d_train}) 不匹配!"
assert d == d_query, f"错误: 基础数据集维度 ({d}) 与查询集维度 ({d_query}) 不匹配!"

nlist = nb // cell_size  # 根据数据集大小计算聚类中心数量
print(f"向量维度 (d): {d}")
print(f"基础数据集大小 (nb): {nb}")
print(f"训练集大小 (ntrain): {ntrain}")
print(f"查询集大小 (nq): {nq}")
print(f"IVF 聚类中心数 (nlist): {nlist}")
print(f"数据分块大小 (chunk_size): {chunk_size}")
print(f"索引将保存在磁盘文件: {index_filename}")
print("="*60)


# ==============================================================================
# 1. 训练量化器 (在内存中完成)
# ==============================================================================
print("\nPhase 1: 训练 HNSW 粗量化器 (in-memory)")
# !! 注意：完全按照你的要求，保留 HNSW+IVF 的索引实现方式 !!
coarse_quantizer = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_L2)
index_for_training = faiss.IndexIVFFlat(coarse_quantizer, d, nlist, faiss.METRIC_L2)
index_for_training.verbose = True

print(f"从 '{learn_file}' 加载训练数据...")
xt = read_fbin(learn_file)
assert xt.shape[0] == ntrain and xt.shape[1] == d

print("训练聚类中心并构建 HNSW 量化器...")
start_time = time.time()
index_for_training.train(xt)
end_time = time.time()

print(f"量化器训练完成，耗时: {end_time - start_time:.2f} 秒")
print(f"粗量化器中的质心数量: {coarse_quantizer.ntotal}")
del xt
# `index_for_training` 包含对 `coarse_quantizer` 的引用，
# 我们需要 `coarse_quantizer` 来构建下一步的空索引，所以只删除 `index_for_training`
del index_for_training


# ==============================================================================
# 2. 创建一个空的、基于磁盘的索引框架
# ==============================================================================
print("\nPhase 2: 创建空的磁盘索引框架")
# !! 注意：使用在第一步中训练好的 coarse_quantizer !!
index_shell = faiss.IndexIVFFlat(coarse_quantizer, d, nlist, faiss.METRIC_L2)
print("将空的索引框架写入磁盘...")
faiss.write_index(index_shell, index_filename)
del index_shell
del coarse_quantizer # `coarse_quantizer` 已写入文件，可以释放内存


# ==============================================================================
# 3. 分块向磁盘索引中添加数据
# ==============================================================================
print("\nPhase 3: 分块添加数据到磁盘索引")

# 兼容不同Faiss版本的IO标志处理
try:
    IO_FLAG_READ_WRITE = faiss.IO_FLAG_READ_WRITE
except AttributeError:
    IO_FLAG_READ_WRITE = 0 # 默认为读写模式

print(f"使用IO标志: {IO_FLAG_READ_WRITE} (读写模式)")

# 以读写模式打开磁盘上的索引文件
index_ondisk = faiss.read_index(index_filename, IO_FLAG_READ_WRITE)
start_time = time.time()

# 保存前5个向量用于后续的健全性检查
sanity_vectors = None
is_first_chunk = True

total_chunks = (nb + chunk_size - 1) // chunk_size
for i, xb_chunk in enumerate(fbin_chunk_generator(base_file, chunk_size)):
    print(f"  -> 正在处理块 {i + 1}/{total_chunks}...")
    
    # 如果是第一个块，保存前5个向量用于Sanity Check
    if is_first_chunk:
        # 确保块内至少有5个向量
        num_to_save = min(5, xb_chunk.shape[0])
        sanity_vectors = xb_chunk[:num_to_save].copy()
        is_first_chunk = False
    
    index_ondisk.add(xb_chunk)
    # 此处不需要 del xb_chunk，因为生成器会在下一次迭代时覆盖它

print(f"\n所有数据块添加完成，总耗时: {time.time() - start_time:.2f} 秒")
print(f"磁盘索引中的向量总数 (ntotal): {index_ondisk.ntotal}")

# ===========================================================
# 添加 Sanity Check - 检查索引是否正常工作
# ===========================================================
if sanity_vectors is not None and sanity_vectors.shape[0] > 0:
    print("\n进行Sanity Check...")
    print(f"在索引中搜索来自 '{base_file}' 的前 {sanity_vectors.shape[0]} 个向量本身:")
    # 对于Sanity Check，我们只关心它自己是不是最近的，所以k=1即可
    D_check, I_check = index_ondisk.search(sanity_vectors, 1)
    
    print("Sanity Check - 索引结果 (I):")
    print(I_check)
    print("Sanity Check - 距离结果 (D):")
    print(D_check)
    
    # 检查结果
    passed = True
    for j in range(sanity_vectors.shape[0]):
        if I_check[j, 0] != j:
            print(f"错误: 第{j}个向量的最近邻居索引是{I_check[j,0]}而不是{j}")
            passed = False
        if not np.isclose(D_check[j, 0], 0.0, atol=1e-5):
            print(f"警告: 第{j}个向量的最近邻居距离是{D_check[j,0]:.6f}而不是0 (可接受的浮点误差)")
    
    if passed:
        print("Sanity Check 通过: 所有测试向量的最近邻居都是自身。")
    else:
        print("Sanity Check 失败: 某些向量的最近邻居不是自身。")
else:
    print("无法进行Sanity Check: 未保存用于测试的向量。")

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
    IO_FLAG_MMAP = faiss.IO_FLAG_MMAP
except AttributeError:
    IO_FLAG_MMAP = 4 # 默认为内存映射模式

print(f"使用IO标志: {IO_FLAG_MMAP} (内存映射模式)")

index_final = faiss.read_index(index_filename, IO_FLAG_MMAP)

index_final.nprobe = 32
print(f"索引已准备好搜索 (nprobe={index_final.nprobe})")

print(f"从 '{query_file}' 加载查询向量并执行搜索...")
xq = read_fbin(query_file)
assert xq.shape[0] == nq and xq.shape[1] == d

start_time = time.time()
D, I = index_final.search(xq, k)
end_time = time.time()
print(f"搜索完成，耗时: {end_time - start_time:.2f} 秒")
print(f"平均查询时间: {(end_time - start_time) * 1000 / nq:.3f} 毫秒/条")


print(f"\n查询结果的前5条 (共 {nq} 条):")
print("索引 (I):")
print(I[:5])
print("距离 (D):")
print(D[:5])


# ==============================================================================
# 5. 报告峰值内存并清理
# ==============================================================================
print("\n" + "="*60)
if RESOURCE_AVAILABLE and platform.system() in ["Linux", "Darwin"]:
    peak_memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Linux":
        # 在 Linux 上，ru_maxrss 单位是 KB
        peak_memory_mb = peak_memory_bytes / 1024
    else: # macOS
        # 在 macOS 上，ru_maxrss 单位是 Bytes
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    print(f"整个程序运行期间的峰值内存占用: {peak_memory_mb:.2f} MB")
else:
    print("当前操作系统非 Linux/macOS 或 'resource' 模块不可用，无法自动报告峰值内存。")
print("="*60)

print(f"\n清理临时索引文件: {index_filename}")
os.remove(index_filename)
print("\n任务完成!")