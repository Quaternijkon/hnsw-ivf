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
ntrain = 100000                  # 用于训练量化器的样本数
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
index_for_training = faiss.IndexIVFFlat(coarse_quantizer, d, 1024, faiss.METRIC_L2)
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
index_shell = faiss.IndexIVFFlat(coarse_quantizer, d, 1024, faiss.METRIC_L2)
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

for i in range(0, nb, chunk_size):
    end_i = min(i + chunk_size, nb)
    print(f"  -> 正在处理块 {i // chunk_size + 1}/ {nb // chunk_size}: 向量 {i} 到 {end_i-1}")
    
    xb_chunk = np.random.random((end_i - i, d)).astype('float32')
    xb_chunk[:, 0] += np.arange(i, end_i) / 1000.
    
    index_ondisk.add(xb_chunk)
    del xb_chunk

print(f"\n所有数据块添加完成，总耗时: {time.time() - start_time:.2f} 秒")
print(f"磁盘索引中的向量总数 (ntotal): {index_ondisk.ntotal}")

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

print("\n查询结果的索引 (I[-5:]):")
print(I[-5:])


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