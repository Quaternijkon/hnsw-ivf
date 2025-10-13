# -*- coding: UTF-8 -*-

import numpy as np
import faiss
import time
import os
import platform
import resource
import struct

# ==============================================================================
# 0. 路径和文件名配置
# ==============================================================================
DATA_DIR = "./data"
LEARN_FILE = os.path.join(DATA_DIR, "sift_learn.fbin")
BASE_FILE = os.path.join(DATA_DIR, "sift_base.fbin")
QUERY_FILE = os.path.join(DATA_DIR, "sift_query.fbin")
INDEX_FILE = "large_ivf_hnsw_on_disk.index"


# ==============================================================================
# 1. 辅助函数：读取.fbin文件
# ==============================================================================
def read_fbin(filename, start_idx=0, chunk_size=None):
    """
    读取.fbin格式的文件
    格式: [nvecs: int32, dim: int32, data: float32[nvecs*dim]]
    """
    with open(filename, 'rb') as f:
        nvecs = struct.unpack('i', f.read(4))[0]
        dim = struct.unpack('i', f.read(4))[0]
        
        if chunk_size is None:
            # 读取整个文件
            data = np.fromfile(f, dtype=np.float32, count=nvecs*dim)
            data = data.reshape(nvecs, dim)
            return data
        else:
            # 读取指定块
            end_idx = min(start_idx + chunk_size, nvecs)
            num_vectors_in_chunk = end_idx - start_idx
            offset = start_idx * dim * 4  # 每个float32占4字节
            f.seek(offset, os.SEEK_CUR)
            data = np.fromfile(f, dtype=np.float32, count=num_vectors_in_chunk*dim)
            data = data.reshape(num_vectors_in_chunk, dim)
            return data, nvecs, dim

# ==============================================================================
# 2. 设置参数与环境
# ==============================================================================

# 从训练文件中获取维度信息
_, nt, d_train = read_fbin(LEARN_FILE, chunk_size=1)  # 只读取元数据

# 获取数据集大小信息
_, nb, d_base = read_fbin(BASE_FILE, chunk_size=1)
_, nq, d_query = read_fbin(QUERY_FILE, chunk_size=1)

# 验证维度一致性
if d_train != d_base or d_train != d_query:
    raise ValueError(f"维度不一致: 训练集{d_train}维, 基础集{d_base}维, 查询集{d_query}维")

# 设置其他参数
cell_size = 128
nlist = nb // cell_size
chunk_size = 100000  # 每次处理的数据块大小
k = 10  # 查找最近的10个邻居

# 确保旧文件被清理
if os.path.exists(INDEX_FILE):
    os.remove(INDEX_FILE)

print("="*60)
print("Phase 0: 环境设置")
print(f"向量维度 (d): {d_train}")
print(f"基础集大小 (nb): {nb}, 训练集大小 (ntrain): {nt}")
print(f"查询集大小 (nq): {nq}, 分块大小 (chunk_size): {chunk_size}")
print(f"索引将保存在磁盘文件: {INDEX_FILE}")
print("="*60)

# ==============================================================================
# 3. 训练量化器 (使用learn.fbin的前ntrain个向量)
# ==============================================================================
print("\nPhase 1: 训练 HNSW 粗量化器 (in-memory)")
coarse_quantizer = faiss.IndexHNSWFlat(d_train, 32, faiss.METRIC_L2)
index_for_training = faiss.IndexIVFFlat(coarse_quantizer, d_train, nlist, faiss.METRIC_L2)
index_for_training.verbose = True

xt = read_fbin(LEARN_FILE)

print("训练聚类中心并构建 HNSW 量化器...")
start_time = time.time()
index_for_training.train(xt)
end_time = time.time()

print(f"量化器训练完成，耗时: {end_time - start_time:.2f} 秒")
print(f"粗量化器中的质心数量: {coarse_quantizer.ntotal}")
del xt
del index_for_training

# ==============================================================================
# 4. 创建一个空的、基于磁盘的索引框架
# ==============================================================================
print("\nPhase 2: 创建空的磁盘索引框架")
index_shell = faiss.IndexIVFFlat(coarse_quantizer, d_train, nlist, faiss.METRIC_L2)
print("将空的索引框架写入磁盘...")
faiss.write_index(index_shell, INDEX_FILE)
del index_shell

# ==============================================================================
# 5. 分块向磁盘索引中添加数据 (从base.fbin)
# ==============================================================================
print("\nPhase 3: 分块添加数据到磁盘索引")

# 兼容不同Faiss版本的IO标志处理
try:
    IO_FLAG_READ_WRITE = faiss.IO_FLAG_READ_WRITE
except AttributeError:
    try:
        IO_FLAG_READ_WRITE = faiss.index_io.IO_FLAG_READ_WRITE
    except AttributeError:
        IO_FLAG_READ_WRITE = 0

print(f"使用IO标志: {IO_FLAG_READ_WRITE} (读写模式)")

index_ondisk = faiss.read_index(INDEX_FILE, IO_FLAG_READ_WRITE)
start_time = time.time()

# 保存前5个向量用于Sanity Check
sanity_vectors = None

num_chunks = (nb + chunk_size - 1) // chunk_size
for i in range(0, nb, chunk_size):
    chunk_idx = i // chunk_size + 1
    print(f"   -> 正在处理块 {chunk_idx}/{num_chunks}: 向量 {i} 到 {min(i+chunk_size, nb)-1}")
    
    # 从base.fbin中读取数据块
    xb_chunk, _, _ = read_fbin(BASE_FILE, i, chunk_size)
    
    # 如果是第一个块，保存前5个向量
    if i == 0 and sanity_vectors is None:
        sanity_vectors = xb_chunk[:5].copy()
    
    index_ondisk.add(xb_chunk)
    del xb_chunk

print(f"\n所有数据块添加完成，总耗时: {time.time() - start_time:.2f} 秒")
print(f"磁盘索引中的向量总数 (ntotal): {index_ondisk.ntotal}")

# ===========================================================
# Sanity Check - 检查索引是否正常工作
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
            print(f"警告: 第{j}个向量的最近邻居索引是{I_check[j,0]}而不是{j}")
            passed = False
        if not np.isclose(D_check[j, 0], 0.0, atol=1e-5):
            print(f"警告: 第{j}个向量的最近邻居距离是{D_check[j,0]}而不是0 (允许误差1e-5)")
    
    if passed:
        print("Sanity Check 通过: 所有向量的最近邻居都是自身")
    else:
        print("Sanity Check 警告: 某些向量的最近邻居不是自身 (可能是索引配置问题)")
else:
    print("无法进行Sanity Check: 未保存前5个向量")

print("正在将最终索引写回磁盘...")
faiss.write_index(index_ondisk, INDEX_FILE)
del index_ondisk

# ==============================================================================
# 6. 使用内存映射 (mmap) 进行搜索 (使用query.fbin)
# ==============================================================================
print("\nPhase 4: 使用内存映射模式进行搜索")
print("以 mmap 模式打开磁盘索引...")

# 兼容不同Faiss版本的IO标志处理
try:
    IO_FLAG_MMAP = faiss.IO_FLAG_MMAP
except AttributeError:
    try:
        IO_FLAG_MMAP = faiss.index_io.IO_FLAG_MMAP
    except AttributeError:
        IO_FLAG_MMAP = 4

print(f"使用IO标志: {IO_FLAG_MMAP} (内存映射模式)")

index_final = faiss.read_index(INDEX_FILE, IO_FLAG_MMAP)
index_final.nprobe = 32
print(f"索引已准备好搜索 (nprobe={index_final.nprobe})")

print("从 query.fbin 加载查询向量...")
xq = read_fbin(QUERY_FILE)

print("执行搜索...")
start_time = time.time()
D, I = index_final.search(xq, k)
end_time = time.time()
print(f"搜索完成，耗时: {end_time - start_time:.2f} 秒")

print("\n查询结果的索引 (I[:5])和距离 (D[:5]):")
print("索引 (I):")
print(I[:5])
print("距离 (D):")
print(D[:5])

# 额外的Sanity Check：检查搜索结果的合理性
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
# 7. 报告峰值内存并清理
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

print(f"\n清理临时索引文件: {INDEX_FILE}")
os.remove(INDEX_FILE)