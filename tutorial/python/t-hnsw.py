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
# --- 请将此路径修改为您的sift数据集所在的目录 ---
DATA_DIR = "./gist"
LEARN_FILE = os.path.join(DATA_DIR, "learn.fbin") # HNSW不需要训练，但我们加载它以获取维度信息
BASE_FILE = os.path.join(DATA_DIR, "base.fbin")
QUERY_FILE = os.path.join(DATA_DIR, "query.fbin")
GROUNDTRUTH_FILE = os.path.join(DATA_DIR, "groundtruth.ivecs")

# ==============================================================================
# 1. 辅助函数：读取文件 (与之前脚本相同)
# ==============================================================================
def read_fbin(filename):
    """
    读取.fbin格式的文件 (一次性读入内存)
    格式: [nvecs: int32, dim: int32, data: float32[nvecs*dim]]
    """
    if not os.path.exists(filename):
        print(f"错误: 文件未找到 {filename}")
        exit(1)
        
    with open(filename, 'rb') as f:
        nvecs = struct.unpack('i', f.read(4))[0]
        dim = struct.unpack('i', f.read(4))[0]
        data = np.fromfile(f, dtype=np.float32, count=nvecs * dim)
        data = data.reshape(nvecs, dim)
    return data, nvecs, dim

def read_ivecs(filename):
    """
    读取.ivecs格式的二进制文件 (一次性读入内存)
    格式: 向量循环 [dim: int32, data: int32[dim]]
    """
    if not os.path.exists(filename):
        print(f"错误: 文件未找到 {filename}")
        exit(1)

    a = np.fromfile(filename, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def calculate_recall(search_results_I, groundtruth_path, k):
    """
    计算ANN搜索结果的召回率，并自行加载Groundtruth文件。

    参数:
    search_results_I (np.array): 形状为 (nq, k) 的搜索结果索引数组。
    groundtruth_path (str): Groundtruth .ivecs文件的路径。
    k (int): 搜索时指定的近邻数。

    返回:
    float: Recall@k 的值，如果文件不存在则返回 -1.0。
    """
    if search_results_I.shape[1] != k:
        raise ValueError(f"搜索结果的列数 ({search_results_I.shape[1]}) 与指定的 k ({k}) 不匹配。")

    print(f"  -> 正在从 {groundtruth_path} 加载Groundtruth数据...")
    if not os.path.exists(groundtruth_path):
        print(f"  -> 错误: Groundtruth文件未找到!")
        return -1.0
        
    # 调用辅助函数读取文件
    ground_truth_I = read_ivecs(groundtruth_path)

    nq = search_results_I.shape[0]
    
    # 验证查询数量是否匹配
    if nq != ground_truth_I.shape[0]:
        print(f"  -> 警告: 搜索结果数量({nq})与Groundtruth数量({ground_truth_I.shape[0]})不匹配!")
        # 截断以匹配较小的那个
        min_nq = min(nq, ground_truth_I.shape[0])
        search_results_I = search_results_I[:min_nq]
        ground_truth_I = ground_truth_I[:min_nq]
        nq = min_nq

    # 从Ground Truth中取出前k个作为比较基准
    gt_k = ground_truth_I[:, :k]

    found_count = sum(len(set(search_results_I[i]).intersection(set(gt_k[i]))) for i in range(nq))
    
    total_possible = nq * k
    recall = found_count / total_possible if total_possible > 0 else 0
    
    print(f"\n查询了 {nq} 个向量, k={k}")
    print(f"在top-{k}的结果中，总共找到了 {found_count} 个真实的近邻。")
    
    return recall

def report_peak_memory():
    """报告程序运行期间的峰值内存占用。"""
    print("\n" + "="*60); print("Phase 6: 性能报告")
    if platform.system() in ["Linux", "Darwin"]:
        peak_memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Linux": peak_memory_bytes *= 1024
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        print(f"整个程序运行期间的峰值内存占用: {peak_memory_mb:.2f} MB")
    else:
        print("无法在当前操作系统上自动获取峰值内存。")
    print("="*60)

# ==============================================================================
# 2. 加载数据集到内存
# ==============================================================================
print("="*60)
print("Phase 1: 从磁盘加载SIFT数据集到内存")
start_time = time.time()

# HNSW不需要独立的训练集，但我们加载learn.fbin以获取维度'd'，并保持流程一致性
_, _, d = read_fbin(LEARN_FILE)

print(f"  -> 正在加载基础集: {BASE_FILE}")
xb, nb, d_base = read_fbin(BASE_FILE)

print(f"  -> 正在加载查询集: {QUERY_FILE}")
xq, nq, d_query = read_fbin(QUERY_FILE)

# 验证维度一致性
if d != d_base or d != d_query:
    raise ValueError(f"维度不一致: 训练集{d}维, 基础集{d_base}维, 查询集{d_query}维")

print(f"\n数据加载完成，耗时: {time.time() - start_time:.2f} 秒")
print(f"向量维度 (d): {d}")
print(f"基础集大小 (nb): {nb}")
print(f"查询集大小 (nq): {nq}")
print("="*60)

# ==============================================================================
# 3. 设置HNSW参数并构建索引
# ==============================================================================
# HNSW 核心参数
M = 32              # 图中每个节点的邻居数(度)。控制索引的质量和内存占用。
efConstruction = 40 # 构建图时的搜索范围。影响构建时间和索引质量。
efSearch = 64      # 搜索时的搜索范围。影响搜索时间和召回率。
k = 10              # k-NN搜索中的k值

print("\nPhase 2: 构建HNSW索引 (in-memory)")
print(f"HNSW M (邻居数): {M}")
print(f"HNSW efConstruction (构建参数): {efConstruction}")

# 1. 创建HNSW索引对象
#    参数: 维度, M, 度量方式 (L2范数)
index_hnsw = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
index_hnsw.verbose = True

# 2. 设置构建时参数
#    efConstruction值越高，构建越慢，但图质量越高
index_hnsw.hnsw.efConstruction = efConstruction

# 3. HNSW索引不需要 .train() 步骤
print("\n注意: HNSW索引是增量构建的，不需要独立的训练(train)步骤。")

# 4. 向索引中添加数据，这一步即是构建图的过程
print("\nPhase 3: 构建HNSW图 (向索引中添加基础向量)")
start_time = time.time()

index_hnsw.add(xb) # 在添加向量的同时，HNSW图被构建起来

end_time = time.time()
print(f"HNSW图构建完成，耗时: {end_time - start_time:.2f} 秒")
print(f"索引中的向量总数 (ntotal): {index_hnsw.ntotal}")


# ==============================================================================
# 4. 执行搜索
# ==============================================================================
print("\n" + "="*60)
print("Phase 4: 执行搜索")
print(f"搜索近邻数 (k): {k}")
print(f"HNSW efSearch (搜索参数): {efSearch}")

# 设置搜索时参数
# efSearch值越高，搜索越慢，但召回率越高
index_hnsw.hnsw.efSearch = efSearch

print("执行搜索...")
start_time = time.time()
D, I = index_hnsw.search(xq, k) # D: 距离, I: 索引ID
end_time = time.time()

print(f"搜索完成，耗时: {end_time - start_time:.2f} 秒")
qps = nq / (end_time - start_time)
print(f"每秒查询数 (QPS): {qps:.2f}")


# ==============================================================================
# 5. 计算召回率
# ==============================================================================
print("\n" + "="*60)
print("Phase 5: 计算召回率")

# 调用函数，直接传入搜索结果I、groundtruth文件路径和k值
recall = calculate_recall(I, GROUNDTRUTH_FILE, k)
print(f"Recall@{k}: {recall:.4f}")

print("="*60)


# ==============================================================================
# 6. 报告峰值内存
# ==============================================================================
print("\n" + "="*60)
report_peak_memory()
print("="*60)