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
DATA_DIR = "./sift"
# --- 新增: 用于存储持久化索引的目录 ---
INDEX_CACHE_DIR = "./index_hnsw"

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

# --- 新增函数：获取CPU时间 ---
def get_cpu_time():
    """
    获取当前进程使用的CPU时间（用户+系统）。
    在Linux/macOS上使用resource模块，在其他系统（如Windows）上回退到time.process_time()。
    """
    if platform.system() in ["Linux", "Darwin"]:
        # resource.getrusage提供了更详细的区分，我们关心用户和系统时间总和
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        return rusage.ru_utime + rusage.ru_stime
    else:
        # time.process_time()是跨平台的标准方式，返回CPU时间
        return time.process_time()

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
# 3. 设置HNSW参数并加载/构建索引
# ==============================================================================
# HNSW 核心参数
M = 32              # 图中每个节点的邻居数(度)。控制索引的质量和内存占用。
efConstruction = 200 # 构建图时的搜索范围。影响构建时间和索引质量。
efSearch = 100       # 搜索时的搜索范围。影响搜索时间和召回率。
k = 10              # k-NN搜索中的k值

# --- 修改部分：根据参数生成索引文件名 ---
dataset_name = os.path.basename(os.path.normpath(DATA_DIR))
index_filename = f"{dataset_name}_HNSW_M{M}_efC{efConstruction}.index"
index_filepath = os.path.join(INDEX_CACHE_DIR, index_filename)

print(f"\nPhase 2: 加载或构建索引")
print(f"HNSW 参数: M={M}, efConstruction={efConstruction}")
print(f"索引文件路径: {index_filepath}")

# --- 修改部分：检查索引文件是否存在 ---
if os.path.exists(index_filepath):
    # 如果文件存在，直接从磁盘加载
    print("\n  -> 检测到预构建的索引文件，正在从磁盘加载...")
    start_load_time = time.time()
    index_hnsw = faiss.read_index(index_filepath)
    end_load_time = time.time()
    print(f"  -> 索引加载完成，耗时: {end_load_time - start_load_time:.2f} 秒")

else:
    # 如果文件不存在，则构建索引并保存
    print("\n  -> 未找到索引文件，开始构建新索引...")
    # 1. 创建HNSW索引对象 (参数: 维度, M, 度量方式)
    index_hnsw = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
    index_hnsw.verbose = True
    index_hnsw.hnsw.efConstruction = efConstruction

    print("   -> 注意: HNSW索引是增量构建的，不需要独立的训练(train)步骤。")
    print("   -> 开始构建HNSW图 (向索引中添加基础向量)...")
    
    # --- 修改部分：同时记录墙上时钟时间和CPU时间 ---
    start_build_wall_time = time.time()
    start_build_cpu_time = get_cpu_time()
    
    index_hnsw.add(xb) # 构建过程
    
    end_build_cpu_time = get_cpu_time()
    end_build_wall_time = time.time()
    # --- 结束修改 ---

    print(f"   -> HNSW图构建完成。")
    print(f"      - 墙上时钟 (Wall-clock) 耗时: {end_build_wall_time - start_build_wall_time:.2f} 秒")
    print(f"      - CPU 耗时 (User + System): {end_build_cpu_time - start_build_cpu_time:.2f} 秒")
    
    print(f"\n   -> 正在将新索引保存到: {index_filepath}")
    os.makedirs(INDEX_CACHE_DIR, exist_ok=True)
    start_save_time = time.time()
    faiss.write_index(index_hnsw, index_filepath)
    end_save_time = time.time()
    print(f"   -> 索引保存完成，耗时: {end_save_time - start_save_time:.2f} 秒")

print(f"索引中的向量总数 (ntotal): {index_hnsw.ntotal}")
print("="*60)

# ==============================================================================
# 4. 执行搜索
# ==============================================================================
print(f"\nPhase 3: 执行搜索")
print(f"搜索近邻数 (k): {k}")
print(f"HNSW efSearch (搜索参数): {efSearch}")

# 设置搜索时参数 (efSearch值越高，搜索越慢，但召回率越高)
index_hnsw.hnsw.efSearch = efSearch

print("   -> 执行搜索...")
# --- 修改部分：同时记录墙上时钟时间和CPU时间 ---
start_search_wall_time = time.time()
start_search_cpu_time = get_cpu_time()

D, I = index_hnsw.search(xq, k)

end_search_cpu_time = get_cpu_time()
end_search_wall_time = time.time()
# --- 结束修改 ---

wall_time_diff = end_search_wall_time - start_search_wall_time
cpu_time_diff = end_search_cpu_time - start_search_cpu_time

qps_wall = nq / wall_time_diff if wall_time_diff > 0 else float('inf')
qps_cpu = nq / cpu_time_diff if cpu_time_diff > 0 else float('inf')

print(f"   -> 搜索完成。")
print(f"      - 墙上时钟 (Wall-clock) 耗时: {wall_time_diff:.2f} 秒")
print(f"      - CPU 耗时 (User + System): {cpu_time_diff:.2f} 秒")
print(f"   -> 每秒查询数 (QPS, 基于墙上时钟): {qps_wall:.2f}")
print(f"   -> 每秒查询数 (QPS, 基于CPU时间):  {qps_cpu:.2f}")
print("="*60)

# ==============================================================================
# 5. 计算召回率
# ==============================================================================
print(f"\nPhase 4: 计算召回率")

# 调用函数，直接传入搜索结果I、groundtruth文件路径和k值
recall = calculate_recall(I, GROUNDTRUTH_FILE, k)
print(f"Recall@{k}: {recall:.4f}")
print("="*60)

# ==============================================================================
# 6. 报告峰值内存
# ==============================================================================
print(f"\nPhase 5: 性能报告")
report_peak_memory()
print("="*60)