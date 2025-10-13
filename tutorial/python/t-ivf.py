import numpy as np
import faiss
import time
import os
import platform
import resource
import struct

# ==============================================================================
# 0. 路径和超参数配置
# ==============================================================================
# --- 请将此路径修改为您的sift数据集所在的目录 ---
DATA_DIR = "./sift"
LEARN_FILE = os.path.join(DATA_DIR, "learn.fbin")
BASE_FILE = os.path.join(DATA_DIR, "base.fbin")
QUERY_FILE = os.path.join(DATA_DIR, "query.fbin")
GROUNDTRUTH_FILE = os.path.join(DATA_DIR, "groundtruth.ivecs")

# --- 索引保存路径 ---
INDEX_DIR = os.path.join(DATA_DIR, "ivf-index")
os.makedirs(INDEX_DIR, exist_ok=True)

# --- 核心超参数配置 --- # <--- MODIFIED: 将超参数移至此处统一管理
nlist = 7812      # IVF倒排列表的数量 (聚类中心数)
k = 10             # k-NN搜索中的k值
nprobe = nlist        # 搜索时要访问的倒排列表数量

# --- 动态生成索引文件名 --- # <--- MODIFIED: 文件名与nlist参数绑定
INDEX_FILENAME = f"sift_ivf_nlist{nlist}.index"
INDEX_FILE = os.path.join(INDEX_DIR, INDEX_FILENAME)


# ==============================================================================
# 1. 辅助函数：读取文件
# ==============================================================================
def read_fbin(filename):
    """
    读取.fbin格式的文件 (一次性读入内存)
    格式: [nvecs: int32, dim: int32, data: float32[nvecs*dim]]
    """
    if not os.path.exists(filename):
        print(f"错误: 文件未找到 {filename}")
        # 尝试提示如何下载数据集
        print("请从 http://corpus-texmex.irisa.fr/ 下载sift.tar.gz并解压到'./sift'目录")
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

# ==============================================================================
# 2. 加载数据集到内存
# ==============================================================================
print("="*60)
print("Phase 1: 从磁盘加载SIFT数据集到内存")
start_time = time.time()

print(f"  -> 正在加载训练集: {LEARN_FILE}")
xt, nt, d = read_fbin(LEARN_FILE)

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
print(f"训练集大小 (nt): {nt}")
print("="*60)

# ==============================================================================
# 3. 准备索引：加载或构建
# ==============================================================================
# <--- MODIFIED: 超参数已移至顶部
print(f"\nPhase 2 & 3: 准备索引 (IVF分区数 nlist: {nlist})")

if os.path.exists(INDEX_FILE):
    # --- 如果索引文件存在，则直接加载 ---
    print(f"  -> 发现现有索引文件，正在从磁盘加载: {INDEX_FILE}")
    start_time = time.time()
    index_ivf = faiss.read_index(INDEX_FILE)
    end_time = time.time()
    print(f"  -> 索引加载完成，耗时: {end_time - start_time:.2f} 秒")

else:
    # --- 如果索引文件不存在，则构建、训练、添加并保存 ---
    print(f"  -> 未找到索引 '{INDEX_FILE}'，开始构建新索引...")
    
    # 3.1. 构建索引
    print("  -> 步骤 3.1: 构建IVF索引结构")
    # quantizer = faiss.IndexFlatL2(d)
    quantizer = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_L2)
    index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    index_ivf.verbose = True

    # 3.2. 训练索引
    print("\n  -> 步骤 3.2: 训练聚类中心...")
    start_time = time.time()
    index_ivf.train(xt)
    end_time = time.time()
    print(f"  -> 索引训练完成，耗时: {end_time - start_time:.2f} 秒")

    # 3.3. 向索引中添加数据
    print("\n  -> 步骤 3.3: 向索引中添加基础向量...")
    start_time = time.time()
    index_ivf.add(xb)
    end_time = time.time()
    print(f"  -> 所有向量添加完成，耗时: {end_time - start_time:.2f} 秒")
    
    # 3.4. 保存索引到磁盘
    print(f"\n  -> 步骤 3.4: 将构建好的索引写入磁盘: {INDEX_FILE}")
    start_time = time.time()
    faiss.write_index(index_ivf, INDEX_FILE)
    end_time = time.time()
    print(f"  -> 索引保存完成，耗时: {end_time - start_time:.2f} 秒")

print(f"\n索引已准备就绪。索引中的向量总数 (ntotal): {index_ivf.ntotal}")

# 释放不再需要的内存
del xt
del xb

# ==============================================================================
# 4. 执行搜索
# ==============================================================================
print("\n" + "="*60)
print("Phase 4: 执行搜索")
print(f"搜索近邻数 (k): {k}")
print(f"搜索分区数 (nprobe): {nprobe}")

# 设置 nprobe 参数
index_ivf.nprobe = nprobe

print("执行搜索...")
start_time = time.time()
D, I = index_ivf.search(xq, k)
end_time = time.time()

print(f"搜索完成，耗时: {end_time - start_time:.2f} 秒")
qps = nq / (end_time - start_time)
print(f"每秒查询数 (QPS): {qps:.2f}")


# ==============================================================================
# 5. 计算召回率
# ==============================================================================
print("\n" + "="*60)
print("Phase 5: 计算召回率 (与Groundtruth文件对比)")

print(f"  -> 正在加载Groundtruth文件: {GROUNDTRUTH_FILE}...")
# I_gt 是一个 (nq, 100) 的数组，groundtruth文件通常包含top 100的真实近邻
I_gt = read_ivecs(GROUNDTRUTH_FILE)

print("  -> 正在计算召回率...")
# 我们只关心我们搜索的 top-k 结果
# 因此，我们将我们的结果与 groundtruth 的前k个结果进行比较
gt_k = I_gt[:, :k] 

found_count = 0
for i in range(nq):
    # 使用集合交集计算重合的ID数量
    ivf_results = set(I[i])
    gt_results = set(gt_k[i])
    found_count += len(ivf_results.intersection(gt_results))

total_possible = nq * k
recall = found_count / total_possible

print(f"\n查询了 {nq} 个向量, k={k}")
print(f"在top-{k}的结果中，总共找到了 {found_count} 个真实的近邻。")
print(f"Recall@{k}: {recall:.4f}")
print("="*60)


# ==============================================================================
# 6. 报告峰值内存
# ==============================================================================
print("\n" + "="*60)
print("Phase 6: 性能报告")
if platform.system() in ["Linux", "Darwin"]:
    peak_memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux返回的是KB, macOS(Darwin)返回的是Bytes
    if platform.system() == "Linux":
        peak_memory_bytes *= 1024
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    print(f"整个程序运行期间的峰值内存占用: {peak_memory_mb:.2f} MB")
else:
    print("无法在当前操作系统上自动获取峰值内存。")
print("="*60)