import numpy as np
import faiss
import time
import os
import platform
import resource
import struct
import re

# ==============================================================================
# 0. 路径和文件名配置 & 调试开关
# ==============================================================================
DATA_DIR = "./sift"
LEARN_FILE = os.path.join(DATA_DIR, "learn.fbin")
BASE_FILE = os.path.join(DATA_DIR, "base.fbin")
QUERY_FILE = os.path.join(DATA_DIR, "query.fbin")
GROUNDTRUTH_FILE = os.path.join(DATA_DIR, "groundtruth.ivecs")

# 调试开关 - 设置为True时输出IVF分区统计信息
ENABLE_IVF_STATS = False  # 控制是否输出IVF分区统计信息

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

def read_ivecs(filename):
    """
    (此函数在此脚本中未用于批量读取，仅作为工具函数保留)
    读取.ivecs格式的二进制文件 (例如SIFT1M的groundtruth)
    格式: 向量循环 [dim: int32, data: int32[dim]]
    """
    a = np.fromfile(filename, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

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
nprobe = 64
chunk_size = 100000  # 每次处理的数据块大小
k = 10  # 查找最近的10个邻居

# 生成基于参数的索引文件名
base_name = os.path.splitext(os.path.basename(BASE_FILE))[0]
# 清理文件名中的特殊字符
clean_base_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name)
INDEX_FILE = os.path.join(DATA_DIR, f"{clean_base_name}_d{d_train}_nlist{nlist}_HNSW32_IVFFlat.index")

print("="*60)
print("Phase 0: 环境设置")
print(f"向量维度 (d): {d_train}")
print(f"基础集大小 (nb): {nb}, 训练集大小 (ntrain): {nt}")
print(f"查询集大小 (nq): {nq}, 分块大小 (chunk_size): {chunk_size}")
print(f"索引将保存在磁盘文件: {INDEX_FILE}")
print(f"IVF统计功能: {'启用' if ENABLE_IVF_STATS else '禁用'}")
print("="*60)

# ==============================================================================
# 3. 检查索引文件是否存在
# ==============================================================================
if os.path.exists(INDEX_FILE):
    print(f"索引文件 {INDEX_FILE} 已存在，跳过索引构建阶段")
    skip_index_building = True
else:
    print("索引文件不存在，将构建新索引")
    skip_index_building = False

# ==============================================================================
# 4. 训练与构建索引 (如果需要)
# ==============================================================================
if not skip_index_building:
    # ------------------- 4.1 训练量化器 -------------------
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

    # ------------------- 4.2 创建索引框架 -------------------
    print("\nPhase 2: 创建空的磁盘索引框架")
    index_shell = faiss.IndexIVFFlat(coarse_quantizer, d_train, nlist, faiss.METRIC_L2)
    print(f"将空的索引框架写入磁盘: {INDEX_FILE}")
    faiss.write_index(index_shell, INDEX_FILE)
    del index_shell

    # ------------------- 4.3 分块添加数据 -------------------
    print("\nPhase 3: 分块添加数据到磁盘索引")

    try:
        IO_FLAG_READ_WRITE = faiss.IO_FLAG_READ_WRITE
    except AttributeError:
        IO_FLAG_READ_WRITE = getattr(faiss.index_io, 'IO_FLAG_READ_WRITE', 0)
    
    print(f"使用IO标志: {IO_FLAG_READ_WRITE} (读写模式)")

    index_ondisk = faiss.read_index(INDEX_FILE, IO_FLAG_READ_WRITE)
    start_time = time.time()

    num_chunks = (nb + chunk_size - 1) // chunk_size
    for i in range(0, nb, chunk_size):
        chunk_idx = i // chunk_size + 1
        print(f"     -> 正在处理块 {chunk_idx}/{num_chunks}: 向量 {i} 到 {min(i+chunk_size, nb)-1}")
        
        xb_chunk, _, _ = read_fbin(BASE_FILE, i, chunk_size)
        
        index_ondisk.add(xb_chunk)
        del xb_chunk

    print(f"\n所有数据块添加完成，总耗时: {time.time() - start_time:.2f} 秒")
    print(f"磁盘索引中的向量总数 (ntotal): {index_ondisk.ntotal}")
    
    # ------------------- 4.4 IVF分区统计 (可选) -------------------
    if ENABLE_IVF_STATS:
        print("\n输出IVF分区统计信息...")
        start_stats_time = time.time()
        
        invlists = index_ondisk.invlists
        
        partition_stats = []
        non_empty_partitions, max_size, min_size, total_vectors = 0, 0, float('inf'), 0
        
        for list_id in range(nlist):
            list_size = invlists.list_size(list_id)
            if list_size > 0:
                non_empty_partitions += 1
                max_size = max(max_size, list_size)
                min_size = min(min_size, list_size)
                total_vectors += list_size
                partition_stats.append((list_id, list_size))
        
        avg_size = total_vectors / non_empty_partitions if non_empty_partitions > 0 else 0
        
        print("IVF分区统计摘要:")
        print(f"  分区总数: {nlist}, 非空分区数: {non_empty_partitions} ({non_empty_partitions/nlist*100:.2f}%)")
        print(f"  最大/最小/平均分区大小: {max_size}/{min_size}/{avg_size:.2f}")
        
        stats_filename = os.path.splitext(INDEX_FILE)[0] + "_ivf_stats.csv"
        with open(stats_filename, 'w') as f:
            f.write("partition_id,vector_count\n")
            for list_id, size in partition_stats:
                f.write(f"{list_id},{size}\n")
        
        print(f"分区统计信息已保存到: {stats_filename}")
        print(f"统计耗时: {time.time() - start_stats_time:.2f}秒")
    
    print(f"正在将最终索引写回磁盘: {INDEX_FILE}")
    faiss.write_index(index_ondisk, INDEX_FILE)
    del index_ondisk

# ==============================================================================
# 5. 加载索引并执行搜索
# ==============================================================================
print("\n" + "="*60)
print("Phase 4: 加载索引与批量搜索")
print(f"以 mmap 模式打开磁盘索引: {INDEX_FILE}")

try:
    IO_FLAG_MMAP = faiss.IO_FLAG_MMAP
except AttributeError:
    IO_FLAG_MMAP = getattr(faiss.index_io, 'IO_FLAG_MMAP', 4)

print(f"使用IO标志: {IO_FLAG_MMAP} (内存映射模式)")

index_final = faiss.read_index(INDEX_FILE, IO_FLAG_MMAP)
index_final.nprobe = nprobe
print(f"索引已准备好搜索 (nprobe={index_final.nprobe})")

print("从 query.fbin 加载查询向量...")
xq = read_fbin(QUERY_FILE)

print("执行批量搜索...")
start_time = time.time()
D, I = index_final.search(xq, k)
end_time = time.time()
total_time_batch = end_time - start_time
print(f"批量搜索完成，总耗时: {total_time_batch:.2f} 秒")
print(f"基于批量搜索的平均QPS: {nq / total_time_batch:.2f}")
print("="*60)


# ==============================================================================
# 6. 新增: 性能测试 (QPS 与 Latency)
# ==============================================================================
print("\n" + "="*60)
print("Phase 5: 性能测试 (QPS 与 Latency)")
print("说明: 为精确计算延迟，将逐个发送查询向量进行测试...")

latencies = []
# 预热一次，避免首次查询的额外开销影响测试结果 (如JIT编译等)
_ = index_final.search(xq[0:1], k)

start_perf_test_time = time.time()
for i in range(nq):
    start_query_time = time.time()
    # Faiss的search方法需要一个二维数组，因此使用xq[i:i+1]来保持维度
    _, _ = index_final.search(xq[i:i+1], k)
    end_query_time = time.time()
    latencies.append(end_query_time - start_query_time)
end_perf_test_time = time.time()

total_perf_test_time = end_perf_test_time - start_perf_test_time

# 计算性能指标
avg_latency_ms = np.mean(latencies) * 1000
p999_latency_ms = np.percentile(latencies, 99.9) * 1000
qps = nq / total_perf_test_time

print("\n--- 性能测试结果 ---")
print(f"查询总数: {nq}")
print(f"测试总耗时: {total_perf_test_time:.2f} 秒")
print(f"QPS (Queries Per Second): {qps:.2f}")
print(f"平均延迟 (Average Latency): {avg_latency_ms:.4f} ms")
print(f"P99.9 延迟 (99.9th Percentile Latency): {p999_latency_ms:.4f} ms")
print("="*60)


# ==============================================================================
# 7. 根据Groundtruth计算召回率
# ==============================================================================
print("\n" + "="*60)
print("Phase 6: 计算召回率 (内存优化版)")

if not os.path.exists(GROUNDTRUTH_FILE):
    print(f"Groundtruth文件未找到: {GROUNDTRUTH_FILE}")
    print("跳过召回率计算。")
else:
    print(f"以流式方式从 {GROUNDTRUTH_FILE} 读取 groundtruth 数据进行计算...")
    
    total_found = 0
    
    with open(GROUNDTRUTH_FILE, 'rb') as f:
        dim_bytes = f.read(4)
        if not dim_bytes:
            raise EOFError("Groundtruth 文件为空或已损坏。")
        k_gt = struct.unpack('i', dim_bytes)[0]
        
        print(f"Groundtruth 维度 (k_gt): {k_gt}")
        
        record_size_bytes = (k_gt + 1) * 4
        
        f.seek(0, os.SEEK_END)
        total_file_size = f.tell()
        num_gt_vectors = total_file_size // record_size_bytes
        if nq != num_gt_vectors:
              print(f"警告: 查询数量({nq})与groundtruth中的数量({num_gt_vectors})不匹配!")

        print(f"正在计算 Recall@{k}...")
        
        for i in range(nq):
            offset = i * record_size_bytes
            f.seek(offset)
            record_data = np.fromfile(f, dtype=np.int32, count=k_gt + 1)
            gt_i = record_data[1:]
            
            # 使用批量搜索时产生的I结果
            found_count = np.isin(I[i], gt_i[:k]).sum()
            total_found += found_count
            
    recall = total_found / (nq * k)
    
    print(f"\n查询了 {nq} 个向量, k={k}")
    print(f"在top-{k}的结果中，总共找到了 {total_found} 个真实的近邻。")
    print(f"Recall@{k}: {recall:.4f}")

print("="*60)


# ==============================================================================
# 8. 报告峰值内存
# ==============================================================================
print("\n" + "="*60)
if platform.system() in ["Linux", "Darwin"]:
    peak_memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Linux":
        peak_memory_bytes *= 1024
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    print(f"整个程序运行期间的峰值内存占用: {peak_memory_mb:.2f} MB")
print("="*60)