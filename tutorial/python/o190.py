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

# 调试开关
ENABLE_IVF_STATS = False  # 控制是否输出IVF分区统计信息

# 新增开关 - 控制是否统计搜索分区信息
ENABLE_SEARCH_PARTITION_STATS = False
SEARCH_STATS_FILENAME = os.path.join(DATA_DIR, "search_partition_ratios.txt")


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
cell_size = 256
nlist = nb // cell_size
nprobe = 32
chunk_size = 500000  # 每次处理的数据块大小
k = 10  # 查找最近的10个邻居

M = 32  # HNSW的连接数
efconstruction = 40 # 默认40
efsearch = 16       # 默认16

# ==============================================================================
# 【重构点】: 在索引文件名中同时体现 M 和 efConstruction 的值
# ==============================================================================
base_name = os.path.splitext(os.path.basename(BASE_FILE))[0]
# 清理文件名中的特殊字符
clean_base_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name)
# 在文件名中添加 M 和 efc 参数，以区分不同参数构建的索引
INDEX_FILE = os.path.join(DATA_DIR, f"{clean_base_name}_d{d_train}_nlist{nlist}_HNSWM{M}_efc{efconstruction}_IVFFlat.index")
# ==============================================================================

print("="*60)
print("Phase 0: 环境设置")
print(f"向量维度 (d): {d_train}")
print(f"基础集大小 (nb): {nb}, 训练集大小 (ntrain): {nt}")
print(f"查询集大小 (nq): {nq}, 分块大小 (chunk_size): {chunk_size}")
print(f"HNSW M (构建参数): {M}")
print(f"HNSW efConstruction (构建参数): {efconstruction}")
print(f"索引将保存在磁盘文件: {INDEX_FILE}")
print(f"IVF统计功能: {'启用' if ENABLE_IVF_STATS else '禁用'}")
print(f"搜索分区统计功能: {'启用' if ENABLE_SEARCH_PARTITION_STATS else '禁用'}")
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
# 4. 训练量化器 
# ==============================================================================
if not skip_index_building:
    print("\nPhase 1: 训练 HNSW 粗量化器 (in-memory)")
    coarse_quantizer = faiss.IndexHNSWFlat(d_train, M, faiss.METRIC_L2)
    coarse_quantizer.hnsw.efConstruction = efconstruction
    coarse_quantizer.hnsw.efSearch = efsearch
    print(f"efconstruction: {coarse_quantizer.hnsw.efConstruction}, efSearch: {coarse_quantizer.hnsw.efSearch}")
    # coarse_quantizer = faiss.IndexFlatL2(d_train)
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
    # 5. 创建一个空的、基于磁盘的索引框架
    # ==============================================================================
    print("\nPhase 2: 创建空的磁盘索引框架")
    index_shell = faiss.IndexIVFFlat(coarse_quantizer, d_train, nlist, faiss.METRIC_L2)
    print(f"将空的索引框架写入磁盘: {INDEX_FILE}")
    faiss.write_index(index_shell, INDEX_FILE)
    del index_shell

    # ==============================================================================
    # 6. 分块向磁盘索引中添加数据 (从base.fbin)
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



    num_chunks = (nb + chunk_size - 1) // chunk_size
    for i in range(0, nb, chunk_size):
        chunk_idx = i // chunk_size + 1
        print(f"       -> 正在处理块 {chunk_idx}/{num_chunks}: 向量 {i} 到 {min(i+chunk_size, nb)-1}")
        
        # 从base.fbin中读取数据块
        xb_chunk, _, _ = read_fbin(BASE_FILE, i, chunk_size)
        
        index_ondisk.add(xb_chunk)
        del xb_chunk

    print(f"\n所有数据块添加完成，总耗时: {time.time() - start_time:.2f} 秒")
    print(f"磁盘索引中的向量总数 (ntotal): {index_ondisk.ntotal}")
    
    # ===========================================================
    # 7. 新增: 输出IVF分区统计信息 (仅在构建索引时执行)
    # ===========================================================
    if ENABLE_IVF_STATS and not skip_index_building:
        print("\n输出IVF分区统计信息...")
        start_stats_time = time.time()
        
        # 获取倒排列表
        invlists = index_ondisk.invlists
        
        # 准备统计信息
        partition_stats = []
        non_empty_partitions = 0
        max_size = 0
        min_size = float('inf')
        total_vectors = 0
        
        # 遍历所有分区
        for list_id in range(nlist):
            list_size = invlists.list_size(list_id)
            
            # 修改点 1：无论分区大小是否为0，都记录下来，以便生成完整的CSV报告
            partition_stats.append((list_id, list_size))
            
            # 仅针对非空分区更新摘要统计信息
            if list_size > 0:
                non_empty_partitions += 1
                max_size = max(max_size, list_size)
                min_size = min(min_size, list_size)
                total_vectors += list_size
                
        # 修改点 2：处理没有非空分区的边缘情况，避免打印 'inf'
        if non_empty_partitions == 0:
            min_size = 0
        
        # 计算非空分区的平均大小 (total_vectors 是非空分区中的向量总数)
        avg_size = total_vectors / non_empty_partitions if non_empty_partitions > 0 else 0
        
        # 输出统计摘要
        print(f"IVF分区统计摘要:")
        print(f"  分区总数: {nlist}")
        print(f"  非空分区数: {non_empty_partitions} ({non_empty_partitions/nlist*100:.2f}%)")
        print(f"  最大分区大小: {max_size}")
        # 修改点 3：为了清晰起见，明确指出这是非空分区的最小值
        print(f"  最小(非空)分区大小: {min_size}")
        # 修改点 3：为了清晰起见，明确指出这是非空分区的平均值
        print(f"  平均(非空)分区大小: {avg_size:.2f}")
        
        # 将详细统计信息写入文件
        # 此部分无需修改，因为它现在会正确处理包含所有分区的 partition_stats 列表
        stats_filename = os.path.splitext(INDEX_FILE)[0] + "_ivf_stats.csv"
        with open(stats_filename, 'w') as f:
            f.write("partition_id,vector_count\n")
            for list_id, size in partition_stats:
                f.write(f"{list_id},{size}\n")
                
        print(f"分区统计信息已保存到: {stats_filename}")
        print(f"统计耗时: {time.time() - start_stats_time:.2f}秒")
    
    # 保存索引到磁盘
    print(f"正在将最终索引写回磁盘: {INDEX_FILE}")
    faiss.write_index(index_ondisk, INDEX_FILE)
    del index_ondisk

# ==============================================================================
# 8. 使用内存映射 (mmap) 进行搜索 (使用query.fbin)
# ==============================================================================
print("\nPhase 4: 使用内存映射模式进行搜索")
print(f"以 mmap 模式打开磁盘索引: {INDEX_FILE}")

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
index_final.nprobe = nprobe
# index_final.quantizer.hnsw.efSearch = 100  # 设置HNSW的efSearch参数以匹配nprobe
faiss.omp_set_num_threads(40)
index_final.parallel_mode = 0
print(f"并行模式线程数: {faiss.omp_get_max_threads()}")
print(f"并行模式: {index_final.parallel_mode}")
print(f"索引已准备好搜索 (nprobe={index_final.nprobe})")
generic_quantizer = index_final.quantizer
quantizer_hnsw = faiss.downcast_index(generic_quantizer)
quantizer_hnsw.hnsw.efSearch = efsearch
print(f"efConstruction: {quantizer_hnsw.hnsw.efConstruction}, efSearch: {quantizer_hnsw.hnsw.efSearch}")


print("从 query.fbin 加载查询向量...")
xq = read_fbin(QUERY_FILE)

# ==============================================================================
# 8.5. 新增: 统计并保存每个查询命中的分区点数占总点数的比例
# ==============================================================================
if ENABLE_SEARCH_PARTITION_STATS:
    print("\n" + "="*60)
    print(f"Phase 4.5: 统计搜索分区信息 (nprobe={nprobe})")
    
    # 检查索引是否为IVF类型，因为该逻辑依赖于quantizer和invlists
    if not isinstance(index_final, faiss.IndexIVF):
        print("错误：索引类型不是IndexIVF，无法执行分区统计。")
    else:
        total_vectors_in_index = index_final.ntotal
        print(f"索引中的总向量数: {total_vectors_in_index}")
        
        if total_vectors_in_index == 0:
            print("警告：索引中没有向量，所有比例将为0。")
        
        print("正在为每个查询向量查找对应的分区...")
        # 1. 对每个查询向量，用粗量化器找到nprobe个最近的簇心(分区)
        # I_quant 的维度是 (nq, nprobe)，存储了每个查询命中的分区ID
        _ , I_quant = index_final.quantizer.search(xq, nprobe)
        
        ratios = []
        print(f"正在计算 {nq} 个查询的命中分区点数比例...")
        
        # 2. 遍历每个查询的结果
        for i in range(nq):
            probed_list_ids = I_quant[i]
            
            # 3. 累加这些分区中的向量总数
            num_vectors_in_probed_partitions = 0
            for list_id in probed_list_ids:
                if list_id >= 0: # 有效的分区ID
                    num_vectors_in_probed_partitions += index_final.invlists.list_size(int(list_id))
            
            # 4. 计算比例
            ratio = num_vectors_in_probed_partitions / total_vectors_in_index if total_vectors_in_index > 0 else 0
            ratios.append(ratio)

        # 5. 将结果写入文件
        try:
            with open(SEARCH_STATS_FILENAME, 'w') as f:
                for ratio in ratios:
                    f.write(f"{ratio:.8f}\n") # 写入时保留8位小数
            print(f"搜索分区统计比例已成功写入文件: {SEARCH_STATS_FILENAME}")
        except IOError as e:
            print(f"错误：无法写入统计文件 {SEARCH_STATS_FILENAME}。原因: {e}")
            
    print("="*60)


print("\n执行搜索...")
start_time = time.time()
D, I = index_final.search(xq, k)
end_time = time.time()

# 从 .indexIVF_stats 属性中获取统计对象
stats = faiss.cvar.indexIVF_stats

print("\n========== 搜索性能统计 ==========")
print(f"查询向量总数 (nq): {stats.nq}")
print(f"总搜索时间 (search_time): {stats.search_time:.3f} ms")
print(f"  - 粗筛阶段用时 (quantization_time): {stats.quantization_time:.3f} ms")
# 精筛时间可以通过总时间减去粗筛时间得到
print(f"  - 精筛阶段用时 (search_time - quantization_time): {stats.search_time - stats.quantization_time:.3f} ms")
print("-" * 30)
print(f"访问的倒排列表总数 (nlist): {stats.nlist}")
print(f"计算的向量距离总数 (ndis): {stats.ndis}")
print(f"结果堆的更新总次数 (nheap_updates): {stats.nheap_updates}")
print("====================================\n")

# --- 新增QPS计算 ---
search_duration = end_time - start_time
print(f"搜索完成，耗时: {search_duration:.2f} 秒")

if search_duration > 0:
    qps = nq / search_duration
    print(f"QPS (每秒查询率): {qps:.2f}")
else:
    print("搜索耗时过短，无法计算QPS")
# --- QPS计算结束 ---


# ==============================================================================
# 9.  新增: 根据Groundtruth计算召回率 (内存优化版)
# ==============================================================================
print("\n" + "="*60)
print("Phase 5: 计算召回率 (内存优化版)")

if not os.path.exists(GROUNDTRUTH_FILE):
    print(f"Groundtruth文件未找到: {GROUNDTRUTH_FILE}")
    print("跳过召回率计算。")
else:
    print(f"以流式方式从 {GROUNDTRUTH_FILE} 读取 groundtruth 数据进行计算...")
    
    total_found = 0
    
    # 使用with语句确保文件被正确关闭
    with open(GROUNDTRUTH_FILE, 'rb') as f:
        # 首先，从文件的第一个整数确定groundtruth的维度 (k_gt)
        dim_bytes = f.read(4)
        if not dim_bytes:
            raise EOFError("Groundtruth 文件为空或已损坏。")
        k_gt = struct.unpack('i', dim_bytes)[0]
        
        print(f"Groundtruth 维度 (k_gt): {k_gt}")
        
        # 计算文件中每条记录的字节大小
        # 每条记录包含1个维度整数和k_gt个ID整数，每个整数4字节
        record_size_bytes = (k_gt + 1) * 4
        
        # 验证文件中的向量数量是否与查询数量(nq)匹配
        f.seek(0, os.SEEK_END)
        total_file_size = f.tell()
        num_gt_vectors = total_file_size // record_size_bytes
        if nq != num_gt_vectors:
              print(f"警告: 查询数量({nq})与groundtruth中的数量({num_gt_vectors})不匹配!")

        print(f"正在计算 Recall@{k}...")
        
        # 遍历每个查询结果
        for i in range(nq):
            # 计算第 i 条记录在文件中的起始位置
            offset = i * record_size_bytes
            f.seek(offset)
            
            # 从该位置读取一条完整的记录 (k_gt + 1 个整数)
            record_data = np.fromfile(f, dtype=np.int32, count=k_gt + 1)
            
            # 记录中的第一个整数是维度，我们提取从第二个元素开始的ID列表
            gt_i = record_data[1:]
            
            found_count = np.isin(I[i], gt_i[:k]).sum()
            total_found += found_count
            
    # 召回率 = (所有查询找到的正确近邻总数) / (所有查询返回的结果总数)
    recall = total_found / (nq * k)
    
    print(f"\n查询了 {nq} 个向量, k={k}")
    print(f"在top-{k}的结果中，总共找到了 {total_found} 个真实的近邻。")
    print(f"Recall@{k}: {recall:.4f}")

print("="*60)


# ==============================================================================
# 10. 报告峰值内存
# ==============================================================================
print("\n" + "="*60)
if platform.system() in ["Linux", "Darwin"]:
    peak_memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Linux":
        peak_memory_bytes *= 1024
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    print(f"整个程序运行期间的峰值内存占用: {peak_memory_mb:.2f} MB")
print("="*60)