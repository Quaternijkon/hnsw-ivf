import numpy as np
import faiss
import time
import os
import platform
import resource
import struct
import re
import heapq
import shutil

# ==============================================================================
# 0. 路径和文件名配置 & 动态聚类控制参数
# ==============================================================================
DATA_DIR = "./data"
LEARN_FILE = os.path.join(DATA_DIR, "sift_learn.fbin")
BASE_FILE = os.path.join(DATA_DIR, "sift_base.fbin")
QUERY_FILE = os.path.join(DATA_DIR, "sift_query.fbin")
GROUNDTRUTH_FILE = os.path.join(DATA_DIR, "sift_groundtruth.ivecs")

# 动态聚类控制参数
MAX_CELL_SIZE = 256  # 分区最大容量阈值
SPLIT_FACTOR = 2       # 分裂因子 (分裂为多少个子聚类)
ENABLE_DYNAMIC_SPLITTING = True  # 是否启用动态聚类分裂
# 调试开关 - 设置为True时输出IVF分区统计信息
ENABLE_IVF_STATS = True  # 控制是否输出IVF分区统计信息

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
# 2. 自定义聚类管理器 - 支持动态分裂
# ==============================================================================
class DynamicClusterManager:
    def __init__(self, d, nlist, max_cell_size=MAX_CELL_SIZE, split_factor=SPLIT_FACTOR):
        self.d = d
        self.nlist = nlist
        self.max_cell_size = max_cell_size
        self.split_factor = split_factor
        
        # 分区统计信息
        self.partition_sizes = np.zeros(nlist, dtype=np.int64)
        self.split_counts = np.zeros(nlist, dtype=np.int32)
        
        # 分区大小监控堆
        self.size_heap = []
        self.partition_map = {}  # 分区ID到堆项的映射
        
        # 初始化堆
        for i in range(nlist):
            self._add_to_heap(i, 0)
    
    def _add_to_heap(self, partition_id, size):
        """添加分区到大小监控堆"""
        # 使用负大小构建最大堆
        entry = [-size, partition_id]
        heapq.heappush(self.size_heap, entry)
        self.partition_map[partition_id] = entry
    
    def _update_heap(self, partition_id, new_size):
        """更新分区在堆中的大小"""
        if partition_id in self.partition_map:
            # 标记旧条目为无效
            old_entry = self.partition_map[partition_id]
            old_entry[1] = -1  # 标记为无效
            
            # 添加新条目
            new_entry = [-new_size, partition_id]
            heapq.heappush(self.size_heap, new_entry)
            self.partition_map[partition_id] = new_entry
    
    def get_largest_partition(self):
        """获取当前最大的分区"""
        while self.size_heap:
            # 获取堆顶元素
            neg_size, partition_id = self.size_heap[0]
            size = -neg_size
            
            # 如果条目有效，返回分区
            if partition_id != -1:
                return partition_id, size
            
            # 弹出无效条目
            heapq.heappop(self.size_heap)
        
        return None, 0
    
    def update_partition(self, partition_id, delta):
        """更新分区大小并检查是否需要分裂"""
        new_size = self.partition_sizes[partition_id] + delta
        self.partition_sizes[partition_id] = new_size
        
        # 更新堆
        self._update_heap(partition_id, new_size)
        
        # 检查是否需要分裂
        if new_size > self.max_cell_size:
            return True
        return False
    
    def split_partition(self, partition_id, vectors, coarse_quantizer):
        """分裂过大的分区"""
        print(f"  -> 分区 {partition_id} 过大 ({self.partition_sizes[partition_id]} 个向量)，进行分裂...")
        
        # 记录分裂前大小
        original_size = self.partition_sizes[partition_id]
        
        # 1. 对分区内的向量进行子聚类
        kmeans = faiss.Kmeans(self.d, self.split_factor, niter=10)
        kmeans.train(vectors)
        
        # 2. 添加新聚类中心到粗量化器
        new_centroids = kmeans.centroids
        coarse_quantizer.add(new_centroids)
        
        # 3. 更新分区统计
        new_partition_ids = np.arange(coarse_quantizer.ntotal - self.split_factor, coarse_quantizer.ntotal)
        
        # 重置原分区大小
        self.partition_sizes[partition_id] = 0
        self._update_heap(partition_id, 0)
        
        # 添加新分区
        for new_id in new_partition_ids:
            self.partition_sizes = np.append(self.partition_sizes, 0)
            self.split_counts = np.append(self.split_counts, self.split_counts[partition_id] + 1)
            self._add_to_heap(new_id, 0)
        
        # 4. 更新nlist
        self.nlist += self.split_factor - 1
        
        print(f"  -> 分区 {partition_id} 分裂为 {self.split_factor} 个子分区: {new_partition_ids}")
        print(f"  -> 新分区数量: {self.nlist}")
        
        return new_partition_ids
    
    def get_stats(self):
        """获取分区统计信息"""
        return {
            "total_partitions": self.nlist,
            "max_size": np.max(self.partition_sizes),
            "min_size": np.min(self.partition_sizes),
            "avg_size": np.mean(self.partition_sizes),
            "split_counts": self.split_counts
        }
    
    def save_stats(self, filename):
        """保存分区统计到文件"""
        with open(filename, 'w') as f:
            f.write("partition_id,vector_count,split_count\n")
            for i in range(len(self.partition_sizes)):
                f.write(f"{i},{self.partition_sizes[i]},{self.split_counts[i]}\n")

# ==============================================================================
# 3. 设置参数与环境
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

# 生成基于参数的索引文件名
base_name = os.path.splitext(os.path.basename(BASE_FILE))[0]
clean_base_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name)
index_suffix = f"d{d_train}_initNlist{nlist}_maxCell{MAX_CELL_SIZE}"
if ENABLE_DYNAMIC_SPLITTING:
    index_suffix += "_dynamic"
INDEX_FILE = os.path.join(DATA_DIR, f"{clean_base_name}_{index_suffix}.index")

print("="*60)
print("Phase 0: 环境设置")
print(f"向量维度 (d): {d_train}")
print(f"基础集大小 (nb): {nb}, 训练集大小 (ntrain): {nt}")
print(f"查询集大小 (nq): {nq}, 分块大小 (chunk_size): {chunk_size}")
print(f"初始分区数 (nlist): {nlist}")
print(f"最大分区大小: {MAX_CELL_SIZE}, 分裂因子: {SPLIT_FACTOR}")
print(f"动态聚类分裂: {'启用' if ENABLE_DYNAMIC_SPLITTING else '禁用'}")
print(f"索引将保存在磁盘文件: {INDEX_FILE}")
print(f"IVF统计功能: {'启用' if ENABLE_IVF_STATS else '禁用'}")
print("="*60)

# ==============================================================================
# 4. 检查索引文件是否存在
# ==============================================================================
if os.path.exists(INDEX_FILE):
    print(f"索引文件 {INDEX_FILE} 已存在，跳过索引构建阶段")
    skip_index_building = True
else:
    print("索引文件不存在，将构建新索引")
    skip_index_building = False

# ==============================================================================
# 5. 训练量化器 (使用learn.fbin的前ntrain个向量)
# ==============================================================================
if not skip_index_building:
    print("\nPhase 1: 训练 HNSW 粗量化器 (in-memory)")
    coarse_quantizer = faiss.IndexHNSWFlat(d_train, 32, faiss.METRIC_L2)
    
    # 初始化动态聚类管理器
    cluster_manager = DynamicClusterManager(
        d=d_train, 
        nlist=nlist,
        max_cell_size=MAX_CELL_SIZE,
        split_factor=SPLIT_FACTOR
    )
    
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
    # 6. 创建一个空的、基于磁盘的索引框架
    # ==============================================================================
    print("\nPhase 2: 创建空的磁盘索引框架")
    index_shell = faiss.IndexIVFFlat(coarse_quantizer, d_train, cluster_manager.nlist, faiss.METRIC_L2)
    print(f"将空的索引框架写入磁盘: {INDEX_FILE}")
    faiss.write_index(index_shell, INDEX_FILE)
    del index_shell

    # ==============================================================================
    # 7. 分块向磁盘索引中添加数据 (从base.fbin)，支持动态聚类分裂
    # ==============================================================================
    print("\nPhase 3: 分块添加数据到磁盘索引 (支持动态聚类分裂)")

    # 兼容不同Faiss版本的IO标志处理
    try:
        IO_FLAG_READ_WRITE = faiss.IO_FLAG_READ_WRITE
    except AttributeError:
        try:
            IO_FLAG_READ_WRITE = faiss.index_io.IO_FLAG_READ_WRITE
        except AttributeError:
            IO_FLAG_READ_WRITE = 0

    print(f"使用IO标志: {IO_FLAG_READ_WRITE} (读写模式)")

    # 创建一个临时索引文件用于构建
    TEMP_INDEX_FILE = INDEX_FILE + ".temp"
    if os.path.exists(TEMP_INDEX_FILE):
        os.remove(TEMP_INDEX_FILE)
    
    # 复制初始索引到临时文件
    shutil.copy(INDEX_FILE, TEMP_INDEX_FILE)
    
    # 打开临时索引文件进行读写
    index_ondisk = faiss.read_index(TEMP_INDEX_FILE, IO_FLAG_READ_WRITE)
    start_time = time.time()

    # 保存前5个向量用于Sanity Check
    sanity_vectors = None
    vectors_to_reassign = []  # 存储需要重新分配的向量
    
    num_chunks = (nb + chunk_size - 1) // chunk_size
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, nb)
        actual_chunk_size = end_idx - start_idx
        
        print(f"\n  -> 处理块 {chunk_idx+1}/{num_chunks}: 向量 {start_idx} 到 {end_idx-1} ({actual_chunk_size} 个向量)")
        
        # 从base.fbin中读取数据块
        xb_chunk, _, _ = read_fbin(BASE_FILE, start_idx, actual_chunk_size)
        
        # 如果是第一个块，保存前5个向量
        if start_idx == 0 and sanity_vectors is None:
            sanity_vectors = xb_chunk[:5].copy()
        
        # 添加当前块到索引
        index_ondisk.add(xb_chunk)
        
        # 获取分配给每个向量的分区ID
        _, partition_ids = coarse_quantizer.search(xb_chunk, 1)
        partition_ids = partition_ids.flatten()
        
        # 更新分区统计
        for pid in np.unique(partition_ids):
            count = np.sum(partition_ids == pid)
            if cluster_manager.update_partition(pid, count) and ENABLE_DYNAMIC_SPLITTING:
                # 获取分区中的所有向量
                partition_vectors = xb_chunk[partition_ids == pid]
                
                # 分裂分区
                new_pids = cluster_manager.split_partition(
                    pid, 
                    partition_vectors, 
                    coarse_quantizer
                )
                
                # 更新索引结构
                print(f"  -> 调整索引结构: nlist 从 {index_ondisk.nlist} -> {cluster_manager.nlist}")
                index_ondisk.nlist = cluster_manager.nlist
                # ！！！关键修复：显式调整内部倒排列表的大小以匹配新的nlist
                index_ondisk.invlists.resize(new_size=cluster_manager.nlist)
                
                # 标记需要重新分配的向量
                vectors_to_reassign.extend(partition_vectors.tolist())
        
        # 重新分配之前分裂的向量
        if vectors_to_reassign:
            print(f"  -> 重新分配 {len(vectors_to_reassign)} 个向量到新分区...")
            reassign_array = np.array(vectors_to_reassign, dtype=np.float32)
            
            # 先移除旧向量
            # 注意: Faiss没有直接支持按ID移除向量，所以我们重新添加它们
            index_ondisk.add(reassign_array)  # 重新添加到索引
            vectors_to_reassign = []  # 清空列表
            
            # 更新分区统计 (重新分配后)
            _, reassign_pids = coarse_quantizer.search(reassign_array, 1)
            reassign_pids = reassign_pids.flatten()
            for pid in np.unique(reassign_pids):
                count = np.sum(reassign_pids == pid)
                cluster_manager.update_partition(pid, count)
        
        # 报告当前最大分区
        largest_pid, largest_size = cluster_manager.get_largest_partition()
        print(f"  当前最大分区: {largest_pid} ({largest_size} 个向量)")
        
        del xb_chunk

    print(f"\n所有数据块添加完成，总耗时: {time.time() - start_time:.2f} 秒")
    print(f"最终分区数量: {cluster_manager.nlist}")
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
            if list_size > 0:
                non_empty_partitions += 1
                max_size = max(max_size, list_size)
                min_size = min(min_size, list_size)
                total_vectors += list_size
                partition_stats.append((list_id, list_size))
        
        # 计算平均值
        avg_size = total_vectors / non_empty_partitions if non_empty_partitions > 0 else 0
        
        # 输出统计摘要
        print(f"IVF分区统计摘要:")
        print(f"  分区总数: {nlist}")
        print(f"  非空分区数: {non_empty_partitions} ({non_empty_partitions/nlist*100:.2f}%)")
        print(f"  最大分区大小: {max_size}")
        print(f"  最小分区大小: {min_size}")
        print(f"  平均分区大小: {avg_size:.2f}")
        
        # 将详细统计信息写入文件
        stats_filename = os.path.splitext(INDEX_FILE)[0] + "_ivf_stats.csv"
        cluster_manager.save_stats(stats_filename)
        with open(stats_filename, 'w') as f:
            f.write("partition_id,vector_count\n")
            for list_id, size in partition_stats:
                f.write(f"{list_id},{size}\n")
        
        print(f"分区统计信息已保存到: {stats_filename}")
        print(f"统计耗时: {time.time() - start_stats_time:.2f}秒")
    
    # 保存索引到磁盘
    print(f"正在将最终索引写回磁盘: {INDEX_FILE}")
    faiss.write_index(index_ondisk, INDEX_FILE)
    
    # 关闭并删除临时索引
    del index_ondisk
    if os.path.exists(TEMP_INDEX_FILE):
        os.remove(TEMP_INDEX_FILE)
        print(f"临时索引文件已删除: {TEMP_INDEX_FILE}")

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

# index_final = faiss.read_index(INDEX_FILE, IO_FLAG_MMAP)
index_final = faiss.read_index(INDEX_FILE, IO_FLAG_MMAP)
index_final.nprobe = 32
print(f"索引已准备好搜索 (nprobe={index_final.nprobe})")
print(f"索引中的分区数量 (nlist): {index_final.nlist}")

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
            
            # I[i] 是我们搜索得到的k个结果ID
            # gt_i 是当前查询的groundtruth ID
            # 使用np.isin高效地计算交集的大小
            found_count = np.isin(I[i], gt_i).sum()
            total_found += found_count
            
    # 召回率 = (所有查询找到的正确近邻总数) / (所有查询返回的结果总数)
    recall = total_found / (nq * k)
    
    print(f"\n查询了 {nq} 个向量, k={k}")
    print(f"在top-{k}的结果中，总共找到了 {total_found} 个真实的近邻。")
    print(f"Recall@{k}: {recall:.4f}")

print("="*60)


# ==============================================================================
# 10. 报告峰值内存并清理
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

# 注释掉删除索引文件的代码，以便后续重用
# print(f"\n清理临时索引文件: {INDEX_FILE}")
# os.remove(INDEX_FILE)