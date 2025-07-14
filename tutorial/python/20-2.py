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
DATA_DIR = "./data"
LEARN_FILE = os.path.join(DATA_DIR, "sift_learn.fbin")
BASE_FILE = os.path.join(DATA_DIR, "sift_base.fbin")
QUERY_FILE = os.path.join(DATA_DIR, "sift_query.fbin")
GROUNDTRUTH_FILE = os.path.join(DATA_DIR, "sift_groundtruth.ivecs")

# 调试开关
ENABLE_IVF_STATS = True  # 控制是否输出IVF分区统计信息

# ==============================================================================
# 1. 新增：索引平衡策略配置
# ==============================================================================
# BALANCING_STRATEGY 控制使用哪种聚类平衡策略
# 1: 'PREVENT' (推荐) - 在训练阶段使用平衡K-Means，从根源上防止大聚类。
# 2: 'POST_SPLIT' - 在数据添加后，识别并分裂过大的聚类。
BALANCING_STRATEGY = 'POST_SPLIT'  # 可选 'PREVENT' 或 'POST_SPLIT'
CLUSTER_SIZE_THRESHOLD_RATIO = 2.5 # 仅用于'POST_SPLIT'策略，定义多大的聚类算“过大”（平均值的倍数）


# ==============================================================================
# 2. 辅助函数 (与原版相同)
# ==============================================================================
def read_fbin(filename, start_idx=0, chunk_size=None):
    """
    读取.fbin格式的文件
    格式: [nvecs: int32, dim: int32, data: float32[nvecs*dim]]
    """
    with open(filename, 'rb') as f:
        nvecs_total = struct.unpack('i', f.read(4))[0]
        dim = struct.unpack('i', f.read(4))[0]
        
        # --- 修复逻辑 ---
        # 1. 处理读取整个文件的情况 (chunk_size is None)
        if chunk_size is None:
            data = np.fromfile(f, dtype=np.float32, count=nvecs_total * dim)
            data = data.reshape(nvecs_total, dim)
            return data  # 正确地只返回Numpy数组

        # 2. 处理只获取元数据的情况 (chunk_size == 0)
        if chunk_size == 0:
            return None, nvecs_total, dim

        # 3. 处理读取数据块的情况
        start_offset = 8 + start_idx * dim * 4
        f.seek(start_offset)
        
        num_vectors_to_read = min(chunk_size, nvecs_total - start_idx)
        if num_vectors_to_read <= 0:
            # 如果起始索引超出了范围，返回一个空数组
            return np.array([], dtype=np.float32).reshape(0, dim), nvecs_total, dim

        data = np.fromfile(f, dtype=np.float32, count=num_vectors_to_read * dim)
        data = data.reshape(num_vectors_to_read, dim)
        
        return data, nvecs_total, dim # 正确地为数据块返回元组


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
# 3. 设置参数与环境
# ==============================================================================
_, nt, d_train = read_fbin(LEARN_FILE, chunk_size=0)
_, nb, d_base = read_fbin(BASE_FILE, chunk_size=0)
_, nq, d_query = read_fbin(QUERY_FILE, chunk_size=0)

if d_train != d_base or d_train != d_query:
    raise ValueError(f"维度不一致: 训练集{d_train}维, 基础集{d_base}维, 查询集{d_query}维")

cell_size = 64
nlist = nb // cell_size
chunk_size = 100000
k = 10
CLUSTER_SIZE_THRESHOLD = cell_size * CLUSTER_SIZE_THRESHOLD_RATIO

# 生成基于参数的索引文件名
base_name = os.path.splitext(os.path.basename(BASE_FILE))[0]
clean_base_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name)
INDEX_FILE = os.path.join(DATA_DIR, f"{clean_base_name}_d{d_train}_nlist{nlist}_HNSW32_IVFFlat_balanced.index")

print("="*60)
print("Phase 0: 环境设置")
print(f"向量维度 (d): {d_train}")
print(f"基础集大小 (nb): {nb}, 训练集大小 (ntrain): {nt}")
print(f"查询集大小 (nq): {nq}, 分块大小 (chunk_size): {chunk_size}")
print(f"索引将保存在磁盘文件: {INDEX_FILE}")
print(f"IVF统计功能: {'启用' if ENABLE_IVF_STATS else '禁用'}")
print(f"聚类平衡策略: {BALANCING_STRATEGY}")
if BALANCING_STRATEGY == 'POST_SPLIT':
    print(f"  -> 过大聚类分裂阈值: > {CLUSTER_SIZE_THRESHOLD} 个向量")
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
# 5. 训练量化器
# ==============================================================================
if not skip_index_building:
    print("\nPhase 1: 训练 HNSW 粗量化器")
    coarse_quantizer = faiss.IndexHNSWFlat(d_train, 32, faiss.METRIC_L2)
    index_for_training = faiss.IndexIVFFlat(coarse_quantizer, d_train, nlist, faiss.METRIC_L2)
    index_for_training.verbose = True

    # --- 核心改进 A: 使用平衡K-Means ('PREVENT' 策略) ---
    if BALANCING_STRATEGY == 'PREVENT':
        print("-> 应用 'PREVENT' 策略: 启用平衡K-Means进行训练。")
        index_for_training.cp.balanced = True

    xt = read_fbin(LEARN_FILE)

    print("训练聚类中心并构建 HNSW 量化器...")
    start_time = time.time()
    index_for_training.train(xt)
    end_time = time.time()

    print(f"量化器训练完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"粗量化器中的质心数量: {coarse_quantizer.ntotal}")
    del xt
    
    # 经过训练后，粗量化器已经包含了质心，可以直接使用
    trained_quantizer = index_for_training.quantizer
    del index_for_training
    
    # ==============================================================================
    # 6. 创建空的磁盘索引框架并分块添加数据
    # ==============================================================================
    print("\nPhase 2: 创建空的磁盘索引框架并分块添加数据")
    # 使用训练好的量化器
    index_shell = faiss.IndexIVFFlat(trained_quantizer, d_train, nlist, faiss.METRIC_L2)
    
    print(f"将空的索引框架写入磁盘: {INDEX_FILE}")
    faiss.write_index(index_shell, INDEX_FILE)
    del index_shell

    # 兼容不同Faiss版本的IO标志处理
    # 为新版Faiss设置IO标志
    IO_FLAG_READ_WRITE = 2
    print(f"使用IO标志: {IO_FLAG_READ_WRITE} (读写模式)")

    index_ondisk = faiss.read_index(INDEX_FILE, IO_FLAG_READ_WRITE)
    start_time = time.time()

    sanity_vectors = None
    num_chunks = (nb + chunk_size - 1) // chunk_size
    for i in range(0, nb, chunk_size):
        chunk_idx = i // chunk_size + 1
        print(f"  -> 正在处理块 {chunk_idx}/{num_chunks}: 向量 {i} 到 {min(i+chunk_size, nb)-1}")
        xb_chunk, _, _ = read_fbin(BASE_FILE, i, chunk_size)
        
        if i == 0 and sanity_vectors is None:
            sanity_vectors = xb_chunk[:5].copy()
            
        index_ondisk.add(xb_chunk)
        del xb_chunk

    print(f"\n所有数据块添加完成，总耗时: {time.time() - start_time:.2f} 秒")
    print(f"磁盘索引中的向量总数 (ntotal): {index_ondisk.ntotal}")
    
    # ==============================================================================
    # 7. 新增：后处理分裂大聚类 ('POST_SPLIT' 策略)
    # ==============================================================================
    if BALANCING_STRATEGY == 'POST_SPLIT':
        print("\n" + "="*60)
        print("Phase 2.5: 应用 'POST_SPLIT' 策略，检查并分裂过大的聚类")
        print("警告: 此操作需要将索引加载到内存，可能消耗大量RAM，且耗时较长。")

        # 将 on-disk 索引对象刷新到磁盘并关闭，然后以内存模式重新读取
        print("将索引加载到内存中进行修改...")
        faiss.write_index(index_ondisk, INDEX_FILE)
        del index_ondisk
        index_mem = faiss.read_index(INDEX_FILE)
        
        ivf_mem = faiss.extract_index_ivf(index_mem)
        quantizer = ivf_mem.quantizer

        # 1. 识别需要分裂的聚类
        print("识别过大的聚类...")
        invlists = ivf_mem.invlists
        oversized_list_ids = []
        for list_id in range(ivf_mem.nlist):
            list_size = invlists.list_size(list_id)
            if list_size > CLUSTER_SIZE_THRESHOLD:
                oversized_list_ids.append(list_id)
                print(f"  - 发现过大聚类: ID={list_id}, 大小={list_size} (阈值={CLUSTER_SIZE_THRESHOLD})")

        if not oversized_list_ids:
            print("没有发现过大的聚类，无需处理。")
        else:
            print(f"共发现 {len(oversized_list_ids)} 个过大聚类，开始进行分裂操作...")
            
            total_readded_vectors = 0
            # 循环分裂
            for list_id in oversized_list_ids:
                print(f"  -> 正在分裂聚类 {list_id}...")
                
                # a. 获取该聚类的所有向量ID和向量本身
                ids_in_list = invlists.get_ids(list_id)
                # 使用 reconstruct_n 比逐个 reconstruct 更高效
                list_size = invlists.list_size(list_id)
                vectors_to_split = index_mem.reconstruct_n(list_id, list_size)

                # b. 从索引中移除这些向量（这会清空旧的倒排列表）
                selector = faiss.IDSelectorArray(ids_in_list)
                removed_count = index_mem.remove_ids(selector)
                if removed_count != len(ids_in_list):
                    print(f"     - 警告: 尝试移除{len(ids_in_list)}个向量，实际移除了{removed_count}个。")

                # c. 对这些向量运行 k-means (k=2) 来找到两个新的中心点
                print(f"     - 对 {vectors_to_split.shape[0]} 个向量运行 K-Means (k=2)...")
                kmeans = faiss.Kmeans(d=d_train, k=2, niter=20, verbose=False)
                kmeans.train(vectors_to_split)
                new_centroids = kmeans.centroids

                # d. 将新的质心添加到HNSW量化器中
                quantizer.add(new_centroids)

                # e. 更新IVF索引以容纳新的列表
                new_nlist_total = quantizer.ntotal
                ivf_mem.resize_invlists(new_nlist_total)
                ivf_mem.nlist = new_nlist_total # 关键：更新nlist计数

                # f. 将分裂的向量重新添加回索引（使用它们原始的ID）
                index_mem.add_with_ids(vectors_to_split, ids_in_list)
                total_readded_vectors += len(ids_in_list)
                
                print(f"     - 聚类 {list_id} 分裂完成。向量已重新分配。当前总列表数: {ivf_mem.nlist}")

            print(f"\n分裂操作完成。共重新分配了 {total_readded_vectors} 个向量。")
            print("将平衡后的索引写回磁盘...")
            faiss.write_index(index_mem, INDEX_FILE)
            # 让后续流程使用这个更新后的内存版索引
            index_ondisk = index_mem
        
        print("="*60)


    # ===========================================================
    # 8. Sanity Check
    # ===========================================================
    if sanity_vectors is not None:
        print("\n进行Sanity Check...")
        # 如果经过了分裂操作，需要重新加载索引以确保一致性
        if BALANCING_STRATEGY == 'POST_SPLIT' and 'index_mem' in locals():
            print("使用内存中已平衡的索引进行检查。")
            checker_index = index_mem
        else:
            print("从磁盘加载索引进行检查。")
            checker_index = faiss.read_index(INDEX_FILE)

        D_check, I_check = checker_index.search(sanity_vectors, k)
        passed = all(I_check[j, 0] == j for j in range(5)) and np.all(np.isclose(D_check[:, 0], 0))
        if passed:
            print("Sanity Check 通过: 所有向量的最近邻居都是自身。")
        else:
            print("Sanity Check 警告: 结果不符合预期。")
            print(I_check)
            print(D_check)
        del checker_index
    
    # ===========================================================
    # 9. 输出IVF分区统计信息
    # ===========================================================
    if ENABLE_IVF_STATS and not skip_index_building:
        print("\n输出IVF分区统计信息...")
        # 重新加载最终的索引
        final_index_for_stats = faiss.read_index(INDEX_FILE)
        invlists = faiss.extract_index_ivf(final_index_for_stats).invlists
        current_nlist = final_index_for_stats.nlist
        
        partition_sizes = [invlists.list_size(i) for i in range(current_nlist)]
        non_empty_partitions = [s for s in partition_sizes if s > 0]
        
        print(f"IVF分区统计摘要:")
        print(f"  分区总数: {current_nlist}")
        if non_empty_partitions:
            print(f"  非空分区数: {len(non_empty_partitions)} ({len(non_empty_partitions)/current_nlist*100:.2f}%)")
            print(f"  最大分区大小: {max(non_empty_partitions)}")
            print(f"  最小分区大小: {min(non_empty_partitions)}")
            print(f"  平均分区大小: {np.mean(non_empty_partitions):.2f}")
            print(f"  分区大小标准差: {np.std(non_empty_partitions):.2f}")
        else:
            print("  所有分区都为空。")

        # 写入文件...
        stats_filename = os.path.splitext(INDEX_FILE)[0] + "_ivf_stats.csv"
        with open(stats_filename, 'w') as f:
            f.write("partition_id,vector_count\n")
            for list_id, size in enumerate(partition_sizes):
                f.write(f"{list_id},{size}\n")
        print(f"分区统计信息已保存到: {stats_filename}")
        del final_index_for_stats

    # 保存最终索引（如果之前没有因为后处理而保存）
    if 'index_ondisk' in locals() and BALANCING_STRATEGY != 'POST_SPLIT':
      print(f"\n正在将最终索引写回磁盘: {INDEX_FILE}")
      faiss.write_index(index_ondisk, INDEX_FILE)
      del index_ondisk

# ==============================================================================
# 10. 使用内存映射进行搜索
# ==============================================================================
print("\nPhase 4: 使用内存映射模式进行搜索")
IO_FLAG_MMAP = 4
print(f"以 mmap 模式 (IO_FLAG={IO_FLAG_MMAP}) 打开磁盘索引: {INDEX_FILE}")

index_final = faiss.read_index(INDEX_FILE, IO_FLAG_MMAP)
index_final.nprobe = 32
print(f"索引已准备好搜索 (nprobe={index_final.nprobe}, nlist={index_final.nlist})")

print("从 query.fbin 加载查询向量...")
xq = read_fbin(QUERY_FILE)

print("执行搜索...")
start_time = time.time()
D, I = index_final.search(xq, k)
end_time = time.time()
print(f"搜索完成，耗时: {end_time - start_time:.2f} 秒")

print("\n查询结果的索引 (I[:5])和距离 (D[:5]):")
print("索引 (I):\n", I[:5])
print("距离 (D):\n", D[:5])

# ==============================================================================
# 11. 根据Groundtruth计算召回率 (与原版类似)
# ==============================================================================
print("\n" + "="*60)
print("Phase 5: 计算召回率")

if not os.path.exists(GROUNDTRUTH_FILE):
    print(f"Groundtruth文件未找到: {GROUNDTRUTH_FILE}，跳过计算。")
else:
    gt_ids = read_ivecs(GROUNDTRUTH_FILE)
    if nq != gt_ids.shape[0]:
        print(f"警告: 查询数量({nq})与groundtruth中的数量({gt_ids.shape[0]})不匹配!")

    recall_at_k = (I[:, :k] == gt_ids[:, :1]).sum() / float(nq)
    print(f"Recall@{k}: {recall_at_k:.4f}")

    # 更精确的召回率计算（检查返回的k个结果中有多少在groundtruth的前k个中）
    gt_top_k = gt_ids[:,:k]
    found_count = 0
    for i in range(nq):
        found_count += np.isin(I[i, :k], gt_top_k[i]).sum()
    
    precise_recall = found_count / (nq * k)
    print(f"Precise Recall@{k} (Intersection over Union): {precise_recall:.4f}")

print("="*60)


# ==============================================================================
# 12. 报告峰值内存
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