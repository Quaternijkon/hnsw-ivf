import numpy as np
import faiss
import time
import os
import struct
import resource
import gc

# 数据文件路径
DATA_DIR = "./sift"
LEARN_FILE = os.path.join(DATA_DIR, "learn.fbin")
BASE_FILE = os.path.join(DATA_DIR, "base.fbin")
QUERY_FILE = os.path.join(DATA_DIR, "query.fbin")
GROUNDTRUTH_FILE = os.path.join(DATA_DIR, "groundtruth.ivecs")

def read_fbin(filename, start_idx=0, chunk_size=None):
    """读取.fbin格式的文件"""
    with open(filename, 'rb') as f:
        nvecs = struct.unpack('i', f.read(4))[0]
        dim = struct.unpack('i', f.read(4))[0]
        
        if chunk_size is None:
            data = np.fromfile(f, dtype=np.float32, count=nvecs*dim)
            return data.reshape(nvecs, dim)
        else:
            end_idx = min(start_idx + chunk_size, nvecs)
            num_vectors_in_chunk = end_idx - start_idx
            offset = start_idx * dim * 4
            f.seek(offset, os.SEEK_CUR)
            data = np.fromfile(f, dtype=np.float32, count=num_vectors_in_chunk*dim)
            return data.reshape(num_vectors_in_chunk, dim), nvecs, dim

def read_ivecs(filename):
    """读取.ivecs格式的二进制文件"""
    a = np.fromfile(filename, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

# 获取IO标志，兼容不同Faiss版本
try:
    IO_FLAG_READ_WRITE = faiss.IO_FLAG_READ_WRITE
    IO_FLAG_MMAP = faiss.IO_FLAG_MMAP
except AttributeError:
    try:
        IO_FLAG_READ_WRITE = faiss.index_io.IO_FLAG_READ_WRITE
        IO_FLAG_MMAP = faiss.index_io.IO_FLAG_MMAP
    except AttributeError:
        IO_FLAG_READ_WRITE = 0
        IO_FLAG_MMAP = 4

def main():
    # 固定线程数为1
    faiss.omp_set_num_threads(1)
    print(f"设置FAISS线程数为: {faiss.omp_get_max_threads()}")
    
    # 获取数据集信息
    _, nt, d_train = read_fbin(LEARN_FILE, chunk_size=1)
    _, nb, d_base = read_fbin(BASE_FILE, chunk_size=1)
    _, nq, d_query = read_fbin(QUERY_FILE, chunk_size=1)
    
    if d_train != d_base or d_train != d_query:
        raise ValueError(f"维度不一致: 训练集{d_train}维, 基础集{d_base}维, 查询集{d_query}维")
    
    # 设置参数
    cell_size = 64
    nlist = nb // cell_size
    nprobe = 32
    chunk_size = 100000  # 减小数据块大小以降低内存占用
    k = 10
    M = 32
    efconstruction = 100
    efsearch = 200
    
    # 强制垃圾回收
    gc.collect()
    
    # 索引文件名
    base_name = os.path.splitext(os.path.basename(BASE_FILE))[0]
    clean_base_name = base_name.replace('.', '_')
    INDEX_FILE = os.path.join(DATA_DIR, f"{clean_base_name}_d{d_train}_nlist{nlist}_HNSWM{M}_efc{efconstruction}_IVFFlat.index")
    
    print(f"数据集: {nb}个向量, 维度{d_train}, 查询{nq}个")
    print(f"索引文件: {INDEX_FILE}")
    
    # 构建索引
    if not os.path.exists(INDEX_FILE):
        print("构建索引...")
        
        # 训练量化器
        coarse_quantizer = faiss.IndexHNSWFlat(d_train, M, faiss.METRIC_L2)
        coarse_quantizer.hnsw.efConstruction = efconstruction
        coarse_quantizer.hnsw.efSearch = efsearch
        
        index_for_training = faiss.IndexIVFFlat(coarse_quantizer, d_train, nlist, faiss.METRIC_L2)
        xt = read_fbin(LEARN_FILE)
        index_for_training.train(xt)
        # 立即释放训练数据
        del xt, index_for_training
        gc.collect()  # 强制垃圾回收
        
        # 创建磁盘索引
        index_shell = faiss.IndexIVFFlat(coarse_quantizer, d_train, nlist, faiss.METRIC_L2)
        faiss.write_index(index_shell, INDEX_FILE)
        del index_shell, coarse_quantizer  # 释放量化器
        
        # 分块添加数据
        index_ondisk = faiss.read_index(INDEX_FILE, IO_FLAG_READ_WRITE)
        
        for i in range(0, nb, chunk_size):
            xb_chunk, _, _ = read_fbin(BASE_FILE, i, chunk_size)
            index_ondisk.add(xb_chunk)
            del xb_chunk  # 立即释放数据块
        
        faiss.write_index(index_ondisk, INDEX_FILE)
        del index_ondisk
        gc.collect()  # 强制垃圾回收
        print("索引构建完成")
    else:
        print("使用现有索引")
    
    # 加载索引
    print("\n" + "="*60)
    print("开始性能对比测试")
    print("="*60)
    index_final = faiss.read_index(INDEX_FILE, IO_FLAG_MMAP)
    index_final.nprobe = nprobe
    
    # 设置HNSW搜索参数
    generic_quantizer = index_final.quantizer
    quantizer_hnsw = faiss.downcast_index(generic_quantizer)
    quantizer_hnsw.hnsw.efSearch = efsearch
    
    # ============================================================
    # 方式1: 批量查询（所有query同时查询）
    # ============================================================
    print("\n方式1: 批量查询（分块处理以控制内存）")
    query_chunk_size = min(1000, nq)  # 每次处理1000个查询
    all_I_batch = np.empty((nq, k), dtype=np.int64)  # 预分配结果数组
    
    start_time_batch = time.time()
    for i in range(0, nq, query_chunk_size):
        end_idx = min(i + query_chunk_size, nq)
        xq_chunk, _, _ = read_fbin(QUERY_FILE, i, query_chunk_size)
        D_chunk, I_chunk = index_final.search(xq_chunk, k)
        all_I_batch[i:end_idx] = I_chunk
        del xq_chunk, D_chunk, I_chunk  # 立即释放块数据
    
    batch_search_time = time.time() - start_time_batch
    batch_avg_latency = (batch_search_time / nq) * 1000  # 转换为毫秒
    print(f"  总耗时: {batch_search_time:.4f}秒")
    print(f"  平均时延: {batch_avg_latency:.4f}毫秒/query")
    print(f"  QPS: {nq/batch_search_time:.2f}")
    
    # ============================================================
    # 方式2: 逐条查询（query一条条处理）
    # ============================================================
    print("\n方式2: 逐条查询")
    all_I_single = np.empty((nq, k), dtype=np.int64)  # 预分配结果数组
    
    start_time_single = time.time()
    for i in range(nq):
        xq_single, _, _ = read_fbin(QUERY_FILE, i, 1)
        D_single, I_single = index_final.search(xq_single, k)
        all_I_single[i] = I_single[0]
        del xq_single, D_single, I_single  # 立即释放
    
    single_search_time = time.time() - start_time_single
    single_avg_latency = (single_search_time / nq) * 1000  # 转换为毫秒
    print(f"  总耗时: {single_search_time:.4f}秒")
    print(f"  平均时延: {single_avg_latency:.4f}毫秒/query")
    print(f"  QPS: {nq/single_search_time:.2f}")
    
    # ============================================================
    # 性能对比总结
    # ============================================================
    print("\n" + "="*60)
    print("性能对比总结")
    print("="*60)
    print(f"查询数量: {nq}")
    print(f"\n批量查询:")
    print(f"  - 平均时延: {batch_avg_latency:.4f} 毫秒/query")
    print(f"  - QPS: {nq/batch_search_time:.2f}")
    print(f"\n逐条查询:")
    print(f"  - 平均时延: {single_avg_latency:.4f} 毫秒/query")
    print(f"  - QPS: {nq/single_search_time:.2f}")
    print(f"\n性能提升:")
    speedup = single_search_time / batch_search_time
    latency_reduction = ((single_avg_latency - batch_avg_latency) / single_avg_latency) * 100
    print(f"  - 批量查询速度是逐条查询的 {speedup:.2f}x")
    print(f"  - 时延降低了 {latency_reduction:.2f}%")
    print("="*60)
    
    # 使用批量查询的结果进行后续计算
    all_I = all_I_batch
    
    # 释放索引对象
    del index_final, all_I_single
    gc.collect()  # 强制垃圾回收
    
    # 计算召回率
    if os.path.exists(GROUNDTRUTH_FILE):
        print("计算召回率...")
        total_found = 0
        
        with open(GROUNDTRUTH_FILE, 'rb') as f:
            k_gt = struct.unpack('i', f.read(4))[0]
            
            record_size_bytes = (k_gt + 1) * 4
            
            for i in range(nq):
                offset = i * record_size_bytes
                f.seek(offset)
                record_data = np.fromfile(f, dtype=np.int32, count=k_gt + 1)
                gt_i = record_data[1:]
                found_count = np.isin(all_I[i], gt_i[:k]).sum()
                total_found += found_count
                # 释放临时变量
                del record_data, gt_i, found_count
        
        recall = total_found / (nq * k)
        print(f"Recall@{k}: {recall:.4f}")
        # 释放groundtruth相关变量
        del total_found, recall
    else:
        print("未找到groundtruth文件，跳过召回率计算")
    
    # 释放结果数组，不再需要
    del all_I
    
    # 报告峰值内存使用
    peak_memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    print(f"峰值内存使用: {peak_memory_mb:.2f} MB")

if __name__ == "__main__":
    main()
