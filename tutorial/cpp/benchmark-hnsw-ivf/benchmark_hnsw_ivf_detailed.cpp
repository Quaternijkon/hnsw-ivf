#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <regex>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <sys/resource.h>
#include <unistd.h>
#include <iomanip>
#include <numeric>
#include <omp.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/utils/Heap.h>
#include <faiss/impl/io.h>
#include <faiss/impl/FaissAssert.h>
#include "../config_parser.h"

using namespace std;
using faiss::QueryLatencyStats;
using idx_t = faiss::idx_t;

// 延迟统计结构体
struct LatencyStats {
    double mean_latency_ms = 0.0;
    double p50_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
    double p95_latency_ms = 0.0;
};

// 单次测试结果结构体
struct TestResult {
    // Build结果
    double training_memory_mb = 0.0;
    double add_memory_mb = 0.0;
    double training_time_s = 0.0;
    double total_time_s = 0.0;
    
    // Search结果
    double recall = 0.0;
    double qps = 0.0;
    double mspq = 0.0;  // 每查询使用多少毫秒 (milliseconds per query)
    double search_memory_mb = 0.0;
    double search_time_s = 0.0;
    LatencyStats latency;
    
    // 参数
    int M = 0;
    int nlist = 0;
    int efconstruction = 0;
    int nprobe = 0;
    int efsearch = 0;
};

// 全局变量
const string DATA_DIR = "../sift";
const string LEARN_FILE = DATA_DIR + "/learn.fbin";
const string BASE_FILE = DATA_DIR + "/base.fbin";
const string QUERY_FILE = DATA_DIR + "/query.fbin";
const string GROUNDTRUTH_FILE = DATA_DIR + "/groundtruth.ivecs";

// 辅助函数
pair<vector<float>, pair<size_t, size_t>> read_fbin(const string& filename, size_t start_idx = 0, size_t chunk_size = 0) {
    ifstream f(filename, ios::binary);
    if (!f.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }

    int32_t nvecs_raw, dim_raw;
    f.read(reinterpret_cast<char*>(&nvecs_raw), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&dim_raw), sizeof(int32_t));
    size_t nvecs = static_cast<size_t>(nvecs_raw);
    size_t dim = static_cast<size_t>(dim_raw);

    size_t num_vectors_in_chunk = nvecs;
    if (chunk_size > 0) {
        size_t end_idx = min(start_idx + chunk_size, nvecs);
        num_vectors_in_chunk = end_idx - start_idx;
        size_t offset = 8 + start_idx * dim * sizeof(float);
        f.seekg(offset, ios::beg);
    }

    vector<float> data(num_vectors_in_chunk * dim);
    f.read(reinterpret_cast<char*>(data.data()), num_vectors_in_chunk * dim * sizeof(float));

    return {data, {nvecs, dim}};
}

vector<vector<int32_t>> read_ivecs(const string& filename) {
    ifstream f(filename, ios::binary | ios::ate);
    if (!f.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }
    size_t file_size = f.tellg();
    f.seekg(0, ios::beg);

    vector<int32_t> a(file_size / sizeof(int32_t));
    f.read(reinterpret_cast<char*>(a.data()), file_size);

    int32_t d = a[0];
    vector<vector<int32_t>> result(a.size() / (d + 1));
    for (size_t i = 0, j = 0; i < result.size(); ++i) {
        result[i].resize(d);
        ++j; // skip dim
        copy(a.begin() + j, a.begin() + j + d, result[i].begin());
        j += d;
    }
    return result;
}

// 获取当前实际内存使用量（通过/proc/self/status）
long getCurrentMemoryMB() {
    ifstream status_file("/proc/self/status");
    string line;
    while (getline(status_file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            istringstream iss(line);
            string key, value, unit;
            iss >> key >> value >> unit;
            return stol(value) / 1024; // 转换为MB
        }
    }
    return 0;
}

// 监控峰值内存的类
class PeakMemoryMonitor {
private:
    long start_memory_mb;
    long peak_memory_mb;
    bool monitoring;
    
public:
    PeakMemoryMonitor() : start_memory_mb(0), peak_memory_mb(0), monitoring(false) {}
    
    void start() {
        start_memory_mb = getCurrentMemoryMB();
        peak_memory_mb = start_memory_mb;
        monitoring = true;
    }
    
    void update() {
        if (monitoring) {
            long current_memory = getCurrentMemoryMB();
            peak_memory_mb = max(peak_memory_mb, current_memory);
        }
    }
    
    long getPeakMemoryMB() {
        return peak_memory_mb;
    }
    
    long getMemoryIncrease() {
        return peak_memory_mb - start_memory_mb;
    }
    
    void stop() {
        monitoring = false;
    }
};

// 计算延迟统计
LatencyStats calculateLatencyStats(const vector<double>& latencies_ms) {
    LatencyStats stats;
    if (latencies_ms.empty()) return stats;
    
    vector<double> sorted_latencies = latencies_ms;
    sort(sorted_latencies.begin(), sorted_latencies.end());
    
    stats.mean_latency_ms = accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) / latencies_ms.size();
    stats.p50_latency_ms = sorted_latencies[sorted_latencies.size() / 2];
    stats.p95_latency_ms = sorted_latencies[static_cast<size_t>(sorted_latencies.size() * 0.95)];
    stats.p99_latency_ms = sorted_latencies[static_cast<size_t>(sorted_latencies.size() * 0.99)];
    
    return stats;
}

// 计算召回率
double calculateRecall(const vector<idx_t>& search_results, const vector<vector<int32_t>>& groundtruth, 
                      size_t nq, size_t k) {
    if (groundtruth.empty()) return 0.0;
    
    size_t total_found = 0;
    for (size_t i = 0; i < nq; ++i) {
        size_t found_count = 0;
        size_t check_size = min(k, static_cast<size_t>(groundtruth[i].size()));
        
        for (size_t j = 0; j < k; ++j) {
            idx_t neighbor = search_results[i * k + j];
            if (find(groundtruth[i].begin(), groundtruth[i].begin() + check_size, neighbor) != 
                groundtruth[i].begin() + check_size) {
                ++found_count;
            }
        }
        total_found += found_count;
    }
    
    return static_cast<double>(total_found) / (nq * k);
}

// 执行build测试，返回索引文件名（on-disk模式，索引保存到磁盘）
pair<TestResult, string> runBuildTest(int M, int nlist, int efconstruction, size_t d, size_t nt, size_t nb) {
    TestResult result;
    result.M = M;
    result.nlist = nlist;
    result.efconstruction = efconstruction;
    
    cout << "\n=== Build测试: M=" << M << ", nlist=" << nlist << ", efconstruction=" << efconstruction << " ===" << endl;
    
    auto total_start_time = chrono::high_resolution_clock::now();
    
    // 开始监控峰值内存
    PeakMemoryMonitor memory_monitor;
    memory_monitor.start();
    
    // 创建HNSW量化器
    faiss::IndexHNSWFlat* coarse_quantizer = new faiss::IndexHNSWFlat(d, M);
    coarse_quantizer->hnsw.efConstruction = efconstruction;
    coarse_quantizer->hnsw.efSearch = 16; // 默认efsearch
    memory_monitor.update();
    
    // 创建IVF索引
    faiss::IndexIVFFlat* index = new faiss::IndexIVFFlat(coarse_quantizer, d, nlist, faiss::METRIC_L2);
    index->verbose = false;
    index->own_fields = true; // 让index管理quantizer的内存
    memory_monitor.update();
    
    // 训练阶段
    auto [xt_data, _] = read_fbin(LEARN_FILE);
    float* xt = xt_data.data();
    memory_monitor.update();
    
    auto train_start = chrono::high_resolution_clock::now();
    index->train(nt, xt);
    auto train_end = chrono::high_resolution_clock::now();
    memory_monitor.update();
    
    result.training_time_s = chrono::duration<double>(train_end - train_start).count();
    result.training_memory_mb = memory_monitor.getPeakMemoryMB();
    
    cout << "训练完成，用时: " << fixed << setprecision(2) << result.training_time_s << "s" << endl;
    cout << "训练阶段峰值内存: " << fixed << setprecision(2) << result.training_memory_mb << "MB" << endl;
    
    // 释放训练数据
    xt_data.clear();
    xt_data.shrink_to_fit();
    memory_monitor.update();
    
    auto add_start = chrono::high_resolution_clock::now();
    
    // 记录添加前的内存
    long memory_before_add = memory_monitor.getPeakMemoryMB();
    
    // 分块添加数据以减少内存占用
    size_t chunk_size = 100000; // 每次添加10万个向量
    size_t num_chunks = (nb + chunk_size - 1) / chunk_size;
    
    cout << "开始分块添加数据，共 " << num_chunks << " 块" << endl;
    cout << "  -> 添加前内存: " << fixed << setprecision(2) << memory_before_add << "MB" << endl;
    cout << "  -> ⚠️  注意: IndexIVFFlat在构建时会将所有向量存储在内存中的倒排列表" << endl;
    cout << "  -> ⚠️  即使分块读取，向量数据也会累积在索引对象中，直到保存到磁盘" << endl;
    
    for (size_t i = 0; i < nb; i += chunk_size) {
        auto [xb_chunk_data, __] = read_fbin(BASE_FILE, i, chunk_size);
        float* xb_chunk = xb_chunk_data.data();
        size_t current_chunk_size = min(chunk_size, nb - i);
        
        long memory_before_chunk = memory_monitor.getPeakMemoryMB();
        
        index->add(current_chunk_size, xb_chunk);
        memory_monitor.update();
        
        long memory_after_chunk = memory_monitor.getPeakMemoryMB();
        long chunk_memory = memory_after_chunk - memory_before_chunk;
        
        // 及时释放chunk数据
        xb_chunk_data.clear();
        xb_chunk_data.shrink_to_fit();
        
        // 计算已添加向量数和内存估算
        size_t vectors_added = min(i + chunk_size, nb);
        size_t vectors_memory_mb = (vectors_added * 128 * 4) / (1024 * 1024); // 已添加向量的理论内存
        
        if ((i / chunk_size + 1) % 10 == 0 || (i / chunk_size + 1) == num_chunks) {
            cout << "已添加 " << (i / chunk_size + 1) << "/" << num_chunks << " 块 (" 
                 << vectors_added << " 向量, 理论内存~" << vectors_memory_mb << "MB, 实际内存" 
                 << fixed << setprecision(2) << memory_after_chunk << "MB, +" << chunk_memory << "MB)" << endl;
        }
    }
    
    auto add_end = chrono::high_resolution_clock::now();
    double add_time = chrono::duration<double>(add_end - add_start).count();
    
    long memory_after_add = memory_monitor.getPeakMemoryMB();
    long total_add_memory = memory_after_add - memory_before_add;
    
    // 计算理论内存占用
    size_t total_vectors_memory_mb = (nb * 128 * 4) / (1024 * 1024); // 所有向量的理论内存
    size_t ids_memory_mb = (nb * sizeof(idx_t)) / (1024 * 1024); // ID列表内存
    
    cout << "数据添加完成，用时: " << fixed << setprecision(2) << add_time << "s" << endl;
    cout << "  -> 添加后内存: " << fixed << setprecision(2) << memory_after_add << "MB (+" 
         << total_add_memory << "MB)" << endl;
    cout << "  -> 理论内存占用: 向量数据~" << total_vectors_memory_mb << "MB, ID列表~" 
         << ids_memory_mb << "MB, 总计~" << (total_vectors_memory_mb + ids_memory_mb) << "MB" << endl;
    cout << "  -> 实际内存占用还包括: HNSW量化器、索引元数据、内存对齐开销等" << endl;
    
    result.add_memory_mb = memory_monitor.getPeakMemoryMB();
    
    // 将索引保存到磁盘（on-disk模式，减少搜索时的内存占用）
    string index_filename = DATA_DIR + "/temp_index_hnsw_ivf_M" + to_string(M) + "_nlist" + to_string(nlist) + "_efC" + to_string(efconstruction) + ".index";
    cout << "正在将索引保存到磁盘: " << index_filename << endl;
    faiss::write_index(index, index_filename.c_str());
    
    // 删除内存中的索引，释放内存
    delete index;
    index = nullptr;
    memory_monitor.update();
    
    auto total_end_time = chrono::high_resolution_clock::now();
    result.total_time_s = chrono::duration<double>(total_end_time - total_start_time).count();
    
    cout << "总时间: " << fixed << setprecision(2) << result.total_time_s << "s" << endl;
    cout << "添加数据阶段峰值内存: " << fixed << setprecision(2) << result.add_memory_mb << "MB" << endl;
    cout << "索引已保存到磁盘并释放内存" << endl;
    
    return {result, index_filename};
}

// 执行search测试（on-disk模式，使用mmap）- 内存优化版
TestResult runSearchTest(int nprobe, int efsearch, const TestResult& build_result, 
                        const string& index_filename, size_t /*d*/, size_t nq, size_t k) {
    TestResult result = build_result;
    result.nprobe = nprobe;
    result.efsearch = efsearch;
    
    cout << "\n=== Search测试 (On-Disk mmap模式): nprobe=" << nprobe << ", efsearch=" << efsearch << " ===" << endl;
    
    // 开始监控搜索阶段的峰值内存
    PeakMemoryMonitor search_memory_monitor;
    search_memory_monitor.start();
    
    // 记录索引加载前的内存
    long memory_before_index = search_memory_monitor.getPeakMemoryMB();
    
    // 使用内存映射加载索引（on-disk模式，减少物理内存占用）
    int IO_FLAG_MMAP = faiss::IO_FLAG_MMAP;
    faiss::Index* index = faiss::read_index(index_filename.c_str(), IO_FLAG_MMAP);
    search_memory_monitor.update();
    
    long memory_after_index = search_memory_monitor.getPeakMemoryMB();
    long index_memory = memory_after_index - memory_before_index;
    
    cout << "索引已通过mmap加载: " << index_filename << endl;
    cout << "  -> 索引加载后内存: " << fixed << setprecision(2) << memory_after_index << "MB (+" << index_memory << "MB)" << endl;
    
    // 设置搜索参数
    faiss::IndexIVF* index_ivf = dynamic_cast<faiss::IndexIVF*>(index);
    index_ivf->nprobe = nprobe;
    faiss::IndexHNSW* quantizer_hnsw = dynamic_cast<faiss::IndexHNSW*>(index_ivf->quantizer);
    
    // 分析HNSW量化器的内存占用
    if (quantizer_hnsw) {
        size_t nlist_actual = quantizer_hnsw->ntotal; // 量化器中的节点数（等于nlist）
        size_t M_actual = build_result.M; // M参数从构建结果获取
        size_t max_level = quantizer_hnsw->hnsw.max_level;
        
        // 获取实际的图结构信息
        size_t neighbors_size = quantizer_hnsw->hnsw.neighbors.size();
        size_t offsets_size = quantizer_hnsw->hnsw.offsets.size();
        size_t levels_size = quantizer_hnsw->hnsw.levels.size();
        
        // 估算HNSW量化器的内存占用
        // 向量数据：nlist × d × 4字节（128维，但实际可能从storage获取）
        size_t vector_data_bytes = nlist_actual * 128 * 4;
        size_t vector_data_mb = vector_data_bytes / (1024 * 1024);
        
        // 图结构实际内存：
        // - neighbors数组：存储所有邻居ID，每个4字节（storage_idx_t = int32_t）
        size_t neighbors_bytes = neighbors_size * sizeof(int32_t);
        // - offsets数组：存储偏移量，每个8字节（size_t）
        size_t offsets_bytes = offsets_size * sizeof(size_t);
        // - levels数组：存储层级，每个4字节（int）
        size_t levels_bytes = levels_size * sizeof(int);
        // - 其他元数据（粗略估算）
        size_t other_metadata_bytes = (nlist_actual * 64); // 粗略估算其他开销
        
        size_t graph_structure_bytes = neighbors_bytes + offsets_bytes + levels_bytes + other_metadata_bytes;
        size_t graph_structure_mb = graph_structure_bytes / (1024 * 1024);
        
        size_t total_quantizer_estimate_mb = vector_data_mb + graph_structure_mb;
        
        cout << "  -> HNSW量化器信息: nlist=" << nlist_actual << ", M=" << M_actual 
             << ", max_level=" << max_level << ", neighbors=" << neighbors_size << endl;
        cout << "  -> HNSW量化器实际内存: 向量数据~" << vector_data_mb 
             << "MB, 图结构~" << graph_structure_mb << "MB (neighbors=" 
             << (neighbors_bytes / (1024 * 1024)) << "MB, offsets=" 
             << (offsets_bytes / (1024 * 1024)) << "MB, levels=" 
             << (levels_bytes / (1024 * 1024)) << "MB)" << endl;
        cout << "  -> HNSW量化器总计估算: ~" << total_quantizer_estimate_mb << "MB" << endl;
        cout << "  -> ⚠️  注意: HNSW图结构即使使用mmap也可能被完全加载到内存（需要频繁随机访问）" << endl;
        cout << "  -> ⚠️  这是HNSW量化器内存占用的主要原因，无法通过mmap完全避免" << endl;
    }
    
    quantizer_hnsw->hnsw.efSearch = efsearch;
    
    // 设置并行模式
    index_ivf->parallel_mode = 0; // 使用OpenMP并行
    
    // 加载查询数据
    auto [xq_data, _] = read_fbin(QUERY_FILE);
    float* xq = xq_data.data();
    search_memory_monitor.update();
    
    long memory_after_query = search_memory_monitor.getPeakMemoryMB();
    long query_memory = memory_after_query - memory_after_index;
    cout << "  -> 查询数据加载后内存: " << fixed << setprecision(2) << memory_after_query << "MB (+" << query_memory << "MB)" << endl;
    
    // 分析IVF倒排列表信息（在搜索前）
    if (index_ivf) {
        size_t nlist_actual = index_ivf->nlist;
        size_t ntotal_actual = index_ivf->ntotal;
        size_t nprobe_actual = index_ivf->nprobe;
        
        // 估算访问的倒排列表大小
        // 假设均匀分布：每个列表约 ntotal/nlist 个向量
        size_t vectors_per_list = (ntotal_actual + nlist_actual - 1) / nlist_actual; // 向上取整
        size_t lists_to_access = min(nprobe_actual, nlist_actual);
        size_t estimated_vectors_accessed = lists_to_access * vectors_per_list;
        
        // 每个向量：128维 × 4字节 + ID（8字节）
        size_t vector_data_bytes = estimated_vectors_accessed * 128 * 4;
        size_t id_data_bytes = estimated_vectors_accessed * sizeof(idx_t);
        size_t estimated_list_memory_mb = (vector_data_bytes + id_data_bytes) / (1024 * 1024);
        
        cout << "  -> IVF索引信息: nlist=" << nlist_actual << ", ntotal=" << ntotal_actual 
             << ", nprobe=" << nprobe_actual << endl;
        cout << "  -> 估算访问: " << lists_to_access << " 个列表, ~" << estimated_vectors_accessed 
             << " 向量, 理论内存~" << estimated_list_memory_mb << "MB" << endl;
        cout << "  -> ⚠️  注意: mmap模式下，操作系统可能预加载大量页面，实际内存可能远超估算" << endl;
    }
    
    // 执行搜索
    vector<idx_t> I(nq * k);
    vector<float> D(nq * k);
    vector<QueryLatencyStats> latency_stats(nq);
    search_memory_monitor.update();
    
    long memory_before_search = search_memory_monitor.getPeakMemoryMB();
    cout << "  -> 搜索开始前内存: " << fixed << setprecision(2) << memory_before_search << "MB" << endl;
    
    auto search_start = chrono::high_resolution_clock::now();
    index_ivf->search_stats(nq, xq, k, D.data(), I.data(), nullptr, latency_stats.data());
    auto search_end = chrono::high_resolution_clock::now();
    search_memory_monitor.update();
    
    long memory_after_search = search_memory_monitor.getPeakMemoryMB();
    long memory_during_search = memory_after_search - memory_before_search;
    
    result.search_time_s = chrono::duration<double>(search_end - search_start).count();
    
    cout << "  -> 搜索完成后内存: " << fixed << setprecision(2) << memory_after_search << "MB (+" << memory_during_search << "MB)" << endl;
    if (memory_during_search > 100) {
        cout << "  -> ⚠️  搜索期间内存增长: +" << memory_during_search << "MB（远超理论估算）" << endl;
        cout << "  -> 可能原因：" << endl;
        cout << "     1. 操作系统mmap页面预加载（访问模式触发大量页面加载）" << endl;
        cout << "     2. Faiss内部缓冲区（距离计算、结果排序等）" << endl;
        cout << "     3. 倒排列表可能被完整加载到物理内存（而非按需加载）" << endl;
        cout << "     4. 内存对齐和碎片化开销" << endl;
        cout << "  -> 这是mmap的系统级行为，无法完全避免" << endl;
    }
    result.qps = nq / result.search_time_s;
    result.mspq = (result.search_time_s * 1000.0) / nq;  // 转换为毫秒
    
    // 记录搜索阶段的峰值内存（在加载groundtruth之前）
    result.search_memory_mb = search_memory_monitor.getPeakMemoryMB();
    
    // 停止内存监控，后续操作（如加载groundtruth）不计入搜索阶段内存
    search_memory_monitor.stop();
    
    // 计算延迟统计
    vector<double> latencies_ms;
    for (size_t i = 0; i < nq; ++i) {
        latencies_ms.push_back(latency_stats[i].total_us / 1000.0);
    }
    result.latency = calculateLatencyStats(latencies_ms);
    
    // 及时删除索引释放内存（在加载groundtruth之前）
    delete index;
    
    // 计算召回率（在内存监控停止后，不计入搜索阶段内存）
    ifstream gt_file_check(GROUNDTRUTH_FILE);
    if (gt_file_check.good()) {
        vector<vector<int32_t>> groundtruth = read_ivecs(GROUNDTRUTH_FILE);
        result.recall = calculateRecall(I, groundtruth, nq, k);
    }
    
    cout << "搜索时间: " << fixed << setprecision(2) << result.search_time_s << "s" << endl;
    cout << "QPS: " << fixed << setprecision(2) << result.qps << endl;
    cout << "mSPQ: " << fixed << setprecision(4) << result.mspq << "ms" << endl;
    cout << "搜索阶段峰值内存: " << fixed << setprecision(2) << result.search_memory_mb << "MB" << endl;
    cout << "平均延迟: " << fixed << setprecision(4) << result.latency.mean_latency_ms << "ms" << endl;
    cout << "P50延迟: " << fixed << setprecision(4) << result.latency.p50_latency_ms << "ms" << endl;
    cout << "P95延迟: " << fixed << setprecision(4) << result.latency.p95_latency_ms << "ms" << endl;
    cout << "P99延迟: " << fixed << setprecision(4) << result.latency.p99_latency_ms << "ms" << endl;
    cout << "召回率: " << fixed << setprecision(4) << result.recall << endl;
    
    return result;
}

// 追加build结果到CSV（首次调用会写入头部）
void appendBuildResultToCSV(const TestResult& result, const string& filename, bool is_first_record = false) {
    ofstream file(filename, ios::app);
    if (!file.is_open()) {
        throw runtime_error("无法打开CSV文件: " + filename);
    }
    
    // 如果是首次写入，写入CSV头部
    if (is_first_record) {
        file << "M,nlist,efconstruction,training_memory_mb,add_memory_mb,training_time_s,total_time_s" << endl;
    }
    
    // 写入数据
    file << result.M << "," << result.nlist << "," << result.efconstruction << ","
         << fixed << setprecision(2) << result.training_memory_mb << ","
         << fixed << setprecision(2) << result.add_memory_mb << ","
         << fixed << setprecision(4) << result.training_time_s << ","
         << fixed << setprecision(4) << result.total_time_s << endl;
    
    // 立即刷新到磁盘，确保数据不会丢失
    file.flush();
    file.close();
}

// 追加search结果到CSV（首次调用会写入头部）
void appendSearchResultToCSV(const TestResult& result, const string& filename, bool is_first_record = false) {
    ofstream file(filename, ios::app);
    if (!file.is_open()) {
        throw runtime_error("无法打开CSV文件: " + filename);
    }
    
    // 如果是首次写入，写入CSV头部
    if (is_first_record) {
        file << "M,nlist,efconstruction,nprobe,efsearch,training_memory_mb,add_memory_mb,training_time_s,total_time_s,"
             << "recall,qps,mspq,search_memory_mb,search_time_s,mean_latency_ms,p50_latency_ms,p95_latency_ms,p99_latency_ms" << endl;
    }
    
    // 写入数据
    file << result.M << "," << result.nlist << "," << result.efconstruction << "," << result.nprobe << "," << result.efsearch << ","
         << fixed << setprecision(2) << result.training_memory_mb << ","
         << fixed << setprecision(2) << result.add_memory_mb << ","
         << fixed << setprecision(4) << result.training_time_s << ","
         << fixed << setprecision(4) << result.total_time_s << ","
         << fixed << setprecision(4) << result.recall << ","
         << fixed << setprecision(2) << result.qps << ","
         << fixed << setprecision(4) << result.mspq << ","
         << fixed << setprecision(2) << result.search_memory_mb << ","
         << fixed << setprecision(4) << result.search_time_s << ","
         << fixed << setprecision(4) << result.latency.mean_latency_ms << ","
         << fixed << setprecision(4) << result.latency.p50_latency_ms << ","
         << fixed << setprecision(4) << result.latency.p95_latency_ms << ","
         << fixed << setprecision(4) << result.latency.p99_latency_ms << endl;
    
    // 立即刷新到磁盘，确保数据不会丢失
    file.flush();
    file.close();
}

int main(int argc, char* argv[]) {
    string config_file = "benchmark.config";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    // 设置OpenMP线程数为20
    omp_set_num_threads(20);
    
    cout << "=== Faiss HNSW-IVF Benchmark测试程序 (详细内存分析版) ===" << endl;
    cout << "配置文件: " << config_file << endl;
    cout << "OpenMP线程数: " << omp_get_max_threads() << endl;
    
    // 解析配置文件
    BenchmarkConfig config = ConfigParser::parseConfig(config_file);
    
    // 获取数据集信息
    auto [_, meta_train] = read_fbin(LEARN_FILE, 0, 1);
    size_t nt = meta_train.first;
    size_t d = meta_train.second;
    
    auto [__, meta_base] = read_fbin(BASE_FILE, 0, 1);
    size_t nb = meta_base.first;
    
    auto [___, meta_query] = read_fbin(QUERY_FILE, 0, 1);
    size_t nq = meta_query.first;
    size_t k = 10; // 查找最近的10个邻居
    
    cout << "\n数据集信息:" << endl;
    cout << "  维度: " << d << endl;
    cout << "  训练集大小: " << nt << endl;
    cout << "  基础集大小: " << nb << endl;
    cout << "  查询集大小: " << nq << endl;
    
    // 生成结果文件名（带时间戳）
    string timestamp = to_string(chrono::duration_cast<chrono::seconds>(
        chrono::system_clock::now().time_since_epoch()).count());
    string build_csv_filename = "benchmark_build_results_" + timestamp + ".csv";
    string search_csv_filename = "benchmark_search_results_" + timestamp + ".csv";
    
    // 标记是否是首次写入（用于写入CSV头部）
    bool first_build_record = true;
    bool first_search_record = true;
    
    int total_test_count = 0;
    int total_build_count = 0;
    
    // 测试流程：对每个build参数组合
    // 1. 构建索引
    // 2. 立即保存build结果到文件（立即刷新到磁盘）
    // 3. 对该索引立即进行所有search参数组合的测试
    // 4. 每个search测试完成后立即保存结果到文件（立即刷新到磁盘）
    // 5. 所有search测试完成后立即删除索引，释放内存
    // 6. 继续下一个build参数组合
    // 
    // 优点：
    // - 避免索引占据大量存储空间（同时只存在一个索引）
    // - 避免缓存大量实验数据（每个测试完成后立即写入文件）
    // - 避免意外中断丢失实验数据（每次写入后立即刷新到磁盘）
    cout << "\n=== 开始Benchmark测试（每构建一个索引立即测试并删除，结果实时保存）===" << endl;
    
    for (int M : config.build.params["M"]) {
        for (int nlist : config.build.params["nlist"]) {
            for (int efconstruction : config.build.params["efconstruction"]) {
                total_build_count++;
                string index_filename;
                
                try {
                    // 1. 构建索引并保存到磁盘（on-disk模式）
                    cout << "\n[构建 " << total_build_count << "] M=" << M << ", nlist=" << nlist << ", efconstruction=" << efconstruction << endl;
                    auto [build_result, index_file] = runBuildTest(M, nlist, efconstruction, d, nt, nb);
                    index_filename = index_file;
                    
                    // 2. 立即保存build结果到文件（立即刷新到磁盘）
                    appendBuildResultToCSV(build_result, build_csv_filename, first_build_record);
                    if (first_build_record) {
                        first_build_record = false;
                        cout << "Build结果已保存到: " << build_csv_filename << endl;
                    }
                    cout << "Build结果已实时保存并刷新到磁盘" << endl;
                    
                    // 3. 对该索引立即进行所有search测试（使用mmap加载）
                    cout << "\n=== 开始Search测试 (On-Disk模式): M=" << M << ", nlist=" << nlist << ", efconstruction=" << efconstruction << " ===" << endl;
                    int search_test_count = 0;
                    
                    for (double nprobe_ratio : config.search.params["nprobe_ratio"]) {
                        // 计算nprobe = nlist * nprobe_ratio，确保至少为1
                        int nprobe = max(1, static_cast<int>(build_result.nlist * nprobe_ratio));
                        
                        for (double efsearch_ratio : config.search.params["efsearch_ratio"]) {
                            // 计算efsearch = nprobe * efsearch_ratio，确保至少为1
                            int efsearch = max(1, static_cast<int>(nprobe * efsearch_ratio));
                            
                            search_test_count++;
                            cout << "\n[Search测试 " << search_test_count << "] "
                                 << "M=" << build_result.M
                                 << ", nlist=" << build_result.nlist
                                 << ", nprobe_ratio=" << nprobe_ratio 
                                 << " -> nprobe=" << nprobe
                                 << ", efsearch_ratio=" << efsearch_ratio 
                                 << " -> efsearch=" << efsearch << endl;
                            
                            TestResult search_result = runSearchTest(nprobe, efsearch, build_result, index_filename, d, nq, k);
                            total_test_count++;
                            
                            // 4. 立即保存search结果到文件（立即刷新到磁盘）
                            appendSearchResultToCSV(search_result, search_csv_filename, first_search_record);
                            if (first_search_record) {
                                first_search_record = false;
                                cout << "Search结果已保存到: " << search_csv_filename << endl;
                            }
                            cout << "Search结果已实时保存并刷新到磁盘" << endl;
                        }
                    }
                    
                    // 5. 所有search测试完成后立即删除索引文件，释放存储空间
                    if (remove(index_filename.c_str()) == 0) {
                        cout << "\n索引文件已删除: " << index_filename << endl;
                    } else {
                        cerr << "警告: 删除索引文件失败: " << index_filename << endl;
                    }
                    
                } catch (const exception& e) {
                    // 错误处理：确保即使出错也能保存已完成的测试结果
                    cerr << "错误: " << e.what() << endl;
                    
                    // 尝试删除索引文件
                    if (!index_filename.empty()) {
                        remove(index_filename.c_str());
                    }
                    
                    cerr << "已清理资源，继续下一个测试..." << endl;
                    continue;
                }
            }
        }
    }
    
    cout << "\n=== Benchmark测试完成 ===" << endl;
    cout << "总共构建了 " << total_build_count << " 个索引" << endl;
    cout << "总共执行了 " << total_test_count << " 次搜索测试" << endl;
    cout << "Build结果保存在: " << build_csv_filename << endl;
    cout << "Search结果保存在: " << search_csv_filename << endl;
    cout << "所有结果已实时保存到磁盘，即使意外中断也不会丢失数据" << endl;
    
    return 0;
}

