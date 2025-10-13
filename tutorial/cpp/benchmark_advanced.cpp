#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <regex>
#include <algorithm>
#include <cmath>
#include <cstdint>
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
#include "config_parser.h"

using namespace std;
using faiss::QueryLatencyStats;
using idx_t = faiss::idx_t;

// 内存监控结构体
struct MemoryStats {
    long peak_memory_mb = 0;
    long current_memory_mb = 0;
};

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
    int nlist = 0;
    int efconstruction = 0;
    int nprobe = 0;
    int efsearch = 0;
};

// 全局变量
const string DATA_DIR = "./sift";
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

// 获取当前内存使用量
MemoryStats getMemoryStats() {
    MemoryStats stats;
    rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    long peak_memory_bytes = usage.ru_maxrss * 1024; // Linux下需要乘以1024
    stats.peak_memory_mb = peak_memory_bytes / (1024 * 1024);
    stats.current_memory_mb = peak_memory_bytes / (1024 * 1024);
    return stats;
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

// 执行build测试
TestResult runBuildTest(int nlist, int efconstruction, size_t d, size_t nt, size_t nb) {
    TestResult result;
    result.nlist = nlist;
    result.efconstruction = efconstruction;
    
    cout << "\n=== Build测试: nlist=" << nlist << ", efconstruction=" << efconstruction << " ===" << endl;
    
    // 开始监控峰值内存
    PeakMemoryMonitor memory_monitor;
    memory_monitor.start();
    
    // 创建HNSW量化器
    auto start_time = chrono::high_resolution_clock::now();
    faiss::IndexHNSWFlat* coarse_quantizer = new faiss::IndexHNSWFlat(d, 32); // M=32固定
    coarse_quantizer->hnsw.efConstruction = efconstruction;
    coarse_quantizer->hnsw.efSearch = 16; // 默认efsearch
    memory_monitor.update();
    
    // 创建IVF索引
    faiss::IndexIVFFlat* index = new faiss::IndexIVFFlat(coarse_quantizer, d, nlist, faiss::METRIC_L2);
    index->verbose = false;
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
    
    // 记录训练阶段的峰值内存
    result.training_memory_mb = memory_monitor.getPeakMemoryMB();
    
    // 添加数据阶段
    auto [xb_data, __] = read_fbin(BASE_FILE);
    float* xb = xb_data.data();
    memory_monitor.update();
    
    index->add(nb, xb);
    memory_monitor.update();
    
    // 记录添加数据阶段的峰值内存
    result.add_memory_mb = memory_monitor.getPeakMemoryMB();
    
    auto end_time = chrono::high_resolution_clock::now();
    result.total_time_s = chrono::duration<double>(end_time - start_time).count();
    
    cout << "训练时间: " << fixed << setprecision(2) << result.training_time_s << "s" << endl;
    cout << "总时间: " << fixed << setprecision(2) << result.total_time_s << "s" << endl;
    cout << "训练阶段峰值内存: " << fixed << setprecision(2) << result.training_memory_mb << "MB" << endl;
    cout << "添加数据阶段峰值内存: " << fixed << setprecision(2) << result.add_memory_mb << "MB" << endl;
    
    // 保存索引供search测试使用
    string index_filename = DATA_DIR + "/temp_index_nlist" + to_string(nlist) + 
                           "_efc" + to_string(efconstruction) + ".index";
    faiss::write_index(index, index_filename.c_str());
    
    delete index;
    delete coarse_quantizer;
    
    return result;
}

// 执行search测试
TestResult runSearchTest(int nprobe, int efsearch, const TestResult& build_result, 
                        size_t /*d*/, size_t nq, size_t k) {
    TestResult result = build_result;
    result.nprobe = nprobe;
    result.efsearch = efsearch;
    
    cout << "\n=== Search测试: nprobe=" << nprobe << ", efsearch=" << efsearch << " ===" << endl;
    
    // 加载索引
    string index_filename = DATA_DIR + "/temp_index_nlist" + to_string(build_result.nlist) + 
                           "_efc" + to_string(build_result.efconstruction) + ".index";
    
    int IO_FLAG_MMAP = faiss::IO_FLAG_MMAP;
    faiss::Index* index = faiss::read_index(index_filename.c_str(), IO_FLAG_MMAP);
    
    // 设置搜索参数
    dynamic_cast<faiss::IndexIVF*>(index)->nprobe = nprobe;
    faiss::IndexHNSW* quantizer_hnsw = dynamic_cast<faiss::IndexHNSW*>(
        dynamic_cast<faiss::IndexIVF*>(index)->quantizer);
    quantizer_hnsw->hnsw.efSearch = efsearch;
    
    // 设置并行模式
    dynamic_cast<faiss::IndexIVF*>(index)->parallel_mode = 0; // 使用OpenMP并行
    
    // 开始监控搜索阶段的峰值内存
    PeakMemoryMonitor search_memory_monitor;
    search_memory_monitor.start();
    
    // 加载查询数据
    auto [xq_data, _] = read_fbin(QUERY_FILE);
    float* xq = xq_data.data();
    search_memory_monitor.update();
    
    // 执行搜索
    vector<idx_t> I(nq * k);
    vector<float> D(nq * k);
    vector<QueryLatencyStats> latency_stats(nq);
    search_memory_monitor.update();
    
    auto search_start = chrono::high_resolution_clock::now();
    dynamic_cast<faiss::IndexIVF*>(index)->search_stats(nq, xq, k, D.data(), I.data(), nullptr, latency_stats.data());
    auto search_end = chrono::high_resolution_clock::now();
    search_memory_monitor.update();
    
    result.search_time_s = chrono::duration<double>(search_end - search_start).count();
    result.qps = nq / result.search_time_s;
    result.mspq = (result.search_time_s * 1000.0) / nq;  // 转换为毫秒
    
    // 记录搜索阶段的峰值内存
    result.search_memory_mb = search_memory_monitor.getPeakMemoryMB();
    
    // 计算延迟统计
    vector<double> latencies_ms;
    for (size_t i = 0; i < nq; ++i) {
        latencies_ms.push_back(latency_stats[i].total_us / 1000.0);
    }
    result.latency = calculateLatencyStats(latencies_ms);
    
    // 计算召回率
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
    cout << "P99延迟: " << fixed << setprecision(4) << result.latency.p99_latency_ms << "ms" << endl;
    cout << "召回率: " << fixed << setprecision(4) << result.recall << endl;
    
    delete index;
    
    return result;
}

// 保存build结果到CSV
void saveBuildResultsToCSV(const vector<TestResult>& build_results, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("无法创建CSV文件: " + filename);
    }
    
    // 写入CSV头部
    file << "nlist,efconstruction,training_memory_mb,add_memory_mb,training_time_s,total_time_s" << endl;
    
    // 写入数据
    for (const auto& result : build_results) {
        file << result.nlist << "," << result.efconstruction << ","
             << fixed << setprecision(2) << result.training_memory_mb << ","
             << fixed << setprecision(2) << result.add_memory_mb << ","
             << fixed << setprecision(4) << result.training_time_s << ","
             << fixed << setprecision(4) << result.total_time_s << endl;
    }
    
    file.close();
    cout << "\nBuild结果已保存到: " << filename << endl;
}

// 保存search结果到CSV
void saveSearchResultsToCSV(const vector<TestResult>& search_results, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("无法创建CSV文件: " + filename);
    }
    
    // 写入CSV头部
    file << "nlist,efconstruction,nprobe,efsearch,training_memory_mb,add_memory_mb,training_time_s,total_time_s,"
         << "recall,qps,mspq,search_memory_mb,search_time_s,mean_latency_ms,p50_latency_ms,p99_latency_ms" << endl;
    
    // 写入数据
    for (const auto& result : search_results) {
        file << result.nlist << "," << result.efconstruction << "," << result.nprobe << "," << result.efsearch << ","
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
             << fixed << setprecision(4) << result.latency.p99_latency_ms << endl;
    }
    
    file.close();
    cout << "\nSearch结果已保存到: " << filename << endl;
}

int main(int argc, char* argv[]) {
    string config_file = "benchmark.config";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    // 设置OpenMP线程数为20
    omp_set_num_threads(20);
    
    cout << "=== Faiss高级Benchmark测试程序 ===" << endl;
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
    
    cout << "数据集信息:" << endl;
    cout << "  维度: " << d << endl;
    cout << "  训练集大小: " << nt << endl;
    cout << "  基础集大小: " << nb << endl;
    cout << "  查询集大小: " << nq << endl;
    
    vector<TestResult> all_results;
    
    // 执行build测试
    cout << "\n=== 开始Build测试 ===" << endl;
    vector<TestResult> build_results;
    
    for (int nlist : config.build.params["nlist"]) {
        for (int efconstruction : config.build.params["efconstruction"]) {
            TestResult build_result = runBuildTest(nlist, efconstruction, d, nt, nb);
            build_results.push_back(build_result);
        }
    }
    
    // 对每个build结果执行search测试
    cout << "\n=== 开始Search测试 ===" << endl;
    
    for (const auto& build_result : build_results) {
        for (double nprobe_ratio : config.search.params["nprobe_ratio"]) {
            // 计算nprobe = nlist * nprobe_ratio，确保至少为1
            int nprobe = max(1, static_cast<int>(build_result.nlist * nprobe_ratio));
            
            for (double efsearch_ratio : config.search.params["efsearch_ratio"]) {
                // 计算efsearch = nprobe * efsearch_ratio，确保至少为1
                int efsearch = max(1, static_cast<int>(nprobe * efsearch_ratio));
                
                cout << "计算参数: nlist=" << build_result.nlist 
                     << ", nprobe_ratio=" << nprobe_ratio 
                     << " -> nprobe=" << nprobe
                     << ", efsearch_ratio=" << efsearch_ratio 
                     << " -> efsearch=" << efsearch << endl;
                
                TestResult search_result = runSearchTest(nprobe, efsearch, build_result, d, nq, k);
                all_results.push_back(search_result);
            }
        }
    }
    
    // 保存结果
    string timestamp = to_string(chrono::duration_cast<chrono::seconds>(
        chrono::system_clock::now().time_since_epoch()).count());
    
    // 保存build结果
    string build_csv_filename = "benchmark_build_results_" + timestamp + ".csv";
    saveBuildResultsToCSV(build_results, build_csv_filename);
    
    // 保存search结果
    string search_csv_filename = "benchmark_search_results_" + timestamp + ".csv";
    saveSearchResultsToCSV(all_results, search_csv_filename);
    
    // 清理临时索引文件
    for (const auto& build_result : build_results) {
        string index_filename = DATA_DIR + "/temp_index_nlist" + to_string(build_result.nlist) + 
                               "_efc" + to_string(build_result.efconstruction) + ".index";
        remove(index_filename.c_str());
    }
    
 
    cout << "\n=== Benchmark测试完成 ===" << endl;
    cout << "总共执行了 " << all_results.size() << " 次测试" << endl;
    
    return 0;
}
