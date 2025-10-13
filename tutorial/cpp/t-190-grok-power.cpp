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

using namespace std;
using faiss::QueryLatencyStats;
using idx_t = faiss::idx_t;

// === 功耗监控结构体 ===
struct PowerStats {
    double total_energy_joules = 0.0;
    double avg_power_watts = 0.0;
    double peak_power_watts = 0.0;
    double power_per_query_mj = 0.0;  // 毫焦耳每查询
    double energy_efficiency = 0.0;   // QPS/瓦特
};

// === 增强的基准测试结果结构体 ===
struct BenchmarkResult {
    double total_wall_time_s = 0.0;
    double qps = 0.0;
    double avg_latency_ms = 0.0;
    double p50_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
    PowerStats power_stats;
};

// ==============================================================================
// 功耗监控类
// ==============================================================================
class PowerMonitor {
private:
    string rapl_package_path;
    string rapl_cores_path;
    bool rapl_available;
    
public:
    PowerMonitor() {
        // 检测RAPL是否可用
        rapl_package_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj";
        rapl_cores_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj";
        
        ifstream test_file(rapl_package_path);
        rapl_available = test_file.good();
        
        if (!rapl_available) {
            cout << "警告: RAPL不可用，将使用替代功耗估算方法" << endl;
        }
    }
    
    // 读取当前能量值（微焦耳）
    uint64_t read_energy_uj(const string& path) {
        ifstream file(path);
        if (!file.is_open()) return 0;
        
        uint64_t energy;
        file >> energy;
        return energy;
    }
    
    // 开始监控
    PowerStats start_monitoring() {
        PowerStats stats;
        if (!rapl_available) {
            return stats;
        }
        
        // 读取初始能量值
        stats.total_energy_joules = 0.0;
        return stats;
    }
    
    // 结束监控并计算功耗统计
    PowerStats end_monitoring(PowerStats start_stats, double duration_s, double qps) {
        PowerStats stats = start_stats;
        
        if (!rapl_available) {
            // 使用CPU使用率估算功耗（粗略估算）
            stats.avg_power_watts = estimate_power_from_cpu_usage();
            stats.total_energy_joules = stats.avg_power_watts * duration_s;
        } else {
            // 使用RAPL读取实际功耗
            uint64_t end_energy_uj = read_energy_uj(rapl_package_path);
            uint64_t start_energy_uj = static_cast<uint64_t>(start_stats.total_energy_joules * 1000000);
            
            if (end_energy_uj > start_energy_uj) {
                stats.total_energy_joules = (end_energy_uj - start_energy_uj) / 1000000.0;
            }
            stats.avg_power_watts = stats.total_energy_joules / duration_s;
        }
        
        // 计算衍生指标
        stats.power_per_query_mj = (stats.total_energy_joules * 1000) / qps;
        stats.energy_efficiency = qps / stats.avg_power_watts;
        
        return stats;
    }
    
private:
    // 基于CPU使用率的功耗估算（粗略方法）
    double estimate_power_from_cpu_usage() {
        // 这是一个简化的估算，实际应用中需要更复杂的模型
        // 假设基础功耗为50W，每个CPU核心增加10W
        int num_cores = omp_get_max_threads();
        return 50.0 + (num_cores * 10.0);
    }
};

// ==============================================================================
// 路径和文件名配置
// ==============================================================================
const string DATA_DIR = "./sift";
const string LEARN_FILE = DATA_DIR + "/learn.fbin";
const string BASE_FILE = DATA_DIR + "/base.fbin";
const string QUERY_FILE = DATA_DIR + "/query.fbin";
const string GROUNDTRUTH_FILE = DATA_DIR + "/groundtruth.ivecs";

// ==============================================================================
// 辅助函数：读取.fbin文件
// ==============================================================================
pair<vector<float>, pair<size_t, size_t>> read_fbin(const string& filename, size_t start_idx = 0, size_t chunk_size = 0) {
    ifstream f(filename, ios::binary);
    if (!f.is_open()) {
        throw runtime_error("无法打开文件: " + filename);
    }
    
    uint32_t d, n;
    f.read(reinterpret_cast<char*>(&d), sizeof(d));
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    
    size_t actual_chunk_size = (chunk_size == 0) ? n : min(chunk_size, static_cast<size_t>(n) - start_idx);
    size_t actual_start = min(start_idx, static_cast<size_t>(n));
    
    f.seekg(8 + actual_start * d * sizeof(float));
    
    vector<float> data(actual_chunk_size * d);
    f.read(reinterpret_cast<char*>(data.data()), actual_chunk_size * d * sizeof(float));
    
    return {data, {d, actual_chunk_size}};
}

int main() {
    cout << string(60, '=') << endl;
    cout << "Phase 0: 环境设置" << endl;
    cout << string(60, '=') << endl;
    
    // 基本参数
    const int d = 128;
    const size_t nb = 1000000;
    const size_t ntrain = 100000;
    const size_t nq = 10000;
    const size_t chunk_size = 100000;
    const int M = 32;
    const int efConstruction = 40;
    const int k = 10;
    
    cout << "向量维度 (d): " << d << endl;
    cout << "基础集大小 (nb): " << nb << ", 训练集大小 (ntrain): " << ntrain << endl;
    cout << "查询集大小 (nq): " << nq << ", 分块大小 (chunk_size): " << chunk_size << endl;
    cout << "HNSW M (构建参数): " << M << endl;
    cout << "HNSW efConstruction (构建参数): " << efConstruction << endl;
    
    // 初始化功耗监控器
    PowerMonitor power_monitor;
    
    // 加载查询向量
    cout << "从 " << QUERY_FILE << " 加载查询向量..." << endl;
    auto [xq, query_dims] = read_fbin(QUERY_FILE);
    cout << "查询向量维度: " << query_dims.first << ", 数量: " << query_dims.second << endl;
    
    // 加载索引
    string index_file = "./sift/base_d128_nlist3906_HNSWM32_efc40_IVFFlat.index";
    cout << "索引将保存在磁盘文件: " << index_file << endl;
    cout << string(60, '=') << endl;
    
    faiss::Index* index_final = nullptr;
    bool skip_index_building = false;
    
    ifstream index_check(index_file);
    if (index_check.good()) {
        cout << "索引文件 " << index_file << " 已存在，跳过索引构建阶段" << endl;
        skip_index_building = true;
    }
    
    if (skip_index_building) {
        cout << "\nPhase 4: 使用内存映射模式进行搜索" << endl;
        cout << "以 mmap 模式打开磁盘索引: " << index_file << endl;
        
        index_final = faiss::read_index(index_file.c_str());
        
        cout << "使用内存映射模式加载索引" << endl;
        cout << "并行模式线程数: " << omp_get_max_threads() << endl;
        cout << "并行模式: " << omp_in_parallel() << endl;
        
        dynamic_cast<faiss::IndexIVFFlat*>(index_final)->nprobe = 32;
        cout << "索引已准备好搜索 (nprobe=" << dynamic_cast<faiss::IndexIVFFlat*>(index_final)->nprobe << ")" << endl;
        cout << "efConstruction: " << efConstruction << ", efSearch: 16" << endl;
    }
    
    // 执行搜索
    cout << "从 " << QUERY_FILE << " 加载查询向量..." << endl;
    cout << "\n执行搜索..." << endl;
    
    vector<idx_t> I(nq * k);
    vector<float> D(nq * k);
    
    double t_start = omp_get_wtime();
    index_final->search(nq, xq.data(), k, D.data(), I.data());
    double t_end = omp_get_wtime();
    
    double search_time = t_end - t_start;
    double qps = nq / search_time;
    
    cout << "搜索完成，耗时: " << fixed << setprecision(2) << search_time << " 秒" << endl;
    cout << "QPS (每秒查询率): " << fixed << setprecision(2) << qps << endl;
    
    // ==============================================================================
    // 功耗基准测试
    // ==============================================================================
    cout << "\n" << string(60, '=') << endl;
    cout << "Phase 6: 功耗基准测试" << endl;
    cout << string(60, '=') << endl;
    
    vector<int> thread_counts = {1, 2, 4, 8, 16, 20, 30, 40, 50, 60};
    const int num_runs_per_thread_setting = 3;
    
    vector<pair<int, BenchmarkResult>> final_results;
    
    cout << "开始功耗基准测试..." << endl;
    
    for (int threads : thread_counts) {
        omp_set_num_threads(threads);
        
        vector<BenchmarkResult> run_results;
        cout << "\n--- 测试 " << threads << " 个OMP线程 ---" << endl;
        
        for (int i = 0; i < num_runs_per_thread_setting; ++i) {
            vector<idx_t> I_bench(nq * k);
            vector<float> D_bench(nq * k);
            QueryLatencyStats* latency_stats = new QueryLatencyStats[nq];
            
            // 开始功耗监控
            PowerStats start_power = power_monitor.start_monitoring();
            
            double t_start = omp_get_wtime();
            dynamic_cast<faiss::IndexIVFFlat*>(index_final)->search_stats(nq, xq.data(), k, D_bench.data(), I_bench.data(), nullptr, latency_stats);
            double t_end = omp_get_wtime();
            
            // 结束功耗监控
            double duration = t_end - t_start;
            double current_qps = nq / duration;
            PowerStats power_stats = power_monitor.end_monitoring(start_power, duration, current_qps);
            
            // 收集结果
            BenchmarkResult current_run;
            current_run.total_wall_time_s = duration;
            current_run.qps = current_qps;
            current_run.power_stats = power_stats;
            
            // 计算延迟统计
            vector<double> latencies_ms;
            for (int j = 0; j < nq; ++j) {
                latencies_ms.push_back(latency_stats[j].total_us / 1000.0);
            }
            sort(latencies_ms.begin(), latencies_ms.end());
            
            current_run.avg_latency_ms = accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) / nq;
            current_run.p50_latency_ms = latencies_ms[nq / 2];
            current_run.p99_latency_ms = latencies_ms[int(nq * 0.99)];
            
            run_results.push_back(current_run);
            
            cout << "Run " << (i + 1) << "/" << num_runs_per_thread_setting 
                 << ": 时间 = " << fixed << setprecision(4) << duration << "s"
                 << ", QPS = " << fixed << setprecision(2) << current_qps
                 << ", 功耗 = " << fixed << setprecision(2) << power_stats.avg_power_watts << "W"
                 << ", 能效 = " << fixed << setprecision(2) << power_stats.energy_efficiency << " QPS/W" << endl;
            
            delete[] latency_stats;
        }
        
        // 计算平均值（忽略第一次热身运行）
        BenchmarkResult avg_result;
        double sum_total_time = 0.0, sum_qps = 0.0, sum_avg_lat = 0.0, sum_p50 = 0.0, sum_p99 = 0.0;
        double sum_energy = 0.0, sum_avg_power = 0.0, sum_peak_power = 0.0, sum_power_per_query = 0.0, sum_energy_efficiency = 0.0;
        
        for (int i = 1; i < num_runs_per_thread_setting; ++i) {
            sum_total_time += run_results[i].total_wall_time_s;
            sum_qps += run_results[i].qps;
            sum_avg_lat += run_results[i].avg_latency_ms;
            sum_p50 += run_results[i].p50_latency_ms;
            sum_p99 += run_results[i].p99_latency_ms;
            sum_energy += run_results[i].power_stats.total_energy_joules;
            sum_avg_power += run_results[i].power_stats.avg_power_watts;
            sum_peak_power += run_results[i].power_stats.peak_power_watts;
            sum_power_per_query += run_results[i].power_stats.power_per_query_mj;
            sum_energy_efficiency += run_results[i].power_stats.energy_efficiency;
        }
        
        int count = num_runs_per_thread_setting - 1;
        avg_result.total_wall_time_s = sum_total_time / count;
        avg_result.qps = sum_qps / count;
        avg_result.avg_latency_ms = sum_avg_lat / count;
        avg_result.p50_latency_ms = sum_p50 / count;
        avg_result.p99_latency_ms = sum_p99 / count;
        avg_result.power_stats.total_energy_joules = sum_energy / count;
        avg_result.power_stats.avg_power_watts = sum_avg_power / count;
        avg_result.power_stats.peak_power_watts = sum_peak_power / count;
        avg_result.power_stats.power_per_query_mj = sum_power_per_query / count;
        avg_result.power_stats.energy_efficiency = sum_energy_efficiency / count;
        
        final_results.push_back({threads, avg_result});
    }
    
    // 打印功耗基准测试结果
    cout << "\n\n===== Faiss 功耗基准测试总结 (后" << (num_runs_per_thread_setting - 1) << "次运行平均值) =====" << endl;
    cout << "------------------------------------------------------------------------------------------------------------------------" << endl;
    cout << "| " << left << setw(8) << "线程数" 
         << " | " << setw(12) << "QPS" 
         << " | " << setw(10) << "平均功耗(W)" 
         << " | " << setw(12) << "总能耗(J)" 
         << " | " << setw(15) << "每查询功耗(mJ)" 
         << " | " << setw(12) << "能效(QPS/W)" 
         << " | " << setw(12) << "平均延迟(ms)" << " |" << endl;
    cout << "------------------------------------------------------------------------------------------------------------------------" << endl;
    
    for (const auto& result_pair : final_results) {
        cout << "| " << left << setw(8) << result_pair.first 
             << " | " << fixed << setprecision(2) << setw(12) << result_pair.second.qps
             << " | " << fixed << setprecision(2) << setw(10) << result_pair.second.power_stats.avg_power_watts
             << " | " << fixed << setprecision(2) << setw(12) << result_pair.second.power_stats.total_energy_joules
             << " | " << fixed << setprecision(2) << setw(15) << result_pair.second.power_stats.power_per_query_mj
             << " | " << fixed << setprecision(2) << setw(12) << result_pair.second.power_stats.energy_efficiency
             << " | " << fixed << setprecision(2) << setw(12) << result_pair.second.avg_latency_ms << " |" << endl;
    }
    cout << "------------------------------------------------------------------------------------------------------------------------" << endl;
    
    // 找出最优配置
    auto max_efficiency = max_element(final_results.begin(), final_results.end(),
        [](const auto& a, const auto& b) {
            return a.second.power_stats.energy_efficiency < b.second.power_stats.energy_efficiency;
        });
    
    auto min_power_per_query = min_element(final_results.begin(), final_results.end(),
        [](const auto& a, const auto& b) {
            return a.second.power_stats.power_per_query_mj < b.second.power_stats.power_per_query_mj;
        });
    
    cout << "\n===== 功耗优化建议 =====" << endl;
    cout << "最高能效配置: " << max_efficiency->first << " 线程 (能效: " 
         << fixed << setprecision(2) << max_efficiency->second.power_stats.energy_efficiency << " QPS/W)" << endl;
    cout << "最低每查询功耗配置: " << min_power_per_query->first << " 线程 (每查询: " 
         << fixed << setprecision(2) << min_power_per_query->second.power_stats.power_per_query_mj << " mJ)" << endl;
    
    delete index_final;
    
    return 0;
}
