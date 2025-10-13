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
    double cpu_energy_joules = 0.0;        // CPU能耗（焦耳）
    double memory_energy_joules = 0.0;     // 内存能耗（焦耳）
    double total_energy_joules = 0.0;      // 总能耗（焦耳）
    double avg_power_watts = 0.0;          // 平均功耗（瓦特）
    double peak_power_watts = 0.0;         // 峰值功耗（瓦特）
    double power_per_query_mj = 0.0;       // 每查询功耗（毫焦耳）
    double energy_efficiency = 0.0;        // 能效比（QPS/瓦特）
    bool rapl_available = false;           // RAPL是否可用
};

// === 增强的基准测试结果结构体 ===
struct BenchmarkResult {
    double total_wall_time_s = 0.0;
    double qps = 0.0;
    double avg_latency_ms = 0.0;
    double p50_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
    double max_latency_ms = 0.0;
    double min_latency_ms = 0.0;
    double peak_memory_mb = 0.0;
    PowerStats power_stats;
};

// ==============================================================================
// 内存监控函数
// ==============================================================================
double get_current_memory_mb() {
    rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    long peak_memory_bytes = usage.ru_maxrss;
    peak_memory_bytes *= 1024; // Linux系统需要乘以1024
    return peak_memory_bytes / (1024.0 * 1024.0);
}

// ==============================================================================
// 功耗监控类
// ==============================================================================
class PowerMonitor {
private:
    string rapl_package0_path;
    string rapl_package1_path;
    string rapl_dram0_path;
    string rapl_dram1_path;
    bool rapl_available;
    
public:
    PowerMonitor() {
        // 检测RAPL是否可用
        rapl_package0_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj";
        rapl_package1_path = "/sys/class/powercap/intel-rapl/intel-rapl:1/energy_uj";
        rapl_dram0_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj";
        rapl_dram1_path = "/sys/class/powercap/intel-rapl/intel-rapl:1/intel-rapl:1:0/energy_uj";
        
        // 检查RAPL接口是否可用
        ifstream test_file(rapl_package0_path);
        rapl_available = test_file.good();
        
        if (!rapl_available) {
            cout << "警告: RAPL不可用，将使用替代功耗估算方法" << endl;
        } else {
            cout << "RAPL功耗监控接口可用" << endl;
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
        stats.rapl_available = rapl_available;
        
        if (!rapl_available) {
            return stats;
        }
        
        // 读取初始能量值（累计值）
        stats.cpu_energy_joules = read_energy_uj(rapl_package0_path) + read_energy_uj(rapl_package1_path);
        stats.memory_energy_joules = read_energy_uj(rapl_dram0_path) + read_energy_uj(rapl_dram1_path);
        stats.total_energy_joules = stats.cpu_energy_joules + stats.memory_energy_joules;
        return stats;
    }
    
    // 结束监控并计算功耗统计
    PowerStats end_monitoring(PowerStats start_stats, double duration_s, double qps) {
        PowerStats stats;
        stats.rapl_available = rapl_available;
        
        if (!rapl_available) {
            // 使用CPU使用率估算功耗（粗略估算）
            stats.avg_power_watts = estimate_power_from_cpu_usage();
            stats.total_energy_joules = stats.avg_power_watts * duration_s;
            stats.cpu_energy_joules = stats.total_energy_joules * 0.8; // 假设80%是CPU
            stats.memory_energy_joules = stats.total_energy_joules * 0.2; // 假设20%是内存
        } else {
            // 使用RAPL读取实际功耗
            uint64_t end_cpu0_uj = read_energy_uj(rapl_package0_path);
            uint64_t end_cpu1_uj = read_energy_uj(rapl_package1_path);
            uint64_t end_dram0_uj = read_energy_uj(rapl_dram0_path);
            uint64_t end_dram1_uj = read_energy_uj(rapl_dram1_path);
            
            // 计算CPU能耗增量（两个package的总和）
            uint64_t total_cpu_uj = end_cpu0_uj + end_cpu1_uj;
            uint64_t start_cpu_uj = static_cast<uint64_t>(start_stats.cpu_energy_joules);
            
            // 处理计数器回绕（假设最大值为2^32微焦耳）
            uint64_t cpu_diff;
            if (total_cpu_uj >= start_cpu_uj) {
                cpu_diff = total_cpu_uj - start_cpu_uj;
            } else {
                // 处理回绕情况
                cpu_diff = (UINT64_MAX - start_cpu_uj) + total_cpu_uj + 1;
            }
            stats.cpu_energy_joules = cpu_diff / 1000000.0; // 转换为焦耳
            
            // 计算内存能耗增量（两个DRAM的总和）
            uint64_t total_dram_uj = end_dram0_uj + end_dram1_uj;
            uint64_t start_dram_uj = static_cast<uint64_t>(start_stats.memory_energy_joules);
            
            // 处理计数器回绕
            uint64_t dram_diff;
            if (total_dram_uj >= start_dram_uj) {
                dram_diff = total_dram_uj - start_dram_uj;
            } else {
                // 处理回绕情况
                dram_diff = (UINT64_MAX - start_dram_uj) + total_dram_uj + 1;
            }
            stats.memory_energy_joules = dram_diff / 1000000.0; // 转换为焦耳
            
            // 总能耗增量
            stats.total_energy_joules = stats.cpu_energy_joules + stats.memory_energy_joules;
            
            // 计算平均功耗
            if (duration_s > 0) {
                stats.avg_power_watts = stats.total_energy_joules / duration_s;
            }
        }
        
        // 计算衍生指标
        if (qps > 0) {
            stats.power_per_query_mj = (stats.total_energy_joules * 1000) / qps;
        }
        if (stats.avg_power_watts > 0) {
            stats.energy_efficiency = qps / stats.avg_power_watts;
        }
        
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
// 0. 路径和文件名配置 & 调试开关
// ==============================================================================
const string DATA_DIR = "./sift";
const string LEARN_FILE = DATA_DIR + "/learn.fbin";
const string BASE_FILE = DATA_DIR + "/base.fbin";
const string QUERY_FILE = DATA_DIR + "/query.fbin";
const string GROUNDTRUTH_FILE = DATA_DIR + "/groundtruth.ivecs";

// ==============================================================================
// 1. 辅助函数：读取.fbin文件
// ==============================================================================
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
        size_t offset = 8 + start_idx * dim * sizeof(float); // Skip header if start_idx > 0, but since we read header, adjust
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

// ==============================================================================
// 2. 设置参数与环境
// ==============================================================================
int main() {
    // 初始化功耗监控器
    PowerMonitor power_monitor;
    
    // 从训练文件中获取维度信息
    auto [_, meta_train] = read_fbin(LEARN_FILE, 0, 1);
    size_t nt = meta_train.first;
    size_t d_train = meta_train.second;

    // 获取数据集大小信息
    auto [__, meta_base] = read_fbin(BASE_FILE, 0, 1);
    size_t nb = meta_base.first;
    size_t d_base = meta_base.second;

    auto [___, meta_query] = read_fbin(QUERY_FILE, 0, 1);
    size_t nq = meta_query.first;
    size_t d_query = meta_query.second;

    // 验证维度一致性
    if (d_train != d_base || d_train != d_query) {
        throw runtime_error("维度不一致: 训练集" + to_string(d_train) + "维, 基础集" + to_string(d_base) + "维, 查询集" + to_string(d_query) + "维");
    }

    // 设置其他参数
    size_t cell_size = 64;
    size_t nlist = nb / cell_size;
    size_t nprobe = 128;
    size_t chunk_size = 100000; // 每次处理的数据块大小
    size_t k = 10; // 查找最近的10个邻居

    size_t M = 32; // HNSW的连接数
    size_t efconstruction = 100; // 默认40
    size_t efsearch = 200;       // 默认16

    // ==============================================================================
    // 【重构点】: 在索引文件名中同时体现 M 和 efConstruction 的值
    // ==============================================================================
    string base_name = BASE_FILE.substr(BASE_FILE.find_last_of("/") + 1);
    base_name = base_name.substr(0, base_name.find_last_of("."));
    regex non_alnum("[^a-zA-Z0-9_]");
    string clean_base_name = regex_replace(base_name, non_alnum, "_");
    string INDEX_FILE = DATA_DIR + "/" + clean_base_name + "_d" + to_string(d_train) + "_nlist" + to_string(nlist) +
                        "_HNSWM" + to_string(M) + "_efc" + to_string(efconstruction) + "_IVFFlat.index";
    // ==============================================================================

    cout << string(60, '=') << endl;
    cout << "Phase 0: 环境设置" << endl;
    cout << "向量维度 (d): " << d_train << endl;
    cout << "基础集大小 (nb): " << nb << ", 训练集大小 (ntrain): " << nt << endl;
    cout << "查询集大小 (nq): " << nq << ", 分块大小 (chunk_size): " << chunk_size << endl;
    cout << "HNSW M (构建参数): " << M << endl;
    cout << "HNSW efConstruction (构建参数): " << efconstruction << endl;
    cout << "索引将保存在磁盘文件: " << INDEX_FILE << endl;
    cout << string(60, '=') << endl;

    // ==============================================================================
    // 3. 检查索引文件是否存在
    // ==============================================================================
    bool skip_index_building = false;
    ifstream index_check(INDEX_FILE);
    if (index_check.good()) {
        cout << "索引文件 " << INDEX_FILE << " 已存在，跳过索引构建阶段" << endl;
        skip_index_building = true;
    } else {
        cout << "索引文件不存在，将构建新索引" << endl;
    }

    faiss::Index* coarse_quantizer = nullptr;

    if (!skip_index_building) {
        // ==============================================================================
        // 4. 训练量化器 
        // ==============================================================================
        cout << "\nPhase 1: 训练 HNSW 粗量化器 (in-memory)" << endl;
        coarse_quantizer = new faiss::IndexHNSWFlat(d_train, M);
        dynamic_cast<faiss::IndexHNSW*>(coarse_quantizer)->hnsw.efConstruction = efconstruction;
        dynamic_cast<faiss::IndexHNSW*>(coarse_quantizer)->hnsw.efSearch = efsearch;
        cout << "efconstruction: " << dynamic_cast<faiss::IndexHNSW*>(coarse_quantizer)->hnsw.efConstruction
             << ", efSearch: " << dynamic_cast<faiss::IndexHNSW*>(coarse_quantizer)->hnsw.efSearch << endl;

        faiss::IndexIVFFlat* index_for_training = new faiss::IndexIVFFlat(coarse_quantizer, d_train, nlist, faiss::METRIC_L2);
        index_for_training->verbose = true;

        auto [xt_data, ___] = read_fbin(LEARN_FILE);
        float* xt = xt_data.data();

        cout << "训练聚类中心并构建 HNSW 量化器..." << endl;
        auto start_time = chrono::high_resolution_clock::now();
        index_for_training->train(nt, xt);
        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end_time - start_time;

        cout << "量化器训练完成，耗时: " << fixed << setprecision(2) << duration.count() << " 秒" << endl;
        cout << "粗量化器中的质心数量: " << coarse_quantizer->ntotal << endl;

        delete index_for_training;

        // ==============================================================================
        // 5. 创建一个空的、基于磁盘的索引框架
        // ==============================================================================
        cout << "\nPhase 2: 创建空的磁盘索引框架" << endl;
        faiss::IndexIVFFlat* index_shell = new faiss::IndexIVFFlat(coarse_quantizer, d_train, nlist, faiss::METRIC_L2);
        cout << "将空的索引框架写入磁盘: " << INDEX_FILE << endl;
        faiss::write_index(index_shell, INDEX_FILE.c_str());
        delete index_shell;

        // ==============================================================================
        // 6. 分块向磁盘索引中添加数据 (从base.fbin)
        // ==============================================================================
        cout << "\nPhase 3: 分块添加数据到磁盘索引" << endl;

        int IO_FLAG_READ_WRITE = 0;
        cout << "使用IO标志: " << IO_FLAG_READ_WRITE << " (读写模式)" << endl;

        faiss::Index* index_ondisk = faiss::read_index(INDEX_FILE.c_str(), IO_FLAG_READ_WRITE);
        auto start_time_add = chrono::high_resolution_clock::now();

        size_t num_chunks = (nb + chunk_size - 1) / chunk_size;
        for (size_t i = 0; i < nb; i += chunk_size) {
            size_t chunk_idx = i / chunk_size + 1;
            cout << "       -> 正在处理块 " << chunk_idx << "/" << num_chunks << ": 向量 " << i << " 到 " << min(i + chunk_size, nb) - 1 << endl;

            auto [xb_chunk_data, ____] = read_fbin(BASE_FILE, i, chunk_size);
            float* xb_chunk = xb_chunk_data.data();

            index_ondisk->add(min(chunk_size, nb - i), xb_chunk);
        }

        auto end_time_add = chrono::high_resolution_clock::now();
        chrono::duration<double> duration_add = end_time_add - start_time_add;
        cout << "\n所有数据块添加完成，总耗时: " << fixed << setprecision(2) << duration_add.count() << " 秒" << endl;
        cout << "磁盘索引中的向量总数 (ntotal): " << index_ondisk->ntotal << endl;

        cout << "正在将最终索引写回磁盘: " << INDEX_FILE << endl;
        faiss::write_index(index_ondisk, INDEX_FILE.c_str());
        delete index_ondisk;
    }

    // ==============================================================================
    // 8. 使用内存映射 (mmap) 进行搜索 (使用query.fbin)
    // ==============================================================================
    cout << "\nPhase 4: 使用内存映射模式进行搜索" << endl;
    cout << "以 mmap 模式打开磁盘索引: " << INDEX_FILE << endl;

    int IO_FLAG_MMAP = faiss::IO_FLAG_MMAP;
    cout << "使用IO标志: " << IO_FLAG_MMAP << " (内存映射模式)" << endl;

    faiss::Index* index_final = faiss::read_index(INDEX_FILE.c_str(), IO_FLAG_MMAP);
    dynamic_cast<faiss::IndexIVF*>(index_final)->nprobe = nprobe;
    omp_set_num_threads(20);
    dynamic_cast<faiss::IndexIVF*>(index_final)->parallel_mode = 0;
    cout << "并行模式线程数: " << omp_get_max_threads() << endl;
    cout << "并行模式: " << dynamic_cast<faiss::IndexIVF*>(index_final)->parallel_mode << endl;
    cout << "索引已准备好搜索 (nprobe=" << dynamic_cast<faiss::IndexIVF*>(index_final)->nprobe << ")" << endl;

    faiss::Index* generic_quantizer = dynamic_cast<faiss::IndexIVF*>(index_final)->quantizer;
    faiss::IndexHNSW* quantizer_hnsw = dynamic_cast<faiss::IndexHNSW*>(generic_quantizer);
    quantizer_hnsw->hnsw.efSearch = efsearch;
    cout << "efConstruction: " << quantizer_hnsw->hnsw.efConstruction << ", efSearch: " << quantizer_hnsw->hnsw.efSearch << endl;

    cout << "从 query.fbin 加载查询向量..." << endl;
    auto [xq_data, _____] = read_fbin(QUERY_FILE);
    float* xq = xq_data.data();

    cout << "\n执行搜索..." << endl;
    vector<faiss::idx_t> I(nq * k);
    vector<float> D(nq * k);
    
    auto start_time_search = chrono::high_resolution_clock::now();
    index_final->search(nq, xq, k, D.data(), I.data());
    auto end_time_search = chrono::high_resolution_clock::now();
    chrono::duration<double> search_duration = end_time_search - start_time_search;

    cout << "搜索完成，耗时: " << fixed << setprecision(2) << search_duration.count() << " 秒" << endl;

    if (search_duration.count() > 0) {
        double qps = nq / search_duration.count();
        cout << "QPS (每秒查询率): " << fixed << setprecision(2) << qps << endl;
    } else {
        cout << "搜索耗时过短，无法计算QPS" << endl;
    }
    
    cout << "索引功能验证完成，详细性能分析将在Phase 6中进行。" << endl;

    // ==============================================================================
    // 9.  新增: 根据Groundtruth计算召回率 (内存优化版)
    // ==============================================================================
    cout << "\n" << string(60, '=') << endl;
    cout << "Phase 5: 计算召回率 (内存优化版)" << endl;

    ifstream gt_file_check(GROUNDTRUTH_FILE);
    if (!gt_file_check.good()) {
        cout << "Groundtruth文件未找到: " << GROUNDTRUTH_FILE << endl;
        cout << "跳过召回率计算。" << endl;
    } else {
        cout << "以流式方式从 " << GROUNDTRUTH_FILE << " 读取 groundtruth 数据进行计算..." << endl;

        ifstream f(GROUNDTRUTH_FILE, ios::binary);
        int32_t k_gt;
        f.read(reinterpret_cast<char*>(&k_gt), sizeof(int32_t));
        cout << "Groundtruth 维度 (k_gt): " << k_gt << endl;

        f.seekg(0, ios::end);
        size_t total_file_size = f.tellg();
        size_t record_size_bytes = (k_gt + 1) * sizeof(int32_t);
        size_t num_gt_vectors = total_file_size / record_size_bytes;
        if (nq != num_gt_vectors) {
            cout << "警告: 查询数量(" << nq << ")与groundtruth中的数量(" << num_gt_vectors << ")不匹配!" << endl;
        }

        cout << "正在计算 Recall@" << k << "..." << endl;

        size_t total_found = 0;
        for (size_t i = 0; i < nq; ++i) {
            size_t offset = i * record_size_bytes;
            f.seekg(offset, ios::beg);

            vector<int32_t> record_data(k_gt + 1);
            f.read(reinterpret_cast<char*>(record_data.data()), (k_gt + 1) * sizeof(int32_t));

            vector<faiss::idx_t> gt_i(k_gt);
            for (int m = 0; m < k_gt; ++m) {
                gt_i[m] = static_cast<faiss::idx_t>(record_data[m + 1]);
            }

            size_t found_count = 0;
            size_t check_size = min(k, static_cast<size_t>(k_gt));
            for (size_t j = 0; j < k; ++j) {
                faiss::idx_t neighbor = I[i * k + j];
                if (find(gt_i.begin(), gt_i.begin() + check_size, neighbor) != gt_i.begin() + check_size) {
                    ++found_count;
                }
            }
            total_found += found_count;
        }

        double recall = static_cast<double>(total_found) / (nq * k);

        cout << "\n查询了 " << nq << " 个向量, k=" << k << endl;
        cout << "在top-" << k << "的结果中，总共找到了 " << total_found << " 个真实的近邻。" << endl;
        cout << "Recall@" << k << ": " << fixed << setprecision(4) << recall << endl;
    }

    cout << string(60, '=') << endl;

    // ==============================================================================
    // Latency Benchmark Module
    // ==============================================================================
    cout << "\n" << string(60, '=') << endl;
    cout << "Phase 6: Latency Benchmark" << endl;
    cout << string(60, '=') << endl;

    // === 基准测试循环设置 ===
    // std::vector<int> thread_counts = {1, 5, 10, 19, 20, 21, 30, 39, 40, 41, 50, 60};
    std::vector<int> thread_counts = {1,2,3,4,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100};
    const int num_runs_per_thread_setting = 5;
    
    // 用于存储最终平均结果的向量
    std::vector<std::pair<int, BenchmarkResult>> final_results;

    cout << "Starting benchmark..." << endl;

    // === 外层循环：遍历不同的线程数设置 ===
    for (int threads : thread_counts) {
        omp_set_num_threads(threads);
        
        std::vector<BenchmarkResult> run_results;
        cout << "\n--- Testing with " << threads << " OMP threads ---" << endl;

        // === 内层循环：为每个线程数设置重复运行多次 ===
        for (int i = 0; i < num_runs_per_thread_setting; ++i) {
            vector<idx_t> I_bench(nq * k);
            vector<float> D_bench(nq * k);
            QueryLatencyStats* latency_stats = new QueryLatencyStats[nq];

            // 开始功耗监控
            PowerStats start_power = power_monitor.start_monitoring();

            double t_start = omp_get_wtime();
            
            dynamic_cast<faiss::IndexIVFFlat*>(index_final)->search_stats(nq, xq, k, D_bench.data(), I_bench.data(), nullptr, latency_stats);

            double t_end = omp_get_wtime();
            
            // 结束功耗监控
            double duration = t_end - t_start;
            double current_qps = nq / duration;
            PowerStats power_stats = power_monitor.end_monitoring(start_power, duration, current_qps);
            
            // --- 数据收集 ---
            BenchmarkResult current_run;
            current_run.total_wall_time_s = duration;
            current_run.qps = current_qps;
            current_run.power_stats = power_stats;

            std::vector<double> latencies_ms;
            for (int j = 0; j < nq; ++j) {
                latencies_ms.push_back(latency_stats[j].total_us / 1000.0);
            }
            std::sort(latencies_ms.begin(), latencies_ms.end());
            
            current_run.avg_latency_ms = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) / nq;
            current_run.p50_latency_ms = latencies_ms[nq / 2];
            current_run.p99_latency_ms = latencies_ms[int(nq * 0.99)];
            current_run.max_latency_ms = latencies_ms.back(); // 最大值（已排序）
            current_run.min_latency_ms = latencies_ms.front(); // 最小值（已排序）
            
            // 记录当前运行的峰值内存
            current_run.peak_memory_mb = get_current_memory_mb();
            
            run_results.push_back(current_run);

            cout << "Run " << (i + 1) << "/" << num_runs_per_thread_setting << ": Total Time = " << fixed << setprecision(4) << current_run.total_wall_time_s << " s, QPS = " << fixed << setprecision(2) << current_run.qps << endl;

            delete[] latency_stats;
        }

        // === 计算平均值（去掉最大值和最小值，剩余3个值取平均） ===
        BenchmarkResult avg_result;
        
        // 创建用于排序的向量，存储QPS值用于排序
        vector<pair<double, int>> qps_with_index;
        for (int i = 0; i < num_runs_per_thread_setting; ++i) {
            qps_with_index.push_back({run_results[i].qps, i});
        }
        
        // 按QPS排序（升序）
        sort(qps_with_index.begin(), qps_with_index.end());
        
        // 去掉最小值（索引0）和最大值（索引4），保留中间3个值（索引1,2,3）
        vector<int> valid_indices = {qps_with_index[1].second, qps_with_index[2].second, qps_with_index[3].second};
        
        // 计算平均值
        double sum_total_time = 0.0, sum_qps = 0.0, sum_avg_lat = 0.0, sum_p50 = 0.0, sum_p99 = 0.0;
        double sum_max_lat = 0.0, sum_min_lat = 0.0;
        double sum_cpu_energy = 0.0, sum_memory_energy = 0.0, sum_total_energy = 0.0;
        double sum_avg_power = 0.0, sum_peak_power = 0.0, sum_power_per_query = 0.0, sum_energy_efficiency = 0.0;
        double sum_peak_memory = 0.0;
        
        for (int idx : valid_indices) {
            sum_total_time += run_results[idx].total_wall_time_s;
            sum_qps += run_results[idx].qps;
            sum_avg_lat += run_results[idx].avg_latency_ms;
            sum_p50 += run_results[idx].p50_latency_ms;
            sum_p99 += run_results[idx].p99_latency_ms;
            sum_max_lat += run_results[idx].max_latency_ms;
            sum_min_lat += run_results[idx].min_latency_ms;
            sum_cpu_energy += run_results[idx].power_stats.cpu_energy_joules;
            sum_memory_energy += run_results[idx].power_stats.memory_energy_joules;
            sum_total_energy += run_results[idx].power_stats.total_energy_joules;
            sum_avg_power += run_results[idx].power_stats.avg_power_watts;
            sum_peak_power += run_results[idx].power_stats.peak_power_watts;
            sum_power_per_query += run_results[idx].power_stats.power_per_query_mj;
            sum_energy_efficiency += run_results[idx].power_stats.energy_efficiency;
            sum_peak_memory += run_results[idx].peak_memory_mb;
        }

        int count = 3; // 使用3个有效值
        avg_result.total_wall_time_s = sum_total_time / count;
        avg_result.qps = sum_qps / count;
        avg_result.avg_latency_ms = sum_avg_lat / count;
        avg_result.p50_latency_ms = sum_p50 / count;
        avg_result.p99_latency_ms = sum_p99 / count;
        avg_result.max_latency_ms = sum_max_lat / count;
        avg_result.min_latency_ms = sum_min_lat / count;
        avg_result.peak_memory_mb = sum_peak_memory / count;
        avg_result.power_stats.cpu_energy_joules = sum_cpu_energy / count;
        avg_result.power_stats.memory_energy_joules = sum_memory_energy / count;
        avg_result.power_stats.total_energy_joules = sum_total_energy / count;
        avg_result.power_stats.avg_power_watts = sum_avg_power / count;
        avg_result.power_stats.peak_power_watts = sum_peak_power / count;
        avg_result.power_stats.power_per_query_mj = sum_power_per_query / count;
        avg_result.power_stats.energy_efficiency = sum_energy_efficiency / count;
        avg_result.power_stats.rapl_available = run_results[0].power_stats.rapl_available;

        final_results.push_back({threads, avg_result});
    }

    // === 最后，打印格式化的总结果表格 ===
    cout << "\n\n===== Faiss Benchmark Summary (Average of middle 3 runs, excluding min/max) =====" << endl;
    cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
    cout << "| " << left << setw(8) << "Threads" 
         << " | " << setw(10) << "Time(s)" 
         << " | " << setw(12) << "QPS" 
         << " | " << setw(10) << "Avg Power(W)" 
         << " | " << setw(12) << "CPU Energy(J)" 
         << " | " << setw(12) << "Mem Energy(J)" 
         << " | " << setw(15) << "Power/Query(mJ)" 
         << " | " << setw(12) << "Efficiency" 
         << " | " << setw(12) << "Avg Lat(ms)" 
         << " | " << setw(10) << "P50 Lat(ms)" 
         << " | " << setw(10) << "P99 Lat(ms)" 
         << " | " << setw(10) << "Max Lat(ms)" 
         << " | " << setw(10) << "Min Lat(ms)" 
         << " | " << setw(12) << "Peak Mem(MB)" << " |" << endl;
    cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;

    // 创建文件输出流
    ofstream output_file("benchmark_results.txt");
    
    // 输出表头到文件
    output_file << "Threads,Time(s),QPS,Avg Power(W),CPU Energy(J),Mem Energy(J),Power/Query(mJ),Efficiency,Avg Lat(ms),P50 Lat(ms),P99 Lat(ms),Max Lat(ms),Min Lat(ms),Peak Mem(MB)" << endl;

    for (const auto& result_pair : final_results) {
        // 输出到终端
        cout << "| " << left << setw(8) << result_pair.first 
             << " | " << fixed << setprecision(4) << setw(10) << result_pair.second.total_wall_time_s
             << " | " << fixed << setprecision(2) << setw(12) << result_pair.second.qps
             << " | " << fixed << setprecision(2) << setw(10) << result_pair.second.power_stats.avg_power_watts
             << " | " << fixed << setprecision(4) << setw(12) << result_pair.second.power_stats.cpu_energy_joules
             << " | " << fixed << setprecision(4) << setw(12) << result_pair.second.power_stats.memory_energy_joules
             << " | " << fixed << setprecision(4) << setw(15) << result_pair.second.power_stats.power_per_query_mj
             << " | " << fixed << setprecision(2) << setw(12) << result_pair.second.power_stats.energy_efficiency
             << " | " << fixed << setprecision(4) << setw(12) << result_pair.second.avg_latency_ms
             << " | " << fixed << setprecision(4) << setw(10) << result_pair.second.p50_latency_ms
             << " | " << fixed << setprecision(4) << setw(10) << result_pair.second.p99_latency_ms
             << " | " << fixed << setprecision(4) << setw(10) << result_pair.second.max_latency_ms
             << " | " << fixed << setprecision(4) << setw(10) << result_pair.second.min_latency_ms
             << " | " << fixed << setprecision(2) << setw(12) << result_pair.second.peak_memory_mb << " |" << endl;
        
        // 输出到文件（CSV格式）
        output_file << result_pair.first << ","
                   << fixed << setprecision(4) << result_pair.second.total_wall_time_s << ","
                   << fixed << setprecision(2) << result_pair.second.qps << ","
                   << fixed << setprecision(2) << result_pair.second.power_stats.avg_power_watts << ","
                   << fixed << setprecision(4) << result_pair.second.power_stats.cpu_energy_joules << ","
                   << fixed << setprecision(4) << result_pair.second.power_stats.memory_energy_joules << ","
                   << fixed << setprecision(4) << result_pair.second.power_stats.power_per_query_mj << ","
                   << fixed << setprecision(2) << result_pair.second.power_stats.energy_efficiency << ","
                   << fixed << setprecision(4) << result_pair.second.avg_latency_ms << ","
                   << fixed << setprecision(4) << result_pair.second.p50_latency_ms << ","
                   << fixed << setprecision(4) << result_pair.second.p99_latency_ms << ","
                   << fixed << setprecision(4) << result_pair.second.max_latency_ms << ","
                   << fixed << setprecision(4) << result_pair.second.min_latency_ms << ","
                   << fixed << setprecision(2) << result_pair.second.peak_memory_mb << endl;
    }
    cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
    
    // 关闭文件
    output_file.close();
    cout << "\n结果已保存到 benchmark_results.txt 文件中" << endl;
    
    // 找出最优配置
    auto max_efficiency = max_element(final_results.begin(), final_results.end(),
        [](const auto& a, const auto& b) {
            return a.second.power_stats.energy_efficiency < b.second.power_stats.energy_efficiency;
        });
    
    auto min_power_per_query = min_element(final_results.begin(), final_results.end(),
        [](const auto& a, const auto& b) {
            return a.second.power_stats.power_per_query_mj < b.second.power_stats.power_per_query_mj;
        });
    
    auto min_total_energy = min_element(final_results.begin(), final_results.end(),
        [](const auto& a, const auto& b) {
            return a.second.power_stats.total_energy_joules < b.second.power_stats.total_energy_joules;
        });
    
    cout << "\n===== 功耗优化建议 =====" << endl;
    cout << "最高能效配置: " << max_efficiency->first << " 线程 (能效: " 
         << fixed << setprecision(2) << max_efficiency->second.power_stats.energy_efficiency << " QPS/W)" << endl;
    cout << "最低每查询功耗配置: " << min_power_per_query->first << " 线程 (每查询: " 
         << fixed << setprecision(4) << min_power_per_query->second.power_stats.power_per_query_mj << " mJ)" << endl;
    cout << "最低总能耗配置: " << min_total_energy->first << " 线程 (总能耗: " 
         << fixed << setprecision(4) << min_total_energy->second.power_stats.total_energy_joules << " J)" << endl;
    
    if (final_results[0].second.power_stats.rapl_available) {
        cout << "\n注意: 功耗数据基于Intel RAPL接口，提供准确的硬件功耗测量" << endl;
    } else {
        cout << "\n注意: 功耗数据基于估算方法，可能不够准确。建议在支持RAPL的系统上运行以获得准确数据" << endl;
    }

    // ==============================================================================
    // 10. 报告峰值内存
    // ==============================================================================
    cout << "\n" << string(60, '=') << endl;
    string sys = "Linux"; // Assuming Linux, adjust if needed
    if (sys == "Linux" || sys == "Darwin") {
        rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        long peak_memory_bytes = usage.ru_maxrss;
        if (sys == "Linux") {
            peak_memory_bytes *= 1024;
        }
        double peak_memory_mb = peak_memory_bytes / (1024.0 * 1024.0);
        cout << "整个程序运行期间的峰值内存占用: " << fixed << setprecision(2) << peak_memory_mb << " MB" << endl;
    }
    cout << string(60, '=') << endl;

    delete index_final;
    if (!skip_index_building) {
        delete coarse_quantizer;
    }

    return 0;
}