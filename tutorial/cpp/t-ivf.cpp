/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * This C++ program is a conversion of the provided Python benchmark script,
 * using a custom `search_stats` function and compatible with older compilers.
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <string>
#include <sys/resource.h>
#include <unordered_set>
#include <vector>

#include <sys/stat.h>
#if defined(_WIN32)
#include <direct.h>
#endif

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <faiss/utils/utils.h>

// 兼容Windows编译环境
#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#endif

// <--- 修正：删除此处的 QueryLatencyStats 重复定义
// 因为它会与 Faiss 库中的官方定义冲突。
// 您的 Faiss 头文件中必须已经包含了这个结构体的定义。

using faiss::QueryLatencyStats;
using idx_t = faiss::idx_t;

// ==============================================================================
// 0. 结构体与辅助函数
// ==============================================================================

struct BenchmarkResult {
    double total_wall_time_s = 0.0;
    double qps = 0.0;
    double avg_latency_ms = 0.0;
    double p50_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
};

inline bool file_exists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

inline void create_directory(const std::string& path) {
#if defined(_WIN32)
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0755);
#endif
}

void read_fbin(
        const std::string& filename,
        std::vector<float>& data,
        int& n,
        int& d) {
    std::ifstream f(filename, std::ios::in | std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "错误: 文件未找到 " << filename << std::endl;
        std::cerr << "请从 http://corpus-texmex.irisa.fr/ 下载sift.tar.gz并解压到'./sift'目录" << std::endl;
        exit(1);
    }
    f.read(reinterpret_cast<char*>(&n), sizeof(int));
    f.read(reinterpret_cast<char*>(&d), sizeof(int));
    data.resize(static_cast<size_t>(n) * d);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<size_t>(n) * d * sizeof(float));
}

void read_ivecs(
        const std::string& filename,
        std::vector<int32_t>& data,
        int& n,
        int& d) {
    std::ifstream f(filename, std::ios::in | std::ios::binary);
     if (!f.is_open()) {
        std::cerr << "错误: 文件未找到 " << filename << std::endl;
        exit(1);
    }
    f.seekg(0, std::ios::end);
    size_t file_size = f.tellg();
    f.seekg(0, std::ios::beg);

    int vec_dim_and_data_size = 0;
    f.read(reinterpret_cast<char*>(&vec_dim_and_data_size), sizeof(int));
    d = vec_dim_and_data_size;

    n = file_size / ((1 + d) * sizeof(int32_t));
    data.resize(static_cast<size_t>(n) * d);
    
    f.seekg(0, std::ios::beg);

    std::vector<int32_t> buffer(d + 1);
    for (int i = 0; i < n; ++i) {
        f.read(reinterpret_cast<char*>(buffer.data()), (d + 1) * sizeof(int32_t));
        for (int j = 0; j < d; ++j) {
            data[static_cast<size_t>(i) * d + j] = buffer[j + 1];
        }
    }
}

// ==============================================================================
// 主函数
// ==============================================================================
int main() {
    // 省略 main 函数前面不变的部分...
    const std::string DATA_DIR = "./sift";
    const std::string LEARN_FILE = DATA_DIR + "/learn.fbin";
    const std::string BASE_FILE = DATA_DIR + "/base.fbin";
    const std::string QUERY_FILE = DATA_DIR + "/query.fbin";
    const std::string GROUNDTRUTH_FILE = DATA_DIR + "/groundtruth.ivecs";

    const std::string INDEX_DIR = DATA_DIR + "/ivf-index";
    create_directory(INDEX_DIR);

    const int nlist = 7812;
    const int k = 10;
    const int nprobe = nlist; 

    const std::string INDEX_FILENAME = "sift_ivf_nlist" + std::to_string(nlist) + ".index";
    const std::string INDEX_FILE = INDEX_DIR + "/" + INDEX_FILENAME;

    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Phase 1: 从磁盘加载SIFT数据集到内存" << std::endl;
    auto phase_start_time = std::chrono::high_resolution_clock::now();

    std::vector<float> xt, xb, xq;
    int nt, d, nb, d_base, nq, d_query;

    std::cout << "  -> 正在加载训练集: " << LEARN_FILE << std::endl;
    read_fbin(LEARN_FILE, xt, nt, d);

    std::cout << "  -> 正在加载基础集: " << BASE_FILE << std::endl;
    read_fbin(BASE_FILE, xb, nb, d_base);

    std::cout << "  -> 正在加载查询集: " << QUERY_FILE << std::endl;
    read_fbin(QUERY_FILE, xq, nq, d_query);
    
    if (d != d_base || d != d_query) {
        std::cerr << "维度不一致: 训练集" << d << "维, 基础集" << d_base << "维, 查询集"
                  << d_query << "维" << std::endl;
        return 1;
    }

    auto phase_end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(phase_end_time - phase_start_time).count();
    
    printf("\n数据加载完成，耗时: %.2f 秒\n", duration);
    printf("向量维度 (d): %d\n", d);
    printf("基础集大小 (nb): %d\n", nb);
    printf("查询集大小 (nq): %d\n", nq);
    printf("训练集大小 (nt): %d\n", nt);
    std::cout << std::string(60, '=') << std::endl;

    printf("\nPhase 2 & 3: 准备索引 (IVF分区数 nlist: %d)\n", nlist);
    
    faiss::Index* index_ptr = nullptr;

    if (file_exists(INDEX_FILE)) {
        printf("  -> 发现现有索引文件，正在从磁盘加载: %s\n", INDEX_FILE.c_str());
        phase_start_time = std::chrono::high_resolution_clock::now();
        index_ptr = faiss::read_index(INDEX_FILE.c_str());
        phase_end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(phase_end_time - phase_start_time).count();
        printf("  -> 索引加载完成，耗时: %.2f 秒\n", duration);
    } else {
        printf("  -> 未找到索引 '%s'，开始构建新索引...\n", INDEX_FILE.c_str());

        printf("  -> 步骤 3.1: 构建IVF索引结构\n");
        faiss::IndexHNSWFlat quantizer(d, 32);
        auto index_ivf = new faiss::IndexIVFFlat(&quantizer, d, nlist, faiss::METRIC_L2);
        index_ivf->verbose = true;

        printf("\n  -> 步骤 3.2: 训练聚类中心...\n");
        phase_start_time = std::chrono::high_resolution_clock::now();
        index_ivf->train(nt, xt.data());
        phase_end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(phase_end_time - phase_start_time).count();
        printf("  -> 索引训练完成，耗时: %.2f 秒\n", duration);

        printf("\n  -> 步骤 3.3: 向索引中添加基础向量...\n");
        phase_start_time = std::chrono::high_resolution_clock::now();
        index_ivf->add(nb, xb.data());
        phase_end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(phase_end_time - phase_start_time).count();
        printf("  -> 所有向量添加完成，耗时: %.2f 秒\n", duration);

        printf("\n  -> 步骤 3.4: 将构建好的索引写入磁盘: %s\n", INDEX_FILE.c_str());
        phase_start_time = std::chrono::high_resolution_clock::now();
        faiss::write_index(index_ivf, INDEX_FILE.c_str());
        phase_end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(phase_end_time - phase_start_time).count();
        printf("  -> 索引保存完成，耗时: %.2f 秒\n", duration);
        
        index_ptr = index_ivf;
    }

    printf("\n索引已准备就绪。索引中的向量总数 (ntotal): %ld\n", index_ptr->ntotal);
    
    xt.clear(); xt.shrink_to_fit();
    xb.clear(); xb.shrink_to_fit();

    faiss::IndexIVF* index_ivf_ptr = dynamic_cast<faiss::IndexIVF*>(index_ptr);
    if (!index_ivf_ptr) {
        std::cerr << "错误: 索引不是IVF类型，无法设置nprobe" << std::endl;
        delete index_ptr;
        return 1;
    }

    std::cout << "\n" << std::string(60, '=') << std::endl;
    printf("Phase 4: 执行搜索与基准测试\n");
    printf("搜索近邻数 (k): %d\n", k);
    printf("搜索分区数 (nprobe): %d\n", nprobe);

    index_ivf_ptr->nprobe = nprobe;
    index_ivf_ptr->parallel_mode = 0;

    std::vector<int> thread_counts = {1, 10, 20, 30, 40, 50, 60};
    const int num_runs_per_thread_setting = 4;
    std::vector<std::pair<int, BenchmarkResult>> final_results;
    
    std::vector<idx_t> I_search_result(k * nq);
    std::vector<float> D_search_result(k * nq);

    for (int threads : thread_counts) {
        omp_set_num_threads(threads);
        
        std::vector<BenchmarkResult> run_results;
        printf("\n--- Testing with %d OMP threads ---\n", threads);

        for (int i = 0; i < num_runs_per_thread_setting; ++i) {
            idx_t* I = new idx_t[k * nq];
            float* D = new float[k * nq];
            auto latency_stats = new QueryLatencyStats[nq];

            double t_start = omp_get_wtime();
            
            index_ivf_ptr->search_stats(nq, xq.data(), k, D, I, nullptr, latency_stats);

            double t_end = omp_get_wtime();
            
            BenchmarkResult current_run;
            current_run.total_wall_time_s = t_end - t_start;
            current_run.qps = nq / current_run.total_wall_time_s;

            std::vector<double> latencies_ms;
            latencies_ms.reserve(nq);
            for (int j = 0; j < nq; ++j) {
                latencies_ms.push_back(latency_stats[j].total_us / 1000.0);
            }
            std::sort(latencies_ms.begin(), latencies_ms.end());
            
            current_run.avg_latency_ms = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) / nq;
            current_run.p50_latency_ms = latencies_ms[nq / 2];
            current_run.p99_latency_ms = latencies_ms[static_cast<int>(nq * 0.99)];
            
            run_results.push_back(current_run);

            printf("Run %d/%d: Total Time = %.4f s, QPS = %.2f\n", 
                   i + 1, num_runs_per_thread_setting, 
                   current_run.total_wall_time_s, current_run.qps);

            if (i == num_runs_per_thread_setting - 1) {
                memcpy(I_search_result.data(), I, (size_t)k * nq * sizeof(idx_t));
                memcpy(D_search_result.data(), D, (size_t)k * nq * sizeof(float));
            }

            delete[] I;
            delete[] D;
            delete[] latency_stats;
        }

        BenchmarkResult avg_result;
        double sum_total_time = 0.0, sum_qps = 0.0, sum_avg_lat = 0.0, sum_p50 = 0.0, sum_p99 = 0.0;
        
        int count = num_runs_per_thread_setting - 1;
        if (count <= 0) count = 1;

        for (int i = 1; i < num_runs_per_thread_setting; ++i) { 
            sum_total_time += run_results[i].total_wall_time_s;
            sum_qps += run_results[i].qps;
            sum_avg_lat += run_results[i].avg_latency_ms;
            sum_p50 += run_results[i].p50_latency_ms;
            sum_p99 += run_results[i].p99_latency_ms;
        }

        avg_result.total_wall_time_s = sum_total_time / count;
        avg_result.qps = sum_qps / count;
        avg_result.avg_latency_ms = sum_avg_lat / count;
        avg_result.p50_latency_ms = sum_p50 / count;
        avg_result.p99_latency_ms = sum_p99 / count;

        final_results.push_back({threads, avg_result});
    }

    printf("\n\n===== Faiss Benchmark Summary (Average of last %d runs) =====\n", num_runs_per_thread_setting - 1);
    printf("--------------------------------------------------------------------------------------------------\n");
    printf("| %-12s | %-20s | %-15s | %-20s | %-15s | %-15s |\n", 
           "OMP Threads", "Total Time (s)", "QPS", "Avg Latency (ms)", "P50 Latency (ms)", "P99 Latency (ms)");
    printf("--------------------------------------------------------------------------------------------------\n");

    for (const auto& result_pair : final_results) {
        printf("| %-12d | %-20.4f | %-15.2f | %-20.4f | %-15.4f | %-15.4f |\n", 
               result_pair.first,
               result_pair.second.total_wall_time_s,
               result_pair.second.qps,
               result_pair.second.avg_latency_ms,
               result_pair.second.p50_latency_ms,
               result_pair.second.p99_latency_ms);
    }
    printf("--------------------------------------------------------------------------------------------------\n");

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Phase 5: 计算召回率 (与Groundtruth文件对比)" << std::endl;

    std::cout << "  -> 正在加载Groundtruth文件: " << GROUNDTRUTH_FILE << "..." << std::endl;
    std::vector<int32_t> I_gt_vec;
    int nq_gt, d_gt;
    read_ivecs(GROUNDTRUTH_FILE, I_gt_vec, nq_gt, d_gt);

    std::cout << "  -> 正在计算召回率..." << std::endl;
    
    long found_count = 0;
    for (int i = 0; i < nq; i++) {
        std::unordered_set<idx_t> gt_results;
        for (int j = 0; j < k; j++) {
            gt_results.insert(I_gt_vec[static_cast<size_t>(i) * d_gt + j]);
        }
        
        for (int j = 0; j < k; j++) {
            if (gt_results.count(I_search_result[static_cast<size_t>(i) * k + j])) {
                found_count++;
            }
        }
    }

    long total_possible = (long)nq * k;
    double recall = (double)found_count / total_possible;

    printf("\n查询了 %d 个向量, k=%d\n", nq, k);
    printf("在top-%d的结果中，总共找到了 %ld 个真实的近邻。\n", k, found_count);
    printf("Recall@%d: %.4f\n", k, recall);
    std::cout << std::string(60, '=') << std::endl;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Phase 6: 性能报告" << std::endl;

    #if defined(__linux__) || defined(__APPLE__)
        struct rusage r_usage;
        getrusage(RUSAGE_SELF, &r_usage);
        #if defined(__APPLE__)
            double peak_memory_mb = r_usage.ru_maxrss / (1024.0 * 1024.0);
        #else
            double peak_memory_mb = r_usage.ru_maxrss / 1024.0;
        #endif
        printf("整个程序运行期间的峰值内存占用: %.2f MB\n", peak_memory_mb);
    #elif defined(_WIN32)
        PROCESS_MEMORY_COUNTERS_EX pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
            double peak_memory_mb = pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
            printf("整个程序运行期间的峰值内存占用: %.2f MB\n", peak_memory_mb);
        } else {
             std::cout << "无法在Windows上获取峰值内存。" << std::endl;
        }
    #else
        std::cout << "无法在当前操作系统上自动获取峰值内存。" << std::endl;
    #endif
    std::cout << std::string(60, '=') << std::endl;

    delete index_ptr;

    return 0;
}