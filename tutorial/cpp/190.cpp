/**
 * @file sift_full_benchmark.cpp
 * @brief A comprehensive C++ benchmark for a modified Faiss IndexIVFFlat.
 *
 * This program integrates a full SIFT benchmark workflow with a rigorous,
 * multi-threaded performance testing harness. It calls a custom
 * `search_with_stats` function to gather detailed per-query latency metrics.
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
// #include <filesystem> // MODIFICATION: Removed this C++17 header
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

#include <omp.h>
#include <sys/resource.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <faiss/utils/utils.h>

// 假设我们修改的 Faiss 源码中已定义此结构体
// 如果编译报错，请确保您在 faiss/IndexIVF.h 中有此定义
using faiss::QueryLatencyStats;

// ==============================================================================
// 1. 辅助函数和结构体
// ==============================================================================

// 用于保存单次运行的完整结果
struct BenchmarkResult {
    double total_wall_time_s = 0.0;
    double qps = 0.0;
    double avg_latency_ms = 0.0; // Per-query "work-time"
    double p50_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
};

// 读取 .fbin 文件的函数
void read_fbin(const std::string& filename,
               std::vector<float>& data,
               int& n,
               int& d,
               long start_idx = 0,
               long chunk_size = -1) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(-1);
    }
    f.read(reinterpret_cast<char*>(&n), sizeof(int));
    f.read(reinterpret_cast<char*>(&d), sizeof(int));

    long read_n = (chunk_size == -1) ? n : std::min((long)n, start_idx + chunk_size);
    if (chunk_size != -1) {
        read_n = std::min((long)n - start_idx, chunk_size);
        long offset = 8 + start_idx * d * sizeof(float);
        f.seekg(offset, std::ios::beg);
    }
    if(read_n <= 0) {
        data.clear();
        return;
    }
    data.resize(read_n * d);
    f.read(reinterpret_cast<char*>(data.data()), read_n * d * sizeof(float));
}


// 读取 .ivecs 文件的函数
void read_ivecs(const std::string& filename, std::vector<int32_t>& data, int& n, int& d) {
    std::ifstream f(filename, std::ios::binary);
     if (!f.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(-1);
    }
    int dim_record;
    std::vector<int32_t> record;
    int count = 0;
    while (f.read(reinterpret_cast<char*>(&dim_record), sizeof(int))) {
        if (count == 0) {
            d = dim_record;
            record.resize(d);
        } else if (dim_record != d) {
            std::cerr << "Error: Inconsistent dimensions in " << filename << std::endl;
            exit(-1);
        }
        f.read(reinterpret_cast<char*>(record.data()), d * sizeof(int32_t));
        data.insert(data.end(), record.begin(), record.end());
        count++;
    }
    n = count;
}


int main(int argc, char** argv) {
    // ==============================================================================
    // 2. 路径和文件名配置
    // ==============================================================================
    const std::string DATA_DIR = "./sift";
    const std::string LEARN_FILE = DATA_DIR + "/learn.fbin";
    const std::string BASE_FILE = DATA_DIR + "/base.fbin";
    const std::string QUERY_FILE = DATA_DIR + "/query.fbin";
    const std::string GROUNDTRUTH_FILE = DATA_DIR + "/groundtruth.ivecs";

    // ==============================================================================
    // 3. 设置参数与环境
    // ==============================================================================
    int d, nt, nb, nq, k_gt;
    std::vector<float> tmp_data;
    read_fbin(LEARN_FILE, tmp_data, nt, d, 0, 1);
    read_fbin(BASE_FILE, tmp_data, nb, d, 0, 1);
    read_fbin(QUERY_FILE, tmp_data, nq, d, 0, 1);
    
    const int cell_size = 256;
    const int nlist = nb / cell_size;
    const int nprobe = 32;
    const long chunk_size = 100000;
    const int k = 10;
    const int M = 32;
    const int efConstruction = 40;
    const int efSearch = 16;
    
    const std::string INDEX_FILE = DATA_DIR + "/sift1m_d" + std::to_string(d) + "_nlist" + std::to_string(nlist) +
                               "_FlatL2_IVFFlat.index";

    printf("==================================================================\n");
    printf("Phase 0: Environment Setup\n");
    printf("Vector dim (d): %d\n", d);
    printf("Base size (nb): %d, Train size (nt): %d\n", nb, nt);
    printf("Query size (nq): %d, Add chunk size: %ld\n", nq, chunk_size);
    printf("HNSW M: %d, efConstruction: %d\n", M, efConstruction);
    printf("Index file path: %s\n", INDEX_FILE.c_str());
    printf("==================================================================\n\n");

    // ==============================================================================
    // 4. 构建索引（如果不存在）
    // ==============================================================================

    // MODIFICATION: Replaced std::filesystem::exists with std::ifstream check
    // ==============================================================================
    // 4. 构建索引（如果不存在）- 修正后的流程
    // ==============================================================================
    std::ifstream f_check(INDEX_FILE);
    if (!f_check.good()) {
        f_check.close();
        printf("Index file not found. Starting build process...\n");
        
        // === 步骤 1: 在内存中创建并训练索引 ===
        printf("Phase 1 & 2: Training HNSW coarse quantizer in memory...\n");
        
        // faiss::IndexHNSWFlat* coarse_quantizer = new faiss::IndexHNSWFlat(d, M, faiss::METRIC_L2);
        // coarse_quantizer->hnsw.efConstruction = efConstruction;
        faiss::Index* coarse_quantizer = new faiss::IndexFlatL2(d);

        // =================================================================
        // === 核心修正：使用 new 在堆上创建 index_in_memory 对象 ===
        faiss::IndexIVFFlat* index_in_memory = 
            new faiss::IndexIVFFlat(coarse_quantizer, d, nlist, faiss::METRIC_L2);
        // =================================================================

        // 使用 -> 箭头操作符访问成员
        index_in_memory->own_fields = true; 
        index_in_memory->verbose = true;

        std::vector<float> xt;
        read_fbin(LEARN_FILE, xt, nt, d);
        
        {
            std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
            // 使用 -> 箭头操作符调用方法
            index_in_memory->train(nt, xt.data());
            std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
            printf("Quantizer training complete, took %.2f seconds\n", std::chrono::duration<double>(t1-t0).count());
        }

        // --- 步骤 2: 继续使用同一个内存中的索引对象来添加数据 ---
        printf("\nPhase 3: Populating index in memory...\n");
        {
            std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
            for (long i = 0; i < nb; i += chunk_size) {
                printf("  -> Processing chunk starting at vector %ld\n", i);
                std::vector<float> xb_chunk;
                int n_chunk, d_chunk;
                read_fbin(BASE_FILE, xb_chunk, n_chunk, d_chunk, i, chunk_size);
                // 使用 -> 箭头操作符调用方法
                index_in_memory->add(n_chunk, xb_chunk.data());
            }
            std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
            printf("\nAll chunks added, took %.2f seconds\n", std::chrono::duration<double>(t1-t0).count());
            printf("Total vectors in memory index: %ld\n", index_in_memory->ntotal);
        }

        // --- 步骤 3: 将构建完成的最终索引一次性写入磁盘 ---
        printf("\nWriting final, populated index to disk: %s\n", INDEX_FILE.c_str());
        faiss::write_index(index_in_memory, INDEX_FILE.c_str());

        // =================================================================
        // === 核心修正：手动释放 index_in_memory ===
        // 因为 own_fields=true，这里会自动 delete 掉 coarse_quantizer
        delete index_in_memory;
        // =================================================================

    } else {
        f_check.close();
        printf("Index file %s already exists, skipping build phase.\n", INDEX_FILE.c_str());
    }

    
    // ==============================================================================
    // 5. 加载索引并执行基准测试
    // ==============================================================================
    printf("\nPhase 4: Loading index for search and running benchmark\n");
    faiss::Index* index_final_generic = faiss::read_index(INDEX_FILE.c_str(), 4 /* IO_FLAG_MMAP */);
    faiss::IndexIVFFlat* index_final = dynamic_cast<faiss::IndexIVFFlat*>(index_final_generic);
    index_final->nprobe = nprobe;
    index_final->parallel_mode = 0;
    
    faiss::IndexHNSW* quantizer_hnsw = dynamic_cast<faiss::IndexHNSW*>(index_final->quantizer);
    if(quantizer_hnsw) {
        quantizer_hnsw->hnsw.efSearch = efSearch;
    }

    std::vector<float> xq;
    read_fbin(QUERY_FILE, xq, nq, d);
    
    std::vector<int> thread_counts = {1, 10, 20, 30, 40, 50, 60};
    const int num_runs_per_thread_setting = 3;
    std::vector<std::pair<int, BenchmarkResult>> final_results;

    for (int threads : thread_counts) {
        omp_set_num_threads(threads);
        
        std::vector<BenchmarkResult> run_results;
        printf("\n--- Testing with %d OMP threads ---\n", threads);

        for (int i = 0; i < num_runs_per_thread_setting; ++i) {
            std::vector<faiss::idx_t> I(k * nq);
            std::vector<float> D(k * nq);
            std::vector<QueryLatencyStats> latency_stats(nq);

            double t_start = omp_get_wtime();
            
            index_final->search_stats(nq, xq.data(), k, D.data(), I.data(), nullptr, latency_stats.data());

            double t_end = omp_get_wtime();
            
            BenchmarkResult current_run;
            current_run.total_wall_time_s = t_end - t_start;
            current_run.qps = nq / current_run.total_wall_time_s;

            std::vector<double> latencies_ms;
            for (int j = 0; j < nq; ++j) {
                latencies_ms.push_back(latency_stats[j].total_us / 1000.0);
            }
            std::sort(latencies_ms.begin(), latencies_ms.end());
            
            current_run.avg_latency_ms = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) / nq;
            current_run.p50_latency_ms = latencies_ms[nq / 2];
            current_run.p99_latency_ms = latencies_ms[int(nq * 0.99)];
            
            run_results.push_back(current_run);

            printf("Run %d/%d: Total Time = %.4f s, QPS = %.2f\n", 
                   i + 1, num_runs_per_thread_setting, 
                   current_run.total_wall_time_s, current_run.qps);
        }

        BenchmarkResult avg_result;
        double sum_total_time = 0.0, sum_qps = 0.0, sum_avg_lat = 0.0, sum_p50 = 0.0, sum_p99 = 0.0;
        
        for (int i = 1; i < num_runs_per_thread_setting; ++i) {
            sum_total_time += run_results[i].total_wall_time_s;
            sum_qps += run_results[i].qps;
            sum_avg_lat += run_results[i].avg_latency_ms;
            sum_p50 += run_results[i].p50_latency_ms;
            sum_p99 += run_results[i].p99_latency_ms;
        }

        int count = num_runs_per_thread_setting > 1 ? num_runs_per_thread_setting - 1 : 1;
        avg_result.total_wall_time_s = sum_total_time / count;
        avg_result.qps = sum_qps / count;
        avg_result.avg_latency_ms = sum_avg_lat / count;
        avg_result.p50_latency_ms = sum_p50 / count;
        avg_result.p99_latency_ms = sum_p99 / count;

        final_results.push_back({threads, avg_result});
    }

    // ==============================================================================
    // 6. 打印最终结果表格
    // ==============================================================================
    printf("\n\n===== Faiss Benchmark Summary (Average of last %d runs) =====\n", num_runs_per_thread_setting > 1 ? num_runs_per_thread_setting - 1 : 1);
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

    // ==============================================================================
    // 7. 计算召回率
    // ==============================================================================
    printf("\nPhase 5: Calculating recall\n");
    std::vector<int32_t> gt_data;
    read_ivecs(GROUNDTRUTH_FILE, gt_data, nq, k_gt);

    std::vector<faiss::idx_t> I_recall(k * nq);
    std::vector<float> D_recall(k * nq);
    omp_set_num_threads(10); // Use a reasonable number of threads for the final search
    index_final->search(nq, xq.data(), k, D_recall.data(), I_recall.data());

    int total_found = 0;
    for (int i = 0; i < nq; ++i) {
        std::unordered_set<int32_t> gt_set;
        // Groundtruth might have more neighbors than k, so only check against the top k
        for (int j = 0; j < k; ++j) { 
            gt_set.insert(gt_data[i * k_gt + j]);
        }

        for (int j = 0; j < k; ++j) {
            if (gt_set.count(I_recall[i * k + j])) {
                total_found++;
            }
        }
    }
    double recall = (double)total_found / (nq * k);
    printf("Recall@%d: %.4f\n", k, recall);

    // ==============================================================================
    // 8. 报告峰值内存
    // ==============================================================================
    printf("\n==================================================================\n");
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    // ru_maxrss is in kilobytes on Linux
    double peak_memory_mb = r_usage.ru_maxrss / 1024.0;
    printf("Peak memory usage: %.2f MB\n", peak_memory_mb);
    printf("==================================================================\n");
    
    delete index_final_generic;
    return 0;
}