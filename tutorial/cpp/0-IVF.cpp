/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>
#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

using faiss::QueryLatencyStats;
using idx_t = faiss::idx_t;

// === 新增一个结构体来保存单次运行的结果 ===
struct BenchmarkResult {
    double total_wall_time_s = 0.0;
    double qps = 0.0;
    double avg_latency_ms = 0.0;
    double p50_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
};

int main() {
    int d = 64;      // dimension
    int nb = 100000; // database size
    int nq = 10000;  // nb of queries

    printf("Generating data...\n");
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    // 使用 std::vector 替代裸指针，更安全
    std::vector<float> xb(d * nb);
    std::vector<float> xq(d * nq);

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }

    int nlist = 100;
    int k = 4;

    printf("Training and adding vectors to index...\n");
    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlat index(&quantizer, d, nlist);
    index.train(nb, xb.data());
    index.add(nb, xb.data());
    index.parallel_mode = 0;

    // === 基准测试循环设置 ===
    std::vector<int> thread_counts = {1, 10, 20, 30, 40, 50, 60};
    const int num_runs_per_thread_setting = 3;
    
    // 用于存储最终平均结果的向量
    std::vector<std::pair<int, BenchmarkResult>> final_results;

    printf("Starting benchmark...\n");

    // === 外层循环：遍历不同的线程数设置 ===
    for (int threads : thread_counts) {
        omp_set_num_threads(threads);
        
        std::vector<BenchmarkResult> run_results;
        printf("\n--- Testing with %d OMP threads ---\n", threads);

        // === 内层循环：为每个线程数设置重复运行多次 ===
        for (int i = 0; i < num_runs_per_thread_setting; ++i) {
            idx_t* I = new idx_t[k * nq];
            float* D = new float[k * nq];
            auto latency_stats = new QueryLatencyStats[nq];

            double t_start = omp_get_wtime();
            
            // 修正函数名: search_stats -> search_with_stats
            index.search_stats(nq, xq.data(), k, D, I, nullptr, latency_stats);

            double t_end = omp_get_wtime();
            
            // --- 数据收集 ---
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

            delete[] I;
            delete[] D;
            delete[] latency_stats;
        }

        // === 计算平均值（忽略第一次热身运行） ===
        BenchmarkResult avg_result;
        double sum_total_time = 0.0, sum_qps = 0.0, sum_avg_lat = 0.0, sum_p50 = 0.0, sum_p99 = 0.0;
        
        for (int i = 1; i < num_runs_per_thread_setting; ++i) { // 从 i=1 开始，忽略第一次
            sum_total_time += run_results[i].total_wall_time_s;
            sum_qps += run_results[i].qps;
            sum_avg_lat += run_results[i].avg_latency_ms;
            sum_p50 += run_results[i].p50_latency_ms;
            sum_p99 += run_results[i].p99_latency_ms;
        }

        int count = num_runs_per_thread_setting - 1;
        avg_result.total_wall_time_s = sum_total_time / count;
        avg_result.qps = sum_qps / count;
        avg_result.avg_latency_ms = sum_avg_lat / count;
        avg_result.p50_latency_ms = sum_p50 / count;
        avg_result.p99_latency_ms = sum_p99 / count;

        final_results.push_back({threads, avg_result});
    }

    // === 最后，打印格式化的总结果表格 ===
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

    return 0;
}