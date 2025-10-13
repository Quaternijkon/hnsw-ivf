// main_simple_benchmark_no_stats.cpp

// ===================================================================================
// 如何编译:
// 确保你已经编译并安装了我们修改过的、支持性能统计的Faiss版本。
// g++ -std=c++17 -O3 -o simple_benchmark main_simple_benchmark_no_stats.cpp \
//     -I/path/to/faiss/source \
//     -L/path/to/faiss/build/faiss \
//     -lfaiss -lopenblas -fopenmp
//
// 运行:
// ./simple_benchmark
// ===================================================================================

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

using idx_t = faiss::idx_t;




// ===================================================================================
// 主程序
// ===================================================================================
int main() {
    // -------------------------------------------------------------------------------
    // 1. 设置参数并生成随机数据
    // -------------------------------------------------------------------------------
    int d = 64;      // 向量维度
    int nb = 100000; // 数据库大小
    int nq = 10000;  // 查询数量

    std::cout << "生成随机数据..." << std::endl;

    std::vector<float> xb(d * nb);
    std::vector<float> xq(d * nq);

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++) {
            xb[d * i + j] = distrib(rng);
        }
        xb[d * i] += i / 1000.;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++) {
            xq[d * i + j] = distrib(rng);
        }
        xq[d * i] += i / 1000.;
    }

    // -------------------------------------------------------------------------------
    // 2. 构建并训练索引
    // -------------------------------------------------------------------------------
    int nlist = 100; // IVF分区数
    int k = 4;       // k-NN搜索的k值
    int nprobe = 10; // 搜索时访问的分区数

    std::cout << "构建并训练 IndexIVFFlat..." << std::endl;

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlat index(&quantizer, d, nlist);

    assert(!index.is_trained);
    index.train(nb, xb.data());
    assert(index.is_trained);
    index.add(nb, xb.data());

    std::cout << "索引已就绪 (ntotal=" << index.ntotal << ")" << std::endl;

    // -------------------------------------------------------------------------------
    // 3. 执行搜索并进行性能测试
    // -------------------------------------------------------------------------------
    std::cout << "\n==================== 性能测试 ====================" << std::endl;
    
    // 设置搜索参数
    index.nprobe = nprobe;
    
    // 准备用于收集性能数据的自定义参数
    faiss::IVFSearchParametersWithQueryStats params;
    params.nprobe = nprobe;
    faiss::QueryTimings timings(nq);   // 创建统计信息收集器
    params.query_timings = &timings;    // 将其指针传递给参数对象

    // 准备结果容器
    std::vector<idx_t> I(k * nq);
    std::vector<float> D(k * nq);

    std::cout << "执行搜索 (nprobe=" << nprobe << ", k=" << k << ")..." << std::endl;
    
    // NEW: 使用外部计时器来测量总耗时，以计算QPS
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // 调用我们修改过的search函数，传入自定义参数
    index.search(nq, xq.data(), k, D.data(), I.data(), &params);
    
    auto t_end = std::chrono::high_resolution_clock::now();
    
    // --- 分析并报告性能数据 ---
    
    // 宏观QPS
    std::chrono::duration<double, std::milli> total_ms = t_end - t_start;
    double qps = (nq / total_ms.count()) * 1000.0;
    
    // 微观Latency
    double sum_latency = std::accumulate(timings.per_query_latencies.begin(), timings.per_query_latencies.end(), 0.0);
    double avg_latency = sum_latency / nq;
    
    std::sort(timings.per_query_latencies.begin(), timings.per_query_latencies.end());
    size_t p999_idx = static_cast<size_t>(nq * 0.999);
    double p999_latency = timings.per_query_latencies[std::min(p999_idx, (size_t)nq - 1)];

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n--- 性能报告 ---" << std::endl;
    std::cout << "查询总数: " << nq << std::endl;
    std::cout << "宏观总耗时: " << total_ms.count() / 1000.0 << " 秒" << std::endl;
    std::cout << "宏观 QPS: " << qps << " queries/sec" << std::endl;
    std::cout << "--------------------" << std::endl;
    std::cout << "微观平均延迟: " << avg_latency << " ms" << std::endl;
    std::cout << "微观 P99.9 延迟: " << p999_latency << " ms" << std::endl;

    // -------------------------------------------------------------------------------
    // 4. 打印部分结果以验证正确性
    // -------------------------------------------------------------------------------
    std::cout << "\n--- 搜索结果验证 (最后5个查询) ---" << std::endl;
    std::cout << "Labels (I):" << std::endl;
    for (int i = nq - 5; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            std::cout << std::setw(5) << I[i * k + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Distances (D):" << std::endl;
    std::cout << std::fixed << std::setprecision(5);
    for (int i = nq - 5; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            std::cout << std::setw(10) << D[i * k + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "====================================================" << std::endl;

    return 0;
}