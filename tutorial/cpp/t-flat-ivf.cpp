/*
 * 此 C++ 程序是所提供 Python 脚本的直接转换，
 * 演示了如何为大规模数据集（如 GIST 1M）构建和查询磁盘上的 Faiss 索引。
 * 核心功能包括：
 * 1. 分块读取数据集并增量式地构建 On-disk IndexIVFFlat 索引。
 * 2. 在索引构建后，选择性地计算并保存 IVF 分区的统计信息。
 * 3. 使用内存映射（mmap）模式加载并查询最终的磁盘索引。
 * 4. 以流式（内存优化）方式计算召回率，以避免加载大型 groundtruth 文件。
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <regex>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#endif

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <faiss/invlists/InvertedLists.h>

// 兼容Windows编译环境
#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#endif

using idx_t = faiss::idx_t;

// ==============================================================================
// 1. 辅助函数
// ==============================================================================

inline bool file_exists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

inline void create_directory(const std::string& path) {
#if defined(_WIN32)
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0755); // POSIX
#endif
}

/**
 * @brief 读取 .fbin 文件，支持全量读取或分块读取。
 * @param filename 文件路径。
 * @param data 输出的向量数据。
 * @param n_chunk 输出的本次读取的向量数。
 * @param d 输出的向量维度。
 * @param n_total 输出的文件中总向量数。
 * @param start_idx 开始读取的向量索引。
 * @param chunk_size 要读取的向量数量，为0则表示读取全部。
 */
void read_fbin(
        const std::string& filename,
        std::vector<float>& data,
        size_t& n_chunk,
        int& d,
        size_t& n_total,
        size_t start_idx = 0,
        size_t chunk_size = 0) {
    std::ifstream f(filename, std::ios::in | std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "错误: 文件未找到 " << filename << std::endl;
        exit(1);
    }
    int n_total_i, d_i;
    f.read(reinterpret_cast<char*>(&n_total_i), sizeof(int));
    f.read(reinterpret_cast<char*>(&d_i), sizeof(int));
    n_total = n_total_i;
    d = d_i;

    size_t num_to_read = (chunk_size == 0) ? n_total : std::min(chunk_size, n_total - start_idx);
    n_chunk = num_to_read;
    
    data.resize(num_to_read * d);

    // 文件指针移动到数据起始位置
    size_t offset = 8 + start_idx * d * sizeof(float);
    f.seekg(offset, std::ios::beg);
    f.read(reinterpret_cast<char*>(data.data()), num_to_read * d * sizeof(float));
}


/**
 * @brief 以流式方式计算召回率，避免将整个 groundtruth 文件加载到内存。
 */
void calculate_recall_stream(
        const std::vector<idx_t>& I_results,
        size_t nq,
        size_t k,
        const std::string& gt_filename) {
    printf("\n============================================================\n");
    printf("Phase 5: 计算召回率 (内存优化版)\n");

    if (!file_exists(gt_filename)) {
        printf("Groundtruth文件未找到: %s\n", gt_filename.c_str());
        printf("跳过召回率计算。\n");
        return;
    }
    printf("以流式方式从 %s 读取 groundtruth 数据进行计算...\n", gt_filename.c_str());
    
    std::ifstream f(gt_filename, std::ios::in | std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "错误: 无法打开 groundtruth 文件 " << gt_filename << std::endl;
        return;
    }

    int k_gt;
    f.read(reinterpret_cast<char*>(&k_gt), sizeof(int));
    printf("Groundtruth 维度 (k_gt): %d\n", k_gt);

    long total_found = 0;
    std::vector<int32_t> gt_buffer(k_gt);

    printf("正在计算 Recall@%zu...\n", k);
    for (size_t i = 0; i < nq; ++i) {
        f.seekg(4, std::ios::cur); // 跳过每行的维度整数
        f.read(reinterpret_cast<char*>(gt_buffer.data()), k_gt * sizeof(int32_t));
        
        // 使用 unordered_set 以获得高效查找
        std::unordered_set<idx_t> gt_set;
        for (size_t j = 0; j < k; ++j) { // 我们只关心 top-k 的 groundtruth
            gt_set.insert(gt_buffer[j]);
        }

        for (size_t j = 0; j < k; ++j) {
            if (gt_set.count(I_results[i * k + j])) {
                total_found++;
            }
        }
    }
    
    double recall = static_cast<double>(total_found) / (nq * k);
    printf("\n查询了 %zu 个向量, k=%zu\n", nq, k);
    printf("在top-%zu的结果中，总共找到了 %ld 个真实的近邻。\n", k, total_found);
    printf("Recall@%zu: %.4f\n", k, recall);
}


// ==============================================================================
// 主函数
// ==============================================================================
int main() {
    // ==============================================================================
    // 0. 路径和文件名配置 & 调试开关
    // ==============================================================================
    const std::string DATA_DIR = "./sift";
    const std::string INDEX_DIR = DATA_DIR + "/ondisk-ivf";
    const std::string LEARN_FILE = DATA_DIR + "/learn.fbin";
    const std::string BASE_FILE = DATA_DIR + "/base.fbin";
    const std::string QUERY_FILE = DATA_DIR + "/query.fbin";
    const std::string GROUNDTRUTH_FILE = DATA_DIR + "/groundtruth.ivecs";
    const bool ENABLE_IVF_STATS = false;

    // ==============================================================================
    // 2. 设置参数与环境
    // ==============================================================================
    std::vector<float> tmp_data;
    size_t nt, nb, nq, n_chunk_tmp;
    int d, d_base, d_query;

    read_fbin(LEARN_FILE, tmp_data, n_chunk_tmp, d, nt, 0, 1);
    read_fbin(BASE_FILE, tmp_data, n_chunk_tmp, d_base, nb, 0, 1);
    read_fbin(QUERY_FILE, tmp_data, n_chunk_tmp, d_query, nq, 0, 1);

    if (d != d_base || d != d_query) {
        fprintf(stderr, "维度不一致: 训练集%d维, 基础集%d维, 查询集%d维\n", d, d_base, d_query);
        return 1;
    }

    const size_t cell_size = 128;
    const size_t nlist = nb / cell_size;
    const size_t nprobe = 128;
    const size_t chunk_size = 100000;
    const size_t k = 10;

    create_directory(INDEX_DIR);

    std::string base_name = std::regex_replace(BASE_FILE, std::regex(".*/"), "");
    base_name = std::regex_replace(base_name, std::regex("\\.fbin$"), "");
    base_name = std::regex_replace(base_name, std::regex("[^a-zA-Z0-9_]"), "_");
    const std::string INDEX_FILE = INDEX_DIR + "/" + base_name + "_d" + std::to_string(d) + "_nlist" + std::to_string(nlist) + "_FLATL2_IVFFlat.index";

    printf("============================================================\n");
    printf("Phase 0: 环境设置\n");
    printf("向量维度 (d): %d\n", d);
    printf("基础集大小 (nb): %zu, 训练集大小 (ntrain): %zu\n", nb, nt);
    printf("查询集大小 (nq): %zu, 分块大小 (chunk_size): %zu\n", nq, chunk_size);
    printf("索引将保存在磁盘文件: %s\n", INDEX_FILE.c_str());
    printf("IVF统计功能: %s\n", ENABLE_IVF_STATS ? "启用" : "禁用");
    printf("============================================================\n");

    // ==============================================================================
    // 3. 检查索引文件是否存在
    // ==============================================================================
    bool skip_index_building = file_exists(INDEX_FILE);
    if (skip_index_building) {
        printf("索引文件 %s 已存在，跳过索引构建阶段\n", INDEX_FILE.c_str());
    } else {
        printf("索引文件不存在，将构建新索引\n");
    }

    if (!skip_index_building) {
        // ==============================================================================
        // 4. 训练量化器并创建索引框架
        // ==============================================================================
        printf("\nPhase 1 & 2: 训练量化器并创建空的磁盘索引框架\n");
        faiss::IndexFlatL2 coarse_quantizer(d);
        faiss::IndexIVFFlat index_shell(&coarse_quantizer, d, nlist, faiss::METRIC_L2);
        index_shell.verbose = true;

        std::vector<float> xt;
        size_t nt_read;
        read_fbin(LEARN_FILE, xt, nt_read, d, nt);

        printf("训练聚类中心并构建 FlatL2 量化器...\n");
        auto start_time = std::chrono::high_resolution_clock::now();
        index_shell.train(nt, xt.data());
        auto end_time = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end_time - start_time).count();
        printf("量化器训练完成，耗时: %.2f 秒\n", duration);
        printf("粗量化器中的质心数量: %lld\n", index_shell.quantizer->ntotal);
        xt.clear(); xt.shrink_to_fit();

        printf("\n将空的索引框架写入磁盘: %s\n", INDEX_FILE.c_str());
        faiss::write_index(&index_shell, INDEX_FILE.c_str());

        // ==============================================================================
        // 6. 分块向磁盘索引中添加数据
        // ==============================================================================
        printf("\nPhase 3: 分块添加数据到磁盘索引\n");
        // <--- 修正 #1：使用 faiss::index_io 命名空间
        faiss::Index* idx_ptr = faiss::read_index(INDEX_FILE.c_str(), faiss::IO_FLAG_MMAP);
        faiss::IndexIVF* index_ondisk = dynamic_cast<faiss::IndexIVF*>(idx_ptr);
        if (!index_ondisk) {
            fprintf(stderr, "错误：磁盘上的索引不是 IVF 类型。\n");
            delete idx_ptr;
            return 1;
        }

        start_time = std::chrono::high_resolution_clock::now();
        size_t num_chunks = (nb + chunk_size - 1) / chunk_size;
        for (size_t i = 0; i < nb; i += chunk_size) {
            size_t chunk_idx = i / chunk_size + 1;
            printf("    -> 正在处理块 %zu/%zu: 向量 %zu 到 %zu\n", chunk_idx, num_chunks, i, std::min(i + chunk_size, nb) - 1);
            std::vector<float> xb_chunk;
            read_fbin(BASE_FILE, xb_chunk, n_chunk_tmp, d, nb, i, chunk_size);
            index_ondisk->add(n_chunk_tmp, xb_chunk.data());
        }
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(end_time - start_time).count();
        printf("\n所有数据块添加完成，总耗时: %.2f 秒\n", duration);
        printf("磁盘索引中的向量总数 (ntotal): %lld\n", index_ondisk->ntotal);

        // ===========================================================
        // 7. 输出IVF分区统计信息
        // ===========================================================
        if (ENABLE_IVF_STATS) {
            printf("\n输出IVF分区统计信息...\n");
            auto start_stats_time = std::chrono::high_resolution_clock::now();
            faiss::InvertedLists* invlists = index_ondisk->invlists;
            
            size_t non_empty_partitions = 0;
            size_t max_size = 0;
            size_t min_size = -1;
            long total_vectors_in_lists = 0;

            std::string stats_filename = std::regex_replace(INDEX_FILE, std::regex("\\.index$"), "") + "_ivf_stats.csv";
            std::ofstream stats_file(stats_filename);
            stats_file << "partition_id,vector_count\n";
            
            for (size_t list_id = 0; list_id < nlist; ++list_id) {
                size_t list_size = invlists->list_size(list_id);
                stats_file << list_id << "," << list_size << "\n";
                if (list_size > 0) {
                    non_empty_partitions++;
                    max_size = std::max(max_size, list_size);
                    if (min_size == -1 || list_size < min_size) {
                        min_size = list_size;
                    }
                    total_vectors_in_lists += list_size;
                }
            }
            stats_file.close();

            double avg_size = non_empty_partitions > 0 ? static_cast<double>(total_vectors_in_lists) / non_empty_partitions : 0.0;
            printf("IVF分区统计摘要:\n");
            printf("  分区总数: %zu\n", nlist);
            printf("  非空分区数: %zu (%.2f%%)\n", non_empty_partitions, static_cast<double>(non_empty_partitions) / nlist * 100.0);
            printf("  最大分区大小: %zu\n", max_size);
            printf("  最小分区大小: %zu\n", min_size == -1 ? 0 : min_size);
            printf("  平均分区大小: %.2f\n", avg_size);
            printf("分区统计信息已保存到: %s\n", stats_filename.c_str());
            auto end_stats_time = std::chrono::high_resolution_clock::now();
            printf("统计耗时: %.2f秒\n", std::chrono::duration<double>(end_stats_time - start_stats_time).count());
        }

        printf("\n正在将最终索引写回磁盘: %s\n", INDEX_FILE.c_str());
        faiss::write_index(index_ondisk, INDEX_FILE.c_str());
        delete index_ondisk; // delete idx_ptr 也可以，因为它是同一个对象
    }

    // ==============================================================================
    // 8. 使用内存映射 (mmap) 进行搜索
    // ==============================================================================
    printf("\nPhase 4: 使用内存映射模式进行搜索\n");
    printf("以 mmap 模式打开磁盘索引: %s\n", INDEX_FILE.c_str());

    // <--- 修正 #1：使用 faiss::index_io 命名空间
    faiss::Index* idx_final_ptr = faiss::read_index(INDEX_FILE.c_str(), faiss::IO_FLAG_MMAP);
    
    // <--- 修正 #2：将基类指针转换为 IndexIVF* 以设置 nprobe
    faiss::IndexIVF* index_ivf_final = dynamic_cast<faiss::IndexIVF*>(idx_final_ptr);
    if (!index_ivf_final) {
        fprintf(stderr, "错误：磁盘上的索引不是 IVF 类型，无法设置 nprobe。\n");
        delete idx_final_ptr;
        return 1;
    }
    index_ivf_final->nprobe = nprobe;
    printf("索引已准备好搜索 (nprobe=%zu)\n", index_ivf_final->nprobe);

    printf("从 query.fbin 加载查询向量...\n");
    std::vector<float> xq_vec;
    size_t nq_read;
    read_fbin(QUERY_FILE, xq_vec, nq_read, d, nq);
    
    std::vector<idx_t> I_results(nq * k);
    std::vector<float> D_results(nq * k);

    printf("执行搜索...\n");
    auto start_time = std::chrono::high_resolution_clock::now();
    // 使用转换后的指针或原始指针进行搜索都可以，因为 search 是虚函数
    index_ivf_final->search(nq, xq_vec.data(), k, D_results.data(), I_results.data());
    auto end_time = std::chrono::high_resolution_clock::now();
    double search_duration = std::chrono::duration<double>(end_time - start_time).count();

    printf("搜索完成，耗时: %.2f 秒\n", search_duration);
    if (search_duration > 0) {
        printf("QPS (每秒查询率): %.2f\n", nq / search_duration);
    } else {
        printf("搜索耗时过短，无法计算QPS\n");
    }

    // ==============================================================================
    // 9. 计算召回率
    // ==============================================================================
    calculate_recall_stream(I_results, nq, k, GROUNDTRUTH_FILE);

    // ==============================================================================
    // 10. 报告峰值内存
    // ==============================================================================
    printf("\n============================================================\n");
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
        }
    #endif
    printf("============================================================\n");

    delete idx_final_ptr;
    return 0;
}