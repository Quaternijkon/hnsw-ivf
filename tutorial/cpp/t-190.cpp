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
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/utils/Heap.h>
#include <faiss/impl/io.h>
#include <faiss/impl/FaissAssert.h>
#include <omp.h>

using namespace std;

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
    size_t cell_size = 256;
    size_t nlist = nb / cell_size;
    size_t nprobe = 32;
    size_t chunk_size = 100000; // 每次处理的数据块大小
    size_t k = 10; // 查找最近的10个邻居

    size_t M = 32; // HNSW的连接数
    size_t efconstruction = 40; // 默认40
    size_t efsearch = 16;       // 默认16

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
    omp_set_num_threads(40);
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