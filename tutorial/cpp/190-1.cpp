#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <faiss/Index.h>
#include <faiss/IVFlib.h>
#include <faiss/AutoTune.h>
#include <faiss/utils/distances.h>
#include <faiss/impl/io.h>
#include <faiss/impl/io_macros.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <memory>
#include <algorithm>
#include <regex>
#include <sys/resource.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <unistd.h>
#endif

using namespace std;
using namespace std::chrono;

// ==============================================================================
// 配置参数
// ==============================================================================
const string DATA_DIR = "./sift";
const string LEARN_FILE = DATA_DIR + "/learn.fbin";
const string BASE_FILE = DATA_DIR + "/base.fbin";
const string QUERY_FILE = DATA_DIR + "/query.fbin";
const string GROUNDTRUTH_FILE = DATA_DIR + "/groundtruth.ivecs";

// 调试开关
const bool ENABLE_IVF_STATS = false;
const bool ENABLE_SEARCH_PARTITION_STATS = false;
const string SEARCH_STATS_FILENAME = DATA_DIR + "/search_partition_ratios.txt";

// 参数设置
int cell_size = 256;
int nprobe = 32;
int chunk_size = 100000;
int k = 10;
int M = 32;
int efconstruction = 40;
int efsearch = 16;

// 全局变量
int d_train = 0, nt = 0;
int d_base = 0, nb = 0;
int d_query = 0, nq = 0;
int nlist = 0;
string INDEX_FILE;

// ==============================================================================
// 辅助函数：读取.fbin文件
// ==============================================================================
vector<vector<float>> read_fbin(const string& filename, int start_idx = 0, int* chunk_size_ptr = nullptr) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("无法打开文件: " + filename);
    }
    
    int nvecs, dim;
    file.read(reinterpret_cast<char*>(&nvecs), sizeof(int));
    file.read(reinterpret_cast<char*>(&dim), sizeof(int));
    
    if (chunk_size_ptr == nullptr) {
        // 读取整个文件
        vector<vector<float>> data(nvecs, vector<float>(dim));
        for (int i = 0; i < nvecs; i++) {
            file.read(reinterpret_cast<char*>(data[i].data()), dim * sizeof(float));
        }
        return data;
    } else {
        // 读取指定块
        int actual_chunk_size = min(*chunk_size_ptr, nvecs - start_idx);
        vector<vector<float>> data(actual_chunk_size, vector<float>(dim));
        
        file.seekg(start_idx * dim * sizeof(float), ios::cur);
        for (int i = 0; i < actual_chunk_size; i++) {
            file.read(reinterpret_cast<char*>(data[i].data()), dim * sizeof(float));
        }
        
        *chunk_size_ptr = actual_chunk_size;
        return data;
    }
}

// 获取文件元数据（向量数量和维度）
void get_fbin_metadata(const string& filename, int* nvecs, int* dim) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("无法打开文件: " + filename);
    }
    
    file.read(reinterpret_cast<char*>(nvecs), sizeof(int));
    file.read(reinterpret_cast<char*>(dim), sizeof(int));
}

// ==============================================================================
// 辅助函数：读取.ivecs文件
// ==============================================================================
vector<vector<int>> read_ivecs(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("无法打开文件: " + filename);
    }
    
    vector<vector<int>> data;
    while (!file.eof()) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (file.eof()) break;
        
        vector<int> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int));
        data.push_back(vec);
    }
    
    return data;
}

// ==============================================================================
// 内存使用统计函数
// ==============================================================================
size_t getPeakMemoryUsage() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.PeakWorkingSetSize;
    }
    return 0;
#else
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss * 1024; // Linux returns KB
#endif
}

// ==============================================================================
// 主函数
// ==============================================================================
int main() {
    // 获取数据集信息
    get_fbin_metadata(LEARN_FILE, &nt, &d_train);
    get_fbin_metadata(BASE_FILE, &nb, &d_base);
    get_fbin_metadata(QUERY_FILE, &nq, &d_query);
    
    // 验证维度一致性
    if (d_train != d_base || d_train != d_query) {
        cerr << "维度不一致: 训练集" << d_train << "维, 基础集" << d_base << "维, 查询集" << d_query << "维" << endl;
        return 1;
    }
    
    // 计算nlist
    nlist = nb / cell_size;
    
    // 创建索引文件名
    string base_name = BASE_FILE.substr(BASE_FILE.find_last_of("/\\") + 1);
    base_name = base_name.substr(0, base_name.find_last_of('.'));
    regex special_chars("[^a-zA-Z0-9_]");
    string clean_base_name = regex_replace(base_name, special_chars, "_");
    INDEX_FILE = DATA_DIR + "/" + clean_base_name + "_d" + to_string(d_train) + 
                "_nlist" + to_string(nlist) + "_HNSWM" + to_string(M) + 
                "_efc" + to_string(efconstruction) + "_IVFFlat.index";
    
    // 打印环境信息
    cout << string(60, '=') << endl;
    cout << "Phase 0: 环境设置" << endl;
    cout << "向量维度 (d): " << d_train << endl;
    cout << "基础集大小 (nb): " << nb << ", 训练集大小 (ntrain): " << nt << endl;
    cout << "查询集大小 (nq): " << nq << ", 分块大小 (chunk_size): " << chunk_size << endl;
    cout << "HNSW M (构建参数): " << M << endl;
    cout << "HNSW efConstruction (构建参数): " << efconstruction << endl;
    cout << "索引将保存在磁盘文件: " << INDEX_FILE << endl;
    cout << "IVF统计功能: " << (ENABLE_IVF_STATS ? "启用" : "禁用") << endl;
    cout << "搜索分区统计功能: " << (ENABLE_SEARCH_PARTITION_STATS ? "启用" : "禁用") << endl;
    cout << string(60, '=') << endl;
    
    // 检查索引文件是否存在
    bool skip_index_building = false;
    ifstream index_file(INDEX_FILE);
    if (index_file.good()) {
        cout << "索引文件 " << INDEX_FILE << " 已存在，跳过索引构建阶段" << endl;
        skip_index_building = true;
    } else {
        cout << "索引文件不存在，将构建新索引" << endl;
        skip_index_building = false;
    }
    
    // 训练量化器和构建索引
    if (!skip_index_building) {
        cout << "\nPhase 1: 训练 HNSW 粗量化器 (in-memory)" << endl;
        
        // 创建HNSW量化器
        faiss::IndexHNSWFlat* coarse_quantizer = new faiss::IndexHNSWFlat(d_train, M);
        coarse_quantizer->hnsw.efConstruction = efconstruction;
        coarse_quantizer->hnsw.efSearch = efsearch;
        cout << "efconstruction: " << coarse_quantizer->hnsw.efConstruction 
             << ", efSearch: " << coarse_quantizer->hnsw.efSearch << endl;
        
        // 创建IVF索引用于训练
        faiss::IndexIVFFlat index_for_training(coarse_quantizer, d_train, nlist);
        index_for_training.verbose = true;
        
        // 读取训练数据
        cout << "读取训练数据..." << endl;
        auto xt = read_fbin(LEARN_FILE);
        
        // 训练量化器
        cout << "训练聚类中心并构建 HNSW 量化器..." << endl;
        auto start_time = high_resolution_clock::now();
        
        // 转换数据格式
        vector<float> train_data_flat;
        for (const auto& vec : xt) {
            train_data_flat.insert(train_data_flat.end(), vec.begin(), vec.end());
        }
        
        index_for_training.train(nt, train_data_flat.data());
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);
        
        cout << "量化器训练完成，耗时: " << duration.count() / 1000.0 << " 秒" << endl;
        cout << "粗量化器中的质心数量: " << coarse_quantizer->ntotal << endl;
        
        // 创建空的索引框架
        cout << "\nPhase 2: 创建空的磁盘索引框架" << endl;
        faiss::IndexIVFFlat index_shell(coarse_quantizer, d_train, nlist);
        cout << "将空的索引框架写入磁盘: " << INDEX_FILE << endl;
        faiss::write_index(&index_shell, INDEX_FILE.c_str());
        
        // 分块添加数据到磁盘索引
        cout << "\nPhase 3: 分块添加数据到磁盘索引" << endl;
        
        // 以读写模式打开索引
        faiss::Index* index_ondisk = faiss::read_index(INDEX_FILE.c_str(), faiss::IO_FLAG_MMAP);
        auto start_add_time = high_resolution_clock::now();
        
        int num_chunks = (nb + chunk_size - 1) / chunk_size;
        for (int i = 0; i < nb; i += chunk_size) {
            int chunk_idx = i / chunk_size + 1;
            int current_chunk_size = min(chunk_size, nb - i);
            cout << "       -> 正在处理块 " << chunk_idx << "/" << num_chunks 
                 << ": 向量 " << i << " 到 " << i + current_chunk_size - 1 << endl;
            
            // 读取数据块
            auto xb_chunk = read_fbin(BASE_FILE, i, &current_chunk_size);
            
            // 转换数据格式
            vector<float> chunk_data_flat;
            for (const auto& vec : xb_chunk) {
                chunk_data_flat.insert(chunk_data_flat.end(), vec.begin(), vec.end());
            }
            
            // 添加到索引
            index_ondisk->add(current_chunk_size, chunk_data_flat.data());
        }
        
        auto end_add_time = high_resolution_clock::now();
        auto add_duration = duration_cast<milliseconds>(end_add_time - start_add_time);
        cout << "\n所有数据块添加完成，总耗时: " << add_duration.count() / 1000.0 << " 秒" << endl;
        cout << "磁盘索引中的向量总数 (ntotal): " << index_ondisk->ntotal << endl;
        
        // 保存索引到磁盘
        cout << "正在将最终索引写回磁盘: " << INDEX_FILE << endl;
        faiss::write_index(index_ondisk, INDEX_FILE.c_str());
        delete index_ondisk;
    }
    
    // 使用内存映射进行搜索
    cout << "\nPhase 4: 使用内存映射模式进行搜索" << endl;
    cout << "以 mmap 模式打开磁盘索引: " << INDEX_FILE << endl;
    
    faiss::Index* index_final_generic = faiss::read_index(INDEX_FILE.c_str(), 4 /* IO_FLAG_MMAP */);
    faiss::IndexIVFFlat* index_final = dynamic_cast<faiss::IndexIVFFlat*>(index_final_generic);
    index_final->nprobe = nprobe;
    
    // 设置线程数
    omp_set_num_threads(40);
    if (auto idx_ivf = dynamic_cast<faiss::IndexIVF*>(index_final)) {
        idx_ivf->parallel_mode = 0;
    }
    
    cout << "并行模式线程数: " << omp_get_max_threads() << endl;
    cout << "索引已准备好搜索 (nprobe=" << index_final->nprobe << ")" << endl;
    
    // 读取查询数据
    cout << "从 query.fbin 加载查询向量..." << endl;
    auto xq = read_fbin(QUERY_FILE);
    
    // 转换查询数据格式
    vector<float> query_data_flat;
    for (const auto& vec : xq) {
        query_data_flat.insert(query_data_flat.end(), vec.begin(), vec.end());
    }
    
    // 执行搜索
    cout << "\n执行搜索..." << endl;
    vector<faiss::idx_t> I(nq * k);
    vector<float> D(nq * k);
    
    auto start_search_time = high_resolution_clock::now();
    index_final->search(nq, query_data_flat.data(), k, D.data(), I.data());
    auto end_search_time = high_resolution_clock::now();
    auto search_duration = duration_cast<milliseconds>(end_search_time - start_search_time);
    
    // 计算QPS
    double search_seconds = search_duration.count() / 1000.0;
    double qps = nq / search_seconds;
    
    cout << "\n========== 搜索性能统计 ==========" << endl;
    cout << "查询向量总数 (nq): " << nq << endl;
    cout << "总搜索时间: " << search_seconds << " 秒" << endl;
    cout << "QPS (每秒查询率): " << qps << endl;
    cout << "====================================" << endl;
    
    // 计算召回率
    cout << "\n" << string(60, '=') << endl;
    cout << "Phase 5: 计算召回率" << endl;
    
    ifstream gt_file(GROUNDTRUTH_FILE, ios::binary);
    if (!gt_file.is_open()) {
        cout << "Groundtruth文件未找到: " << GROUNDTRUTH_FILE << endl;
        cout << "跳过召回率计算。" << endl;
    } else {
        // 读取groundtruth数据
        auto gt_data = read_ivecs(GROUNDTRUTH_FILE);
        
        int total_found = 0;
        for (int i = 0; i < nq; i++) {
            vector<faiss::idx_t> query_result(I.begin() + i * k, I.begin() + (i + 1) * k);
            
            int found_count = 0;
            for (int j = 0; j < k; j++) {
                if (find(gt_data[i].begin(), gt_data[i].end(), query_result[j]) != gt_data[i].end()) {
                    found_count++;
                }
            }
            total_found += found_count;
        }
        
        double recall = static_cast<double>(total_found) / (nq * k);
        cout << "\n查询了 " << nq << " 个向量, k=" << k << endl;
        cout << "在top-" << k << "的结果中，总共找到了 " << total_found << " 个真实的近邻。" << endl;
        cout << "Recall@" << k << ": " << recall << endl;
    }
    
    cout << string(60, '=') << endl;
    
    // 报告峰值内存
    cout << "\n" << string(60, '=') << endl;
    size_t peak_memory = getPeakMemoryUsage();
    cout << "整个程序运行期间的峰值内存占用: " << peak_memory / (1024.0 * 1024.0) << " MB" << endl;
    cout << string(60, '=') << endl;
    
    // 清理资源
    delete index_final;
    
    return 0;
}