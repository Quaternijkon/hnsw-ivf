#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <regex>
#include <algorithm>
#include <cmath>
#include <cstdint>
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

// 路径和文件名配置
const string DATA_DIR = "./sift";
const string LEARN_FILE = DATA_DIR + "/learn.fbin";
const string BASE_FILE = DATA_DIR + "/base.fbin";
const string QUERY_FILE = DATA_DIR + "/query.fbin";
const string GROUNDTRUTH_FILE = DATA_DIR + "/groundtruth.ivecs";

// 读取.fbin文件
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

int main() {
    // 获取数据集信息
    auto [_, meta_train] = read_fbin(LEARN_FILE, 0, 1);
    size_t nt = meta_train.first;
    size_t d_train = meta_train.second;

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

    // 设置参数
    size_t cell_size = 256;
    size_t nlist = nb / cell_size;
    size_t nprobe = 32;
    size_t chunk_size = 100000;
    size_t k = 10;
    size_t M = 32;
    size_t efconstruction = 40;
    size_t efsearch = 16;

    // 生成索引文件名
    string base_name = BASE_FILE.substr(BASE_FILE.find_last_of("/") + 1);
    base_name = base_name.substr(0, base_name.find_last_of("."));
    regex non_alnum("[^a-zA-Z0-9_]");
    string clean_base_name = regex_replace(base_name, non_alnum, "_");
    string INDEX_FILE = DATA_DIR + "/" + clean_base_name + "_d" + to_string(d_train) + "_nlist" + to_string(nlist) +
                        "_HNSWM" + to_string(M) + "_efc" + to_string(efconstruction) + "_IVFFlat.index";


    // 检查索引文件是否存在
    bool skip_index_building = false;
    ifstream index_check(INDEX_FILE);
    if (index_check.good()) {
        skip_index_building = true;
    }

    faiss::Index* coarse_quantizer = nullptr;

    if (!skip_index_building) {
        // 训练量化器
        coarse_quantizer = new faiss::IndexHNSWFlat(d_train, M);
        dynamic_cast<faiss::IndexHNSW*>(coarse_quantizer)->hnsw.efConstruction = efconstruction;
        dynamic_cast<faiss::IndexHNSW*>(coarse_quantizer)->hnsw.efSearch = efsearch;

        faiss::IndexIVFFlat* index_for_training = new faiss::IndexIVFFlat(coarse_quantizer, d_train, nlist, faiss::METRIC_L2);
        index_for_training->verbose = false;

        auto [xt_data, ___] = read_fbin(LEARN_FILE);
        float* xt = xt_data.data();

        auto start_time = chrono::high_resolution_clock::now();
        index_for_training->train(nt, xt);
        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end_time - start_time;

        delete index_for_training;

        // 创建空的磁盘索引框架
        faiss::IndexIVFFlat* index_shell = new faiss::IndexIVFFlat(coarse_quantizer, d_train, nlist, faiss::METRIC_L2);
        faiss::write_index(index_shell, INDEX_FILE.c_str());
        delete index_shell;

        // 分块添加数据
        int IO_FLAG_READ_WRITE = 0;
        faiss::Index* index_ondisk = faiss::read_index(INDEX_FILE.c_str(), IO_FLAG_READ_WRITE);
        auto start_time_add = chrono::high_resolution_clock::now();

        size_t num_chunks = (nb + chunk_size - 1) / chunk_size;
        for (size_t i = 0; i < nb; i += chunk_size) {
            auto [xb_chunk_data, ____] = read_fbin(BASE_FILE, i, chunk_size);
            float* xb_chunk = xb_chunk_data.data();
            index_ondisk->add(min(chunk_size, nb - i), xb_chunk);
        }

        auto end_time_add = chrono::high_resolution_clock::now();
        chrono::duration<double> duration_add = end_time_add - start_time_add;

        faiss::write_index(index_ondisk, INDEX_FILE.c_str());
        delete index_ondisk;
    }

    // 使用内存映射进行搜索
    int IO_FLAG_MMAP = faiss::IO_FLAG_MMAP;
    faiss::Index* index_final = faiss::read_index(INDEX_FILE.c_str(), IO_FLAG_MMAP);
    dynamic_cast<faiss::IndexIVF*>(index_final)->nprobe = nprobe;
    omp_set_num_threads(40);
    dynamic_cast<faiss::IndexIVF*>(index_final)->parallel_mode = 0;

    faiss::Index* generic_quantizer = dynamic_cast<faiss::IndexIVF*>(index_final)->quantizer;
    faiss::IndexHNSW* quantizer_hnsw = dynamic_cast<faiss::IndexHNSW*>(generic_quantizer);
    quantizer_hnsw->hnsw.efSearch = efsearch;

    auto [xq_data, _____] = read_fbin(QUERY_FILE);
    float* xq = xq_data.data();

    vector<faiss::idx_t> I(nq * k);
    vector<float> D(nq * k);
    auto start_time_search = chrono::high_resolution_clock::now();
    index_final->search(nq, xq, k, D.data(), I.data());
    auto end_time_search = chrono::high_resolution_clock::now();
    chrono::duration<double> search_duration = end_time_search - start_time_search;

    cout << "耗时: " << fixed << setprecision(2) << search_duration.count() << " 秒" << endl;
    if (search_duration.count() > 0) {
        double qps = nq / search_duration.count();
        cout << "QPS: " << fixed << setprecision(2) << qps << endl;
    }

    // 计算召回率
    ifstream gt_file_check(GROUNDTRUTH_FILE);
    if (!gt_file_check.good()) {
    } else {
        ifstream f(GROUNDTRUTH_FILE, ios::binary);
        int32_t k_gt;
        f.read(reinterpret_cast<char*>(&k_gt), sizeof(int32_t));

        f.seekg(0, ios::end);
        size_t total_file_size = f.tellg();
        size_t record_size_bytes = (k_gt + 1) * sizeof(int32_t);
        size_t num_gt_vectors = total_file_size / record_size_bytes;

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

        size_t check_size = min(k, static_cast<size_t>(k_gt));
        double recall = static_cast<double>(total_found) / (nq * check_size);
        cout << "Recall@" << k << ": " << fixed << setprecision(4) << recall << endl;
    }


    delete index_final;
    if (!skip_index_building) {
        delete coarse_quantizer;
    }

    return 0;
}

