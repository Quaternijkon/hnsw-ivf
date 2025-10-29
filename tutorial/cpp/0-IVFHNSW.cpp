/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <cassert>
#include <regex>
#include <faiss/IndexIVFHNSW.h>
#include <faiss/index_io.h>
#include <omp.h>

using namespace std;
using idx_t = faiss::idx_t;

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

    size_t d = d_train;

    cout << "数据集: 训练集 " << nt << " 个, 基础集 " << nb << " 个, 查询集 " << nq << " 个, 维度 " << d << endl;

    // 设置参数
    size_t cell_size = 256;
    size_t nlist = nb / cell_size;
    size_t nprobe = 32;
    size_t chunk_size = 100000;  // 分块大小
    size_t k = 10;
    size_t M = 32;
    size_t efConstruction = 40;
    size_t efSearch = 16;

    cout << "索引参数: nlist=" << nlist << ", nprobe=" << nprobe << ", M=" << M 
         << ", efC=" << efConstruction << ", efS=" << efSearch << ", k=" << k << endl;

    // 生成索引文件名
    string base_name = BASE_FILE.substr(BASE_FILE.find_last_of("/") + 1);
    base_name = base_name.substr(0, base_name.find_last_of("."));
    regex non_alnum("[^a-zA-Z0-9_]");
    string clean_base_name = regex_replace(base_name, non_alnum, "_");
    string INDEX_FILE = DATA_DIR + "/" + clean_base_name + "_d" + to_string(d) + 
                        "_nlist" + to_string(nlist) + "_HNSWM" + to_string(M) + 
                        "_efc" + to_string(efConstruction) + "_IndexIVFHNSW.index";

    // 检查索引文件是否存在
    bool skip_index_building = false;
    ifstream index_check(INDEX_FILE);
    if (index_check.good()) {
        skip_index_building = true;
        cout << "加载已有索引: " << INDEX_FILE << endl;
    }

    if (!skip_index_building) {
        cout << "构建新索引: " << INDEX_FILE << endl;
        // 读取训练数据
        auto [xt_data, ____] = read_fbin(LEARN_FILE);
        float* xt = xt_data.data();

        // 训练索引
        auto start_time = chrono::high_resolution_clock::now();
        faiss::IndexIVFHNSW* index_for_training = new faiss::IndexIVFHNSW(d, nlist, M);
        index_for_training->set_hnsw_parameters(M, efConstruction, efSearch);
        index_for_training->verbose = false;
        index_for_training->train(nt, xt);
        auto end_time = chrono::high_resolution_clock::now();
        cout << "  训练耗时: " << fixed << setprecision(2) 
             << chrono::duration<double>(end_time - start_time).count() << "s" << endl;

        delete index_for_training;

        // 保存训练好的索引框架
        faiss::IndexIVFHNSW* index_shell = new faiss::IndexIVFHNSW(d, nlist, M);
        index_shell->set_hnsw_parameters(M, efConstruction, efSearch);
        index_shell->train(nt, xt);
        faiss::write_index(index_shell, INDEX_FILE.c_str());
        delete index_shell;

        // 分块添加向量
        int IO_FLAG_READ_WRITE = 0;
        faiss::Index* index_ondisk = faiss::read_index(INDEX_FILE.c_str(), IO_FLAG_READ_WRITE);
        
        auto start_time_add = chrono::high_resolution_clock::now();
        for (size_t i = 0; i < nb; i += chunk_size) {
            auto [xb_chunk_data, _____] = read_fbin(BASE_FILE, i, chunk_size);
            index_ondisk->add(min(chunk_size, nb - i), xb_chunk_data.data());
        }
        auto end_time_add = chrono::high_resolution_clock::now();
        cout << "  添加耗时: " << fixed << setprecision(2) 
             << chrono::duration<double>(end_time_add - start_time_add).count() << "s" << endl;
        
        faiss::write_index(index_ondisk, INDEX_FILE.c_str());
        delete index_ondisk;
    }

    // 使用mmap加载索引并搜索
    int IO_FLAG_MMAP = faiss::IO_FLAG_MMAP;
    faiss::Index* index_final = faiss::read_index(INDEX_FILE.c_str(), IO_FLAG_MMAP);
    
    faiss::IndexIVFHNSW* ivfhnsw_index = dynamic_cast<faiss::IndexIVFHNSW*>(index_final);
    if (ivfhnsw_index) {
        ivfhnsw_index->nprobe = nprobe;
        ivfhnsw_index->parallel_mode = 0;
        faiss::IndexHNSW* hnsw_quantizer = ivfhnsw_index->get_hnsw_quantizer();
        if (hnsw_quantizer) {
            hnsw_quantizer->hnsw.efSearch = efSearch;
        }
    }
    omp_set_num_threads(40);

    auto [xq_data, ______] = read_fbin(QUERY_FILE);
    float* xq = xq_data.data();
    vector<idx_t> I(nq * k);
    vector<float> D(nq * k);
    
    auto start_search = chrono::high_resolution_clock::now();
    index_final->search(nq, xq, k, D.data(), I.data());
    auto end_search = chrono::high_resolution_clock::now();
    double search_time = chrono::duration<double>(end_search - start_search).count();

    cout << "搜索: " << fixed << setprecision(2) << search_time << "s, QPS: " 
         << fixed << setprecision(0) << nq / search_time << endl;

    // 计算召回率
    ifstream gt_file_check(GROUNDTRUTH_FILE);
    if (gt_file_check.good()) {
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

            vector<idx_t> gt_i(k_gt);
            for (int m = 0; m < k_gt; ++m) {
                gt_i[m] = static_cast<idx_t>(record_data[m + 1]);
            }

            size_t found_count = 0;
            size_t check_size = min(k, static_cast<size_t>(k_gt));
            for (size_t j = 0; j < k; ++j) {
                idx_t neighbor = I[i * k + j];
                if (find(gt_i.begin(), gt_i.begin() + check_size, neighbor) != gt_i.begin() + check_size) {
                    ++found_count;
                }
            }
            total_found += found_count;
        }

        double recall = static_cast<double>(total_found) / (nq * k);
        cout << "Recall@" << k << ": " << fixed << setprecision(4) << recall << endl;
    }

    // 清理资源
    delete index_final;

    return 0;
}

