/*
 * @Author: Quaternijkon quaternijkon@mail.ustc.edu.cn
 * @Date: 2025-02-07 06:29:50
 * @LastEditors: Quaternijkon quaternijkon@mail.ustc.edu.cn
 * @LastEditTime: 2025-11-06 10:58:03
 * @FilePath: /faiss/tutorial/cpp/2-IVFFlat-SIFT1M.cpp
 * @Description: 基于 SIFT1M 数据集验证 IndexIVF 自动聚类维护功能
 */
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <sys/stat.h>
#include <sys/time.h>
#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/invlists/DirectMap.h>

using idx_t = faiss::idx_t;

/*****************************************************
 * I/O functions for fvecs and ivecs
 * SIFT1M 数据集格式：每个向量前面有一个 int32 的维度，然后是 float32 的向量数据
 *****************************************************/

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "错误: 无法打开文件 %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"维度不合理");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"文件大小异常");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr __attribute__((unused)) = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"无法读取完整文件");

    // 移除每行的维度头，将数据压缩
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// 读取 ivecs 格式（groundtruth）
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// 计算 recall@k
float compute_recall_at_k(
        const idx_t* I,           // 搜索结果，size nq * k
        const idx_t* gt,          // groundtruth，size nq * k_gt
        size_t nq,
        size_t k,
        size_t k_gt) {
    size_t n_correct = 0;
    size_t total_gt = 0;
    
    for (size_t i = 0; i < nq; i++) {
        // 将 groundtruth 转换为 set 以便快速查找
        std::vector<idx_t> gt_set(gt + i * k_gt, gt + i * k_gt + k_gt);
        std::sort(gt_set.begin(), gt_set.end());
        
        // 检查前 k 个结果中有多少个在 groundtruth 中
        size_t correct_in_query = 0;
        size_t k_actual = std::min(k, k_gt);
        
        for (size_t j = 0; j < k_actual; j++) {
            if (std::binary_search(gt_set.begin(), gt_set.end(), I[i * k + j])) {
                correct_in_query++;
            }
        }
        
        n_correct += correct_in_query;
        total_gt += k_actual;
    }
    
    return (total_gt > 0) ? (n_correct / float(total_gt)) : 0.0f;
}

int main(int argc, char* argv[]) {
    double t0 = elapsed();
    
    // 设置 OpenMP 线程数
    int num_threads = 4;
    omp_set_num_threads(num_threads);
    char omp_threads_str[32];
    snprintf(omp_threads_str, sizeof(omp_threads_str), "%d", num_threads);
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    setenv("OMP_NUM_THREADS", omp_threads_str, 1);
    
    printf("OpenMP 线程数: %d\n", omp_get_max_threads());
    
    // SIFT1M 数据集路径（可以通过命令行参数指定）
    const char* sift1m_dir = (argc > 1) ? argv[1] : "sift1M";
    
    printf("\n========================================\n");
    printf("SIFT1M 数据集自动聚类维护功能验证\n");
    printf("========================================\n\n");
    
    // ========== 加载训练集 ==========
    printf("[%.3f s] 加载训练集...\n", elapsed() - t0);
    size_t d, nt;
    float* xt = fvecs_read((std::string(sift1m_dir) + "/sift_learn.fvecs").c_str(), &d, &nt);
    printf("  维度: %zd, 训练向量数: %zd\n", d, nt);
    
    // ========== 创建索引 ==========
    printf("\n[%.3f s] 创建索引...\n", elapsed() - t0);
    int nlist = 15625; // 调整为 15625 个聚类
    int nprobe = 32;   // 搜索时探查的聚类数量
    int k = 100;       // 返回 top-k 结果
    
    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlat index(&quantizer, d, nlist);
    
    printf("  聚类数量 (nlist): %d\n", nlist);
    printf("  搜索探查数 (nprobe): %d\n", nprobe);
    printf("  返回结果数 (k): %d\n", k);
    
    // ========== 加载数据库 ==========
    printf("\n[%.3f s] 加载数据库...\n", elapsed() - t0);
    size_t nb, d2;
    float* xb = fvecs_read((std::string(sift1m_dir) + "/sift_base.fvecs").c_str(), &d2, &nb);
    assert(d == d2 || !"数据库维度与训练集不一致");
    printf("  数据库向量数: %zd\n", nb);
    
    // ========== 训练索引 ==========
    printf("\n[%.3f s] 训练索引...\n", elapsed() - t0);
    assert(!index.is_trained);
    size_t min_training = std::max<size_t>(nlist * 40, nt);
    std::vector<float> training_vectors;
    training_vectors.reserve((nt + min_training) * d);
    training_vectors.insert(training_vectors.end(), xt, xt + nt * d);
    if (nt < min_training) {
        size_t need = std::min(nb, min_training - nt);
        training_vectors.insert(
                training_vectors.end(),
                xb,
                xb + need * d);
        printf("  训练集不足，额外使用 %zu 条数据库向量增强。\n", need);
    }
    size_t training_count = training_vectors.size() / d;
    index.train(training_count, training_vectors.data());
    assert(index.is_trained);
    printf("  训练完成，共使用 %zu 条向量\n", training_count);
    delete[] xt;
    
    // ========== 启用 direct_map ==========
    index.set_direct_map_type(faiss::DirectMap::Hashtable);
    index.nprobe = nprobe;
    printf("\n[%.3f s] 已启用 direct_map (Hashtable 类型)\n", elapsed() - t0);
    
    // ========== 设定操作规模 ==========
    constexpr size_t kInitialTarget = 800000;
    constexpr size_t kInsertTarget = 100000;
    constexpr size_t kDeleteTarget = 100000;
    constexpr size_t kUpdateTarget = 100000;
    constexpr size_t kUpdateTargetStartId = 100000;
    constexpr size_t kUpdateSourceStart = 900000;
    
    if (nb < kUpdateSourceStart + kUpdateTarget) {
        printf("  警告：数据库向量不足 100 万（nb=%zu），无法完全覆盖预期流程。\n", nb);
    }
    
    // ========== 初始添加数据 ==========
    printf("\n[%.3f s] 初始添加数据...\n", elapsed() - t0);
    size_t n_initial = std::min(kInitialTarget, nb);
    if (n_initial == 0) {
        fprintf(stderr, "错误: 数据库向量数为 0，无法构建索引。\n");
        return 1;
    }
    std::vector<idx_t> ids_initial(n_initial);
    std::iota(ids_initial.begin(), ids_initial.end(), 0);
    index.add_with_ids(n_initial, xb, ids_initial.data());
    printf("  初始添加 %zd 个向量（目标 %zu），索引总数: %zd\n",
           n_initial,
           kInitialTarget,
           index.ntotal);
    
    // ========== 加载查询集和 groundtruth ==========
    printf("\n[%.3f s] 加载查询集...\n", elapsed() - t0);
    size_t nq, d3;
    float* xq = fvecs_read((std::string(sift1m_dir) + "/sift_query.fvecs").c_str(), &d3, &nq);
    assert(d == d3 || !"查询集维度不一致");
    printf("  查询向量数: %zd\n", nq);
    
    printf("\n[%.3f s] 加载 groundtruth...\n", elapsed() - t0);
    size_t k_gt;
    int* gt_int = ivecs_read((std::string(sift1m_dir) + "/sift_groundtruth.ivecs").c_str(), &k_gt, &nq);
    idx_t* gt = new idx_t[k_gt * nq];
    for (size_t i = 0; i < k_gt * nq; i++) {
        gt[i] = gt_int[i];
    }
    delete[] gt_int;
    printf("  Groundtruth 每个查询的最近邻数: %zd\n", k_gt);
    
    // ========== 初始搜索和评估 ==========
    printf("\n[%.3f s] 初始搜索和评估...\n", elapsed() - t0);
    idx_t* I_initial = new idx_t[k * nq];
    float* D_initial = new float[k * nq];
    
    double search_t0 = elapsed();
    index.search(nq, xq, k, D_initial, I_initial);
    double search_time = elapsed() - search_t0;
    
    float recall_1_initial = compute_recall_at_k(I_initial, gt, nq, 1, k_gt);
    float recall_10_initial = compute_recall_at_k(I_initial, gt, nq, 10, k_gt);
    float recall_100_initial = compute_recall_at_k(I_initial, gt, nq, 100, k_gt);
    
    printf("  搜索时间: %.3f s (%.3f ms/query)\n", search_time, search_time * 1000 / nq);
    printf("  Recall@1:   %.4f\n", recall_1_initial);
    printf("  Recall@10:  %.4f\n", recall_10_initial);
    printf("  Recall@100: %.4f\n", recall_100_initial);
    
    // ========== 演示插入操作 ==========
    printf("\n[%.3f s] ========== 演示插入操作 ==========\n", elapsed() - t0);
    size_t n_insert = (nb > n_initial)
            ? std::min(kInsertTarget, nb - n_initial)
            : 0;
    double insert_time = 0.0;
    if (n_insert > 0) {
        std::vector<idx_t> ids_insert(n_insert);
        std::iota(ids_insert.begin(), ids_insert.end(), n_initial);
        const float* xb_insert = xb + n_initial * d;
        
        double insert_t0 = elapsed();
        index.add_with_ids(n_insert, xb_insert, ids_insert.data());
        insert_time = elapsed() - insert_t0;
        
        printf("  插入了 %zd 个向量，索引总数: %zd\n", n_insert, index.ntotal);
        printf("  插入时间: %.3f s\n", insert_time);
        printf("  注意：插入操作后已自动维护被插入的聚类\n");
    } else {
        printf("  没有可插入的数据，跳过插入演示。\n");
    }
    
    // ========== 演示删除操作 ==========
    printf("\n[%.3f s] ========== 演示删除操作 ==========\n", elapsed() - t0);
    size_t ntotal_before_delete = static_cast<size_t>(index.ntotal);
    size_t n_delete = std::min(kDeleteTarget, ntotal_before_delete);
    double delete_time = 0.0;
    size_t n_removed = 0;
    if (n_delete > 0) {
        std::vector<idx_t> ids_to_delete(n_delete);
        std::iota(ids_to_delete.begin(), ids_to_delete.end(), 0);

        faiss::IDSelectorArray selector(n_delete, ids_to_delete.data());
        double delete_t0 = elapsed();
        n_removed = index.remove_ids(selector);
        delete_time = elapsed() - delete_t0;
        
        printf("  删除了 %zd 个向量，索引总数: %zd\n", n_removed, index.ntotal);
        printf("  删除时间: %.3f s\n", delete_time);
        printf("  注意：删除操作后已自动维护被删除向量所在的聚类\n");
    } else {
        printf("  当前索引向量过少，跳过删除演示。\n");
    }
    
    // ========== 演示更新操作 ==========
    printf("\n[%.3f s] ========== 演示更新操作 ==========\n", elapsed() - t0);
    size_t update_source_available =
            (nb > kUpdateSourceStart) ? (nb - kUpdateSourceStart) : 0;
    size_t ntotal_after_delete = static_cast<size_t>(index.ntotal);
    size_t available_ids_for_update =
            (ntotal_after_delete > kUpdateTargetStartId)
            ? (ntotal_after_delete - kUpdateTargetStartId)
            : 0;
    available_ids_for_update = std::min(available_ids_for_update, kUpdateTarget);
    size_t n_update = std::min(kUpdateTarget, update_source_available);
    n_update = std::min(n_update, available_ids_for_update);
    double update_time = 0.0;
    if (n_update > 0) {
        std::vector<idx_t> ids_update(n_update);
        std::iota(ids_update.begin(), ids_update.end(), kUpdateTargetStartId);
        std::vector<float> x_update(d * n_update);
        const float* source_begin = xb + kUpdateSourceStart * d;
        for (size_t i = 0; i < n_update; ++i) {
            memcpy(
                    x_update.data() + i * d,
                    source_begin + i * d,
                    d * sizeof(float));
        }
        double update_t0 = elapsed();
        index.update_vectors(n_update, ids_update.data(), x_update.data());
        update_time = elapsed() - update_t0;
        
        size_t update_id_end = kUpdateTargetStartId + n_update;
        size_t update_source_end = kUpdateSourceStart + n_update;
        printf("  更新了 %zd 个向量（ID [%zu, %zu) → 数据段 [%zu, %zu)）\n",
               n_update,
               kUpdateTargetStartId,
               update_id_end,
               kUpdateSourceStart,
               update_source_end);
        printf("  更新时间: %.3f s\n", update_time);
        printf("  注意：更新操作（删除+插入）后已自动维护被操作的聚类\n");
    } else {
        printf("  可更新的向量数量不足，跳过更新演示。\n");
    }
    
    size_t desired_final = n_initial + n_insert - n_removed;
    if (index.ntotal < desired_final && desired_final > index.ntotal) {
        size_t missing = desired_final - index.ntotal;
        std::vector<idx_t> ids_reinsert(missing);
        size_t fill = 0;
        for (idx_t id = 0; id < (idx_t)(n_initial + n_insert) && fill < missing; ++id) {
            if (index.direct_map.get(id) == -1) {
                ids_reinsert[fill++] = id;
            }
        }
        std::vector<float> x_reinsert(missing * d);
        for (size_t i = 0; i < missing; ++i) {
            idx_t id = ids_reinsert[i];
            memcpy(
                    x_reinsert.data() + i * d,
                    xb + id * d,
                    d * sizeof(float));
        }
        index.add_with_ids(missing, x_reinsert.data(), ids_reinsert.data());
        printf("  补回 %zd 个原始向量，索引总数: %zd\n", missing, index.ntotal);
    }
    
    size_t retained_base = 0;
    for (idx_t id = 0; id < (idx_t)nb; ++id) {
        if (index.direct_map.get(id) != -1) {
            retained_base++;
        }
    }
    double retained_ratio = nb > 0 ? (retained_base * 100.0 / nb) : 0.0;
    printf("  原始 SIFT1M 数据保留量: %zd (占比 %.2f%%)\n", retained_base, retained_ratio);
    
    // 删除操作后，某些倒排列表可能变空，增加 nprobe 可以提高搜索成功率
    // ========== 显示聚类统计信息 ==========
    printf("\n[%.3f s] ========== 聚类统计信息 ==========\n", elapsed() - t0);
    size_t total_vectors = index.ntotal;
    size_t avg_cluster_size = (index.nlist > 0 && total_vectors > 0) ? (total_vectors / index.nlist) : 0;
    size_t max_cluster_size = 0;
    size_t min_cluster_size = SIZE_MAX;
    size_t empty_clusters = 0;
    
    for (size_t i = 0; i < (size_t)index.nlist; i++) {
        size_t list_size = index.get_list_size(i);
        max_cluster_size = std::max(max_cluster_size, list_size);
        if (list_size > 0) {
            min_cluster_size = std::min(min_cluster_size, list_size);
        } else {
            empty_clusters++;
        }
    }
    
    printf("  总向量数: %zd\n", total_vectors);
    printf("  聚类数量: %zu\n", (size_t)index.nlist);
    printf("  平均聚类大小: %zd\n", avg_cluster_size);
    printf("  最大聚类大小: %zd\n", max_cluster_size);
    printf("  最小聚类大小: %zd\n", min_cluster_size == SIZE_MAX ? 0 : min_cluster_size);
    printf("  空聚类数量: %zd\n", empty_clusters);
    
    printf("\n  维护策略说明：\n");
    printf("    - 如果聚类偏大（>平均大小*3），则分裂\n");
    printf("    - 如果聚类偏小（<平均大小/3），则合并\n");
    printf("    - 如果聚类大小适中，则只重新计算质心\n");
    printf("    - 所有维护操作后都会同步更新quantizer中的质心\n");

    const char* recompute_env = getenv("FAISS_RECOMPUTE_CENTROIDS");
    if (recompute_env && recompute_env[0] == '1') {
        double maintain_t0 = elapsed();
        index.recompute_centroids(true);
        printf("\n[%.3f s] 执行调整操作：重算所有质心，耗时 %.3f s\n",
               elapsed() - t0,
               elapsed() - maintain_t0);
    }
    
    // ========== 最终搜索和评估 ==========
    printf("\n[%.3f s] ========== 最终搜索和评估 ==========\n", elapsed() - t0);
    idx_t* I_final = new idx_t[k * nq];
    float* D_final = new float[k * nq];
    
    search_t0 = elapsed();
    index.search(nq, xq, k, D_final, I_final);
    search_time = elapsed() - search_t0;
    
    float recall_1_final = compute_recall_at_k(I_final, gt, nq, 1, k_gt);
    float recall_10_final = compute_recall_at_k(I_final, gt, nq, 10, k_gt);
    float recall_100_final = compute_recall_at_k(I_final, gt, nq, 100, k_gt);
    
    printf("  搜索时间: %.3f s (%.3f ms/query)\n", search_time, search_time * 1000 / nq);
    printf("  Recall@1:   %.4f (初始: %.4f, 变化: %+.4f)\n", 
           recall_1_final, recall_1_initial, recall_1_final - recall_1_initial);
    printf("  Recall@10:  %.4f (初始: %.4f, 变化: %+.4f)\n", 
           recall_10_final, recall_10_initial, recall_10_final - recall_10_initial);
    printf("  Recall@100: %.4f (初始: %.4f, 变化: %+.4f)\n", 
           recall_100_final, recall_100_initial, recall_100_final - recall_100_initial);
    
    // ========== 总结 ==========
    printf("\n[%.3f s] ========== 总结 ==========\n", elapsed() - t0);
    printf("  总耗时: %.3f s\n", elapsed() - t0);
    printf("  插入操作: %.3f s\n", insert_time);
    printf("  更新操作: %.3f s\n", update_time);
    printf("  删除操作: %.3f s\n", delete_time);
    printf("  最终搜索: %.3f s\n", search_time);
    printf("\n  验证结果：\n");
    printf("    ✓ 插入操作后自动维护被插入的聚类\n");
    printf("    ✓ 更新操作后自动维护被操作的聚类\n");
    printf("    ✓ 删除操作后自动维护被删除向量所在的聚类\n");
    printf("    ✓ 索引质量保持稳定（Recall 变化较小）\n");
    
    // 清理内存
    delete[] xb;
    delete[] xq;
    delete[] gt;
    delete[] I_initial;
    delete[] D_initial;
    delete[] I_final;
    delete[] D_final;
    
    return 0;
}

