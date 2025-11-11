/*
 * @Author: Quaternijkon quaternijkon@mail.ustc.edu.cn
 * @Date: 2025-02-07 06:29:50
 * @LastEditors: Quaternijkon quaternijkon@mail.ustc.edu.cn
 * @LastEditTime: 2025-11-06 10:58:03
 * @FilePath: /faiss/tutorial/cpp/2-IVFFlat.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
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
#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/invlists/DirectMap.h>

using idx_t = faiss::idx_t;

int main() {
    // 设置 OpenMP 线程数，避免 OpenBLAS 内存分配问题
    // OpenBLAS 会为每个线程分配内存区域，线程数过多会导致分配失败
    int num_threads = 4;  // 根据你的 CPU 核心数调整，建议不超过 CPU 核心数
    omp_set_num_threads(num_threads);
    
    // 设置 OpenBLAS 线程数环境变量（如果支持）
    // OpenBLAS 使用单线程，避免内存分配问题
    // 注意：setenv 需要在程序启动时设置，这里设置可能不会立即生效
    // 建议在运行程序前设置：export OPENBLAS_NUM_THREADS=1
    char omp_threads_str[32];
    snprintf(omp_threads_str, sizeof(omp_threads_str), "%d", num_threads);
    setenv("OPENBLAS_NUM_THREADS", "1", 1);  // 覆盖已存在的值
    setenv("OMP_NUM_THREADS", omp_threads_str, 1);
    
    printf("OpenMP 线程数: %d\n", omp_get_max_threads());
    int d = 64;      // dimension
    int nb = 100000; // database size
    int nq = 10000;  // nb of queries

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

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

    faiss::IndexFlatL2 quantizer(d); // the other index
    faiss::IndexIVFFlat index(&quantizer, d, nlist);
    assert(!index.is_trained);
    index.train(nb, xb);
    assert(index.is_trained);
    
    // 设置 nprobe：搜索时探查的倒排列表数量
    // 默认值是 1，删除向量后某些列表可能变空，需要增加 nprobe 以确保能找到结果
    int nprobe = 10;  // 建议设置为 nlist 的 10% 左右，至少为 1
    index.nprobe = nprobe;
    printf("已设置 nprobe = %d (搜索时将探查 %d 个倒排列表)\n", nprobe, nprobe);
    
    // 启用 direct_map 以支持插入、更新和删除操作
    // 使用 Hashtable 类型以支持任意 ID
    // 注意：应该在添加数据之前设置，这样添加时会自动维护映射
    index.set_direct_map_type(faiss::DirectMap::Hashtable);
    printf("已启用 direct_map (Hashtable 类型)\n");
    
    // 初始添加数据（使用显式 ID，确保 direct_map 正确维护）
    // 在 Hashtable 模式下，使用 add_with_ids 可以确保所有 ID 都被正确添加到 direct_map
    idx_t* ids_initial = new idx_t[nb];
    for (int i = 0; i < nb; i++) {
        ids_initial[i] = i;  // ID 从 0 开始
    }
    index.add_with_ids(nb, xb, ids_initial);
    printf("初始添加了 %d 个向量，索引总数: %zd\n", nb, index.ntotal);
    delete[] ids_initial;

    { // 初始搜索
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];

        index.search(nq, xq, k, D, I);

        printf("\n=== 初始搜索结果 ===\n");
        printf("I=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }

        printf("D=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++)
                printf("%5f ", D[i * k + j]);
            printf("\n");
        }

        delete[] I;
        delete[] D;
    }

    // ========== 演示插入操作 ==========
    printf("\n=== 演示插入操作 ===\n");
    int n_insert = 10000;  // 要插入的向量数量
    float* x_insert = new float[d * n_insert];
    idx_t* ids_insert = new idx_t[n_insert];
    
    // 生成要插入的向量和 ID
    for (int i = 0; i < n_insert; i++) {
        for (int j = 0; j < d; j++)
            x_insert[d * i + j] = distrib(rng);
        x_insert[d * i] += (nb + i) / 1000.;
        ids_insert[i] = nb + i;  // 使用新的 ID
    }
    
    index.add_with_ids(n_insert, x_insert, ids_insert);
    printf("插入了 %d 个向量，索引总数: %zd\n", n_insert, index.ntotal);
    printf("注意：插入操作后已自动维护被插入的聚类\n");

    // ========== 演示更新操作 ==========
    printf("\n=== 演示更新操作 ===\n");
    int n_update = 10000;  // 要更新的向量数量
    float* x_update = new float[d * n_update];
    idx_t* ids_update = new idx_t[n_update];
    
    // 选择要更新的向量（选择前 n_update 个）
    for (int i = 0; i < n_update; i++) {
        ids_update[i] = i;  // 更新 ID 为 0 到 n_update-1 的向量
        for (int j = 0; j < d; j++)
            x_update[d * i + j] = distrib(rng);  // 生成新的向量值
        x_update[d * i] += i / 1000.;
    }
    
    index.update_vectors(n_update, ids_update, x_update);
    printf("更新了 %d 个向量\n", n_update);
    printf("注意：更新操作（删除+插入）后已自动维护被操作的聚类\n");

    // ========== 演示删除操作 ==========
    printf("\n=== 演示删除操作 ===\n");
    int n_delete = 10000;  // 要删除的向量数量
    std::vector<idx_t> ids_to_delete(n_delete);
    
    // 选择要删除的向量 ID（选择 ID 为 5000 到 9999 的向量）
    // 注意：避免删除刚更新的向量，这里删除的是 ID 5000-9999
    for (int i = 0; i < n_delete; i++) {
        ids_to_delete[i] = 5000 + i;  // ID 范围：5000 到 9999
    }
    
    faiss::IDSelectorArray selector(n_delete, ids_to_delete.data());
    size_t n_removed = index.remove_ids(selector);
    printf("删除了 %zd 个向量，索引总数: %zd\n", n_removed, index.ntotal);
    printf("注意：删除操作后已自动维护被删除向量所在的聚类\n");
    
    // 删除操作后，某些倒排列表可能变空，增加 nprobe 可以提高搜索成功率
    // 如果删除的向量很多，建议适当增加 nprobe
    if (n_removed > 0) {
        int new_nprobe = std::min((int)index.nlist, nprobe + 5);  // 增加 nprobe 以确保能找到结果
        index.nprobe = new_nprobe;
        printf("删除操作后，已将 nprobe 调整为 %d\n", new_nprobe);
    }

    // ========== 演示动态聚类维护 ==========
    printf("\n=== 聚类维护说明 ===\n");
    printf("注意：插入、更新和删除操作后，系统会自动维护被操作的聚类。\n");
    printf("维护策略：\n");
    printf("  - 如果聚类偏大（>平均大小*3），则分裂\n");
    printf("  - 如果聚类偏小（<平均大小/3），则合并\n");
    printf("  - 如果聚类大小适中，则只重新计算质心\n");
    printf("  - 所有维护操作后都会同步更新quantizer中的质心\n");
    
    // 显示当前聚类统计信息
    size_t total_vectors = index.ntotal;
    size_t avg_cluster_size = (nlist > 0 && total_vectors > 0) ? (total_vectors / nlist) : 0;
    size_t max_cluster_size = 0;
    size_t min_cluster_size = SIZE_MAX;
    
    for (size_t i = 0; i < (size_t)nlist; i++) {
        size_t list_size = index.get_list_size(i);
        max_cluster_size = std::max(max_cluster_size, list_size);
        if (list_size > 0) {
            min_cluster_size = std::min(min_cluster_size, list_size);
        }
    }
    
    printf("\n当前聚类统计信息:\n");
    printf("  - 总向量数: %zd\n", total_vectors);
    printf("  - 聚类数量: %zu\n", (size_t)nlist);
    printf("  - 平均聚类大小: %zd\n", avg_cluster_size);
    printf("  - 最大聚类大小: %zd\n", max_cluster_size);
    printf("  - 最小聚类大小: %zd\n", min_cluster_size == SIZE_MAX ? 0 : min_cluster_size);
    
    // 注意：不需要手动调用维护函数，因为插入/删除/更新操作已经自动维护了被影响的聚类

    // ========== 最终搜索 ==========
    { // 最终搜索
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];

        index.search(nq, xq, k, D, I);

        printf("\n=== 最终搜索结果（插入/更新/删除后）===\n");
        printf("I=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }

        printf("D=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++)
                printf("%5f ", D[i * k + j]);
            printf("\n");
        }

        delete[] I;
        delete[] D;
    }

    delete[] x_insert;
    delete[] ids_insert;
    delete[] x_update;
    delete[] ids_update;

    delete[] xb;
    delete[] xq;

    return 0;
}
