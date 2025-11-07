/*
 * @Author: Quaternijkon quaternijkon@mail.ustc.edu.cn
 * @Date: 2025-02-07 06:29:50
 * @LastEditors: Quaternijkon quaternijkon@mail.ustc.edu.cn
 * @LastEditTime: 2025-11-06 09:42:32
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
 
 #include <faiss/IndexFlat.h>
 #include <faiss/IndexIVFFlat.h>
 #include <faiss/impl/IDSelector.h>
 #include <faiss/invlists/DirectMap.h>
 
 using idx_t = faiss::idx_t;
 
 int main() {
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
     
     // 启用 direct_map 以支持插入、更新和删除操作
     // 使用 Hashtable 类型以支持任意 ID
     // 注意：应该在添加数据之前设置，这样添加时会自动维护映射
     index.set_direct_map_type(faiss::DirectMap::Hashtable);
     printf("已启用 direct_map (Hashtable 类型)\n");
     
     // 初始添加数据（使用自动生成的 ID）
     index.add(nb, xb);
     printf("初始添加了 %d 个向量，索引总数: %zd\n", nb, index.ntotal);
 
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
     int n_insert = 1000;  // 要插入的向量数量
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
 
     // ========== 演示更新操作 ==========
     printf("\n=== 演示更新操作 ===\n");
     int n_update = 100;  // 要更新的向量数量
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
 
     // ========== 演示删除操作 ==========
     printf("\n=== 演示删除操作 ===\n");
     int n_delete = 200;  // 要删除的向量数量
     std::vector<idx_t> ids_to_delete(n_delete);
     
     // 选择要删除的向量 ID（选择 ID 为 5000 到 5199 的向量）
     for (int i = 0; i < n_delete; i++) {
         ids_to_delete[i] = 5000 + i;
     }
     
     faiss::IDSelectorArray selector(n_delete, ids_to_delete.data());
     size_t n_removed = index.remove_ids(selector);
     printf("删除了 %zd 个向量，索引总数: %zd\n", n_removed, index.ntotal);
 
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
 