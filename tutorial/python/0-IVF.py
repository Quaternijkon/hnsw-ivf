# -*- coding: utf-8 -*-

"""
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
"""

import numpy as np
import faiss
import time

# 维度
d = 64
# 数据库大小
nb = 100000
# 查询向量数量
nq = 10000

# --- 1. 生成数据 ---
print("Generating fake data...")
# 设置随机种子以保证结果可复现
np.random.seed(1234)

# 创建数据库向量 xb
xb = np.random.random((nb, d)).astype('float32')
# 对数据进行一些轻微的、非均匀的调整，使其不完全是随机分布
xb[:, 0] += np.arange(nb) / 1000.

# 创建查询向量 xq
xq = np.random.random((nq, d)).astype('float32')
# 同样对查询向量进行调整
xq[:, 0] += np.arange(nq) / 1000.

print("Data generated.")

# --- 2. 构建并训练索引 ---
nlist = 100  # 将数据库分为100个倒排列表
k = 4        # 我们想要查找最近的4个邻居

# 量化器，这里使用精确的L2距离索引
quantizer = faiss.IndexFlatL2(d)
# 创建IVFFlat索引
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# 训练索引
assert not index.is_trained
print("Training index...")
index.train(xb)
assert index.is_trained
print("Training complete.")

# 添加向量到索引
print("Adding vectors to index...")
index.add(xb)
print(f"Index contains {index.ntotal} vectors.")


# --- 3. 设置并行模式并准备延迟统计 ---

# 1. 设置为 parallel_mode = 0 (在IVF搜索中禁用查询间的并行)
#    这使得我们可以更清晰地测量单个查询的延迟
index.parallel_mode = 0

# 2. 将 OMP 线程数设为 1，以精确测量单线程下的真实挂钟延迟
#    这模拟了典型的 Web 服务中单个工作进程的行为
faiss.omp_set_num_threads(1)

print(f"Starting search with nq={nq}, k={k}, nlist={nlist}, "
      f"parallel_mode={index.parallel_mode}, OMP_threads={faiss.omp_get_max_threads()}")


# --- 4. 逐个执行查询并收集延迟数据 ---
latencies_ms = []
print("Measuring per-query latency...")

# 为了避免首次查询的“冷启动”开销影响统计，可以先进行一次预热查询
index.search(xq[:1], k)

for i in range(nq):
    # 获取单个查询向量，并确保其为 2D 数组
    query_vector = xq[i:i+1]

    # 记录开始时间
    start_time = time.perf_counter()
    # 执行搜索
    D, I = index.search(query_vector, k)
    # 记录结束时间
    end_time = time.perf_counter()

    # 计算耗时并转换为毫秒，然后添加到列表中
    latency = (end_time - start_time) * 1000
    latencies_ms.append(latency)

print("Latency measurement complete.")

# --- 5. 分析并打印延迟统计结果 ---
latencies_np = np.array(latencies_ms)

avg_latency = np.mean(latencies_np)
min_latency = np.min(latencies_np)
max_latency = np.max(latencies_np)
p50_latency = np.median(latencies_np) # P50 就是中位数
p99_latency = np.percentile(latencies_np, 99)

print("\n--- Latency Stats (ms) ---")
print(f"Average : {avg_latency:.4f} ms")
print(f"Min     : {min_latency:.4f} ms")
print(f"Max     : {max_latency:.4f} ms")
print(f"P50 (Median) : {p50_latency:.4f} ms")
print(f"P99     : {p99_latency:.4f} ms")
print("--------------------------\n")

# 注意：标准的 Faiss Python API 无法直接提供 C++ 版本中
# quantization_us 和 list_scan_us 这样的内部耗时分解。
# 我们这里测量的是每个 search() 调用的总延迟。


# --- 6. (可选) 批量搜索并验证结果正确性 ---
print("Performing batch search for verification...")
# 为了验证结果，我们进行一次完整的批量搜索
D_batch, I_batch = index.search(xq, k)

# 打印最后5个查询的结果，与 C++ 示例的输出格式保持一致
print("I (last 5 results)=")
for i in range(nq - 5, nq):
    for j in range(k):
        print(f"{I_batch[i][j]:5d}", end=" ")
    print()