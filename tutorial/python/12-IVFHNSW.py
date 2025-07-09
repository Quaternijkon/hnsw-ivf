'''
Author: superestos superestos@gmail.com
Date: 2025-07-09 02:06:49
LastEditors: superestos superestos@gmail.com
LastEditTime: 2025-07-09 03:00:28
FilePath: /dry/faiss/tutorial/python/12-IVFHNSW_with_sampled_training.py
Description: 在原脚本基础上，增加了 ntrain 参数，使用数据样本进行训练。
'''
# -*- coding: UTF-8 -*-

import numpy as np
import faiss
import time
import platform # 导入模块以检查操作系统

# 1. 设置参数
d = 128                          # 向量维度 (dimension)
nb = 1000000                     # 数据集大小 (database size)
nq = 10000                       # 查询集大小 (number of queries)
ntrain = 100000                  # ★★★ 用于训练的样本数量 ★★★
k = 10                           # 我们想要查找最近的 10 个邻居
np.random.seed(1234)             # 设置随机种子以保证结果可复现

# 2. 生成随机数据
print("正在生成数据...")
# 生成数据集
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
# 生成查询集
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

# 3. 构建 IVF-HNSW 索引
print("\n正在构建 IVF-HNSW 索引...")

# --- HNSW 相关参数 ---
M = 32                           # HNSW 中每个节点的最大连接数

# --- IVF 相关参数 ---
nlist = 1024                     # 将数据集划分成 1024 个 cell (簇)

# --- 构建粗量化器 (Coarse Quantizer) ---
coarse_quantizer = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)

# --- 构建核心的 IVF-HNSW 索引 ---
index = faiss.IndexIVFFlat(coarse_quantizer, d, nlist, faiss.METRIC_L2)

# --- 设置 HNSW 和 IVF 的参数 ---
coarse_quantizer.hnsw.efConstruction = 40
coarse_quantizer.hnsw.efSearch = 16
index.nprobe = 32

# 4. 训练索引 ★★★ 此部分已修改 ★★★
print(f"索引是否已经训练过 (is_trained): {index.is_trained}")
print(f"正在从 {nb} 个向量中随机采样 {ntrain} 个用于训练...")

# 从 xb 中随机抽取 ntrain 个向量作为训练集 xt
random_indices = np.random.permutation(nb)
xt = xb[random_indices[:ntrain]]

print("正在使用样本训练索引 (计算聚类中心)...")
start_time = time.time()
index.train(xt) # ★★★ 使用样本 xt 而不是全部 xb进行训练 ★★★
end_time = time.time()
print(f"训练完成，耗时: {end_time - start_time:.2f} 秒")
print(f"索引是否已经训练过 (is_trained): {index.is_trained}")

# 5. 添加向量到索引
print(f"\n正在向索引中添加全部 {nb} 个向量...")
start_time = time.time()
index.add(xb) # ★★★ 添加的是全部数据 xb ★★★
end_time = time.time()
print(f"添加完成，耗时: {end_time - start_time:.2f} 秒")
print(f"索引中的向量总数 (ntotal): {index.ntotal}")


# 6. Sanity Check (健全性检查)
print("\n正在进行 Sanity Check...")
# 使用数据集中的前5个向量进行搜索
# 期望它们各自的最近邻是它们自己（索引为 0,1,2,3,4），且距离为0
D_check, I_check = index.search(xb[:5], k)
print("Sanity Check - 索引结果 (I):")
print(I_check)
print("Sanity Check - 距离结果 (D):")
print(D_check)
print("-> 检查第一列的索引是否为 0, 1, 2, 3, 4，且对应的距离是否为 0。")


# 7. 对查询集进行搜索
print(f"\n正在对查询集进行搜索 (nprobe = {index.nprobe})...")
start_time = time.time()
D, I = index.search(xq, k)
end_time = time.time()
print(f"搜索完成，耗时: {end_time - start_time:.2f} 秒")

# 8. 打印真实查询结果
print("\n真实查询结果的索引 (I[-5:]):")
print(I[-5:])
print("\n真实查询结果的距离 (D[-5:]):")
print(D[-5:])

# ==============================================================================
# ★★★ 在程序末尾报告整个生命周期的峰值内存占用 ★★★
# ==============================================================================
print("\n" + "="*60)
# 检查操作系统，因为 resource 模块在 Windows 上不可用
if platform.system() in ["Linux", "Darwin"]:
    import resource
    # resource.getrusage(resource.RUSAGE_SELF).ru_maxrss 获取峰值常驻集大小 (Peak RSS)
    peak_memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Linux":
        peak_memory_bytes *= 1024 # 统一转换为 Bytes
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    print(f"整个程序运行期间的峰值内存占用: {peak_memory_mb:.2f} MB")
else:
    print("当前操作系统非 Linux/macOS，无法自动报告峰值内存。")
    print("请使用系统自带的监控工具（如 Windows 任务管理器）进行观察。")
print("="*60)