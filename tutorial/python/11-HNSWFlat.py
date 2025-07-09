'''
Author: superestos superestos@gmail.com
Date: 2025-07-09 01:59:37
LastEditors: superestos superestos@gmail.com
LastEditTime: 2025-07-09 01:59:40
FilePath: /dry/faiss/tutorial/python/11-HNSW.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# -*- coding: UTF-8 -*-

import numpy as np
import faiss

# 1. 设置参数
d = 64                           # 向量维度 (dimension)
nb = 100000                      # 数据集大小 (database size)
nq = 10000                       # 查询集大小 (number of queries)
np.random.seed(1234)             # 设置随机种子以保证结果可复现

# 2. 生成随机数据
# 生成数据集
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
# 生成查询集
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

# 3. 构建 HNSW 索引
print("正在构建 HNSW 索引...")

# 定义索引，这里使用 L2 距离
# M 是每个节点的最大连接数，是构建图时的一个关键参数
M = 32  # M 的典型值是 16、32、48
index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)

# efConstruction 控制索引构建时的搜索深度，值越大，索引质量越高，但构建时间越长
index.hnsw.efConstruction = 40

print(f"索引是否已经训练过 (is_trained): {index.is_trained}") # HNSW 不需要训练，所以总是 True
print(f"索引中的向量总数 (ntotal): {index.ntotal}")

# 4. 添加向量到索引
print(f"正在向索引中添加 {nb} 个向量...")
index.add(xb)
print(f"添加后索引中的向量总数 (ntotal): {index.ntotal}")

# 5. 进行搜索
k = 4  # 我们想要查找最近的 4 个邻居

# efSearch 控制搜索时的搜索深度，值越大，精度越高，但搜索时间越长
# 这个参数可以在搜索时动态调整
index.hnsw.efSearch = 16

print("\n正在搜索...")
D, I = index.search(xq, k)  # D 是距离矩阵，I 是索引矩阵

# 6. 打印结果
# 打印最后 5 个查询的结果
print("\n查询结果的索引 (I):")
print(I[-5:])
print("\n查询结果的距离 (D):")
print(D[-5:])

# sanity check: 搜索数据集中的前5个向量，它们自己应该是自己的最近邻 (距离为0)
print("\nSanity check:")
D_check, I_check = index.search(xb[:5], k)
print("索引:")
print(I_check)
print("距离:")
print(D_check)