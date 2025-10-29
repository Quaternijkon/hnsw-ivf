# IVF Benchmark 测试说明

## 概述

`benchmark_ivf.cpp` 是针对 Faiss IVF (Inverted File) 索引的性能测试工具，参照 `benchmark_advanced.cpp` 的框架开发。该版本使用标准的 `IndexIVFFlat` 配合 `IndexFlatL2` 量化器，是最基础的 IVF 索引实现。

## 主要特性

### Build 阶段测试
- **参数**: nlist（聚类中心数量）
- **指标**: 
  - `training_memory_mb`: 训练阶段峰值内存
  - `add_memory_mb`: 添加数据阶段峰值内存
  - `training_time_s`: 训练时间（k-means 聚类时间）
  - `total_time_s`: 总构建时间

### Search 阶段测试
- **参数**: nprobe（搜索时访问的聚类中心数量）= nlist × nprobe_ratio
- **指标**:
  - `recall`: 召回率
  - `QPS`: 每秒查询数
  - `mSPQ`: 每查询毫秒数
  - `search_memory_mb`: 搜索阶段峰值内存
  - `search_time_s`: 搜索总时间
  - `mean_latency_ms`: 平均延迟
  - `P50_latency_ms`: P50延迟
  - `P99_latency_ms`: P99延迟

## 编译方法

```bash
# 基础编译
g++ -std=c++17 -O3 -fopenmp benchmark_ivf.cpp -o benchmark_ivf \
    -I/path/to/faiss/include \
    -L/path/to/faiss/lib \
    -lfaiss -lopenblas -lgomp

# 或者使用相对路径（在 tutorial/cpp 目录下）
g++ -std=c++17 -O3 -o benchmark_ivf benchmark_ivf.cpp \
    -I ../.. -L ../../build/faiss \
    -Wl,-rpath,../../build/faiss \
    -lfaiss -lopenblas -fopenmp
```

## 使用方法

### 1. 准备数据
确保 `./sift` 目录下包含以下文件：
- `learn.fbin`: 训练数据集（用于 k-means 聚类）
- `base.fbin`: 基础数据集
- `query.fbin`: 查询数据集
- `groundtruth.ivecs`: 真实最近邻结果（用于计算召回率）

### 2. 配置测试参数
编辑 `benchmark-ivf.config` 文件：

```
build
  param
    nlist:1953,3906,7812,15625,31250
  metric
    training_memory
    add_memory
    training_time
    total_time
search
  param
    nprobe_ratio:0.004096,0.008192,0.016384,0.032768,0.065536
  metric
    recall
    QPS
    mSPQ
    search_memory
    search_time
    mean_latency
    P50_latency
    P99_latency
```

### 3. 运行测试

```bash
# 使用默认配置文件 benchmark-ivf.config
./benchmark_ivf

# 或指定配置文件
./benchmark_ivf my_ivf_config.config
```

### 4. 查看结果
程序会生成两个CSV文件：
- `benchmark_ivf_build_results_<timestamp>.csv`: Build阶段结果
- `benchmark_ivf_search_results_<timestamp>.csv`: Search阶段结果

## 三种索引类型对比

| 特性 | benchmark_ivf.cpp | benchmark_advanced.cpp | benchmark_hnsw.cpp |
|------|------------------|----------------------|-------------------|
| **索引类型** | IVFFlat + FlatL2 | IVFFlat + HNSW | HNSWFlat |
| **量化器** | IndexFlatL2 | IndexHNSWFlat | N/A |
| **Build参数** | nlist | nlist, efconstruction | M, efconstruction |
| **Search参数** | nprobe | nprobe, efsearch | efsearch |
| **训练阶段** | k-means 聚类 | k-means + HNSW构建 | 无需训练 |
| **内存占用** | 低 | 中等 | 高 |
| **构建速度** | 快 | 中等 | 慢 |
| **搜索速度** | 中等 | 快 | 快 |
| **适用场景** | 大规模数据集，内存受限 | 高性能要求 | 小规模高精度 |

## 参数调优建议

### nlist（聚类中心数量）
- **推荐范围**: `sqrt(nb)` 到 `4 * sqrt(nb)`，其中 nb 是基础数据集大小
- **示例**（对于 1M 数据集）:
  - 最小值: `sqrt(1000000) ≈ 1000`
  - 推荐值: `2000 - 4000`
  - 最大值: `4 * sqrt(1000000) ≈ 4000`
- **影响**:
  - 较小的 nlist: 训练快，内存少，但搜索精度可能降低
  - 较大的 nlist: 训练慢，内存多，但可以获得更好的搜索精度

### nprobe（搜索的聚类数量）
- **通过 nprobe_ratio 控制**: nprobe = nlist × nprobe_ratio
- **推荐范围**: 0.001 - 0.1（相对于 nlist）
- **示例**（nlist=2000）:
  - 低精度快速搜索: nprobe = 8-16 (ratio ≈ 0.004-0.008)
  - 平衡: nprobe = 32-64 (ratio ≈ 0.016-0.032)
  - 高精度: nprobe = 128-256 (ratio ≈ 0.064-0.128)
- **影响**:
  - 较小的 nprobe: 搜索快，但召回率低
  - 较大的 nprobe: 召回率高，但搜索慢

### 配置示例

#### 快速搜索配置（低精度）
```
build
  param
    nlist:2000,4000
search
  param
    nprobe_ratio:0.004,0.008,0.016
```

#### 平衡配置（中等精度）
```
build
  param
    nlist:2000,4000,8000
search
  param
    nprobe_ratio:0.016,0.032,0.064
```

#### 高精度配置（慢速搜索）
```
build
  param
    nlist:4000,8000,16000
search
  param
    nprobe_ratio:0.032,0.064,0.128
```

## IVF 工作原理

### Build 阶段
1. **训练（k-means 聚类）**: 使用训练数据集对数据空间进行聚类，生成 nlist 个聚类中心
2. **添加数据**: 将基础数据集中的每个向量分配到最近的聚类中心，形成倒排列表

### Search 阶段
1. **查找最近的聚类中心**: 对于每个查询向量，找到 nprobe 个最近的聚类中心
2. **在倒排列表中搜索**: 只在这 nprobe 个倒排列表中进行精确搜索
3. **返回最近邻**: 合并结果并返回最近的 k 个邻居

## 性能分析

### 时间复杂度
- **训练**: O(nt × d × nlist × iterations)，其中 nt 是训练集大小，iterations 是 k-means 迭代次数
- **添加**: O(nb × d × 1)，需要计算每个向量到最近聚类中心的距离
- **搜索**: O(nq × (d × nprobe + k × avg_list_size))

### 空间复杂度
- **量化器**: O(nlist × d)
- **倒排列表**: O(nb × d)
- **总计**: 约为原始数据的 1.0-1.1 倍

## 注意事项

1. **训练数据集大小**: 建议训练集大小 ≥ 30 × nlist，以确保聚类质量
2. **内存使用**: IVF 索引需要存储完整的原始向量，因此内存占用与数据集大小成正比
3. **OpenMP 并行**: 默认使用 20 个线程，可以根据 CPU 核心数调整
4. **临时文件**: 索引文件会在测试期间保存到磁盘，测试完成后自动清理
5. **精度-速度权衡**: nprobe 越大，召回率越高但速度越慢，需要根据应用场景选择合适的值

## 输出示例

```
=== Faiss IVF Benchmark测试程序 ===
配置文件: benchmark-ivf.config
OpenMP线程数: 20
数据集信息:
  维度: 128
  训练集大小: 100000
  基础集大小: 1000000
  查询集大小: 10000

=== Build测试: nlist=2000 ===
训练时间: 45.23s
总时间: 78.56s
训练阶段峰值内存: 523.45MB
添加数据阶段峰值内存: 1245.67MB

=== Search测试: nprobe=32 ===
搜索时间: 8.34s
QPS: 1199.04
mSPQ: 0.8340ms
搜索阶段峰值内存: 1256.78MB
平均延迟: 0.8345ms
P50延迟: 0.7823ms
P99延迟: 1.4567ms
召回率: 0.8934
```

## 与其他索引的选择建议

### 使用 IVF (benchmark_ivf.cpp) 当:
- ✅ 数据集较大（百万级以上）
- ✅ 内存受限
- ✅ 可以接受中等的召回率（80-95%）
- ✅ 需要快速构建索引

### 使用 IVF+HNSW (benchmark_advanced.cpp) 当:
- ✅ 需要高性能搜索
- ✅ 可以接受较长的构建时间
- ✅ 有足够的内存
- ✅ 需要较高的召回率（90-98%）

### 使用 HNSW (benchmark_hnsw.cpp) 当:
- ✅ 数据集相对较小（百万级以下）
- ✅ 需要最高的召回率（95-99%+）
- ✅ 有充足的内存
- ✅ 对构建时间不敏感

## 故障排除

### 问题: 内存不足
**解决方案**: 减小 nlist 或使用 PQ 压缩版本（IndexIVFPQ）

### 问题: 训练时间过长
**解决方案**: 
- 减小 nlist
- 减小训练集大小（但要保持 nt ≥ 30 × nlist）
- 使用预训练的量化器

### 问题: 召回率过低
**解决方案**:
- 增大 nprobe
- 增大 nlist（改善聚类质量）
- 检查数据分布是否适合 IVF

### 问题: 搜索速度过慢
**解决方案**:
- 减小 nprobe
- 增大 nlist（使每个倒排列表更短）
- 确保 OpenMP 并行正常工作

