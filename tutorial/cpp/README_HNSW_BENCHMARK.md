# HNSW Benchmark 测试说明

## 概述

`benchmark_hnsw.cpp` 是针对 Faiss HNSW 索引的性能测试工具，参照 `benchmark_advanced.cpp` 的框架开发。

## 主要特性

### Build 阶段测试
- **参数**: M（HNSW图的连接数）, efConstruction（构建时的搜索范围）
- **指标**: 
  - `training_memory_mb`: 初始化阶段峰值内存（HNSW无需训练，值为初始化后内存）
  - `add_memory_mb`: 添加数据阶段峰值内存
  - `training_time_s`: 训练时间（HNSW为0）
  - `total_time_s`: 总构建时间

### Search 阶段测试
- **参数**: efsearch（搜索时的候选集大小）= efConstruction × efsearch_ratio
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
# 需要C++17支持
g++ -std=c++17 -O3 -fopenmp benchmark_hnsw.cpp -o benchmark_hnsw \
    -I/path/to/faiss/include \
    -L/path/to/faiss/lib \
    -lfaiss -lopenblas -lgomp
```

或使用CMake（如果项目已配置）：

```bash
mkdir build && cd build
cmake ..
make benchmark_hnsw
```

## 使用方法

### 1. 准备数据
确保 `./sift` 目录下包含以下文件：
- `base.fbin`: 基础数据集
- `query.fbin`: 查询数据集
- `groundtruth.ivecs`: 真实最近邻结果（用于计算召回率）

### 2. 配置测试参数
编辑 `benchmark-hnsw.config` 文件：

```
build
  param
    M:4,8,16,32,64
    efconstruction:40,80,160,200
  metric
    training_memory
    add_memory
    training_time
    total_time
search
  param
    efsearch_ratio:0.5,0.9,1.0,1.1,1.5,2.0
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
# 使用默认配置文件 benchmark-hnsw.config
./benchmark_hnsw

# 或指定配置文件
./benchmark_hnsw my_config.config
```

### 4. 查看结果
程序会生成两个CSV文件：
- `benchmark_hnsw_build_results_<timestamp>.csv`: Build阶段结果
- `benchmark_hnsw_search_results_<timestamp>.csv`: Search阶段结果

## 与 IVF+HNSW 版本的主要区别

| 特性 | benchmark_advanced.cpp | benchmark_hnsw.cpp |
|------|----------------------|-------------------|
| 索引类型 | IndexIVFFlat + HNSW quantizer | IndexHNSWFlat |
| Build参数 | nlist, efconstruction | M, efconstruction |
| Search参数 | nprobe, efsearch | efsearch |
| 训练阶段 | 需要训练 | 无需训练 |
| 参数计算 | nprobe = nlist × ratio, efsearch = nprobe × ratio | efsearch = efconstruction × ratio |

## 参数调优建议

### M（连接数）
- 范围: 4-64
- 较大的M: 更高的召回率和搜索速度，但占用更多内存
- 推荐: 16-32 用于一般场景

### efConstruction（构建时搜索范围）
- 范围: 40-200+
- 较大的efConstruction: 更好的图质量，但构建时间更长
- 推荐: 80-160 用于一般场景

### efsearch（搜索时候选集大小）
- 通过 efsearch_ratio 控制: efsearch = efConstruction × efsearch_ratio
- 较大的efsearch: 更高的召回率，但搜索速度较慢
- 推荐: 1.0-1.5 倍的 efConstruction

## 注意事项

1. 程序默认使用20个OpenMP线程，可以在代码中修改
2. 临时索引文件会在测试完成后自动清理
3. 内存监控基于 `/proc/self/status`，仅适用于Linux系统
4. 确保有足够的磁盘空间存储临时索引文件

## 输出示例

```
=== Faiss HNSW Benchmark测试程序 ===
配置文件: benchmark-hnsw.config
OpenMP线程数: 20
数据集信息:
  维度: 128
  基础集大小: 1000000
  查询集大小: 10000

=== Build测试: M=32, efconstruction=80 ===
训练时间: 0.00s (HNSW无需训练)
总时间: 125.34s
初始化阶段峰值内存: 245.32MB
添加数据阶段峰值内存: 1234.56MB

=== Search测试: efsearch=80 ===
搜索时间: 12.45s
QPS: 803.21
mSPQ: 1.2450ms
搜索阶段峰值内存: 1256.78MB
平均延迟: 1.2456ms
P50延迟: 1.1234ms
P99延迟: 2.3456ms
召回率: 0.9876
```

