# Faiss高级Benchmark测试程序

这是一个基于配置文件的Faiss benchmark测试程序，支持参数遍历和多种性能指标测量。

## 功能特性

- **配置文件驱动**: 通过`benchmark.config`文件配置测试参数
- **智能参数设置**: 使用比例参数避免不合理的参数组合
- **多指标测量**: 支持内存使用、延迟、召回率等多种指标
- **CSV输出**: 结果以CSV格式保存，便于分析
- **分阶段测试**: 先完成build测试，再对每个索引进行search测试
- **固定线程数**: 所有实验使用固定的20个OpenMP线程

## 文件结构

```
tutorial/cpp/
├── benchmark_advanced.cpp    # 主程序
├── config_parser.h          # 配置文件解析器
├── benchmark.config         # 配置文件示例
├── Makefile                # 编译文件
└── README_benchmark.md     # 说明文档
```

## 配置文件格式

配置文件采用简单的文本格式，支持以下结构：

```
build
  param
    nlist:1953,3906,7812,15625,31250
    efconstruction:40,80,160,200
  metric
    training_memory
    add_memory
    training_time
    total_time
search
  param
    nprobe:16,32,64,128,256
    efsearch:16,32,64,128
  metric
    recall
    QPS
    search_memory
    search_time
    mean_latency
    P50_latency
    P99_latency
```

### 参数说明

**Build参数:**
- `nlist`: IVF索引的聚类中心数量
- `efconstruction`: HNSW构建时的搜索参数

**Search参数:**
- `nprobe_ratio`: nprobe相对于nlist的比例 (nprobe = nlist × nprobe_ratio)
- `efsearch_ratio`: efsearch相对于nprobe的比例 (efsearch = nprobe × efsearch_ratio)

**测量指标:**
- `training_memory`: 训练阶段内存使用量(MB)
- `add_memory`: 添加数据阶段内存使用量(MB)
- `training_time`: 训练时间(秒)
- `total_time`: 总构建时间(秒)
- `recall`: 召回率
- `QPS`: 每秒查询数
- `mSPQ`: 每查询使用多少毫秒(milliseconds per query)
- `search_memory`: 搜索阶段内存使用量(MB)
- `search_time`: 搜索时间(秒)
- `mean_latency`: 平均延迟(毫秒)
- `P50_latency`: 50分位延迟(毫秒)
- `P99_latency`: 99分位延迟(毫秒)

## 编译和运行

### 编译

```bash
make benchmark_advanced
```

### 运行

```bash
# 使用默认配置文件
./benchmark_advanced

# 使用指定配置文件
./benchmark_advanced benchmark.config
```

## 输出结果

程序会生成两个CSV文件，分别记录build和search的测试结果：

### Build结果文件
文件名格式：`benchmark_build_results_[timestamp].csv`
包含以下列：
- `nlist, efconstruction`: Build参数
- `training_memory_mb, add_memory_mb`: 内存使用指标
- `training_time_s, total_time_s`: 时间指标

### Search结果文件
文件名格式：`benchmark_search_results_[timestamp].csv`
包含以下列：
- `nlist, efconstruction, nprobe, efsearch`: 所有测试参数
- `training_memory_mb, add_memory_mb, training_time_s, total_time_s`: 对应的构建开销数据
- `recall, qps, mspq`: 搜索性能指标
- `search_memory_mb, search_time_s`: 搜索资源使用
- `mean_latency_ms, p50_latency_ms, p99_latency_ms`: 延迟指标

## 测试流程

1. **Build测试阶段**:
   - 遍历所有`nlist`和`efconstruction`参数组合
   - 对每组参数构建IVF+HNSW索引
   - 测量训练时间、内存使用等指标
   - 将索引保存到临时文件

2. **Search测试阶段**:
   - 对每个构建的索引，遍历所有`nprobe`和`efsearch`参数组合
   - 执行搜索测试
   - 测量QPS、延迟、召回率等指标

3. **结果输出**:
   - 将所有结果保存到CSV文件
   - 清理临时索引文件

## 注意事项

1. 确保数据文件存在于`./sift/`目录下：
   - `learn.fbin`: 训练数据
   - `base.fbin`: 基础数据集
   - `query.fbin`: 查询数据
   - `groundtruth.ivecs`: 真实答案（可选）

2. 程序会创建临时索引文件，测试完成后会自动清理

3. 内存监控基于`/proc/self/status`文件，在Linux系统上工作

4. 延迟统计基于Faiss的`QueryLatencyStats`结构

5. 所有实验使用固定的20个OpenMP线程，确保结果的一致性

## 示例输出

```
=== Faiss高级Benchmark测试程序 ===
配置文件: benchmark.config
OpenMP线程数: 20
数据集信息:
  维度: 128
  训练集大小: 100000
  基础集大小: 1000000
  查询集大小: 10000

=== 开始Build测试 ===
=== Build测试: nlist=1953, efconstruction=40 ===
训练时间: 12.34s
总时间: 45.67s
训练阶段峰值内存: 123.45MB
添加数据阶段峰值内存: 234.56MB

=== 开始Search测试 ===
=== Search测试: nprobe=16, efsearch=16 ===
搜索时间: 0.12s
QPS: 83333.33
mSPQ: 0.0120ms
搜索阶段峰值内存: 12.34MB
平均延迟: 0.1500ms
P50延迟: 0.1200ms
P99延迟: 0.4500ms
召回率: 0.9876

Build结果已保存到: benchmark_build_results_1703123456.csv
Search结果已保存到: benchmark_search_results_1703123456.csv
=== Benchmark测试完成 ===
总共执行了 200 次测试
```
