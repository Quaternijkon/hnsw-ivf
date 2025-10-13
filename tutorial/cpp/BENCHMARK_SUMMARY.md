# Faiss高级Benchmark系统总结

## 系统概述

基于您的要求，我已经创建了一个完整的benchmark测试系统，该系统能够：

1. **基于配置文件驱动测试** - 通过`benchmark.config`文件配置所有测试参数
2. **参数遍历测试** - 自动遍历所有参数组合进行测试
3. **分阶段测试** - 先完成build测试，再对每个索引进行search测试
4. **多指标测量** - 测量内存使用、延迟、召回率等多种性能指标
5. **CSV结果输出** - 将所有测试结果保存为CSV格式便于分析

## 创建的文件

### 核心程序文件
- `benchmark_advanced.cpp` - 主程序，实现完整的benchmark测试逻辑
- `config_parser.h` - 配置文件解析器，支持自定义格式的配置文件
- `Makefile` - 编译配置文件

### 配置文件
- `benchmark.config` - 基础配置文件，包含您要求的参数组合
- `benchmark_example.config` - 详细示例配置文件，包含更多参数选项

### 文档和脚本
- `README_benchmark.md` - 详细使用说明文档
- `test_benchmark.sh` - 测试脚本，用于验证程序功能
- `BENCHMARK_SUMMARY.md` - 本总结文档

## 功能特性

### 1. 配置文件格式
支持简单的文本格式，易于编辑和理解：
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

### 2. 测试流程
1. **Build阶段**: 遍历所有`nlist`和`efconstruction`参数组合
   - 构建IVF+HNSW索引
   - 测量训练时间、内存使用等指标
   - 保存索引到临时文件

2. **Search阶段**: 对每个构建的索引，遍历所有`nprobe`和`efsearch`参数组合
   - 执行搜索测试
   - 测量QPS、延迟、召回率等指标

3. **结果输出**: 将所有结果保存到CSV文件

### 3. 测量指标

**Build指标:**
- `training_memory`: 训练阶段内存使用量(MB)
- `add_memory`: 添加数据阶段内存使用量(MB)
- `training_time`: 训练时间(秒)
- `total_time`: 总构建时间(秒)

**Search指标:**
- `recall`: 召回率
- `QPS`: 每秒查询数
- `search_memory`: 搜索阶段内存使用量(MB)
- `search_time`: 搜索时间(秒)
- `mean_latency`: 平均延迟(毫秒)
- `P50_latency`: 50分位延迟(毫秒)
- `P99_latency`: 99分位延迟(毫秒)

### 4. 技术实现

**内存监控**: 使用`getrusage()`系统调用监控内存使用
**延迟统计**: 基于Faiss的`QueryLatencyStats`结构计算延迟指标
**召回率计算**: 与groundtruth数据对比计算召回率
**CSV输出**: 自动生成带时间戳的CSV结果文件

## 使用方法

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

### 测试
```bash
./test_benchmark.sh
```

## 输出示例

程序会生成类似以下的CSV文件：
```csv
nlist,efconstruction,nprobe,efsearch,training_memory_mb,add_memory_mb,training_time_s,total_time_s,recall,qps,search_memory_mb,search_time_s,mean_latency_ms,p50_latency_ms,p99_latency_ms
1953,40,16,16,123.45,234.56,12.34,45.67,0.9876,83333.33,12.34,0.12,0.15,0.12,0.45
1953,40,16,32,123.45,234.56,12.34,45.67,0.9889,78947.37,12.34,0.13,0.16,0.13,0.48
...
```

## 参数说明

- **nlist**: IVF索引的聚类中心数量，影响索引精度和内存使用
- **efconstruction**: HNSW构建时的搜索参数，影响构建时间和索引质量
- **nprobe**: 搜索时访问的聚类中心数量，影响搜索精度和速度
- **efsearch**: HNSW搜索时的搜索参数，影响搜索精度和速度

## 注意事项

1. 确保数据文件存在于`./sift/`目录下
2. 程序会创建临时索引文件，测试完成后自动清理
3. 测试时间取决于参数组合数量，建议先用小参数集测试
4. 内存监控基于Linux系统，其他系统可能需要调整

这个benchmark系统完全满足您的需求，能够系统性地测试不同参数组合下的Faiss性能，并生成详细的CSV报告供分析使用。
