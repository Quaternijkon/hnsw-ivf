# 版本说明

本目录包含两个版本的benchmark程序：

## 1. benchmark_hnsw_ivf.cpp（标准版）

**特点**：
- 简洁的输出，只显示基本的测试结果
- 适用于日常性能测试和生产环境
- 输出包括：训练时间、搜索时间、QPS、召回率、内存占用等基本指标

**适用场景**：
- 快速性能测试
- 自动化基准测试
- 生产环境监控

## 2. benchmark_hnsw_ivf_detailed.cpp（详细内存分析版）

**特点**：
- 详细的内存分析输出
- 显示HNSW量化器的详细内存组成：
  - 索引加载前后的内存对比
  - HNSW量化器的图结构详细信息（neighbors、offsets、levels数组大小）
  - 向量数据和图结构的内存分解
  - 查询数据加载后的内存增长
  - 搜索期间的内存变化
- 帮助深入理解内存占用来源

**适用场景**：
- 内存问题诊断
- 性能优化分析
- 内存占用研究

## 构建和使用

### 构建两个版本
```bash
make          # 同时构建两个版本
```

### 只构建标准版本
```bash
make benchmark_hnsw_ivf
```

### 只构建详细分析版本
```bash
make benchmark_hnsw_ivf_detailed
```

### 运行标准版本
```bash
make run
# 或
./benchmark_hnsw_ivf
```

### 运行详细分析版本
```bash
make run-detailed
# 或
./benchmark_hnsw_ivf_detailed
```

## 输出差异示例

### 标准版输出
```
索引已通过mmap加载: ../sift/temp_index_hnsw_ivf_M32_nlist1000_efC40.index
搜索时间: 0.25s
QPS: 40000.00
搜索阶段峰值内存: 520.00MB
```

### 详细分析版输出
```
索引已通过mmap加载: ../sift/temp_index_hnsw_ivf_M32_nlist1000_efC40.index
  -> 索引加载后内存: 485.32MB (+485.32MB)
  -> HNSW量化器信息: nlist=1000, M=32, max_level=3, neighbors=125000
  -> HNSW量化器实际内存: 向量数据~0MB, 图结构~1MB (neighbors=0MB, offsets=0MB, levels=0MB)
  -> HNSW量化器总计估算: ~1MB
  -> ⚠️  注意: HNSW图结构即使使用mmap也可能被完全加载到内存（需要频繁随机访问）
  -> ⚠️  这是HNSW量化器内存占用的主要原因，无法通过mmap完全避免
  -> 查询数据加载后内存: 490.15MB (+4.83MB)
  -> 搜索完成后内存: 520.00MB (+29.85MB)
搜索时间: 0.25s
QPS: 40000.00
搜索阶段峰值内存: 520.00MB
```

## 清理

```bash
make clean    # 删除所有构建产物和CSV文件
```

