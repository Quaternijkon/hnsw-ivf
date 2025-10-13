# CSV文件格式说明

## 文件结构

程序现在会生成两个独立的CSV文件，分别记录build和search阶段的测试结果。

## Build结果文件

**文件名**: `benchmark_build_results_[timestamp].csv`

**列说明**:
- `nlist`: IVF索引的聚类中心数量
- `efconstruction`: HNSW构建时的搜索参数
- `training_memory_mb`: 训练阶段峰值内存使用量(MB)
- `add_memory_mb`: 添加数据阶段峰值内存使用量(MB)
- `training_time_s`: 训练时间(秒)
- `total_time_s`: 总构建时间(秒)

**示例内容**:
```csv
nlist,efconstruction,training_memory_mb,add_memory_mb,training_time_s,total_time_s
1953,40,123.45,234.56,12.34,45.67
1953,80,145.67,256.78,15.23,52.34
3906,40,167.89,278.90,18.12,59.45
3906,80,189.01,300.12,21.01,66.56
```

## Search结果文件

**文件名**: `benchmark_search_results_[timestamp].csv`

**列说明**:
- `nlist`: IVF索引的聚类中心数量
- `efconstruction`: HNSW构建时的搜索参数
- `nprobe`: 搜索时访问的聚类中心数量
- `efsearch`: HNSW搜索时的搜索参数
- `training_memory_mb`: 训练阶段峰值内存使用量(MB)
- `add_memory_mb`: 添加数据阶段峰值内存使用量(MB)
- `training_time_s`: 训练时间(秒)
- `total_time_s`: 总构建时间(秒)
- `recall`: 召回率
- `qps`: 每秒查询数
- `mspq`: 每查询使用多少毫秒(milliseconds per query)
- `search_memory_mb`: 搜索阶段峰值内存使用量(MB)
- `search_time_s`: 搜索时间(秒)
- `mean_latency_ms`: 平均延迟(毫秒)
- `p50_latency_ms`: 50分位延迟(毫秒)
- `p99_latency_ms`: 99分位延迟(毫秒)

**示例内容**:
```csv
nlist,efconstruction,nprobe,efsearch,training_memory_mb,add_memory_mb,training_time_s,total_time_s,recall,qps,mspq,search_memory_mb,search_time_s,mean_latency_ms,p50_latency_ms,p99_latency_ms
1953,40,7,3,123.45,234.56,12.3400,45.6700,0.9876,83333.33,0.0120,12.34,0.1200,0.1500,0.1200,0.4500
1953,40,7,6,123.45,234.56,12.3400,45.6700,0.9889,78947.37,0.0127,12.45,0.1266,0.1600,0.1300,0.4800
1953,40,15,7,123.45,234.56,12.3400,45.6700,0.9923,76923.08,0.0130,12.56,0.1300,0.1700,0.1400,0.5100
1953,40,15,13,123.45,234.56,12.3400,45.6700,0.9934,74074.07,0.0135,12.67,0.1351,0.1800,0.1500,0.5400
```

## 优势

### 1. 数据分离
- **Build数据**: 专注于索引构建性能分析
- **Search数据**: 专注于搜索性能分析
- 避免数据冗余，便于独立分析

### 2. 文件大小
- Build文件较小，只包含构建相关的数据
- Search文件包含完整的搜索测试结果
- 便于不同阶段的分析需求

### 3. 分析便利
- 可以单独分析构建性能（内存使用、构建时间）
- 可以单独分析搜索性能（QPS、延迟、召回率）
- 便于生成针对性的性能报告

## 使用建议

### Build结果分析
- 分析不同`nlist`和`efconstruction`参数对构建性能的影响
- 重点关注内存使用和构建时间的关系
- 选择最优的构建参数组合

### Search结果分析
- 分析不同搜索参数对性能的影响
- 平衡QPS、延迟和召回率
- 选择最优的搜索参数组合

### 综合分析
- 结合两个文件的数据进行综合分析
- 考虑构建成本和搜索性能的权衡
- 为实际应用选择最佳参数组合
