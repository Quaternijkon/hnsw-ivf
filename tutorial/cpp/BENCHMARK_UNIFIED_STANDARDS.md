# Benchmark 脚本统一标准

本文档定义了所有 benchmark 脚本的统一流程和检测标准，确保测试结果的一致性和可比性。

## 统一的流程标准

### 1. 搜索阶段内存监控流程

所有 benchmark 脚本现在都遵循以下统一的流程：

```
1. 开始内存监控
   ↓
2. 加载查询数据（计入监控）
   ↓
3. 分配搜索结果缓冲区（计入监控）
   ↓
4. 执行搜索操作（计入监控）
   ↓
5. 记录搜索阶段的峰值内存
   ↓
6. **停止内存监控** ⬅️ 关键步骤
   ↓
7. 计算延迟统计（不计入监控）
   ↓
8. 加载 groundtruth 数据（不计入监控）
   ↓
9. 计算召回率（不计入监控）
```

### 2. 代码模板

所有脚本的 `runSearchTest` 函数都应遵循以下模板：

```cpp
// 1. 开始监控搜索阶段的峰值内存
PeakMemoryMonitor search_memory_monitor;
search_memory_monitor.start();

// 2. 加载/创建索引（如适用）

// 3. 加载查询数据
auto [xq_data, _] = read_fbin(QUERY_FILE);
search_memory_monitor.update();

// 4. 执行搜索
index->search_stats(...);
search_memory_monitor.update();

// 5. 记录搜索阶段的峰值内存
result.search_memory_mb = search_memory_monitor.getPeakMemoryMB();

// 6. 停止内存监控 ⬅️ 统一标准
search_memory_monitor.stop();

// 7. 计算延迟统计（不计入监控）
// ...

// 8. 删除索引（如适用）

// 9. 加载 groundtruth 并计算召回率（不计入监控）
vector<vector<int32_t>> groundtruth = read_ivecs(GROUNDTRUTH_FILE);
result.recall = calculateRecall(I, groundtruth, nq, k);
```

## 统一的检测标准

### 1. 内存监测方法

**所有脚本统一使用**：
- 监测方式：`/proc/self/status` 读取 `VmRSS`
- 监测时机：搜索阶段开始到搜索结束
- **不包括**：groundtruth 数据加载

### 2. 内存监控范围

**计入搜索阶段内存**：
- ✅ 索引对象（通过 mmap 或内存）
- ✅ 查询数据
- ✅ 搜索结果缓冲区
- ✅ 搜索过程中的临时数据结构

**不计入搜索阶段内存**：
- ❌ Groundtruth 数据
- ❌ 延迟统计计算
- ❌ 召回率计算

### 3. 数据释放时机

**所有脚本统一标准**：
- ✅ 训练数据：训练完成后立即释放
- ✅ 基础数据：添加完成后立即释放（或分块处理，每块立即释放）
- ✅ 索引：搜索测试完成后删除（on-disk 模式：删除文件）

## 已统一的脚本

### ✅ benchmark_ivf.cpp
- ✅ Groundtruth 在内存监控停止后加载
- ✅ 内存监控在记录峰值后停止
- ✅ 训练数据和基础数据及时释放

### ✅ benchmark_hnsw.cpp
- ✅ Groundtruth 在内存监控停止后加载
- ✅ 内存监控在记录峰值后停止
- ✅ 基础数据在索引构建后释放

### ✅ benchmark_ivf_ondisk.cpp
- ✅ Groundtruth 在内存监控停止后加载
- ✅ 内存监控在记录峰值后停止
- ✅ 索引在加载 groundtruth 之前删除
- ✅ 训练数据及时释放，分块处理基础数据

### ✅ benchmark_hnsw_ivf.cpp
- ✅ Groundtruth 在内存监控停止后加载
- ✅ 内存监控在记录峰值后停止
- ✅ 索引在加载 groundtruth 之前删除（on-disk 模式）
- ✅ 训练数据及时释放，分块处理基础数据

## 关键优化点总结

### 1. Groundtruth 延迟加载

**原因**：
- Groundtruth 不是搜索操作本身需要的
- 不应该计入搜索阶段的内存占用
- 确保内存监控只反映实际搜索操作

**实现**：
```cpp
// 记录峰值内存
result.search_memory_mb = search_memory_monitor.getPeakMemoryMB();

// 停止监控
search_memory_monitor.stop();

// 然后加载 groundtruth（不计入监控）
vector<vector<int32_t>> groundtruth = read_ivecs(GROUNDTRUTH_FILE);
```

### 2. 内存监控停止

**原因**：
- 确保后续操作（如加载 groundtruth）不计入搜索阶段内存
- 使内存监控只反映实际搜索相关的内存占用

**实现**：
```cpp
// 在记录峰值内存后立即停止
search_memory_monitor.stop();
```

### 3. 数据及时释放

**标准**：
- 训练数据：训练完成后立即释放
- 基础数据：添加完成后立即释放（或分块处理）
- 索引：搜索测试完成后删除

## 内存监测一致性验证

### 检查清单

确保所有脚本都遵循：

- [x] ✅ 使用 `/proc/self/status` 读取 `VmRSS`
- [x] ✅ 在搜索阶段开始时启动内存监控
- [x] ✅ 在记录峰值内存后停止监控
- [x] ✅ Groundtruth 在监控停止后加载
- [x] ✅ 训练数据在训练后立即释放
- [x] ✅ 基础数据及时释放（或分块处理）

### 预期效果

所有脚本的搜索阶段内存报告应该：
- ✅ 只包含搜索操作本身的内存占用
- ✅ 不包含 groundtruth 数据
- ✅ 不包含不必要的训练/基础数据
- ✅ 可以通过相同的标准进行对比

## 对比测试

使用统一标准后，所有脚本的内存报告可以直接对比：

| 脚本 | 索引类型 | 量化器类型 | 搜索阶段内存（示例） |
|------|---------|-----------|---------------------|
| benchmark_ivf.cpp | IVF | IndexFlatL2 | ~50-200MB（取决于nprobe） |
| benchmark_hnsw.cpp | HNSW | N/A | ~700-800MB |
| benchmark_ivf_ondisk.cpp | IVF (mmap) | IndexFlatL2 (mmap) | ~50-200MB（取决于nprobe） |
| benchmark_hnsw_ivf.cpp | IVF (mmap) | IndexHNSW (mmap) | ~100-300MB（取决于nprobe和M） |

**注意**：内存占用差异主要来自：
- 索引结构的差异（HNSW vs IVF）
- 量化器类型的差异（HNSW vs Flat）
- mmap 的实际行为（按需加载）

这些差异是正常的，反映了不同索引类型的内存特性。

## 维护指南

### 添加新的 Benchmark 脚本时

1. **必须遵循统一的流程**：
   - 使用相同的内存监控类 `PeakMemoryMonitor`
   - 在记录峰值内存后停止监控
   - Groundtruth 在监控停止后加载

2. **代码检查清单**：
   ```cpp
   // ✅ 正确
   result.search_memory_mb = search_memory_monitor.getPeakMemoryMB();
   search_memory_monitor.stop();
   // 然后加载 groundtruth
   
   // ❌ 错误
   result.search_memory_mb = search_memory_monitor.getPeakMemoryMB();
   // 直接加载 groundtruth（仍在监控中）
   ```

3. **测试验证**：
   - 运行脚本并检查内存报告
   - 确保内存占用合理
   - 与相同类型的脚本对比

## 总结

所有 benchmark 脚本现在都遵循统一的：
- ✅ 内存监控流程
- ✅ 数据释放时机
- ✅ Groundtruth 加载时机
- ✅ 内存监测方法

这确保了所有测试结果的一致性和可比性。






