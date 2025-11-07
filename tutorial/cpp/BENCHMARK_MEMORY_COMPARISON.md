# Benchmark 内存监控一致性分析报告

## 概述

本文档详细分析了四个 benchmark 文件的内存监控机制，识别了影响数据可比性的关键差异。

## 关键发现：不一致性问题

### 1. Build阶段 - Training Memory 的不一致性

#### benchmark_hnsw.cpp (lines 209-251)
- **training_memory_mb**: 记录的是**索引初始化后**的内存（HNSW不需要训练）
- **内存中包含**: 索引对象本身
- **内存中不包含**: 训练数据（HNSW不需要训练）、基础数据（尚未加载）

#### benchmark_hnsw_ivf.cpp (lines 208-329)
- **training_memory_mb**: 记录的是**训练完成后**的峰值内存
- **内存中包含**: 索引对象、量化器、**训练数据（LEARN_FILE）仍在内存中**
- **训练数据释放时机**: 在记录 training_memory_mb **之后**（line 251-252）

#### benchmark_ivf.cpp (lines 208-275)
- **training_memory_mb**: 记录的是**训练完成后**的峰值内存
- **内存中包含**: 索引对象、量化器、**训练数据（LEARN_FILE）仍在内存中**
- **训练数据释放时机**: 在记录 training_memory_mb **之后**（line 245-247）

#### benchmark_ivf_ondisk.cpp (lines 208-321)
- **training_memory_mb**: 记录的是**训练完成后**的峰值内存
- **内存中包含**: 索引对象、量化器、**训练数据（LEARN_FILE）仍在内存中**
- **训练数据释放时机**: 在记录 training_memory_mb **之后**（line 250-252）

**问题**: 
- `benchmark_hnsw.cpp` 的 training_memory 不包括训练数据，但其他三个都包括
- 这导致 HNSW 的 training_memory 数值偏小，不具有可比性

### 2. Build阶段 - Add Memory 的不一致性

#### benchmark_hnsw.cpp (lines 232-240)
- **add_memory_mb**: 记录的是**添加数据后**的峰值内存
- **内存中包含**: 索引对象、**基础数据（BASE_FILE）仍在内存中**（xb_data是局部变量，函数结束前不会释放）
- **问题**: 基础数据会一直保留在内存中，直到函数返回

#### benchmark_hnsw_ivf.cpp (lines 288-314)
- **add_memory_mb**: 记录的是**添加数据后**的峰值内存
- **内存中包含**: 索引对象（已从磁盘加载，使用 ArrayInvertedLists）、**部分基础数据可能在内存中**（分块加载，每个chunk添加后立即释放）
- **分块策略**: 每个chunk在添加后立即释放（line 297-298）

#### benchmark_ivf.cpp (lines 250-264)
- **add_memory_mb**: 记录的是**添加数据后**的峰值内存
- **内存中包含**: 索引对象、**基础数据（BASE_FILE）仍在内存中**
- **基础数据释放时机**: 在记录 add_memory_mb **之后**（line 262-263）

#### benchmark_ivf_ondisk.cpp (lines 283-308)
- **add_memory_mb**: 记录的是**添加数据后**的峰值内存
- **内存中包含**: 索引对象（已从磁盘加载，使用 ArrayInvertedLists）、**部分基础数据可能在内存中**（分块加载，每个chunk添加后立即释放）
- **分块策略**: 每个chunk在添加后立即释放（line 292-293）

**问题**:
- `benchmark_hnsw.cpp` 和 `benchmark_ivf.cpp` 的 add_memory 包括完整的基础数据
- `benchmark_hnsw_ivf.cpp` 和 `benchmark_ivf_ondisk.cpp` 使用分块策略，峰值内存可能不包括完整基础数据
- 这导致 add_memory 的可比性存在问题

### 3. Search阶段 - 索引状态的不一致性

#### benchmark_hnsw.cpp (lines 254-322)
- **索引状态**: 内存中的完整索引（包含所有向量数据）
- **基础数据**: **仍然在内存中**（从 runBuildTest 返回时未释放）
- **问题**: search_memory 包括了索引 + 基础数据 + 查询数据

#### benchmark_hnsw_ivf.cpp (lines 332-414)
- **索引状态**: 通过 mmap 从磁盘加载（on-disk模式）
- **基础数据**: 不在内存中（索引文件存储在磁盘）
- **索引删除**: 在加载 groundtruth **之前**删除（line 394）

#### benchmark_ivf.cpp (lines 278-346)
- **索引状态**: 内存中的完整索引（包含所有向量数据）
- **基础数据**: **不在内存中**（在 runBuildTest 中已释放，line 262-263）
- **索引删除**: 不删除（索引在 main 函数中统一管理）

#### benchmark_ivf_ondisk.cpp (lines 324-404)
- **索引状态**: 通过 mmap 从磁盘加载（on-disk模式）
- **基础数据**: 不在内存中（索引文件存储在磁盘）
- **索引删除**: 在加载 groundtruth **之前**删除（line 384）

**问题**:
- `benchmark_hnsw.cpp` 的 search_memory 包括基础数据，其他三个不包括
- 这导致 HNSW 的 search_memory 数值偏大，不具有可比性

### 4. Search阶段 - 内存监控时机的一致性

所有四个文件都遵循相同的流程：
1. ✅ 开始监控
2. ✅ 加载/创建索引
3. ✅ 加载查询数据
4. ✅ 执行搜索
5. ✅ 记录 search_memory（在加载 groundtruth 之前）
6. ✅ 停止监控
7. ✅ 加载 groundtruth 并计算召回率

**这部分是一致的** ✅

## 总结：影响可比性的关键问题

### 问题1: training_memory 不一致
- **benchmark_hnsw.cpp**: 不包括训练数据（HNSW不需要训练）
- **其他三个**: 包括训练数据
- **影响**: HNSW 的 training_memory 偏小，不能直接比较

### 问题2: add_memory 不一致
- **benchmark_hnsw.cpp**: 包括完整基础数据
- **benchmark_ivf.cpp**: 包括完整基础数据
- **benchmark_hnsw_ivf.cpp**: 分块加载，峰值可能不包括完整基础数据
- **benchmark_ivf_ondisk.cpp**: 分块加载，峰值可能不包括完整基础数据
- **影响**: on-disk 版本的 add_memory 可能偏小

### 问题3: search_memory 不一致
- **benchmark_hnsw.cpp**: 包括基础数据（仍在内存中）
- **其他三个**: 不包括基础数据
- **影响**: HNSW 的 search_memory 偏大，不能直接比较

## 建议的修复方案

### 方案1: 统一内存监控范围（推荐）

#### Build阶段 - Training Memory
- **所有benchmark**: training_memory 应该在**训练数据释放之前**记录，包括训练数据
- **例外**: HNSW 不需要训练，training_memory 可以设为 0 或记录索引初始化后的内存（但需要在文档中说明）

#### Build阶段 - Add Memory
- **所有benchmark**: add_memory 应该在**基础数据释放之前**记录，包括基础数据
- **统一策略**: 所有benchmark都应该在记录 add_memory 后再释放基础数据

#### Search阶段 - Search Memory
- **所有benchmark**: search_memory 应该**只包括索引、查询数据和搜索结果缓冲区**
- **统一策略**: 确保基础数据在 search 阶段开始前已释放（或使用 on-disk 模式）

### 方案2: 记录更细粒度的指标

可以同时记录：
- `training_memory_with_data`: 包括训练数据
- `training_memory_index_only`: 只包括索引
- `add_memory_with_data`: 包括基础数据
- `add_memory_index_only`: 只包括索引
- `search_memory_with_base`: 包括基础数据（如适用）
- `search_memory_query_only`: 只包括查询相关

这样用户可以根据需要选择比较哪个指标。

## 已完成的修复

### ✅ 修复 benchmark_hnsw.cpp

1. **runBuildTest**: 
   - ✅ 在记录 add_memory 后，显式释放基础数据（与其他benchmark对齐）
   - ✅ 添加注释说明 add_memory 包括完整基础数据
   - ✅ 添加注释说明 HNSW 的 training_memory 不包括训练数据（因为HNSW不需要训练）

2. **runSearchTest**: 
   - ✅ 添加注释说明基础数据已在 runBuildTest 中释放，search_memory 不包括基础数据

### ✅ 统一所有 benchmark 的注释

1. **training_memory**: 
   - ✅ 所有需要训练的benchmark都添加注释说明"包括训练数据"
   - ✅ HNSW添加注释说明"不包括训练数据"（因为不需要训练）

2. **add_memory**: 
   - ✅ 内存模式（benchmark_hnsw.cpp, benchmark_ivf.cpp）添加注释说明"包括完整基础数据"
   - ✅ On-disk模式（benchmark_hnsw_ivf.cpp, benchmark_ivf_ondisk.cpp）添加注释说明"不包括完整基础数据"（分块加载）

3. **search_memory**: 
   - ✅ 所有benchmark都确保基础数据在search阶段开始前已释放
   - ✅ 添加注释说明内存监控的范围

## 修复后的状态

### ✅ 已解决的不一致性问题

1. **search_memory 现在一致**: 
   - 所有benchmark的search_memory都不包括基础数据
   - 所有benchmark都在search阶段开始前释放基础数据

2. **add_memory 已明确说明**: 
   - 内存模式的add_memory包括完整基础数据
   - On-disk模式的add_memory不包括完整基础数据（这是预期的，因为使用分块加载）

3. **training_memory 已明确说明**: 
   - HNSW的training_memory不包括训练数据（因为不需要训练）
   - 其他benchmark的training_memory包括训练数据

### ⚠️ 仍需注意的差异（这些是合理的）

1. **add_memory**: 
   - 内存模式 vs On-disk模式的数据范围不同是合理的
   - On-disk模式的目的就是减少内存占用，所以分块加载是预期行为

2. **training_memory**: 
   - HNSW不需要训练，所以training_memory不包括训练数据是合理的
   - 在比较时应意识到这个差异

## 数据可比性建议

### 可以直接比较的指标

1. **search_memory**: ✅ 现在所有benchmark都一致（只包括索引、查询数据、搜索结果缓冲区）

2. **training_memory**: ⚠️ 需要区分：
   - HNSW: 只包括索引对象
   - 其他: 包括索引 + 训练数据
   - **建议**: 比较时应明确说明HNSW的training_memory不包括训练数据

3. **add_memory**: ⚠️ 需要区分：
   - 内存模式: 包括索引 + 完整基础数据
   - On-disk模式: 只包括索引 + 最后一个chunk（可能）
   - **建议**: 比较时应明确说明On-disk模式的add_memory不包括完整基础数据

### 建议的对比方式

1. **按模式分组比较**:
   - 内存模式 vs 内存模式（benchmark_hnsw.cpp vs benchmark_ivf.cpp）
   - On-disk模式 vs On-disk模式（benchmark_hnsw_ivf.cpp vs benchmark_ivf_ondisk.cpp）

2. **关注实际内存占用**:
   - search_memory 可以直接比较（所有benchmark一致）
   - add_memory 需要结合模式说明（On-disk模式较小是预期的）
   - training_memory 需要结合算法特性说明（HNSW不需要训练）

