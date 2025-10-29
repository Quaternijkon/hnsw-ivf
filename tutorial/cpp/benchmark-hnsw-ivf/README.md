# HNSW-IVF Benchmark (内存优化版)

本程序是针对 Faiss HNSW-IVF 索引的基准测试工具，采用了多种内存优化技术。

## 主要特性

### 内存优化策略

1. **磁盘索引存储**：索引在构建和搜索时始终保存在磁盘中，避免全部加载到内存
2. **MMAP 方式读取**：搜索时使用内存映射（mmap）读取索引，大幅减少内存占用
3. **分块数据添加**：将基础数据集分块添加到索引中（每次10万向量），避免一次性加载
4. **及时对象销毁**：在每个阶段结束后立即删除不再使用的对象
5. **内存监控**：实时监控各阶段的峰值内存使用情况

### 与原始实现的区别

相比 `benchmark_advanced.cpp`，本实现遵循 `demo.cpp` 的方式：

- **训练阶段**：训练完成后立即删除训练索引，只保留量化器
- **构建阶段**：创建空索引框架写入磁盘，然后以读写模式打开，分块添加数据
- **搜索阶段**：使用 mmap 从磁盘读取索引，而非全部加载到内存

## 编译

```bash
make
```

编译命令使用相对路径指向 Faiss 源码和编译库：
- 包含路径: `-I../../..` (指向 faiss 根目录)
- 库路径: `-L../../../build/faiss`
- 运行时库路径: `-Wl,-rpath,../../../build/faiss`

## 运行

```bash
./benchmark_hnsw_ivf [config_file]
```

如果不指定配置文件，默认使用 `benchmark.config`。

## 配置文件格式

见 `benchmark.config` 示例：

```
build
  param
    nlist:1953,3906,7812,15625,31250
    efconstruction:40,100,150,200
  metric
    training_memory
    add_memory
    training_time
    total_time
search
  param
    nprobe_ratio:0.004096,0.008192,0.016384,0.032768
    efsearch_ratio:0.5,1.0,1.5,2.0
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

### 参数说明

**Build 参数：**
- `nlist`: IVF 聚类中心数量
- `efconstruction`: HNSW 构建时的 efConstruction 参数

**Search 参数：**
- `nprobe_ratio`: nprobe/nlist 的比例
- `efsearch_ratio`: efsearch/nprobe 的比例

实际搜索时：
- `nprobe = nlist × nprobe_ratio`
- `efsearch = nprobe × efsearch_ratio`

## 输出

程序会生成两个 CSV 文件：

1. `benchmark_build_results_<timestamp>.csv`：构建阶段结果
   - nlist, efconstruction
   - training_memory_mb, add_memory_mb
   - training_time_s, total_time_s

2. `benchmark_search_results_<timestamp>.csv`：搜索阶段结果
   - nlist, efconstruction, nprobe, efsearch
   - training_memory_mb, add_memory_mb, training_time_s, total_time_s
   - recall, qps, mspq
   - search_memory_mb, search_time_s
   - mean_latency_ms, p50_latency_ms, p95_latency_ms, p99_latency_ms

## 索引文件

构建的索引保存在 `indices/` 目录下，文件名格式：
```
index_nlist<N>_efc<E>.index
```

例如：`index_nlist1953_efc40.index`

如果索引文件已存在，会跳过构建阶段直接进行搜索测试。

## 清理

清理编译产物和结果：
```bash
make clean
```

这将删除：
- 编译生成的目标文件和可执行文件
- 所有 CSV 结果文件
- indices 目录及其中的所有索引文件

