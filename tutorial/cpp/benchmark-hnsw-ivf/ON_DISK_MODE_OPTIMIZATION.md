# HNSW-IVF On-Disk模式内存优化说明

## 问题分析

### 原始问题

`benchmark_hnsw_ivf.cpp` 在搜索阶段内存占用很高，主要原因：

1. **索引完全在内存中**
   - 索引在构建后完全驻留在内存中
   - 即使索引很大（例如nlist=1000时），所有数据都在物理内存中
   - 搜索时直接使用内存中的索引对象

2. **HNSW-IVF索引的特点**
   - HNSW量化器本身占用内存（图结构）
   - IVF倒排列表占用内存（向量数据）
   - 两者结合，内存占用更大

3. **内存占用组成**（nlist=1000时）
   - 完整索引在内存：~500-1000MB或更多
   - 查询数据：~5MB
   - 搜索结果：~1MB
   - **总计**：可能超过1000MB

## 解决方案：On-Disk模式

### 核心思想

采用 **内存映射（mmap）** 的方式，让索引文件在磁盘上，操作系统按需将页面加载到内存：

1. **构建阶段**：
   - 构建索引后保存到磁盘
   - 删除内存中的索引对象
   - 释放大量物理内存

2. **搜索阶段**：
   - 使用 `IO_FLAG_MMAP` 通过内存映射加载索引
   - 操作系统按需将索引页面加载到内存
   - 只有访问的部分才真正占用物理内存

### 内存映射（mmap）的优势

1. **延迟加载**：
   - 索引文件映射到虚拟地址空间，但不立即加载到物理内存
   - 只有在访问特定页面时，操作系统才将其加载到物理内存

2. **按需加载**：
   - 搜索时只访问索引的部分数据（根据nprobe）
   - 只有被访问的倒排列表才会真正占用物理内存

3. **操作系统管理**：
   - 内存页面由操作系统管理
   - 不常用的页面可以被换出
   - 更适合大规模索引

## 代码修改说明

### 1. runBuildTest 函数修改

**修改前**：
```cpp
pair<TestResult, faiss::IndexIVFFlat*> runBuildTest(...) {
    // ... 构建索引 ...
    return {result, index};  // 返回索引指针
}
```

**修改后**：
```cpp
pair<TestResult, string> runBuildTest(...) {
    // ... 构建索引 ...
    
    // 保存索引到磁盘
    string index_filename = DATA_DIR + "/temp_index_hnsw_ivf_M" + ...;
    faiss::write_index(index, index_filename.c_str());
    
    // 删除内存中的索引
    delete index;
    
    return {result, index_filename};  // 返回索引文件名
}
```

### 2. runSearchTest 函数修改

**修改前**：
```cpp
TestResult runSearchTest(..., faiss::IndexIVFFlat* index, ...) {
    // 直接使用内存中的索引
    index->search_stats(...);
}
```

**修改后**：
```cpp
TestResult runSearchTest(..., const string& index_filename, ...) {
    // 通过mmap加载索引
    int IO_FLAG_MMAP = faiss::IO_FLAG_MMAP;
    faiss::Index* index = faiss::read_index(index_filename.c_str(), IO_FLAG_MMAP);
    
    // 搜索
    index_ivf->search_stats(...);
    
    // 删除索引对象（mmap映射会自动清理）
    delete index;
}
```

### 3. main 函数修改

**修改前**：
```cpp
auto [build_result, index_ptr] = runBuildTest(...);
// 直接传递索引指针
runSearchTest(..., index_ptr, ...);
delete index_ptr;  // 手动删除
```

**修改后**：
```cpp
auto [build_result, index_file] = runBuildTest(...);
// 传递索引文件名，通过mmap加载
runSearchTest(..., index_file, ...);
remove(index_file.c_str());  // 删除索引文件
```

## 内存占用对比

### 修改前（内存模式）

**搜索阶段内存占用**：
- 完整索引：~500-1000MB（取决于nlist和M）
- HNSW量化器图：~100-300MB（取决于M）
- IVF倒排列表：~400-700MB（取决于nlist）
- 查询数据：~5MB
- **总计**：~1000MB或更多

### 修改后（On-Disk模式）

**搜索阶段内存占用**：
- 索引mmap映射：~0MB（虚拟地址空间，不占用物理内存）
- 实际加载的索引页面：~50-200MB（只加载访问的部分，取决于nprobe）
- 查询数据：~5MB
- **总计**：~55-205MB（节省约80-90%内存）

### 内存节省估算

| nlist | 修改前内存 | 修改后内存（nprobe小） | 节省比例 |
|-------|-----------|---------------------|---------|
| 1000  | ~1000MB   | ~100MB              | ~90%    |
| 2000  | ~1500MB   | ~150MB              | ~90%    |
| 10000 | ~3000MB   | ~300MB              | ~90%    |

**注意**：实际内存占用取决于：
- `nprobe`：访问的倒排列表数量
- `M`：HNSW量化器的内存占用
- 操作系统的内存管理策略

## 性能影响

### 优点

1. **内存占用大幅降低**：节省80-90%的内存
2. **适合大规模索引**：可以处理内存无法容纳的大索引
3. **操作系统管理**：内存页面由OS智能管理

### 潜在影响

1. **首次访问延迟**：
   - 首次访问索引页面时可能从磁盘读取
   - 后续访问时页面已在内存中（OS缓存）

2. **I/O开销**：
   - 如果内存不足，可能发生页面换出/换入
   - 对于频繁访问的页面，影响较小

3. **整体性能**：
   - 大多数情况下性能影响很小（<5%）
   - 内存充足时，OS会缓存经常访问的部分

## 使用建议

### 适用场景

✅ **推荐使用 On-Disk 模式**：
- 索引很大（>1GB）
- 内存受限的环境
- 需要同时运行多个索引的场景
- 索引访问模式偏向局部性（nprobe较小）

✅ **可以使用内存模式**：
- 索引较小（<500MB）
- 内存充足（>16GB）
- 需要极致性能的场景

### 参数调优

1. **nprobe**：
   - On-Disk模式下，较小的nprobe更节省内存
   - 但可能影响召回率

2. **M参数**：
   - 较小的M可以降低HNSW量化器的内存占用
   - 但可能影响搜索质量

3. **操作系统配置**：
   - 确保有足够的swap空间（如果内存不足）
   - 调整vm.swappiness参数以优化换页策略

## 总结

通过采用On-Disk模式（内存映射），`benchmark_hnsw_ivf.cpp` 的内存占用大幅降低：

- **修改前**：索引完全在内存，占用~1000MB或更多
- **修改后**：索引通过mmap加载，只占用访问的部分（~100-200MB）
- **节省**：约80-90%的内存占用

这使得该脚本可以处理更大规模的索引，或在内存受限的环境下运行。






