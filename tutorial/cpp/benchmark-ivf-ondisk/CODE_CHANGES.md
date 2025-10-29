# benchmark_ivf_ondisk.cpp 代码变更对比

## runBuildTest() 函数

### 变更前（原始版本）
```cpp
TestResult runBuildTest(int nlist, size_t d, size_t nt, size_t nb) {
    // 创建索引
    faiss::IndexIVFFlat* index = new faiss::IndexIVFFlat(...);
    
    // ❌ 一次性加载全部训练数据
    auto train_data = read_fbin(LEARN_FILE);
    index->train(nt, train_data.first.data());
    
    // ❌ 一次性加载全部基础数据
    auto base_data = read_fbin(BASE_FILE);
    
    // ❌ 一次性添加
    index->add(nb, base_data.first.data());
    
    // 最后才保存
    faiss::write_index(index, index_filename.c_str());
    delete index;
}
```

**问题：训练数据 + 基础数据 + 索引同时在内存**

### 变更后（优化版本）
```cpp
TestResult runBuildTest(int nlist, size_t d, size_t nt, size_t nb) {
    // ✓ 检查索引是否已存在
    ifstream index_check(index_filename);
    if (index_check.good()) {
        return result;  // 跳过重复构建
    }
    
    // ✓ 1. 训练阶段
    faiss::IndexIVFFlat* index_for_training = new faiss::IndexIVFFlat(...);
    auto [xt_data, _] = read_fbin(LEARN_FILE);
    index_for_training->train(nt, xt_data.data());
    
    // ✓ 2. 立即删除训练索引
    delete index_for_training;
    
    // ✓ 3. 创建空索引框架并保存到磁盘
    faiss::IndexIVFFlat* index_shell = new faiss::IndexIVFFlat(...);
    faiss::write_index(index_shell, index_filename.c_str());
    delete index_shell;
    
    // ✓ 4. 释放训练数据
    xt_data.clear();
    xt_data.shrink_to_fit();
    
    // ✓ 5. 以读写模式从磁盘加载
    faiss::Index* index_ondisk = faiss::read_index(index_filename.c_str(), 0);
    
    // ✓ 6. 分块添加数据
    size_t chunk_size = 100000;
    for (size_t i = 0; i < nb; i += chunk_size) {
        auto [xb_chunk, __] = read_fbin(BASE_FILE, i, chunk_size);
        index_ondisk->add(min(chunk_size, nb - i), xb_chunk.data());
        
        // 立即释放
        xb_chunk.clear();
        xb_chunk.shrink_to_fit();
    }
    
    // ✓ 7. 写回磁盘
    faiss::write_index(index_ondisk, index_filename.c_str());
    delete index_ondisk;
}
```

**优势：内存占用 = max(训练数据, 单个chunk)**

## runSearchTest() 函数

### 变更前
```cpp
TestResult runSearchTest(...) {
    // 使用mmap（已经很好）
    faiss::Index* index = faiss::read_index(index_filename.c_str(), IO_FLAG_MMAP);
    
    // ❌ 内存监控开始较晚
    PeakMemoryMonitor search_memory_monitor;
    search_memory_monitor.start();
    
    // 加载查询数据
    auto query_data = read_fbin(QUERY_FILE);
    float* xq = query_data.first.data();
    
    // 搜索...
    delete index;
}
```

### 变更后
```cpp
TestResult runSearchTest(...) {
    // ✓ 更早开始监控
    PeakMemoryMonitor search_memory_monitor;
    search_memory_monitor.start();
    
    // ✓ 使用mmap（保持）
    faiss::Index* index = faiss::read_index(index_filename.c_str(), IO_FLAG_MMAP);
    search_memory_monitor.update();
    
    // ✓ 使用结构化绑定
    auto [xq_data, _] = read_fbin(QUERY_FILE);
    search_memory_monitor.update();
    
    // 搜索...
    search_memory_monitor.update();
    
    // ✓ 及时删除
    delete index;
}
```

## 关键代码模式

### 1. 结构化绑定（C++17）
```cpp
// 优化前
auto train_data = read_fbin(LEARN_FILE);
float* xt = train_data.first.data();

// 优化后
auto [xt_data, _] = read_fbin(LEARN_FILE);
float* xt = xt_data.data();
```

### 2. 及时释放内存
```cpp
// 训练数据用完后
xt_data.clear();            // 清空内容
xt_data.shrink_to_fit();    // 释放容量

// chunk用完后
xb_chunk_data.clear();
xb_chunk_data.shrink_to_fit();
```

### 3. own_fields 管理
```cpp
// 训练索引：不拥有量化器
index_for_training->own_fields = false;
delete index_for_training;  // 不会删除quantizer

// 空索引：拥有量化器
index_shell->own_fields = true;
delete index_shell;  // 会删除quantizer
```

### 4. 索引存在性检查
```cpp
ifstream index_check(index_filename);
if (index_check.good()) {
    cout << "索引文件已存在，跳过构建" << endl;
    return result;
}
```

### 5. 分块处理模式
```cpp
size_t chunk_size = 100000;
size_t num_chunks = (nb + chunk_size - 1) / chunk_size;

for (size_t i = 0; i < nb; i += chunk_size) {
    // 读取一个chunk
    auto [chunk_data, __] = read_fbin(BASE_FILE, i, chunk_size);
    size_t current_size = min(chunk_size, nb - i);
    
    // 处理chunk
    index_ondisk->add(current_size, chunk_data.data());
    
    // 立即释放
    chunk_data.clear();
    chunk_data.shrink_to_fit();
    
    // 进度提示
    if ((i / chunk_size + 1) % 10 == 0) {
        cout << "已处理 " << (i / chunk_size + 1) << "/" << num_chunks << endl;
    }
}
```

## 统计数据

### 代码行数
- 原始版本: 489 行
- 优化版本: ~520 行
- 增加: ~6% (主要是注释和进度提示)

### 修改函数
- `runBuildTest()`: 大幅重构（~60行 -> ~130行）
- `runSearchTest()`: 小幅优化（~75行 -> ~85行）
- 其他函数: 无改动

### 新增特性
- ✓ 索引存在性检查
- ✓ 分块数据处理
- ✓ 及时内存释放
- ✓ 更准确的内存监控
- ✓ 进度提示输出
- ✓ C++17结构化绑定

## 编译要求

```bash
# 需要C++17支持
-std=c++17

# 需要OpenMP
-fopenmp

# 建议的优化选项
-O3 -march=native
```

## 总结

优化版本通过**渐进式内存管理**和**分块处理**，将内存占用从**峰值同时占用**改为**分阶段占用**，实现了显著的内存节省，同时保持了代码的可读性和可维护性。
