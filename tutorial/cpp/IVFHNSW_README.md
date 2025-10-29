# IndexIVFHNSW 使用说明

## 概述

`IndexIVFHNSW` 是一个结合了 IVFFlat 和 HNSW 量化器的新索引类型。它使用 HNSW 图结构进行快速的粗量化，同时保持 IVFFlat 的存储方式（原始向量存储）。

## 文件说明

- **faiss/IndexIVFHNSW.h** - 头文件，定义了 `IndexIVFHNSW` 类
- **faiss/IndexIVFHNSW.cpp** - 实现文件，包含所有索引方法的实现
- **tutorial/cpp/0-IVFHNSW.cpp** - 示例程序，展示如何使用这个索引

## 主要特性

### 1. 继承结构
- 继承自 `IndexIVF`
- 使用 `IndexHNSWFlat` 作为粗量化器
- 支持 L2 和内积距离度量

### 2. 关键参数

#### HNSW 参数
- **M** (默认 32): HNSW 图中每个节点的邻居数量
- **efConstruction** (默认 40): 构建时的搜索深度
- **efSearch** (默认 16): 搜索时的搜索深度

#### IVF 参数
- **nlist**: 聚类中心的数量
- **nprobe**: 搜索时访问的聚类数量

## 使用示例

```cpp
#include <faiss/IndexIVFHNSW.h>

int main() {
    int d = 64;              // 向量维度
    int nb = 100000;         // 数据库大小
    int nlist = 100;         // 聚类数量
    int M = 32;              // HNSW M 参数
    
    // 创建索引
    faiss::IndexIVFHNSW index(d, nlist, M);
    
    // 设置 HNSW 参数
    index.set_hnsw_parameters(32, 40, 16);
    
    // 训练索引
    index.train(nb, xb);
    
    // 添加向量
    index.add(nb, xb);
    
    // 设置搜索参数
    index.nprobe = 10;
    
    // 搜索
    index.search(nq, xq, k, D, I);
    
    return 0;
}
```

## 与 demo.cpp 的对比

### demo.cpp 的原始结构
```cpp
// 在主函数中混合了以下内容：
// - 数据读取和处理
// - 索引创建和训练
// - 索引保存和加载
// - 搜索和评估
```

### 重构后的结构

#### IndexIVFHNSW 类（封装的索引方法）
- `train()` - 训练索引
- `add()` / `add_core()` - 添加向量
- `search()` - 搜索
- `encode_vectors()` - 向量编码
- `reconstruct_from_offset()` - 向量重建
- `set_hnsw_parameters()` - 设置 HNSW 参数

#### 外层调用程序（保留在 0-IVFHNSW.cpp）
- 数据生成和处理
- 参数配置
- 性能评估
- 结果输出

## 核心方法说明

### 构造函数
```cpp
// 方式1：自动创建 HNSW 量化器
IndexIVFHNSW(size_t d, size_t nlist, size_t M = 32, MetricType metric = METRIC_L2);

// 方式2：使用外部量化器
IndexIVFHNSW(Index* quantizer, size_t d, size_t nlist, MetricType metric = METRIC_L2);
```

### 训练方法
```cpp
void train(idx_t n, const float* x);
```
- 训练 HNSW 量化器
- 训练 IVF 结构

### 添加方法
```cpp
void add_core(idx_t n, const float* x, const idx_t* xids, 
              const idx_t* precomputed_idx, void* inverted_list_context);
```
- 支持多线程并行添加
- 每个线程处理不同的倒排列表

### 搜索方法
```cpp
void search(idx_t n, const float* x, idx_t k, 
            float* distances, idx_t* labels, const SearchParameters* params);
```
- 使用 HNSW 进行粗量化
- 在选定的倒排列表中进行精确搜索

## 编译说明

确保将新文件添加到编译系统中：

```bash
# 如果使用 CMake
# 在 faiss/CMakeLists.txt 中添加：
# faiss/IndexIVFHNSW.cpp

# 编译示例程序
g++ -o 0-IVFHNSW tutorial/cpp/0-IVFHNSW.cpp -lfaiss -fopenmp
```

## 性能特点

### 优势
1. **快速粗量化**: HNSW 提供比 Flat 更快的聚类中心查找
2. **高召回率**: 结合了 HNSW 的高质量近似和 IVFFlat 的精确距离计算
3. **可扩展性**: 适用于大规模数据集

### 适用场景
- 需要高召回率的应用
- 向量维度中等（64-512维）
- 数据库规模在百万到千万级别

## 参数调优建议

### nlist（聚类数量）
- 推荐值: `sqrt(nb)` 到 `4*sqrt(nb)`
- 较大的 nlist 提高召回率但降低速度

### nprobe（搜索时访问的聚类）
- 推荐值: 10-100
- 增大 nprobe 提高召回率但降低速度

### M（HNSW 邻居数）
- 推荐值: 16-64
- 较大的 M 提高搜索质量但增加内存使用

### efConstruction
- 推荐值: 40-500
- 影响索引构建质量和时间

### efSearch
- 推荐值: 16-512
- 影响搜索质量和速度

## 注意事项

1. 该索引默认不使用残差编码（`by_residual = false`）
2. HNSW 量化器在训练时自动构建
3. 索引支持序列化和反序列化（通过 `write_index` / `read_index`）
4. 多线程搜索通过 `parallel_mode` 参数控制

## 扩展建议

可以进一步扩展的功能：
- 支持残差编码（`by_residual = true`）
- 添加量化器训练的自定义选项
- 实现范围搜索的优化
- 添加增量更新支持

