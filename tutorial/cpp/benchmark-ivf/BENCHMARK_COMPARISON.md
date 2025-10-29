# Faiss Benchmark 工具对比参考

本文档提供三种 Faiss benchmark 工具的快速对比参考。

## 文件清单

| 文件 | 索引类型 | 配置文件 | 说明 |
|------|---------|---------|------|
| `benchmark_ivf.cpp` | IVFFlat + FlatL2 | `benchmark-ivf.config` | 基础 IVF 索引 |
| `benchmark_advanced.cpp` | IVFFlat + HNSW | `benchmark.config` | IVF + HNSW 混合索引 |
| `benchmark_hnsw.cpp` | HNSWFlat | `benchmark-hnsw.config` | 纯 HNSW 索引 |

## 快速对比表

### 索引特性对比

| 特性 | IVF | IVF+HNSW | HNSW |
|------|-----|----------|------|
| **Faiss 类** | `IndexIVFFlat` | `IndexIVFFlat` | `IndexHNSWFlat` |
| **量化器** | `IndexFlatL2` | `IndexHNSWFlat` | N/A |
| **需要训练** | ✅ 是 | ✅ 是 | ❌ 否 |
| **构建速度** | ⭐⭐⭐ 快 | ⭐⭐ 中等 | ⭐ 慢 |
| **搜索速度** | ⭐⭐ 中等 | ⭐⭐⭐ 快 | ⭐⭐⭐ 快 |
| **内存占用** | ⭐⭐⭐ 低 | ⭐⭐ 中等 | ⭐ 高 |
| **召回率** | ⭐⭐ 中等 | ⭐⭐⭐ 高 | ⭐⭐⭐ 高 |

### 参数对比

| 参数类型 | IVF | IVF+HNSW | HNSW |
|---------|-----|----------|------|
| **Build参数1** | `nlist` (聚类数) | `nlist` (聚类数) | `M` (图连接数) |
| **Build参数2** | - | `efconstruction` (HNSW构建) | `efconstruction` (HNSW构建) |
| **Search参数1** | `nprobe` (访问聚类数) | `nprobe` (访问聚类数) | `efsearch` (搜索候选集) |
| **Search参数2** | - | `efsearch` (HNSW搜索) | - |

### 参数计算方式

| 索引类型 | 参数计算公式 |
|---------|-------------|
| **IVF** | `nprobe = nlist × nprobe_ratio` |
| **IVF+HNSW** | `nprobe = nlist × nprobe_ratio`<br>`efsearch = nprobe × efsearch_ratio` |
| **HNSW** | `efsearch = efconstruction × efsearch_ratio` |

### 性能指标对比（示例：1M SIFT 数据集）

| 指标 | IVF | IVF+HNSW | HNSW |
|------|-----|----------|------|
| **构建时间** | ~80s | ~150s | ~250s |
| **索引大小** | ~512MB | ~650MB | ~800MB |
| **QPS (nprobe=32/64)** | ~1200 | ~2500 | ~3000 |
| **召回率@10** | 0.85-0.92 | 0.92-0.97 | 0.95-0.99 |
| **P99延迟** | ~1.5ms | ~0.8ms | ~0.6ms |

*注: 以上数据为典型值，实际性能取决于硬件、数据分布和参数设置*

## 推荐配置

### IVF - 快速平衡配置
```
build
  param
    nlist:2000,4000,8000
search
  param
    nprobe_ratio:0.008,0.016,0.032
```

**适用场景**: 大规模数据集，内存受限，需要快速构建

### IVF+HNSW - 高性能配置
```
build
  param
    nlist:1024,2048,4096
    efconstruction:40,80,160
search
  param
    nprobe_ratio:0.0625,0.125,0.25
    efsearch_ratio:0.5,1.0,2.0
```

**适用场景**: 高性能要求，可接受较长构建时间

### HNSW - 高精度配置
```
build
  param
    M:16,32,64
    efconstruction:40,80,160,200
search
  param
    efsearch_ratio:0.9,1.0,1.1,1.5,2.0
```

**适用场景**: 中小规模数据集，需要最高精度

## 应用场景选择指南

### 场景1: 电商商品推荐（百万级商品）
**推荐**: IVF
- 数据规模: 1M-10M
- 延迟要求: <5ms
- 召回率要求: >85%
- 理由: 快速构建，内存占用低，可频繁更新索引

### 场景2: 图像搜索（千万级图片）
**推荐**: IVF+HNSW
- 数据规模: 10M-100M
- 延迟要求: <2ms
- 召回率要求: >92%
- 理由: 高性能搜索，良好的精度-速度平衡

### 场景3: 人脸识别（百万级人脸）
**推荐**: HNSW
- 数据规模: 100K-1M
- 延迟要求: <1ms
- 召回率要求: >95%
- 理由: 最高精度，快速搜索，人脸库相对较小

### 场景4: 视频指纹（亿级特征）
**推荐**: IVF
- 数据规模: 100M+
- 延迟要求: <10ms
- 召回率要求: >80%
- 理由: 唯一能处理如此大规模数据的选项

### 场景5: 实时推荐系统
**推荐**: IVF+HNSW 或 HNSW
- 数据规模: 取决于具体业务
- 延迟要求: <1ms (P99)
- 召回率要求: >90%
- 理由: 低延迟，高吞吐量

## 参数调优策略

### IVF 调优步骤
1. **确定 nlist**: 从 `sqrt(nb)` 开始，通常在 `[1000, 10000]` 范围
2. **测试 nprobe**: 从小到大测试 `[8, 16, 32, 64, 128]`
3. **权衡精度与速度**: 选择满足召回率要求的最小 nprobe

### IVF+HNSW 调优步骤
1. **确定 nlist**: 比纯 IVF 可以更小，推荐 `[512, 4096]`
2. **设置 efconstruction**: 通常 `[40, 160]`，越大图质量越好
3. **调整 nprobe**: 从 `nlist/32` 开始测试
4. **调整 efsearch**: 从 `nprobe` 开始，逐步增加到 `2*nprobe`

### HNSW 调优步骤
1. **选择 M**: 从 `32` 开始，内存充足可以用 `64`
2. **设置 efconstruction**: 通常 `[80, 160]`
3. **测试 efsearch**: 从 `efconstruction` 开始，根据精度需求调整

## 性能优化技巧

### 通用优化
1. **OpenMP 线程数**: 设置为物理核心数或略少
2. **数据预加载**: 使用 `mmap` 或预先加载到内存
3. **批量查询**: 利用并行处理提高吞吐量

### IVF 特定优化
1. **训练数据采样**: 可以使用数据集的采样进行训练（保持 ≥30×nlist）
2. **倒排列表优化**: 考虑使用 `IndexIVFPQ` 进行压缩
3. **预计算**: 预先计算聚类中心的范数

### HNSW 特定优化
1. **M 值权衡**: M=32 是大多数场景的最佳选择
2. **构建时内存**: 构建时预留 1.5-2 倍最终索引大小的内存
3. **层级结构**: efConstruction 越大，层级结构越优

## 编译和运行

### 统一编译命令
```bash
# 在 tutorial/cpp 目录下
g++ -std=c++17 -O3 -o benchmark_ivf benchmark_ivf.cpp \
    -I ../.. -L ../../build/faiss -Wl,-rpath,../../build/faiss \
    -lfaiss -lopenblas -fopenmp

g++ -std=c++17 -O3 -o benchmark_advanced benchmark_advanced.cpp \
    -I ../.. -L ../../build/faiss -Wl,-rpath,../../build/faiss \
    -lfaiss -lopenblas -fopenmp

g++ -std=c++17 -O3 -o benchmark_hnsw benchmark_hnsw.cpp \
    -I ../.. -L ../../build/faiss -Wl,-rpath,../../build/faiss \
    -lfaiss -lopenblas -fopenmp
```

### 运行测试
```bash
# IVF
./benchmark_ivf benchmark-ivf.config

# IVF+HNSW
./benchmark_advanced benchmark.config

# HNSW
./benchmark_hnsw benchmark-hnsw.config
```

## 输出文件

每个工具都会生成两个 CSV 文件：
- `benchmark_XXX_build_results_<timestamp>.csv` - 构建阶段结果
- `benchmark_XXX_search_results_<timestamp>.csv` - 搜索阶段结果

### CSV 字段说明

#### Build Results
| 字段 | IVF | IVF+HNSW | HNSW | 说明 |
|------|-----|----------|------|------|
| `nlist` | ✅ | ✅ | - | 聚类数量 |
| `M` | - | - | ✅ | 图连接数 |
| `efconstruction` | - | ✅ | ✅ | HNSW构建参数 |
| `training_memory_mb` | ✅ | ✅ | ✅ | 训练阶段内存 |
| `add_memory_mb` | ✅ | ✅ | ✅ | 添加数据内存 |
| `training_time_s` | ✅ | ✅ | 0 | 训练时间 |
| `total_time_s` | ✅ | ✅ | ✅ | 总构建时间 |

#### Search Results
| 字段 | 说明 |
|------|------|
| `recall` | 召回率 (0-1) |
| `qps` | 每秒查询数 |
| `mspq` | 每查询毫秒数 |
| `search_memory_mb` | 搜索阶段内存 |
| `search_time_s` | 搜索总时间 |
| `mean_latency_ms` | 平均延迟 |
| `p50_latency_ms` | P50延迟 |
| `p99_latency_ms` | P99延迟 |

## 常见问题

### Q: 如何选择合适的索引类型？
**A**: 
- 数据量 < 1M: HNSW
- 1M < 数据量 < 10M: IVF+HNSW
- 数据量 > 10M: IVF

### Q: 如何提高召回率？
**A**:
- IVF: 增大 nprobe 或 nlist
- IVF+HNSW: 增大 nprobe 和 efsearch
- HNSW: 增大 efsearch 或 efconstruction

### Q: 如何降低内存占用？
**A**:
- 使用 IVF 而不是 HNSW
- 使用量化技术（PQ, SQ）
- 减小 M 值（HNSW）
- 减小 nlist（IVF）

### Q: 构建时间太长怎么办？
**A**:
- 减小 nlist 或 efconstruction
- 使用更少的训练数据
- 考虑增量构建策略

### Q: 搜索速度不够快怎么办？
**A**:
- 减小 nprobe 或 efsearch
- 增加 OpenMP 线程数
- 使用 IVF+HNSW 代替纯 IVF
- 考虑 GPU 加速

## 进阶主题

### 混合策略
某些场景可以结合多种索引：
- **冷热数据分离**: 热数据用 HNSW，冷数据用 IVF
- **多级检索**: 第一级用 IVF 粗筛，第二级用 HNSW 精排
- **动态切换**: 根据负载动态调整 nprobe/efsearch

### GPU 加速
所有三种索引都支持 GPU 加速（需要编译 GPU 版本）：
```cpp
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
```

### 量化压缩
当内存受限时，可以使用量化版本：
- `IndexIVFPQ`: IVF + Product Quantization
- `IndexIVFScalarQuantizer`: IVF + Scalar Quantization
- `IndexHNSWPQ`: HNSW + Product Quantization

## 参考资料

- [Faiss官方文档](https://github.com/facebookresearch/faiss/wiki)
- [HNSW论文](https://arxiv.org/abs/1603.09320)
- [IVF方法介绍](https://hal.inria.fr/inria-00514462/document)
- 更多 benchmark 脚本: `tutorial/python/` 目录

## 贡献

如果您有改进建议或发现问题，欢迎提交 Issue 或 Pull Request。

