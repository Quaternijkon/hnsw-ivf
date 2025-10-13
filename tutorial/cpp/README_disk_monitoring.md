# Faiss 基准测试程序 - 磁盘监控版本

## 概述

这是一个增强版的Faiss基准测试程序，在原有功能基础上新增了磁盘I/O监控功能，能够全面分析系统性能瓶颈，特别是磁盘相关的性能指标。

## 新增功能

### 1. 磁盘I/O监控
- **读取/写入速度**: 监控磁盘的读取和写入速度 (MB/s)
- **IOPS监控**: 每秒输入/输出操作数统计
- **IO延迟**: 平均I/O操作延迟时间
- **磁盘利用率**: 磁盘使用率百分比
- **磁盘空间**: 可用空间和总空间统计

### 2. 磁盘瓶颈检测
- **自动瓶颈识别**: 基于多个指标自动判断是否存在磁盘瓶颈
- **瓶颈阈值**: 
  - 磁盘利用率 > 80%
  - 读取/写入速度 < 100 MB/s
  - IO延迟 > 10 ms
  - 总IOPS < 1000

### 3. 性能优化建议
- **磁盘升级建议**: 当检测到瓶颈时提供具体的硬件升级建议
- **配置优化**: 基于测试结果推荐最优的线程配置
- **系统调优**: 提供I/O调度器和缓存优化建议

## 文件结构

```
benchmark-thread-disk.cpp     # 主程序文件
compile_disk_benchmark.sh     # 编译脚本
README_disk_monitoring.md     # 说明文档
```

## 编译和运行

### 编译
```bash
cd /home/gpu/dry/faiss/tutorial/cpp/
./compile_disk_benchmark.sh
```

### 运行
```bash
./benchmark-thread-disk
```

## 输出结果

### 1. 控制台输出
程序会实时显示：
- 每个线程配置的测试进度
- 实时的磁盘I/O指标
- 磁盘瓶颈检测结果
- 性能优化建议

### 2. 文件输出
结果会保存到 `benchmark_results_with_disk.txt` 文件中，包含以下列：

| 列名 | 描述 |
|------|------|
| Threads | 线程数 |
| Time(s) | 总运行时间 |
| QPS | 每秒查询数 |
| Avg Power(W) | 平均功耗 |
| CPU Energy(J) | CPU能耗 |
| Mem Energy(J) | 内存能耗 |
| Power/Query(mJ) | 每查询功耗 |
| Efficiency | 能效比 |
| Avg Lat(ms) | 平均延迟 |
| P50 Lat(ms) | 50分位延迟 |
| P99 Lat(ms) | 99分位延迟 |
| Max Lat(ms) | 最大延迟 |
| Min Lat(ms) | 最小延迟 |
| Peak Mem(MB) | 峰值内存 |
| **Disk Read(MB/s)** | **磁盘读取速度** |
| **Disk Write(MB/s)** | **磁盘写入速度** |
| **Total IOPS** | **总IOPS** |
| **IO Latency(ms)** | **IO延迟** |
| **Disk Util(%)** | **磁盘利用率** |
| **Bottleneck** | **是否瓶颈** |

## 磁盘监控技术细节

### 1. 监控方法
- 基于Linux `/proc/diskstats` 接口
- 实时读取磁盘统计信息
- 计算增量值获得准确的I/O指标

### 2. 监控指标
```cpp
struct DiskStats {
    double read_bytes = 0.0;              // 读取字节数
    double write_bytes = 0.0;             // 写入字节数
    double read_ops = 0.0;                // 读取操作数
    double write_ops = 0.0;               // 写入操作数
    double read_speed_mbps = 0.0;         // 读取速度 (MB/s)
    double write_speed_mbps = 0.0;        // 写入速度 (MB/s)
    double read_iops = 0.0;               // 读取IOPS
    double write_iops = 0.0;               // 写入IOPS
    double total_iops = 0.0;              // 总IOPS
    double avg_io_latency_ms = 0.0;        // 平均IO延迟 (ms)
    double disk_utilization = 0.0;         // 磁盘利用率 (%)
    double queue_depth = 0.0;             // 队列深度
    bool disk_bottleneck = false;         // 是否为磁盘瓶颈
    string disk_device = "";              // 磁盘设备名
    double available_space_gb = 0.0;      // 可用空间 (GB)
    double total_space_gb = 0.0;          // 总空间 (GB)
    double space_utilization = 0.0;       // 空间利用率 (%)
};
```

### 3. 瓶颈检测算法
```cpp
void detect_disk_bottleneck(DiskStats& stats) {
    bool high_utilization = stats.disk_utilization > 80.0;
    bool low_speed = stats.read_speed_mbps < 100.0 && stats.write_speed_mbps < 100.0;
    bool high_latency = stats.avg_io_latency_ms > 10.0;
    bool low_iops = stats.total_iops < 1000.0;
    
    stats.disk_bottleneck = high_utilization || (low_speed && high_latency) || low_iops;
}
```

## 使用场景

### 1. 性能调优
- 识别系统性能瓶颈
- 优化线程配置
- 硬件升级建议

### 2. 容量规划
- 评估存储需求
- 预测I/O性能
- 制定升级计划

### 3. 故障诊断
- 磁盘健康检查
- 性能问题定位
- 系统优化建议

## 注意事项

1. **权限要求**: 需要读取 `/proc/diskstats` 的权限
2. **系统兼容性**: 主要针对Linux系统设计
3. **磁盘设备检测**: 自动检测根文件系统设备
4. **监控精度**: 基于系统统计信息，精度取决于系统更新频率

## 扩展功能

### 1. 自定义阈值
可以修改 `detect_disk_bottleneck` 函数中的阈值来适应不同的硬件环境。

### 2. 多磁盘监控
可以扩展代码来监控多个磁盘设备。

### 3. 历史数据
可以添加数据持久化功能来保存历史监控数据。

## 故障排除

### 1. 编译错误
- 确保安装了Faiss库
- 检查OpenMP支持
- 验证C++17编译器

### 2. 运行时错误
- 检查磁盘设备权限
- 验证数据文件存在
- 确认系统支持RAPL接口

### 3. 监控数据异常
- 检查磁盘设备名称
- 验证系统统计信息更新
- 确认监控时间间隔合理

## 性能影响

磁盘监控功能对系统性能的影响很小：
- CPU开销: < 1%
- 内存开销: < 1MB
- 磁盘I/O: 仅读取系统统计信息

## 总结

这个增强版的基准测试程序提供了全面的性能分析能力，特别适合：
- 大规模向量搜索系统
- 高并发I/O应用
- 性能调优和容量规划
- 系统瓶颈诊断

通过磁盘监控功能，用户可以更好地理解系统性能瓶颈，制定针对性的优化策略。
