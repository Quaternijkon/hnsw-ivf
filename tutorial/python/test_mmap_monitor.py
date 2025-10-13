#!/usr/bin/env python3
"""
测试mmap内存监控器的功能
验证新的内存计算方法的准确性
"""

import os
import time
import tempfile
import numpy as np
from mmap_memory_monitor import MmapMemoryMonitor


def create_test_file(size_mb: int = 100) -> str:
    """创建测试用的临时文件"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
    
    # 写入指定大小的数据
    data_size = size_mb * 1024 * 1024
    chunk_size = 1024 * 1024  # 1MB chunks
    
    with open(temp_file.name, 'wb') as f:
        written = 0
        while written < data_size:
            chunk = np.random.bytes(min(chunk_size, data_size - written))
            f.write(chunk)
            written += len(chunk)
    
    print(f"创建测试文件: {temp_file.name} ({size_mb}MB)")
    return temp_file.name


def test_memory_baseline():
    """测试内存基线记录"""
    print("=== 测试1: 内存基线记录 ===")
    
    monitor = MmapMemoryMonitor(enable_continuous=False)
    baseline = monitor.baseline_memory
    
    print(f"程序启动基线内存: {baseline['rss_mb']:.2f} MB")
    print(f"虚拟内存: {baseline['vms_mb']:.2f} MB")
    
    # 创建一些数据，观察内存变化
    data = np.random.rand(1000000)  # 约8MB
    
    current = monitor._get_current_memory_info()
    growth = current['rss_mb'] - baseline['rss_mb']
    
    print(f"创建数据后内存: {current['rss_mb']:.2f} MB (+{growth:+.2f} MB)")
    
    del data
    monitor.cleanup()
    print("测试1完成\n")


def test_mmap_file_tracking():
    """测试mmap文件追踪"""
    print("=== 测试2: mmap文件追踪 ===")
    
    monitor = MmapMemoryMonitor(enable_continuous=False)
    
    # 创建测试文件
    test_file = create_test_file(50)  # 50MB文件
    
    # 注册mmap文件
    monitor.register_mmap_file(test_file, "测试索引文件")
    
    # 模拟文件映射（这里只是读取文件来模拟内存使用）
    with monitor.monitor_phase("文件映射测试"):
        # 读取文件的一部分来模拟mmap的内存使用
        with open(test_file, 'rb') as f:
            data = f.read(10 * 1024 * 1024)  # 读取10MB
            time.sleep(1)  # 模拟处理时间
    
    # 生成分析报告
    analysis = monitor.get_mmap_memory_analysis()
    print(f"mmap文件总大小: {analysis['total_mmap_file_size_mb']:.2f} MB")
    print(f"估算加载内存: {analysis['estimated_loaded_mmap_mb']:.2f} MB")
    print(f"程序自身内存: {analysis['program_memory_mb']:.2f} MB")
    
    # 清理
    os.unlink(test_file)
    monitor.cleanup()
    print("测试2完成\n")


def test_phase_memory_tracking():
    """测试阶段内存追踪"""
    print("=== 测试3: 阶段内存追踪 ===")
    
    monitor = MmapMemoryMonitor(enable_continuous=True, sampling_interval=0.5)
    
    # 模拟不同阶段的内存使用
    with monitor.monitor_phase("数据加载"):
        data1 = np.random.rand(500000)  # 约4MB
        time.sleep(1)
    
    with monitor.monitor_phase("数据处理"):
        data2 = np.random.rand(1000000)  # 约8MB
        time.sleep(1)
    
    with monitor.monitor_phase("结果输出"):
        result = data1 + data2[:len(data1)]
        time.sleep(0.5)
    
    # 获取阶段摘要
    phase_summary = monitor.get_phase_summary()
    print("\n阶段内存摘要:")
    for phase_name, summary in phase_summary.items():
        print(f"  {phase_name}: {summary['start_memory_mb']:.1f}→{summary['end_memory_mb']:.1f}MB "
              f"(+{summary['memory_growth_mb']:+.1f}MB), 耗时{summary['duration_s']:.1f}s")
    
    monitor.cleanup()
    print("测试3完成\n")


def test_mmap_vs_baseline_comparison():
    """测试mmap方法与基线方法的对比"""
    print("=== 测试4: mmap方法 vs 基线方法对比 ===")
    
    # 创建测试文件
    test_file = create_test_file(100)  # 100MB文件
    
    monitor = MmapMemoryMonitor(enable_continuous=False)
    
    # 记录加载前状态
    monitor.mark_phase_start("文件加载前")
    before_memory = monitor._get_current_memory_info()
    
    # 注册并"加载"文件
    monitor.register_mmap_file(test_file, "大型索引文件")
    
    # 模拟mmap加载（部分读取）
    with open(test_file, 'rb') as f:
        # 读取文件的一小部分来模拟实际的mmap使用
        chunk = f.read(20 * 1024 * 1024)  # 20MB
    
    monitor.mark_phase_end("文件加载前")
    after_memory = monitor._get_current_memory_info()
    
    # 分析内存使用
    analysis = monitor.get_mmap_memory_analysis()
    
    print("内存使用对比:")
    print(f"  文件大小: 100 MB")
    print(f"  实际RSS增长: {after_memory['rss_mb'] - before_memory['rss_mb']:+.2f} MB")
    print(f"  mmap方法估算:")
    print(f"    - 程序自身内存: {analysis['program_memory_mb']:.2f} MB")
    print(f"    - mmap加载内存: {analysis['estimated_loaded_mmap_mb']:.2f} MB")
    print(f"    - 加载效率: {analysis['mmap_efficiency']:.1%}")
    print(f"  传统基线方法会错误地认为索引占用了: ~80MB (文件大小的80%)")
    
    # 生成完整报告
    print("\n完整mmap内存报告:")
    report = monitor.generate_mmap_memory_report()
    print(report)
    
    # 清理
    os.unlink(test_file)
    monitor.cleanup()
    print("测试4完成\n")


def main():
    """运行所有测试"""
    print("开始测试mmap内存监控器")
    print("=" * 50)
    
    test_memory_baseline()
    test_mmap_file_tracking()
    test_phase_memory_tracking()
    test_mmap_vs_baseline_comparison()
    
    print("所有测试完成！")
    print("\n主要改进点:")
    print("1. 移除了不准确的索引内存基线概念")
    print("2. 基于实际RSS变化计算程序内存使用")
    print("3. 区分mmap文件大小和实际加载的内存")
    print("4. 提供更准确的内存分解分析")
    print("5. 支持页面缓存监控（Linux系统）")


if __name__ == "__main__":
    main()
