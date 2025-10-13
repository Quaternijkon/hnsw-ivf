#!/usr/bin/env python3
"""
内存碎片化演示脚本
演示VMS/RSS比例与内存碎片化的关系
"""

import numpy as np
import psutil
import gc
import os
import time

def get_memory_info():
    """获取当前内存信息"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / (1024 * 1024),
        'vms_mb': memory_info.vms / (1024 * 1024),
        'ratio': memory_info.vms / memory_info.rss if memory_info.rss > 0 else 0
    }

def demonstrate_fragmentation():
    """演示内存碎片化对VMS/RSS比例的影响"""
    
    print("=" * 60)
    print("内存碎片化演示")
    print("=" * 60)
    
    # 初始状态
    initial = get_memory_info()
    print(f"初始状态:")
    print(f"  RSS: {initial['rss_mb']:.2f} MB")
    print(f"  VMS: {initial['vms_mb']:.2f} MB")
    print(f"  VMS/RSS比例: {initial['ratio']:.2f}")
    print()
    
    # 模拟正常的内存分配模式
    print("1. 正常内存分配模式:")
    arrays_normal = []
    for i in range(10):
        # 分配连续的大块内存
        arr = np.zeros((1000, 1000), dtype=np.float32)  # 4MB
        arrays_normal.append(arr)
        
        if i % 3 == 0:
            mem = get_memory_info()
            print(f"  分配 {i+1} 个数组后: RSS={mem['rss_mb']:.1f}MB, VMS={mem['vms_mb']:.1f}MB, 比例={mem['ratio']:.2f}")
    
    normal_mem = get_memory_info()
    print(f"  正常分配完成: VMS/RSS比例 = {normal_mem['ratio']:.2f}")
    print()
    
    # 模拟碎片化内存分配模式
    print("2. 碎片化内存分配模式:")
    arrays_fragmented = []
    
    # 先分配一些内存
    for i in range(5):
        arr = np.zeros((500, 500), dtype=np.float32)  # 1MB
        arrays_fragmented.append(arr)
    
    # 删除一些内存，创建碎片
    del arrays_fragmented[1]
    del arrays_fragmented[3]
    gc.collect()
    
    # 再分配不同大小的内存，利用碎片
    for i in range(8):
        if i % 2 == 0:
            arr = np.zeros((200, 200), dtype=np.float32)  # 0.16MB
        else:
            arr = np.zeros((800, 800), dtype=np.float32)  # 2.56MB
        arrays_fragmented.append(arr)
        
        if i % 2 == 0:
            mem = get_memory_info()
            print(f"  碎片化分配 {i+1} 次后: RSS={mem['rss_mb']:.1f}MB, VMS={mem['vms_mb']:.1f}MB, 比例={mem['ratio']:.2f}")
    
    fragmented_mem = get_memory_info()
    print(f"  碎片化分配完成: VMS/RSS比例 = {fragmented_mem['ratio']:.2f}")
    print()
    
    # 分析Faiss搜索的内存模式
    print("3. Faiss搜索内存模式分析:")
    print("   Faiss搜索期间的内存分配特点:")
    print("   - 大量小对象分配（距离计算、结果堆等）")
    print("   - 频繁的内存分配和释放")
    print("   - 多线程并发分配")
    print("   - 不同大小的内存块混合分配")
    print()
    
    # 模拟Faiss的内存分配模式
    print("4. 模拟Faiss内存分配模式:")
    faiss_arrays = []
    
    # 模拟距离计算矩阵
    for i in range(20):
        # 不同大小的距离矩阵
        size = np.random.randint(100, 1000)
        arr = np.random.rand(size, size).astype(np.float32)
        faiss_arrays.append(arr)
        
        # 模拟一些内存释放
        if i % 5 == 0 and len(faiss_arrays) > 3:
            del faiss_arrays[0]
            gc.collect()
        
        if i % 5 == 0:
            mem = get_memory_info()
            print(f"  模拟Faiss分配 {i+1} 次后: RSS={mem['rss_mb']:.1f}MB, VMS={mem['vms_mb']:.1f}MB, 比例={mem['ratio']:.2f}")
    
    faiss_mem = get_memory_info()
    print(f"  模拟Faiss完成: VMS/RSS比例 = {faiss_mem['ratio']:.2f}")
    print()
    
    # 总结
    print("5. 总结:")
    print(f"  正常分配: VMS/RSS = {normal_mem['ratio']:.2f}")
    print(f"  碎片化分配: VMS/RSS = {fragmented_mem['ratio']:.2f}")
    print(f"  模拟Faiss: VMS/RSS = {faiss_mem['ratio']:.2f}")
    print(f"  您的Faiss: VMS/RSS = 15.21")
    print()
    print("  结论: VMS/RSS比例越高，说明:")
    print("  - 虚拟地址空间使用效率低")
    print("  - 内存分配碎片化严重")
    print("  - 可能存在内存泄漏或过度分配")
    print("  - 系统内存管理效率低下")

if __name__ == "__main__":
    demonstrate_fragmentation()
