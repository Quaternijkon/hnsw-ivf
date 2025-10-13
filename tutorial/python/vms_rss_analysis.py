#!/usr/bin/env python3
"""
VMS/RSS比例分析脚本
深入分析为什么高VMS/RSS比例表明内存碎片化
"""

import numpy as np
import psutil
import gc
import os
import time
import subprocess

def get_memory_info():
    """获取当前内存信息"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / (1024 * 1024),
        'vms_mb': memory_info.vms / (1024 * 1024),
        'ratio': memory_info.vms / memory_info.rss if memory_info.rss > 0 else 0
    }

def analyze_memory_mapping():
    """分析内存映射情况"""
    try:
        # 读取/proc/self/maps来了解内存映射
        with open('/proc/self/maps', 'r') as f:
            maps = f.readlines()
        
        print("内存映射分析:")
        print("=" * 50)
        
        total_virtual = 0
        total_physical = 0
        heap_count = 0
        stack_count = 0
        mmap_count = 0
        
        for line in maps:
            parts = line.split()
            if len(parts) >= 2:
                address_range = parts[0]
                permissions = parts[1]
                
                # 计算地址范围大小
                start, end = address_range.split('-')
                size = int(end, 16) - int(start, 16)
                total_virtual += size
                
                # 统计不同类型的映射
                if 'heap' in line:
                    heap_count += 1
                    total_physical += size
                elif 'stack' in line:
                    stack_count += 1
                    total_physical += size
                elif 'rw' in permissions and 'p' in permissions:  # 私有可读写
                    mmap_count += 1
                    total_physical += size
        
        print(f"总虚拟地址空间: {total_virtual / (1024*1024):.2f} MB")
        print(f"总物理映射: {total_physical / (1024*1024):.2f} MB")
        print(f"堆映射数量: {heap_count}")
        print(f"栈映射数量: {stack_count}")
        print(f"内存映射数量: {mmap_count}")
        print(f"虚拟/物理比例: {total_virtual / total_physical:.2f}")
        
    except FileNotFoundError:
        print("无法读取/proc/self/maps（非Linux系统）")

def demonstrate_fragmentation_causes():
    """演示导致高VMS/RSS比例的原因"""
    
    print("=" * 60)
    print("VMS/RSS比例分析")
    print("=" * 60)
    
    # 初始状态
    initial = get_memory_info()
    print(f"初始状态:")
    print(f"  RSS: {initial['rss_mb']:.2f} MB")
    print(f"  VMS: {initial['vms_mb']:.2f} MB")
    print(f"  VMS/RSS比例: {initial['ratio']:.2f}")
    print()
    
    print("高VMS/RSS比例的原因分析:")
    print("=" * 40)
    
    print("1. 内存映射文件 (Memory-mapped files):")
    print("   - 当使用mmap()映射大文件时，VMS会增加但RSS可能很小")
    print("   - 只有实际访问的页面才会加载到物理内存")
    print("   - 您的Faiss使用了IO_FLAG_MMAP，这是主要原因！")
    print()
    
    print("2. 共享库和代码段:")
    print("   - 动态链接库占用虚拟地址空间")
    print("   - 但多个进程可以共享同一物理内存")
    print("   - 导致VMS高但RSS相对较低")
    print()
    
    print("3. 内存碎片化:")
    print("   - 频繁分配和释放不同大小的内存块")
    print("   - 导致虚拟地址空间不连续")
    print("   - 物理内存使用效率降低")
    print()
    
    print("4. 过度分配:")
    print("   - 程序分配了比实际需要更多的虚拟内存")
    print("   - 但实际使用的物理内存较少")
    print("   - 常见于内存池或预分配策略")
    print()
    
    # 模拟Faiss的mmap使用
    print("5. 模拟Faiss的mmap使用:")
    print("   创建大文件并映射到内存...")
    
    # 创建一个大文件
    filename = "test_mmap_file.bin"
    file_size = 100 * 1024 * 1024  # 100MB
    
    with open(filename, 'wb') as f:
        f.write(b'\x00' * file_size)
    
    # 使用mmap映射文件
    import mmap
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # 只访问文件的一小部分
            data = mm[0:1024]  # 只读取1KB
            
            mem_after_mmap = get_memory_info()
            print(f"   文件大小: {file_size / (1024*1024):.2f} MB")
            print(f"   实际访问: 1 KB")
            print(f"   RSS变化: {mem_after_mmap['rss_mb'] - initial['rss_mb']:.2f} MB")
            print(f"   VMS变化: {mem_after_mmap['vms_mb'] - initial['vms_mb']:.2f} MB")
            print(f"   VMS/RSS比例: {mem_after_mmap['ratio']:.2f}")
    
    # 清理
    os.remove(filename)
    print()
    
    print("6. 您的Faiss情况分析:")
    print(f"   您的VMS/RSS比例: 15.21")
    print(f"   这主要由于:")
    print(f"   - Faiss使用mmap加载索引文件")
    print(f"   - 索引文件大小: ~500MB")
    print(f"   - 但实际使用的物理内存: ~150MB")
    print(f"   - 这是正常的mmap行为，不是内存碎片化！")
    print()
    
    print("7. 正确的判断标准:")
    print("   对于使用mmap的程序:")
    print("   - VMS/RSS比例 5-20 是正常的")
    print("   - 比例 > 50 才可能表示内存碎片化")
    print("   - 需要结合其他指标判断")
    print()
    
    print("8. 真正的内存碎片化指标:")
    print("   - 频繁的内存分配/释放")
    print("   - 内存使用率低但分配失败")
    print("   - 大量小内存块分配")
    print("   - 内存使用模式不规律")

if __name__ == "__main__":
    demonstrate_fragmentation_causes()
    print("\n" + "=" * 60)
    analyze_memory_mapping()
