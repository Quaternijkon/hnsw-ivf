#!/usr/bin/env python3
"""
Faiss 性能对比分析脚本
分析普通磁盘 vs 内存磁盘的性能差异
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def load_and_clean_data(csv_file):
    """加载并清理CSV数据"""
    try:
        # 尝试不同的分隔符
        for sep in [',', '\t', ';']:
            try:
                df = pd.read_csv(csv_file, sep=sep)
                if len(df.columns) > 1:  # 确保有多个列
                    break
            except:
                continue
        
        # 清理列名
        df.columns = df.columns.str.strip()
        
        # 查找关键列
        thread_col = None
        qps_col = None
        latency_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'thread' in col_lower:
                thread_col = col
            elif 'qps' in col_lower:
                qps_col = col
            elif 'lat' in col_lower and 'avg' in col_lower:
                latency_col = col
        
        if thread_col is None:
            print(f"警告: 在 {csv_file} 中未找到线程数列")
            return None
            
        return df, thread_col, qps_col, latency_col
        
    except Exception as e:
        print(f"错误: 无法读取文件 {csv_file}: {e}")
        return None

def analyze_performance(disk_file, memory_file):
    """分析性能差异"""
    print("=== Faiss 性能对比分析 ===")
    
    # 加载数据
    disk_data = load_and_clean_data(disk_file)
    memory_data = load_and_clean_data(memory_file)
    
    if disk_data is None or memory_data is None:
        print("错误: 无法加载数据文件")
        return
    
    disk_df, disk_thread_col, disk_qps_col, disk_latency_col = disk_data
    memory_df, memory_thread_col, memory_qps_col, memory_latency_col = memory_data
    
    print(f"普通磁盘数据: {len(disk_df)} 行")
    print(f"内存磁盘数据: {len(memory_df)} 行")
    
    # 创建对比表
    comparison_data = []
    
    # 获取线程数列表
    disk_threads = sorted(disk_df[disk_thread_col].unique())
    memory_threads = sorted(memory_df[memory_thread_col].unique())
    common_threads = sorted(set(disk_threads) & set(memory_threads))
    
    print(f"共同测试的线程数: {common_threads}")
    
    for thread in common_threads:
        disk_row = disk_df[disk_df[disk_thread_col] == thread].iloc[0]
        memory_row = memory_df[memory_df[memory_thread_col] == thread].iloc[0]
        
        disk_qps = disk_row[disk_qps_col] if disk_qps_col else 0
        memory_qps = memory_row[memory_qps_col] if memory_qps_col else 0
        
        disk_latency = disk_row[disk_latency_col] if disk_latency_col else 0
        memory_latency = memory_row[memory_latency_col] if memory_latency_col else 0
        
        # 计算性能提升
        qps_improvement = 0
        if disk_qps > 0:
            qps_improvement = ((memory_qps - disk_qps) / disk_qps) * 100
        
        latency_improvement = 0
        if disk_latency > 0:
            latency_improvement = ((disk_latency - memory_latency) / disk_latency) * 100
        
        comparison_data.append({
            'Threads': thread,
            'Disk_QPS': disk_qps,
            'Memory_QPS': memory_qps,
            'QPS_Improvement_%': qps_improvement,
            'Disk_Latency_ms': disk_latency,
            'Memory_Latency_ms': memory_latency,
            'Latency_Improvement_%': latency_improvement
        })
    
    # 创建对比DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # 保存对比结果
    comparison_df.to_csv('performance_comparison.csv', index=False)
    print("✅ 对比结果已保存到: performance_comparison.csv")
    
    # 生成分析报告
    generate_analysis_report(comparison_df)
    
    return comparison_df

def generate_analysis_report(df):
    """生成分析报告"""
    report = []
    report.append("# Faiss 磁盘 vs 内存磁盘性能分析报告")
    report.append("")
    report.append(f"## 测试概览")
    report.append(f"- 测试线程数范围: {df['Threads'].min()} - {df['Threads'].max()}")
    report.append(f"- 平均QPS提升: {df['QPS_Improvement_%'].mean():.2f}%")
    report.append(f"- 平均延迟改善: {df['Latency_Improvement_%'].mean():.2f}%")
    report.append("")
    
    # 找出最佳性能配置
    best_disk_idx = df['Disk_QPS'].idxmax()
    best_memory_idx = df['Memory_QPS'].idxmax()
    
    report.append("## 最佳性能配置")
    report.append(f"### 普通磁盘最佳配置")
    report.append(f"- 线程数: {df.loc[best_disk_idx, 'Threads']}")
    report.append(f"- QPS: {df.loc[best_disk_idx, 'Disk_QPS']:.2f}")
    report.append(f"- 延迟: {df.loc[best_disk_idx, 'Disk_Latency_ms']:.4f} ms")
    report.append("")
    
    report.append(f"### 内存磁盘最佳配置")
    report.append(f"- 线程数: {df.loc[best_memory_idx, 'Threads']}")
    report.append(f"- QPS: {df.loc[best_memory_idx, 'Memory_QPS']:.2f}")
    report.append(f"- 延迟: {df.loc[best_memory_idx, 'Memory_Latency_ms']:.4f} ms")
    report.append("")
    
    # 瓶颈分析
    report.append("## 磁盘I/O瓶颈分析")
    
    # 计算高线程数下的性能差异
    high_threads = df[df['Threads'] >= df['Threads'].quantile(0.7)]
    if len(high_threads) > 0:
        avg_improvement_high = high_threads['QPS_Improvement_%'].mean()
        report.append(f"- 高线程数(≥{high_threads['Threads'].min()})平均QPS提升: {avg_improvement_high:.2f}%")
        
        if avg_improvement_high > 10:
            report.append("- **结论**: 存在明显的磁盘I/O瓶颈，内存磁盘在高线程数下表现显著更好")
        elif avg_improvement_high > 5:
            report.append("- **结论**: 存在轻微的磁盘I/O瓶颈，建议优化存储配置")
        else:
            report.append("- **结论**: 磁盘I/O不是主要瓶颈，性能差异较小")
    
    report.append("")
    
    # 性能对比表
    report.append("## 详细性能对比")
    report.append("")
    report.append("| 线程数 | 普通磁盘QPS | 内存磁盘QPS | QPS提升% | 普通磁盘延迟(ms) | 内存磁盘延迟(ms) | 延迟改善% |")
    report.append("|--------|-------------|-------------|----------|------------------|------------------|-----------|")
    
    for _, row in df.iterrows():
        report.append(f"| {int(row['Threads'])} | {row['Disk_QPS']:.2f} | {row['Memory_QPS']:.2f} | {row['QPS_Improvement_%']:.2f} | {row['Disk_Latency_ms']:.4f} | {row['Memory_Latency_ms']:.4f} | {row['Latency_Improvement_%']:.2f} |")
    
    report.append("")
    report.append("## 优化建议")
    
    max_improvement = df['QPS_Improvement_%'].max()
    if max_improvement > 20:
        report.append("- **强烈建议**: 升级到SSD或使用内存存储，性能提升超过20%")
    elif max_improvement > 10:
        report.append("- **建议**: 考虑使用更快的存储设备或增加内存缓存")
    else:
        report.append("- **当前配置**: 存储性能基本满足需求，可继续使用当前配置")
    
    # 找出最优线程数
    optimal_disk_threads = df.loc[df['Disk_QPS'].idxmax(), 'Threads']
    optimal_memory_threads = df.loc[df['Memory_QPS'].idxmax(), 'Threads']
    
    report.append(f"- **普通磁盘最优线程数**: {int(optimal_disk_threads)}")
    report.append(f"- **内存磁盘最优线程数**: {int(optimal_memory_threads)}")
    
    # 保存报告
    with open('detailed_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("✅ 详细分析报告已保存到: detailed_analysis_report.md")

def main():
    """主函数"""
    if len(sys.argv) != 3:
        print("使用方法: python3 analyze_results.py <disk_results.csv> <memory_results.csv>")
        sys.exit(1)
    
    disk_file = sys.argv[1]
    memory_file = sys.argv[2]
    
    if not os.path.exists(disk_file):
        print(f"错误: 文件 {disk_file} 不存在")
        sys.exit(1)
    
    if not os.path.exists(memory_file):
        print(f"错误: 文件 {memory_file} 不存在")
        sys.exit(1)
    
    # 执行分析
    comparison_df = analyze_performance(disk_file, memory_file)
    
    if comparison_df is not None:
        print("\n=== 分析完成 ===")
        print("生成的文件:")
        print("- performance_comparison.csv: 性能对比数据")
        print("- detailed_analysis_report.md: 详细分析报告")
        
        # 显示关键统计信息
        print(f"\n关键统计:")
        print(f"- 平均QPS提升: {comparison_df['QPS_Improvement_%'].mean():.2f}%")
        print(f"- 最大QPS提升: {comparison_df['QPS_Improvement_%'].max():.2f}%")
        print(f"- 平均延迟改善: {comparison_df['Latency_Improvement_%'].mean():.2f}%")

if __name__ == "__main__":
    main()
