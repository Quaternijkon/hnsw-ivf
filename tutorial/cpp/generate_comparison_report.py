#!/usr/bin/env python3

import csv
import sys
from datetime import datetime

def read_csv(filename):
    """读取CSV文件并返回数据"""
    data = []
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"❌ 错误: 文件 {filename} 不存在")
        return None
    return data

def main():
    print("=== 生成性能对比报告 ===\n")
    
    # 读取数据
    disk_data = read_csv('disk_results.csv')
    memory_data = read_csv('memory_results.csv')
    
    if disk_data is None or memory_data is None:
        sys.exit(1)
    
    # 生成Markdown报告
    report = []
    report.append("# Faiss 磁盘 vs 内存磁盘性能对比分析")
    report.append("")
    report.append("## 测试环境")
    report.append(f"- 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("- 系统: Linux 5.15.0-97-generic")
    report.append("- 内存: 219Gi")
    report.append("- CPU: 80 核心")
    report.append("")
    report.append("## 存储介质")
    report.append("- **普通磁盘**: 本地文件系统（NVMe SSD）")
    report.append("- **内存磁盘**: tmpfs (4GB)")
    report.append("")
    
    # QPS对比
    report.append("## 性能对比表")
    report.append("")
    report.append("### 1. QPS（每秒查询数）对比")
    report.append("")
    report.append("| 线程数 | 普通磁盘 QPS | 内存磁盘 QPS | 性能提升 | 提升百分比 |")
    report.append("|--------|--------------|--------------|----------|-----------|")
    
    improvements = []
    for disk, memory in zip(disk_data, memory_data):
        threads = disk['Threads']
        disk_qps = float(disk['QPS'])
        memory_qps = float(memory['QPS'])
        diff = memory_qps - disk_qps
        percent = (memory_qps / disk_qps - 1) * 100
        improvements.append(percent)
        report.append(f"| {threads:>6} | {disk_qps:>12.2f} | {memory_qps:>12.2f} | {diff:>+8.2f} | {percent:>+7.2f}% |")
    
    report.append("")
    
    # 延迟对比
    report.append("### 2. 延迟（毫秒）对比")
    report.append("")
    report.append("| 线程数 | 指标 | 普通磁盘 | 内存磁盘 | 改善 |")
    report.append("|--------|------|---------|---------|------|")
    
    for disk, memory in zip(disk_data, memory_data):
        threads = disk['Threads']
        
        # 平均延迟
        disk_avg = float(disk['Avg Lat(ms)'])
        memory_avg = float(memory['Avg Lat(ms)'])
        avg_diff = disk_avg - memory_avg
        report.append(f"| {threads:>6} | 平均 | {disk_avg:>7.4f} | {memory_avg:>7.4f} | {avg_diff:>+6.4f} |")
        
        # P99延迟
        disk_p99 = float(disk['P99 Lat(ms)'])
        memory_p99 = float(memory['P99 Lat(ms)'])
        p99_diff = disk_p99 - memory_p99
        report.append(f"| {threads:>6} | P99  | {disk_p99:>7.4f} | {memory_p99:>7.4f} | {p99_diff:>+6.4f} |")
    
    report.append("")
    
    # 功耗对比
    report.append("### 3. 功耗（瓦特）和能效对比")
    report.append("")
    report.append("| 线程数 | 普通磁盘功耗 | 内存磁盘功耗 | 普通磁盘能效 | 内存磁盘能效 |")
    report.append("|--------|-------------|-------------|-------------|-------------|")
    
    for disk, memory in zip(disk_data, memory_data):
        threads = disk['Threads']
        disk_power = float(disk['Avg Power(W)'])
        memory_power = float(memory['Avg Power(W)'])
        disk_eff = float(disk['Efficiency'])
        memory_eff = float(memory['Efficiency'])
        report.append(f"| {threads:>6} | {disk_power:>11.2f}W | {memory_power:>11.2f}W | {disk_eff:>11.2f} | {memory_eff:>11.2f} |")
    
    report.append("")
    report.append("*能效 = QPS/瓦特，越高越好*")
    report.append("")
    
    # 分析结论
    avg_improvement = sum(improvements) / len(improvements)
    max_improvement = max(improvements)
    max_idx = improvements.index(max_improvement)
    max_threads = disk_data[max_idx]['Threads']
    
    # 找出最优配置
    best_disk_idx = max(range(len(disk_data)), key=lambda i: float(disk_data[i]['QPS']))
    best_memory_idx = max(range(len(memory_data)), key=lambda i: float(memory_data[i]['QPS']))
    
    best_disk_threads = disk_data[best_disk_idx]['Threads']
    best_disk_qps = float(disk_data[best_disk_idx]['QPS'])
    best_memory_threads = memory_data[best_memory_idx]['Threads']
    best_memory_qps = float(memory_data[best_memory_idx]['QPS'])
    
    report.append("## 关键发现")
    report.append("")
    report.append("### 1. QPS性能分析")
    report.append(f"- 平均性能提升: **{avg_improvement:+.2f}%**")
    report.append(f"- 最大性能提升: **{max_threads} 线程时达到 {max_improvement:+.2f}%**")
    report.append("")
    
    report.append("### 2. 延迟分析")
    avg_disk_p99 = sum(float(d['P99 Lat(ms)']) for d in disk_data) / len(disk_data)
    avg_memory_p99 = sum(float(d['P99 Lat(ms)']) for d in memory_data) / len(memory_data)
    report.append(f"- 普通磁盘平均P99延迟: **{avg_disk_p99:.4f} ms**")
    report.append(f"- 内存磁盘平均P99延迟: **{avg_memory_p99:.4f} ms**")
    report.append(f"- 延迟改善: **{avg_disk_p99 - avg_memory_p99:+.4f} ms** ({(1 - avg_memory_p99/avg_disk_p99)*100:+.2f}%)")
    report.append("")
    
    report.append("### 3. 磁盘I/O瓶颈评估")
    if avg_improvement > 10:
        report.append(f"- ⚠️ **存在明显的磁盘I/O瓶颈**（平均性能提升 {avg_improvement:.2f}%）")
        report.append("- **建议**: 考虑使用更快的SSD或增加系统缓存")
    elif avg_improvement > 5:
        report.append(f"- ⚡ **存在轻微的磁盘I/O瓶颈**（平均性能提升 {avg_improvement:.2f}%）")
        report.append("- **建议**: 当前配置基本合理，可根据需求优化")
    else:
        report.append(f"- ✅ **磁盘I/O不是性能瓶颈**（平均性能提升仅 {avg_improvement:.2f}%）")
        report.append("- **建议**: 当前存储配置已经很好，无需额外优化")
    report.append("")
    
    report.append("### 4. 最优配置建议")
    report.append(f"- **普通磁盘最优配置**: {best_disk_threads} 线程 (QPS: {best_disk_qps:.2f})")
    report.append(f"- **内存磁盘最优配置**: {best_memory_threads} 线程 (QPS: {best_memory_qps:.2f})")
    overall_improvement = (best_memory_qps / best_disk_qps - 1) * 100
    report.append(f"- **最优配置下的性能提升**: {overall_improvement:+.2f}%")
    report.append("")
    
    # 能效分析
    report.append("### 5. 能效分析")
    best_disk_eff_idx = max(range(len(disk_data)), key=lambda i: float(disk_data[i]['Efficiency']))
    best_memory_eff_idx = max(range(len(memory_data)), key=lambda i: float(memory_data[i]['Efficiency']))
    
    best_disk_eff_threads = disk_data[best_disk_eff_idx]['Threads']
    best_disk_eff = float(disk_data[best_disk_eff_idx]['Efficiency'])
    best_memory_eff_threads = memory_data[best_memory_eff_idx]['Threads']
    best_memory_eff = float(memory_data[best_memory_eff_idx]['Efficiency'])
    
    report.append(f"- **普通磁盘最高能效**: {best_disk_eff_threads} 线程 (能效: {best_disk_eff:.2f} QPS/W)")
    report.append(f"- **内存磁盘最高能效**: {best_memory_eff_threads} 线程 (能效: {best_memory_eff:.2f} QPS/W)")
    report.append("")
    
    # 原始数据
    report.append("## 原始数据")
    report.append("")
    report.append("### 普通磁盘测试结果")
    report.append("```csv")
    with open('disk_results.csv', 'r') as f:
        report.append(f.read().strip())
    report.append("```")
    report.append("")
    report.append("### 内存磁盘测试结果")
    report.append("```csv")
    with open('memory_results.csv', 'r') as f:
        report.append(f.read().strip())
    report.append("```")
    
    # 写入文件
    output_file = 'performance_comparison_detailed.md'
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✅ 详细对比报告已生成: {output_file}")
    print(f"\n关键结果:")
    print(f"  - 平均性能提升: {avg_improvement:+.2f}%")
    print(f"  - 最优磁盘配置: {best_disk_threads} 线程 (QPS: {best_disk_qps:.2f})")
    print(f"  - 最优内存配置: {best_memory_threads} 线程 (QPS: {best_memory_qps:.2f})")

if __name__ == '__main__':
    main()

