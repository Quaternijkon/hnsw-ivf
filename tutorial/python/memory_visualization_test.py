#!/usr/bin/env python3
"""
内存可视化功能测试脚本
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import time
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def generate_memory_visualization():
    """生成内存使用情况可视化图表"""
    
    # 模拟内存数据
    timestamps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    phases = ['程序开始', '索引加载', '查询加载', '搜索开始', '搜索进行1', '搜索进行2', '搜索进行3', '搜索完成', '召回率计算', '程序结束']
    rss_values = [50, 60, 70, 80, 200, 300, 400, 500, 480, 460, 450]
    vms_values = [100, 200, 300, 400, 800, 1200, 1600, 2000, 1900, 1800, 1700]
    index_memory = [0, 0, 0, 0, 50, 100, 150, 200, 200, 200, 200]
    other_memory = [50, 60, 70, 80, 150, 200, 250, 300, 280, 260, 250]
    python_objects = [1000, 1200, 1400, 1600, 2000, 2500, 3000, 3500, 3400, 3300, 3200]
    
    # 创建图表
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Faiss内存使用情况分析', fontsize=16, fontweight='bold')
    
    # 子图1: 内存使用趋势
    ax1.plot(timestamps, rss_values, 'b-', linewidth=2, label='RSS内存', marker='o', markersize=4)
    ax1.plot(timestamps, vms_values, 'r--', linewidth=2, label='VMS内存', marker='s', markersize=4)
    ax1.fill_between(timestamps, rss_values, alpha=0.3, color='blue')
    ax1.fill_between(timestamps, vms_values, alpha=0.1, color='red')
    
    # 添加阶段分割线
    phase_boundaries = [(3, '搜索阶段'), (7, '评估阶段')]
    for x_pos, label in phase_boundaries:
        ax1.axvline(x=x_pos, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax1.text(x_pos, max(rss_values) * 0.9, label, rotation=90, 
                verticalalignment='top', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('内存使用 (MB)', fontsize=12)
    ax1.set_title('内存使用趋势', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 内存分解分析
    ax2.bar(timestamps, index_memory, width=0.3, label='索引内存', color='green', alpha=0.7)
    ax2.bar(timestamps, other_memory, width=0.3, bottom=index_memory, label='其他内存', color='orange', alpha=0.7)
    
    # 添加阶段分割线
    for x_pos, label in phase_boundaries:
        ax2.axvline(x=x_pos, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax2.text(x_pos, max(rss_values) * 0.9, label, rotation=90, 
                verticalalignment='top', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('内存分解 (MB)', fontsize=12)
    ax2.set_title('内存使用分解分析', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: Python对象数量
    ax3.plot(timestamps, python_objects, 'g-', linewidth=2, label='Python对象数', marker='^', markersize=4)
    ax3.fill_between(timestamps, python_objects, alpha=0.3, color='green')
    
    # 添加阶段分割线
    for x_pos, label in phase_boundaries:
        ax3.axvline(x=x_pos, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax3.text(x_pos, max(python_objects) * 0.9, label, rotation=90, 
                verticalalignment='top', fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('运行时间 (秒)', fontsize=12)
    ax3.set_ylabel('Python对象数量', fontsize=12)
    ax3.set_title('Python对象数量变化', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 添加阶段标签
    ax_phase = fig.add_axes([0.1, 0.95, 0.8, 0.03])
    ax_phase.set_xlim(0, 10)
    ax_phase.set_ylim(0, 1)
    ax_phase.axis('off')
    
    # 添加阶段条
    phases_info = [
        ('training', 0, 3, '训练/构建阶段'),
        ('search', 3, 7, '搜索阶段'),
        ('evaluation', 7, 10, '评估阶段')
    ]
    
    colors = {'training': 'lightblue', 'search': 'lightgreen', 'evaluation': 'lightcoral'}
    for phase_type, start_time, end_time, label in phases_info:
        width = end_time - start_time
        ax_phase.add_patch(Rectangle((start_time, 0), width, 1, 
                                   facecolor=colors.get(phase_type, 'lightgray'), 
                                   alpha=0.7, edgecolor='black'))
        ax_phase.text(start_time + width/2, 0.5, label, ha='center', va='center', 
                     fontweight='bold', fontsize=10)
    
    # 添加内存分析注释
    peak_rss = max(rss_values)
    peak_vms = max(vms_values)
    final_index_memory = index_memory[-1]
    final_other_memory = other_memory[-1]
    vms_rss_ratio = peak_vms / peak_rss
    
    analysis_text = f"""内存使用分析:
峰值RSS内存: {peak_rss:.1f} MB
峰值VMS内存: {peak_vms:.1f} MB
VMS/RSS比例: {vms_rss_ratio:.2f}
索引内存占比: {final_index_memory/(final_index_memory+final_other_memory)*100:.1f}%
其他内存占比: {final_other_memory/(final_index_memory+final_other_memory)*100:.1f}%"""
    
    fig.text(0.02, 0.02, analysis_text, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig('memory_usage_plot_test.png', dpi=300, bbox_inches='tight')
    print("内存使用情况图表已保存到: memory_usage_plot_test.png")
    plt.close()

if __name__ == "__main__":
    generate_memory_visualization()
