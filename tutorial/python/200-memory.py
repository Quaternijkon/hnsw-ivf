import numpy as np
import faiss
import time
import os
import platform
import resource
import struct
import re
import psutil
import tracemalloc
import gc
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np

# ==============================================================================
# 0. 路径和文件名配置 & 调试开关
# ==============================================================================
DATA_DIR = "./sift"
LEARN_FILE = os.path.join(DATA_DIR, "learn.fbin")
BASE_FILE = os.path.join(DATA_DIR, "base.fbin")
QUERY_FILE = os.path.join(DATA_DIR, "query.fbin")
GROUNDTRUTH_FILE = os.path.join(DATA_DIR, "groundtruth.ivecs")

# 调试开关
ENABLE_IVF_STATS = False  # 控制是否输出IVF分区统计信息

# 新增开关 - 控制是否统计搜索分区信息
ENABLE_SEARCH_PARTITION_STATS = False
SEARCH_STATS_FILENAME = os.path.join(DATA_DIR, "search_partition_ratios.txt")

# 内存监控开关
ENABLE_DETAILED_MEMORY_MONITORING = True
MEMORY_LOG_FILENAME = os.path.join(DATA_DIR, "memory_usage_log.txt")

# 内存优化配置
MEMORY_OPTIMIZATION_CONFIG = {
    'enable_gc_before_search': True,  # 搜索前进行垃圾回收
    'enable_gc_after_search': True,  # 搜索后进行垃圾回收
    'gc_threshold_mb': 100,  # 内存增长超过此值时触发GC
    'max_memory_mb': 1000,  # 最大内存使用限制
    'enable_memory_compression': False,  # 启用内存压缩（实验性）
    'chunk_size_optimization': True,  # 启用分块大小优化
}

# 内存可视化配置
ENABLE_MEMORY_VISUALIZATION = True
MEMORY_PLOT_FILENAME = os.path.join(DATA_DIR, "memory_usage_plot.png")

# 运行结束时内存图表配置
ENABLE_FINAL_MEMORY_PLOT = True  # 控制是否在运行结束时生成内存使用情况图表
FINAL_MEMORY_PLOT_FILENAME = os.path.join(DATA_DIR, "final_memory_usage_plot.png")


# ==============================================================================
# 1. 高级内存监控类
# ==============================================================================
class AdvancedMemoryMonitor:
    """高级内存监控类，提供多种内存监控功能"""

    def __init__(self, enable_tracemalloc: bool = True, log_file: Optional[str] = None):
        self.enable_tracemalloc = enable_tracemalloc
        self.log_file = log_file
        self.memory_snapshots: List[Dict] = []
        self.process = psutil.Process()
        self.index_memory_baseline = None  # 索引加载后的内存基线
        self.search_memory_breakdown = []  # 搜索期间内存分解

        if self.enable_tracemalloc:
            tracemalloc.start()

        if self.log_file:
            with open(self.log_file, 'w') as f:
                f.write("时间戳,阶段,RSS_MB,VMS_MB,内存百分比,Python对象数,垃圾回收次数,索引内存_MB,其他内存_MB\n")

    def get_memory_info(self) -> Dict:
        """获取详细的内存信息"""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()

        # 获取Python对象统计
        gc_stats = gc.get_stats()
        total_objects = sum(stat['collected'] for stat in gc_stats)

        info = {
            'timestamp': time.time(),
            'rss_mb': memory_info.rss / (1024 * 1024),  # 实际物理内存
            'vms_mb': memory_info.vms / (1024 * 1024),  # 虚拟内存
            'memory_percent': memory_percent,
            'python_objects': len(gc.get_objects()),
            'gc_collections': total_objects
        }

        # 如果启用了tracemalloc，添加更详细的信息
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            info.update({
                'traced_current_mb': current / (1024 * 1024),
                'traced_peak_mb': peak / (1024 * 1024)
            })

        # 计算索引内存和其他内存的分解
        if self.index_memory_baseline is not None:
            index_memory = self.estimate_index_memory()
            other_memory = info['rss_mb'] - index_memory
            info.update({
                'index_memory_mb': index_memory,
                'other_memory_mb': other_memory
            })
        else:
            info.update({
                'index_memory_mb': 0,
                'other_memory_mb': info['rss_mb']
            })

        return info

    def estimate_index_memory(self) -> float:
        """估算索引占用的内存"""
        if self.index_memory_baseline is None:
            return 0

        # 基于索引大小估算内存使用
        # 这是一个简化的估算，实际可能需要更复杂的计算
        current_rss = self.process.memory_info().rss / (1024 * 1024)
        return min(current_rss - self.index_memory_baseline, current_rss * 0.8)

    def set_index_memory_baseline(self, baseline_mb: float):
        """设置索引内存基线"""
        self.index_memory_baseline = baseline_mb
        print(f"[内存监控] 设置索引内存基线: {baseline_mb:.2f} MB")

    def analyze_memory_growth_pattern(self) -> Dict:
        """分析内存增长模式"""
        if len(self.memory_snapshots) < 2:
            return {}

        analysis = {
            'total_growth_mb': 0,
            'growth_phases': [],
            'peak_usage_mb': 0,
            'memory_efficiency': 0
        }

        # 计算总增长
        first_snapshot = self.memory_snapshots[0]
        last_snapshot = self.memory_snapshots[-1]
        analysis['total_growth_mb'] = last_snapshot['rss_mb'] - first_snapshot['rss_mb']
        analysis['peak_usage_mb'] = max(s['rss_mb'] for s in self.memory_snapshots)

        # 分析各阶段增长
        for i in range(1, len(self.memory_snapshots)):
            prev = self.memory_snapshots[i - 1]
            curr = self.memory_snapshots[i]
            growth = curr['rss_mb'] - prev['rss_mb']
            analysis['growth_phases'].append({
                'phase': curr.get('phase', f'阶段{i}'),
                'growth_mb': growth,
                'rss_mb': curr['rss_mb']
            })

        return analysis

    def log_memory_snapshot(self, phase: str):
        """记录内存快照"""
        info = self.get_memory_info()
        info['phase'] = phase
        self.memory_snapshots.append(info)

        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{info['timestamp']:.2f},{phase},{info['rss_mb']:.2f},"
                        f"{info['vms_mb']:.2f},{info['memory_percent']:.2f},"
                        f"{info['python_objects']},{info['gc_collections']},"
                        f"{info['index_memory_mb']:.2f},{info['other_memory_mb']:.2f}\n")

        print(f"[内存监控] {phase}: RSS={info['rss_mb']:.2f}MB, "
              f"VMS={info['vms_mb']:.2f}MB, 对象数={info['python_objects']}, "
              f"索引内存={info['index_memory_mb']:.2f}MB, 其他内存={info['other_memory_mb']:.2f}MB")

    @contextmanager
    def monitor_phase(self, phase_name: str):
        """上下文管理器，用于监控特定代码段的内存使用"""
        print(f"\n[内存监控] 开始阶段: {phase_name}")
        self.log_memory_snapshot(f"{phase_name}_开始")

        start_info = self.get_memory_info()
        start_time = time.time()

        try:
            yield
        finally:
            end_time = time.time()
            end_info = self.get_memory_info()

            # 计算内存变化
            rss_diff = end_info['rss_mb'] - start_info['rss_mb']
            vms_diff = end_info['vms_mb'] - start_info['vms_mb']
            objects_diff = end_info['python_objects'] - start_info['python_objects']

            print(f"[内存监控] 结束阶段: {phase_name}")
            print(f"  耗时: {end_time - start_time:.2f}秒")
            print(f"  RSS变化: {rss_diff:+.2f}MB")
            print(f"  VMS变化: {vms_diff:+.2f}MB")
            print(f"  Python对象变化: {objects_diff:+d}")

            self.log_memory_snapshot(f"{phase_name}_结束")

    def get_memory_summary(self) -> Dict:
        """获取内存使用摘要"""
        if not self.memory_snapshots:
            return {}

        rss_values = [s['rss_mb'] for s in self.memory_snapshots]
        vms_values = [s['vms_mb'] for s in self.memory_snapshots]

        return {
            'peak_rss_mb': max(rss_values),
            'peak_vms_mb': max(vms_values),
            'avg_rss_mb': sum(rss_values) / len(rss_values),
            'avg_vms_mb': sum(vms_values) / len(vms_values),
            'total_snapshots': len(self.memory_snapshots)
        }

    def force_gc_and_log(self, phase: str):
        """强制垃圾回收并记录内存变化"""
        before_info = self.get_memory_info()
        gc.collect()
        after_info = self.get_memory_info()

        rss_freed = before_info['rss_mb'] - after_info['rss_mb']
        objects_freed = before_info['python_objects'] - after_info['python_objects']

        print(f"[垃圾回收] {phase}: 释放RSS={rss_freed:.2f}MB, 对象={objects_freed}")
        self.log_memory_snapshot(f"{phase}_GC后")

    def get_tracemalloc_top_stats(self, limit: int = 10):
        """获取tracemalloc统计信息（如果启用）"""
        if not self.enable_tracemalloc or not tracemalloc.is_tracing():
            return None

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        print(f"\n[内存分析] Top {limit} 内存使用最多的代码行:")
        for index, stat in enumerate(top_stats[:limit], 1):
            print(f"{index}. {stat}")

        return top_stats[:limit]

    def get_memory_optimization_suggestions(self) -> List[str]:
        """获取内存优化建议"""
        suggestions = []

        if not self.memory_snapshots:
            return suggestions

        # 分析内存增长模式
        analysis = self.analyze_memory_growth_pattern()

        if analysis.get('total_growth_mb', 0) > 500:  # 如果总增长超过500MB
            suggestions.append("⚠️  内存增长较大，建议在搜索前进行垃圾回收")

        # 检查VMS/RSS比例
        last_snapshot = self.memory_snapshots[-1]
        vms_rss_ratio = last_snapshot['vms_mb'] / last_snapshot['rss_mb']
        if vms_rss_ratio > 10:  # VMS比RSS大10倍以上
            suggestions.append("⚠️  VMS/RSS比例过高，可能存在内存碎片化")

        # 检查Python对象数量
        if last_snapshot['python_objects'] > 100000:
            suggestions.append("⚠️  Python对象数量较多，建议检查是否有对象泄漏")

        # 检查索引内存占比
        if last_snapshot.get('index_memory_mb', 0) > 0:
            index_ratio = last_snapshot['index_memory_mb'] / last_snapshot['rss_mb']
            if index_ratio < 0.3:  # 索引内存占比小于30%
                suggestions.append("💡 索引内存占比较低，其他内存使用可能过多")
            elif index_ratio > 0.8:  # 索引内存占比大于80%
                suggestions.append("💡 索引内存占比很高，这是正常的")

        return suggestions

    def generate_final_memory_plot(self, plot_filename: str):
        """生成最终的内存使用情况图表（仅RSS内存，按阶段分布）"""
        if not self.memory_snapshots or len(self.memory_snapshots) < 2:
            print("内存快照数据不足，无法生成最终内存图表")
            return

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建单个图表，专注于RSS内存使用
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        fig.suptitle('Faiss程序运行内存使用情况分析', fontsize=18, fontweight='bold')

        # 准备数据
        timestamps = [s['timestamp'] for s in self.memory_snapshots]
        phases = [s.get('phase', 'Unknown') for s in self.memory_snapshots]
        rss_values = [s['rss_mb'] for s in self.memory_snapshots]
        index_memory = [s.get('index_memory_mb', 0) for s in self.memory_snapshots]
        other_memory = [s.get('other_memory_mb', 0) for s in self.memory_snapshots]

        # 计算相对时间（以秒为单位）
        start_time = timestamps[0]
        relative_times = [(t - start_time) for t in timestamps]

        # 识别主要阶段
        phase_boundaries = self._identify_main_phases(phases, relative_times)
        
        # 绘制RSS内存曲线
        ax.plot(relative_times, rss_values, 'b-', linewidth=3, label='RSS物理内存', marker='o', markersize=6)
        ax.fill_between(relative_times, rss_values, alpha=0.3, color='blue')

        # 绘制内存分解的堆叠区域图
        ax.fill_between(relative_times, 0, index_memory, alpha=0.6, color='green', label='索引内存')
        ax.fill_between(relative_times, index_memory, [idx + other for idx, other in zip(index_memory, other_memory)], 
                       alpha=0.6, color='orange', label='其他内存')

        # 添加阶段分隔线和颜色背景
        self._add_phase_backgrounds(ax, phase_boundaries, relative_times, max(rss_values))
        
        # 添加内存增长点标注
        self._add_memory_growth_annotations(ax, relative_times, rss_values, phases)

        # 设置图表属性
        ax.set_xlabel('运行时间 (秒)', fontsize=14)
        ax.set_ylabel('内存使用量 (MB)', fontsize=14)
        ax.set_title('内存使用时间线 - 按运行阶段分析', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, alpha=0.3)

        # 添加详细的内存分析文本框
        self._add_detailed_memory_analysis(fig, rss_values, index_memory, other_memory, phase_boundaries, relative_times)

        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"最终内存使用情况图表已保存到: {plot_filename}")
        plt.close()

    def _identify_main_phases(self, phases, relative_times):
        """识别主要的运行阶段（训练、构建、搜索）"""
        boundaries = []
        current_phase = None
        
        for i, phase in enumerate(phases):
            phase_lower = phase.lower()
            
            # 训练阶段
            if ('训练' in phase or 'train' in phase_lower or '量化器' in phase) and current_phase != 'training':
                boundaries.append(('training', i, relative_times[i] if i < len(relative_times) else 0, '训练阶段'))
                current_phase = 'training'
            
            # 构建阶段（包括添加数据）
            elif ('构建' in phase or '添加' in phase or 'build' in phase_lower or 'add' in phase_lower) and current_phase != 'building':
                boundaries.append(('building', i, relative_times[i] if i < len(relative_times) else 0, '构建阶段'))
                current_phase = 'building'
            
            # 搜索阶段
            elif ('搜索' in phase or 'search' in phase_lower or '查询' in phase) and current_phase != 'searching':
                boundaries.append(('searching', i, relative_times[i] if i < len(relative_times) else 0, '搜索阶段'))
                current_phase = 'searching'
                
            # 评估阶段
            elif ('召回' in phase or '计算' in phase or 'recall' in phase_lower or 'evaluation' in phase_lower) and current_phase != 'evaluation':
                boundaries.append(('evaluation', i, relative_times[i] if i < len(relative_times) else 0, '评估阶段'))
                current_phase = 'evaluation'
        
        return boundaries

    def _add_phase_backgrounds(self, ax, phase_boundaries, relative_times, max_value):
        """添加阶段背景色和分隔线"""
        if not phase_boundaries:
            return
            
        # 定义阶段颜色
        phase_colors = {
            'training': 'lightblue',
            'building': 'lightgreen', 
            'searching': 'lightyellow',
            'evaluation': 'lightcoral'
        }
        
        # 添加背景色
        for i, (phase_type, idx, time_point, label) in enumerate(phase_boundaries):
            # 计算阶段的时间范围
            start_time = time_point
            if i + 1 < len(phase_boundaries):
                end_time = phase_boundaries[i + 1][2]
            else:
                end_time = relative_times[-1] if relative_times else start_time + 1
                
            # 添加背景色
            ax.axvspan(start_time, end_time, alpha=0.2, color=phase_colors.get(phase_type, 'lightgray'))
            
            # 添加分隔线
            ax.axvline(x=start_time, color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            # 添加阶段标签
            label_x = start_time + (end_time - start_time) / 2
            ax.text(label_x, max_value * 0.95, label, ha='center', va='top', 
                   fontsize=12, fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=phase_colors.get(phase_type, 'lightgray'), alpha=0.8))

    def _add_memory_growth_annotations(self, ax, relative_times, rss_values, phases):
        """添加内存增长关键点的标注"""
        if len(rss_values) < 2:
            return
            
        # 找到内存显著增长的点
        growth_points = []
        for i in range(1, len(rss_values)):
            growth = rss_values[i] - rss_values[i-1]
            if growth > 50:  # 增长超过50MB的点
                growth_points.append((i, growth, relative_times[i], rss_values[i], phases[i]))
        
        # 标注显著增长点
        for idx, growth, time_point, memory_value, phase in growth_points[:5]:  # 最多显示5个点
            ax.annotate(f'+{growth:.0f}MB\n{phase}', 
                       xy=(time_point, memory_value), 
                       xytext=(time_point + max(relative_times) * 0.05, memory_value + max(rss_values) * 0.05),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                       fontsize=10, ha='left', va='bottom',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))

    def _add_detailed_memory_analysis(self, fig, rss_values, index_memory, other_memory, phase_boundaries, relative_times):
        """添加详细的内存分析信息"""
        if not rss_values:
            return
            
        # 计算关键指标
        peak_rss = max(rss_values)
        final_rss = rss_values[-1]
        total_growth = final_rss - rss_values[0]
        final_index_memory = index_memory[-1] if index_memory else 0
        final_other_memory = other_memory[-1] if other_memory else 0
        
        # 计算各阶段的内存使用
        phase_memory_info = []
        for i, (phase_type, idx, time_point, label) in enumerate(phase_boundaries):
            if idx < len(rss_values):
                phase_memory = rss_values[idx]
                phase_memory_info.append(f"{label}: {phase_memory:.1f}MB")
        
        # 创建分析文本
        analysis_text = f"""内存使用详细分析:

峰值内存: {peak_rss:.1f} MB
总内存增长: {total_growth:.1f} MB
最终内存: {final_rss:.1f} MB

内存构成:
• 索引内存: {final_index_memory:.1f} MB ({final_index_memory/final_rss*100:.1f}%)
• 其他内存: {final_other_memory:.1f} MB ({final_other_memory/final_rss*100:.1f}%)

各阶段内存:
{chr(10).join(phase_memory_info)}

内存效率分析:
• 内存利用率: {'良好' if final_index_memory/final_rss > 0.5 else '可优化'}
• 内存增长: {'平稳' if total_growth < peak_rss * 0.5 else '较大'}"""

        # 添加分析文本框
        fig.text(0.02, 0.02, analysis_text, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
                verticalalignment='bottom', horizontalalignment='left')

    def generate_memory_visualization(self, plot_filename: str):
        """生成内存使用情况可视化图表"""
        if not self.memory_snapshots or len(self.memory_snapshots) < 2:
            print("内存快照数据不足，无法生成可视化图表")
            return

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'sans-serif'] # 使用新安装的字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Faiss内存使用情况分析', fontsize=16, fontweight='bold')

        # 准备数据``
        timestamps = [s['timestamp'] for s in self.memory_snapshots]
        phases = [s.get('phase', 'Unknown') for s in self.memory_snapshots]
        rss_values = [s['rss_mb'] for s in self.memory_snapshots]
        vms_values = [s['vms_mb'] for s in self.memory_snapshots]
        index_memory = [s.get('index_memory_mb', 0) for s in self.memory_snapshots]
        other_memory = [s.get('other_memory_mb', 0) for s in self.memory_snapshots]
        python_objects = [s['python_objects'] for s in self.memory_snapshots]

        # 计算相对时间（以秒为单位）
        start_time = timestamps[0]
        relative_times = [(t - start_time) for t in timestamps]

        # 识别阶段
        phase_boundaries = self._identify_phase_boundaries(phases)

        # 子图1: 内存使用趋势
        ax1.plot(relative_times, rss_values, 'b-', linewidth=2, label='RSS内存', marker='o', markersize=4)
        ax1.plot(relative_times, vms_values, 'r--', linewidth=2, label='VMS内存', marker='s', markersize=4)
        ax1.fill_between(relative_times, rss_values, alpha=0.3, color='blue')
        ax1.fill_between(relative_times, vms_values, alpha=0.1, color='red')

        # 添加阶段分割线
        self._add_phase_boundaries(ax1, phase_boundaries, relative_times, rss_values)

        ax1.set_ylabel('内存使用 (MB)', fontsize=12)
        ax1.set_title('内存使用趋势', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: 内存分解分析
        ax2.bar(relative_times, index_memory, width=0.1, label='索引内存', color='green', alpha=0.7)
        ax2.bar(relative_times, other_memory, width=0.1, bottom=index_memory, label='其他内存', color='orange', alpha=0.7)

        # 添加阶段分割线
        self._add_phase_boundaries(ax2, phase_boundaries, relative_times, rss_values)

        ax2.set_ylabel('内存分解 (MB)', fontsize=12)
        ax2.set_title('内存使用分解分析', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 子图3: Python对象数量
        ax3.plot(relative_times, python_objects, 'g-', linewidth=2, label='Python对象数', marker='^', markersize=4)
        ax3.fill_between(relative_times, python_objects, alpha=0.3, color='green')

        # 添加阶段分割线
        self._add_phase_boundaries(ax3, phase_boundaries, relative_times, python_objects)

        ax3.set_xlabel('运行时间 (秒)', fontsize=12)
        ax3.set_ylabel('Python对象数量', fontsize=12)
        ax3.set_title('Python对象数量变化', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 添加阶段标签
        self._add_phase_labels(fig, phase_boundaries, relative_times)

        # 添加内存分析注释
        self._add_memory_analysis_annotations(fig, rss_values, vms_values, index_memory, other_memory)

        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"内存使用情况图表已保存到: {plot_filename}")
        plt.close()

    def _identify_phase_boundaries(self, phases):
        """识别阶段边界"""
        boundaries = []
        current_phase = None

        for i, phase in enumerate(phases):
            if '训练' in phase or '构建' in phase:
                if current_phase != 'training':
                    boundaries.append(('training', i, '训练/构建阶段'))
                    current_phase = 'training'
            elif '搜索' in phase:
                if current_phase != 'search':
                    boundaries.append(('search', i, '搜索阶段'))
                    current_phase = 'search'
            elif '召回率' in phase or '计算' in phase:
                if current_phase != 'evaluation':
                    boundaries.append(('evaluation', i, '评估阶段'))
                    current_phase = 'evaluation'

        return boundaries

    def _add_phase_boundaries(self, ax, phase_boundaries, relative_times, values):
        """添加阶段分割线"""
        for phase_type, idx, label in phase_boundaries:
            if idx < len(relative_times):
                x_pos = relative_times[idx]
                y_max = max(values) if values else 0
                ax.axvline(x=x_pos, color='red', linestyle='--', alpha=0.7, linewidth=2)
                ax.text(x_pos, y_max * 0.9, label, rotation=90,
                        verticalalignment='top', fontsize=10, fontweight='bold')

    def _add_phase_labels(self, fig, phase_boundaries, relative_times):
        """添加阶段标签"""
        if not phase_boundaries:
            return

        # 计算阶段时间范围
        phases_info = []
        for i, (phase_type, idx, label) in enumerate(phase_boundaries):
            start_time = relative_times[idx] if idx < len(relative_times) else 0
            end_time = relative_times[phase_boundaries[i + 1][1]] if i + 1 < len(phase_boundaries) else relative_times[-1]
            phases_info.append((phase_type, start_time, end_time, label))

        # 在图表顶部添加阶段条
        ax_phase = fig.add_axes([0.1, 0.95, 0.8, 0.03])
        ax_phase.set_xlim(0, relative_times[-1] if relative_times else 1)
        ax_phase.set_ylim(0, 1)
        ax_phase.axis('off')

        colors = {'training': 'lightblue', 'search': 'lightgreen', 'evaluation': 'lightcoral'}
        for phase_type, start_time, end_time, label in phases_info:
            width = end_time - start_time
            ax_phase.add_patch(Rectangle((start_time, 0), width, 1,
                                         facecolor=colors.get(phase_type, 'lightgray'),
                                         alpha=0.7, edgecolor='black'))
            ax_phase.text(start_time + width / 2, 0.5, label, ha='center', va='center',
                          fontweight='bold', fontsize=10)

    def _add_memory_analysis_annotations(self, fig, rss_values, vms_values, index_memory, other_memory):
        """添加内存分析注释"""
        if not rss_values or not vms_values:
            return

        # 计算关键指标
        peak_rss = max(rss_values)
        peak_vms = max(vms_values)
        final_index_memory = index_memory[-1] if index_memory else 0
        final_other_memory = other_memory[-1] if other_memory else 0
        vms_rss_ratio = peak_vms / peak_rss if peak_rss > 0 else 0

        # 添加分析文本框
        analysis_text = f"""内存使用分析:
峰值RSS内存: {peak_rss:.1f} MB
峰值VMS内存: {peak_vms:.1f} MB
VMS/RSS比例: {vms_rss_ratio:.2f}
索引内存占比: {final_index_memory/(final_index_memory+final_other_memory)*100:.1f}%
其他内存占比: {final_other_memory/(final_index_memory+final_other_memory)*100:.1f}%"""

        fig.text(0.02, 0.02, analysis_text, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
                 verticalalignment='bottom')


# 创建全局内存监控器
memory_monitor = AdvancedMemoryMonitor(
    enable_tracemalloc=ENABLE_DETAILED_MEMORY_MONITORING,
    log_file=MEMORY_LOG_FILENAME if ENABLE_DETAILED_MEMORY_MONITORING else None
)


# ==============================================================================
# 2. 辅助函数：读取.fbin文件
# ==============================================================================
def read_fbin(filename, start_idx=0, chunk_size=None):
    """
    读取.fbin格式的文件
    格式: [nvecs: int32, dim: int32, data: float32[nvecs*dim]]
    """
    with open(filename, 'rb') as f:
        nvecs = struct.unpack('i', f.read(4))[0]
        dim = struct.unpack('i', f.read(4))[0]

        if chunk_size is None:
            # 读取整个文件
            data = np.fromfile(f, dtype=np.float32, count=nvecs * dim)
            data = data.reshape(nvecs, dim)
            return data
        else:
            # 读取指定块
            end_idx = min(start_idx + chunk_size, nvecs)
            num_vectors_in_chunk = end_idx - start_idx
            offset = start_idx * dim * 4  # 每个float32占4字节
            f.seek(offset, os.SEEK_CUR)
            data = np.fromfile(f, dtype=np.float32, count=num_vectors_in_chunk * dim)
            data = data.reshape(num_vectors_in_chunk, dim)
            return data, nvecs, dim


def read_ivecs(filename):
    """
    (此函数在此脚本中未用于批量读取，仅作为工具函数保留)
    读取.ivecs格式的二进制文件 (例如SIFT1M的groundtruth)
    格式: 向量循环 [dim: int32, data: int32[dim]]
    """
    a = np.fromfile(filename, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


# ==============================================================================
# 2. 设置参数与环境
# ==============================================================================

# 从训练文件中获取维度信息
_, nt, d_train = read_fbin(LEARN_FILE, chunk_size=1)  # 只读取元数据

# 获取数据集大小信息
_, nb, d_base = read_fbin(BASE_FILE, chunk_size=1)
_, nq, d_query = read_fbin(QUERY_FILE, chunk_size=1)

# 验证维度一致性
if d_train != d_base or d_train != d_query:
    raise ValueError(f"维度不一致: 训练集{d_train}维, 基础集{d_base}维, 查询集{d_query}维")

# 设置其他参数
cell_size = 256
nlist = nb // cell_size
nprobe = 32
chunk_size = 100000  # 每次处理的数据块大小
k = 10  # 查找最近的10个邻居

M = 32  # HNSW的连接数
efconstruction = 40  # 默认40
efsearch = 16  # 默认16

# ==============================================================================
# 【重构点】: 在索引文件名中同时体现 M 和 efConstruction 的值
# ==============================================================================
base_name = os.path.splitext(os.path.basename(BASE_FILE))[0]
# 清理文件名中的特殊字符
clean_base_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name)
# 在文件名中添加 M 和 efc 参数，以区分不同参数构建的索引
INDEX_FILE = os.path.join(DATA_DIR,
                          f"{clean_base_name}_d{d_train}_nlist{nlist}_HNSWM{M}_efc{efconstruction}_IVFFlat.index")
# ==============================================================================

print("=" * 60)
print("Phase 0: 环境设置")
print(f"向量维度 (d): {d_train}")
print(f"基础集大小 (nb): {nb}, 训练集大小 (ntrain): {nt}")
print(f"查询集大小 (nq): {nq}, 分块大小 (chunk_size): {chunk_size}")
print(f"HNSW M (构建参数): {M}")
print(f"HNSW efConstruction (构建参数): {efconstruction}")
print(f"索引将保存在磁盘文件: {INDEX_FILE}")
print(f"IVF统计功能: {'启用' if ENABLE_IVF_STATS else '禁用'}")
print(f"搜索分区统计功能: {'启用' if ENABLE_SEARCH_PARTITION_STATS else '禁用'}")
print(f"详细内存监控: {'启用' if ENABLE_DETAILED_MEMORY_MONITORING else '禁用'}")
print("=" * 60)

# 记录初始内存状态
if ENABLE_DETAILED_MEMORY_MONITORING:
    memory_monitor.log_memory_snapshot("程序开始")

# ==============================================================================
# 3. 检查索引文件是否存在
# ==============================================================================
if os.path.exists(INDEX_FILE):
    print(f"索引文件 {INDEX_FILE} 已存在，跳过索引构建阶段")
    skip_index_building = True
else:
    print("索引文件不存在，将构建新索引")
    skip_index_building = False

# ==============================================================================
# 4. 训练量化器
# ==============================================================================
if not skip_index_building:
    if ENABLE_DETAILED_MEMORY_MONITORING:
        with memory_monitor.monitor_phase("训练HNSW粗量化器"):
            print("\nPhase 1: 训练 HNSW 粗量化器 (in-memory)")
            coarse_quantizer = faiss.IndexHNSWFlat(d_train, M, faiss.METRIC_L2)
            coarse_quantizer.hnsw.efConstruction = efconstruction
            coarse_quantizer.hnsw.efSearch = efsearch
            print(f"efconstruction: {coarse_quantizer.hnsw.efConstruction}, efSearch: {coarse_quantizer.hnsw.efSearch}")
            # coarse_quantizer = faiss.IndexFlatL2(d_train)
            index_for_training = faiss.IndexIVFFlat(coarse_quantizer, d_train, nlist, faiss.METRIC_L2)
            index_for_training.verbose = True

            xt = read_fbin(LEARN_FILE)

            print("训练聚类中心并构建 HNSW 量化器...")
            start_time = time.time()
            index_for_training.train(xt)
            end_time = time.time()

            print(f"量化器训练完成，耗时: {end_time - start_time:.2f} 秒")
            print(f"粗量化器中的质心数量: {coarse_quantizer.ntotal}")
            del xt
            del index_for_training

            # 强制垃圾回收
            memory_monitor.force_gc_and_log("训练完成后")
    else:
        print("\nPhase 1: 训练 HNSW 粗量化器 (in-memory)")
        coarse_quantizer = faiss.IndexHNSWFlat(d_train, M, faiss.METRIC_L2)
        coarse_quantizer.hnsw.efConstruction = efconstruction
        coarse_quantizer.hnsw.efSearch = efsearch
        print(f"efconstruction: {coarse_quantizer.hnsw.efConstruction}, efSearch: {coarse_quantizer.hnsw.efSearch}")
        # coarse_quantizer = faiss.IndexFlatL2(d_train)
        index_for_training = faiss.IndexIVFFlat(coarse_quantizer, d_train, nlist, faiss.METRIC_L2)
        index_for_training.verbose = True

        xt = read_fbin(LEARN_FILE)

        print("训练聚类中心并构建 HNSW 量化器...")
        start_time = time.time()
        index_for_training.train(xt)
        end_time = time.time()

        print(f"量化器训练完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"粗量化器中的质心数量: {coarse_quantizer.ntotal}")
        del xt
        del index_for_training

    # ==============================================================================
    # 5. 创建一个空的、基于磁盘的索引框架
    # ==============================================================================
    print("\nPhase 2: 创建空的磁盘索引框架")
    index_shell = faiss.IndexIVFFlat(coarse_quantizer, d_train, nlist, faiss.METRIC_L2)
    print(f"将空的索引框架写入磁盘: {INDEX_FILE}")
    faiss.write_index(index_shell, INDEX_FILE)
    del index_shell

    # ==============================================================================
    # 6. 分块向磁盘索引中添加数据 (从base.fbin)
    # ==============================================================================
    if ENABLE_DETAILED_MEMORY_MONITORING:
        with memory_monitor.monitor_phase("分块添加数据到磁盘索引"):
            print("\nPhase 3: 分块添加数据到磁盘索引")

            # 兼容不同Faiss版本的IO标志处理
            try:
                IO_FLAG_READ_WRITE = faiss.IO_FLAG_READ_WRITE
            except AttributeError:
                try:
                    IO_FLAG_READ_WRITE = faiss.index_io.IO_FLAG_READ_WRITE
                except AttributeError:
                    IO_FLAG_READ_WRITE = 0

            print(f"使用IO标志: {IO_FLAG_READ_WRITE} (读写模式)")

            index_ondisk = faiss.read_index(INDEX_FILE, IO_FLAG_READ_WRITE)
            start_time = time.time()

            num_chunks = (nb + chunk_size - 1) // chunk_size
            for i in range(0, nb, chunk_size):
                chunk_idx = i // chunk_size + 1
                print(f"       -> 正在处理块 {chunk_idx}/{num_chunks}: 向量 {i} 到 {min(i + chunk_size, nb) - 1}")

                # 从base.fbin中读取数据块
                xb_chunk, _, _ = read_fbin(BASE_FILE, i, chunk_size)

                index_ondisk.add(xb_chunk)
                del xb_chunk

                # 每处理10个块进行一次内存监控
                if chunk_idx % 10 == 0:
                    memory_monitor.log_memory_snapshot(f"处理块{chunk_idx}")

            print(f"\n所有数据块添加完成，总耗时: {time.time() - start_time:.2f} 秒")
            print(f"磁盘索引中的向量总数 (ntotal): {index_ondisk.ntotal}")

            # 强制垃圾回收
            memory_monitor.force_gc_and_log("分块添加完成后")
    else:
        print("\nPhase 3: 分块添加数据到磁盘索引")

        # 兼容不同Faiss版本的IO标志处理
        try:
            IO_FLAG_READ_WRITE = faiss.IO_FLAG_READ_WRITE
        except AttributeError:
            try:
                IO_FLAG_READ_WRITE = faiss.index_io.IO_FLAG_READ_WRITE
            except AttributeError:
                IO_FLAG_READ_WRITE = 0

        print(f"使用IO标志: {IO_FLAG_READ_WRITE} (读写模式)")

        index_ondisk = faiss.read_index(INDEX_FILE, IO_FLAG_READ_WRITE)
        start_time = time.time()

        num_chunks = (nb + chunk_size - 1) // chunk_size
        for i in range(0, nb, chunk_size):
            chunk_idx = i // chunk_size + 1
            print(f"       -> 正在处理块 {chunk_idx}/{num_chunks}: 向量 {i} 到 {min(i + chunk_size, nb) - 1}")

            # 从base.fbin中读取数据块
            xb_chunk, _, _ = read_fbin(BASE_FILE, i, chunk_size)

            index_ondisk.add(xb_chunk)
            del xb_chunk

        print(f"\n所有数据块添加完成，总耗时: {time.time() - start_time:.2f} 秒")
        print(f"磁盘索引中的向量总数 (ntotal): {index_ondisk.ntotal}")

    # ===========================================================
    # 7. 新增: 输出IVF分区统计信息 (仅在构建索引时执行)
    # ===========================================================
    if ENABLE_IVF_STATS and not skip_index_building:
        print("\n输出IVF分区统计信息...")
        start_stats_time = time.time()

        # 获取倒排列表
        invlists = index_ondisk.invlists

        # 准备统计信息
        partition_stats = []
        non_empty_partitions = 0
        max_size = 0
        min_size = float('inf')
        total_vectors = 0

        # 遍历所有分区
        for list_id in range(nlist):
            list_size = invlists.list_size(list_id)

            # 修改点 1：无论分区大小是否为0，都记录下来，以便生成完整的CSV报告
            partition_stats.append((list_id, list_size))

            # 仅针对非空分区更新摘要统计信息
            if list_size > 0:
                non_empty_partitions += 1
                max_size = max(max_size, list_size)
                min_size = min(min_size, list_size)
                total_vectors += list_size

        # 修改点 2：处理没有非空分区的边缘情况，避免打印 'inf'
        if non_empty_partitions == 0:
            min_size = 0

        # 计算非空分区的平均大小 (total_vectors 是非空分区中的向量总数)
        avg_size = total_vectors / non_empty_partitions if non_empty_partitions > 0 else 0

        # 输出统计摘要
        print(f"IVF分区统计摘要:")
        print(f"  分区总数: {nlist}")
        print(f"  非空分区数: {non_empty_partitions} ({non_empty_partitions / nlist * 100:.2f}%)")
        print(f"  最大分区大小: {max_size}")
        # 修改点 3：为了清晰起见，明确指出这是非空分区的最小值
        print(f"  最小(非空)分区大小: {min_size}")
        # 修改点 3：为了清晰起见，明确指出这是非空分区的平均值
        print(f"  平均(非空)分区大小: {avg_size:.2f}")

        # 将详细统计信息写入文件
        # 此部分无需修改，因为它现在会正确处理包含所有分区的 partition_stats 列表
        stats_filename = os.path.splitext(INDEX_FILE)[0] + "_ivf_stats.csv"
        with open(stats_filename, 'w') as f:
            f.write("partition_id,vector_count\n")
            for list_id, size in partition_stats:
                f.write(f"{list_id},{size}\n")

        print(f"分区统计信息已保存到: {stats_filename}")
        print(f"统计耗时: {time.time() - start_stats_time:.2f}秒")

    # 保存索引到磁盘
    print(f"正在将最终索引写回磁盘: {INDEX_FILE}")
    faiss.write_index(index_ondisk, INDEX_FILE)
    del index_ondisk

# ==============================================================================
# 8. 使用内存映射 (mmap) 进行搜索 (使用query.fbin)
# ==============================================================================
print("\nPhase 4: 使用内存映射模式进行搜索")
print(f"以 mmap 模式打开磁盘索引: {INDEX_FILE}")

# 兼容不同Faiss版本的IO标志处理
try:
    IO_FLAG_MMAP = faiss.IO_FLAG_MMAP
except AttributeError:
    try:
        IO_FLAG_MMAP = faiss.index_io.IO_FLAG_MMAP
    except AttributeError:
        IO_FLAG_MMAP = 4

print(f"使用IO标志: {IO_FLAG_MMAP} (内存映射模式)")

index_final = faiss.read_index(INDEX_FILE, IO_FLAG_MMAP)
index_final.nprobe = nprobe
# index_final.quantizer.hnsw.efSearch = 100  # 设置HNSW的efSearch参数以匹配nprobe
faiss.omp_set_num_threads(40)
index_final.parallel_mode = 0
print(f"并行模式线程数: {faiss.omp_get_max_threads()}")
print(f"并行模式: {index_final.parallel_mode}")
print(f"索引已准备好搜索 (nprobe={index_final.nprobe})")
generic_quantizer = index_final.quantizer
quantizer_hnsw = faiss.downcast_index(generic_quantizer)
quantizer_hnsw.hnsw.efSearch = efsearch
print(f"efConstruction: {quantizer_hnsw.hnsw.efConstruction}, efSearch: {quantizer_hnsw.hnsw.efSearch}")

# 设置索引内存基线
if ENABLE_DETAILED_MEMORY_MONITORING:
    memory_monitor.log_memory_snapshot("索引加载完成")
    # 估算索引大小（基于索引文件大小）
    index_file_size_mb = os.path.getsize(INDEX_FILE) / (1024 * 1024)
    memory_monitor.set_index_memory_baseline(index_file_size_mb * 0.8)  # 假设80%的索引文件被加载到内存

print("从 query.fbin 加载查询向量...")
xq = read_fbin(QUERY_FILE)

# 记录查询向量加载后的内存状态
if ENABLE_DETAILED_MEMORY_MONITORING:
    memory_monitor.log_memory_snapshot("查询向量加载完成")

# 在搜索前进行内存优化
if ENABLE_DETAILED_MEMORY_MONITORING:
    memory_monitor.force_gc_and_log("搜索前优化")

# ==============================================================================
# 8.5. 新增: 统计并保存每个查询命中的分区点数占总点数的比例
# ==============================================================================
if ENABLE_SEARCH_PARTITION_STATS:
    print("\n" + "=" * 60)
    print(f"Phase 4.5: 统计搜索分区信息 (nprobe={nprobe})")

    # 检查索引是否为IVF类型，因为该逻辑依赖于quantizer和invlists
    if not isinstance(index_final, faiss.IndexIVF):
        print("错误：索引类型不是IndexIVF，无法执行分区统计。")
    else:
        total_vectors_in_index = index_final.ntotal
        print(f"索引中的总向量数: {total_vectors_in_index}")

        if total_vectors_in_index == 0:
            print("警告：索引中没有向量，所有比例将为0。")

        print("正在为每个查询向量查找对应的分区...")
        # 1. 对每个查询向量，用粗量化器找到nprobe个最近的簇心(分区)
        # I_quant 的维度是 (nq, nprobe)，存储了每个查询命中的分区ID
        _, I_quant = index_final.quantizer.search(xq, nprobe)

        ratios = []
        print(f"正在计算 {nq} 个查询的命中分区点数比例...")

        # 2. 遍历每个查询的结果
        for i in range(nq):
            probed_list_ids = I_quant[i]

            # 3. 累加这些分区中的向量总数
            num_vectors_in_probed_partitions = 0
            for list_id in probed_list_ids:
                if list_id >= 0:  # 有效的分区ID
                    num_vectors_in_probed_partitions += index_final.invlists.list_size(int(list_id))

            # 4. 计算比例
            ratio = num_vectors_in_probed_partitions / total_vectors_in_index if total_vectors_in_index > 0 else 0
            ratios.append(ratio)

        # 5. 将结果写入文件
        try:
            with open(SEARCH_STATS_FILENAME, 'w') as f:
                for ratio in ratios:
                    f.write(f"{ratio:.8f}\n")  # 写入时保留8位小数
            print(f"搜索分区统计比例已成功写入文件: {SEARCH_STATS_FILENAME}")
        except IOError as e:
            print(f"错误：无法写入统计文件 {SEARCH_STATS_FILENAME}。原因: {e}")

    print("=" * 60)


if ENABLE_DETAILED_MEMORY_MONITORING:
    with memory_monitor.monitor_phase("执行搜索"):
        print("\n执行搜索...")
        start_time = time.time()
        D, I = index_final.search(xq, k)
        end_time = time.time()

        # 搜索后立即记录内存状态
        memory_monitor.log_memory_snapshot("搜索完成")
else:
    print("\n执行搜索...")
    start_time = time.time()
    D, I = index_final.search(xq, k)
    end_time = time.time()

# 从 .indexIVF_stats 属性中获取统计对象
stats = faiss.cvar.indexIVF_stats

print("\n========== 搜索性能统计 ==========")
print(f"查询向量总数 (nq): {stats.nq}")
print(f"总搜索时间 (search_time): {stats.search_time:.3f} ms")
print(f"  - 粗筛阶段用时 (quantization_time): {stats.quantization_time:.3f} ms")
# 精筛时间可以通过总时间减去粗筛时间得到
print(f"  - 精筛阶段用时 (search_time - quantization_time): {stats.search_time - stats.quantization_time:.3f} ms")
print("-" * 30)
print(f"访问的倒排列表总数 (nlist): {stats.nlist}")
print(f"计算的向量距离总数 (ndis): {stats.ndis}")
print(f"结果堆的更新总次数 (nheap_updates): {stats.nheap_updates}")
print("====================================\n")

# --- 新增QPS计算 ---
search_duration = end_time - start_time
print(f"搜索完成，耗时: {search_duration:.2f} 秒")

if search_duration > 0:
    qps = nq / search_duration
    print(f"QPS (每秒查询率): {qps:.2f}")
else:
    print("搜索耗时过短，无法计算QPS")
# --- QPS计算结束 ---


# ==============================================================================
# 9.  新增: 根据Groundtruth计算召回率 (内存优化版)
# ==============================================================================
print("\n" + "=" * 60)
print("Phase 5: 计算召回率 (内存优化版)")

if not os.path.exists(GROUNDTRUTH_FILE):
    print(f"Groundtruth文件未找到: {GROUNDTRUTH_FILE}")
    print("跳过召回率计算。")
else:
    print(f"以流式方式从 {GROUNDTRUTH_FILE} 读取 groundtruth 数据进行计算...")

    total_found = 0

    # 使用with语句确保文件被正确关闭
    with open(GROUNDTRUTH_FILE, 'rb') as f:
        # 首先，从文件的第一个整数确定groundtruth的维度 (k_gt)
        dim_bytes = f.read(4)
        if not dim_bytes:
            raise EOFError("Groundtruth 文件为空或已损坏。")
        k_gt = struct.unpack('i', dim_bytes)[0]

        print(f"Groundtruth 维度 (k_gt): {k_gt}")

        # 计算文件中每条记录的字节大小
        # 每条记录包含1个维度整数和k_gt个ID整数，每个整数4字节
        record_size_bytes = (k_gt + 1) * 4

        # 验证文件中的向量数量是否与查询数量(nq)匹配
        f.seek(0, os.SEEK_END)
        total_file_size = f.tell()
        num_gt_vectors = total_file_size // record_size_bytes
        if nq != num_gt_vectors:
            print(f"警告: 查询数量({nq})与groundtruth中的数量({num_gt_vectors})不匹配!")

        print(f"正在计算 Recall@{k}...")

        # 遍历每个查询结果
        for i in range(nq):
            # 计算第 i 条记录在文件中的起始位置
            offset = i * record_size_bytes
            f.seek(offset)

            # 从该位置读取一条完整的记录 (k_gt + 1 个整数)
            record_data = np.fromfile(f, dtype=np.int32, count=k_gt + 1)

            # 记录中的第一个整数是维度，我们提取从第二个元素开始的ID列表
            gt_i = record_data[1:]

            found_count = np.isin(I[i], gt_i[:k]).sum()
            total_found += found_count

    # 召回率 = (所有查询找到的正确近邻总数) / (所有查询返回的结果总数)
    recall = total_found / (nq * k)

    print(f"\n查询了 {nq} 个向量, k={k}")
    print(f"在top-{k}的结果中，总共找到了 {total_found} 个真实的近邻。")
    print(f"Recall@{k}: {recall:.4f}")

    # 记录召回率计算后的内存状态
    if ENABLE_DETAILED_MEMORY_MONITORING:
        memory_monitor.log_memory_snapshot("召回率计算完成")

print("=" * 60)

# ==============================================================================
# 10. 报告峰值内存
# ==============================================================================
print("\n" + "=" * 60)
print("内存使用情况报告")
print("=" * 60)

# 传统方法（resource.getrusage）
if platform.system() in ["Linux", "Darwin"]:
    peak_memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Linux":
        peak_memory_bytes *= 1024
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    print(f"传统方法 - 整个程序运行期间的峰值内存占用: {peak_memory_mb:.2f} MB")

# 新的详细内存分析
if ENABLE_DETAILED_MEMORY_MONITORING:
    print("\n详细内存分析:")
    summary = memory_monitor.get_memory_summary()
    if summary:
        print(f"  峰值RSS内存: {summary['peak_rss_mb']:.2f} MB")
        print(f"  峰值VMS内存: {summary['peak_vms_mb']:.2f} MB")
        print(f"  平均RSS内存: {summary['avg_rss_mb']:.2f} MB")
        print(f"  平均VMS内存: {summary['avg_vms_mb']:.2f} MB")
        print(f"  总监控快照数: {summary['total_snapshots']}")

    # 内存增长模式分析
    print("\n内存增长模式分析:")
    growth_analysis = memory_monitor.analyze_memory_growth_pattern()
    if growth_analysis:
        print(f"  总内存增长: {growth_analysis['total_growth_mb']:.2f} MB")
        print(f"  峰值使用: {growth_analysis['peak_usage_mb']:.2f} MB")
        print("  各阶段内存增长:")
        for phase_info in growth_analysis['growth_phases']:
            print(f"    {phase_info['phase']}: +{phase_info['growth_mb']:.2f} MB (总计: {phase_info['rss_mb']:.2f} MB)")

    # 内存分解分析
    print("\n内存使用分解:")
    final_info = memory_monitor.get_memory_info()
    print(
        f"  索引内存: {final_info['index_memory_mb']:.2f} MB ({final_info['index_memory_mb'] / final_info['rss_mb'] * 100:.1f}%)")
    print(
        f"  其他内存: {final_info['other_memory_mb']:.2f} MB ({final_info['other_memory_mb'] / final_info['rss_mb'] * 100:.1f}%)")

    # 显示tracemalloc统计信息
    print("\n内存使用热点分析:")
    memory_monitor.get_tracemalloc_top_stats(10)

    # 最终内存状态
    print(f"\n最终内存状态:")
    print(f"  RSS内存: {final_info['rss_mb']:.2f} MB")
    print(f"  VMS内存: {final_info['vms_mb']:.2f} MB")
    print(f"  VMS/RSS比例: {final_info['vms_mb'] / final_info['rss_mb']:.2f}")
    print(f"  Python对象数: {final_info['python_objects']}")
    if 'traced_current_mb' in final_info:
        print(f"  Traced内存: {final_info['traced_current_mb']:.2f} MB")

    # 内存优化建议
    print("\n内存优化建议:")
    suggestions = memory_monitor.get_memory_optimization_suggestions()
    if suggestions:
        for suggestion in suggestions:
            print(f"  {suggestion}")
    else:
        print("  ✅ 内存使用情况良好，无需特别优化")

    print(f"\n内存日志已保存到: {MEMORY_LOG_FILENAME}")

    # 生成内存可视化图表
    if ENABLE_MEMORY_VISUALIZATION:
        print("\n正在生成内存使用情况可视化图表...")
        memory_monitor.generate_memory_visualization(MEMORY_PLOT_FILENAME)
    
    # 生成最终的内存使用情况图表
    if ENABLE_FINAL_MEMORY_PLOT:
        print("\n正在生成最终内存使用情况图表...")
        memory_monitor.generate_final_memory_plot(FINAL_MEMORY_PLOT_FILENAME)
else:
    print("详细内存监控已禁用")

print("=" * 60)