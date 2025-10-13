#!/usr/bin/env python3
"""
改进的内存监控类 - 针对mmap模式的Faiss索引
移除索引内存基线概念，采用基于阶段的内存增长追踪
"""

import time
import psutil
import tracemalloc
import gc
import threading
import queue
from contextlib import contextmanager
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

class ImprovedMemoryMonitor:
    """改进的内存监控类，专门针对mmap模式下的Faiss应用"""
    
    def __init__(self, enable_tracemalloc: bool = True, log_file: Optional[str] = None, 
                 enable_continuous: bool = False, sampling_interval: float = 1.0, 
                 change_threshold: float = 5.0):
        self.enable_tracemalloc = enable_tracemalloc
        self.log_file = log_file
        self.memory_snapshots: List[Dict] = []
        self.process = psutil.Process()
        self.start_time = time.time()
        self.phase_markers = []
        
        # 新的内存追踪方案：基于阶段的内存记录
        self.memory_phases = {}  # 存储各阶段的内存使用情况
        self.baseline_memory = 0  # 程序启动时的基线内存
        self.pre_index_memory = 0  # 索引加载前的内存
        self.post_index_memory = 0  # 索引加载后的内存
        
        # 连续监控相关
        self.enable_continuous = enable_continuous
        self.sampling_interval = sampling_interval
        self.change_threshold = change_threshold
        self.monitoring_thread = None
        self.monitoring_queue = queue.Queue()
        self.stop_monitoring = threading.Event()
        self.last_rss_memory = 0
        self.current_phase = "初始化"
        self.monitoring_active = False
        
        if self.enable_tracemalloc:
            tracemalloc.start()
        
        if self.log_file:
            with open(self.log_file, 'w') as f:
                f.write("时间戳,相对时间,阶段,RSS_MB,内存百分比,Python对象数,垃圾回收次数,阶段增长_MB,总增长_MB,内存类型,监控类型\n")
        
        # 记录初始基线
        self.baseline_memory = self._get_current_rss()
        
        # 启动连续监控
        if self.enable_continuous:
            self.start_continuous_monitoring()
    
    def _get_current_rss(self) -> float:
        """获取当前RSS内存（MB）"""
        return self.process.memory_info().rss / (1024 * 1024)
    
    def get_memory_info(self) -> Dict:
        """获取详细的内存信息"""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        current_time = time.time()
        current_rss = memory_info.rss / (1024 * 1024)
        
        # 获取Python对象统计
        gc_stats = gc.get_stats()
        total_objects = sum(stat['collected'] for stat in gc_stats)
        
        # 计算内存增长
        total_growth = current_rss - self.baseline_memory
        phase_growth = self._calculate_phase_growth(current_rss)
        memory_type = self._classify_memory_usage(current_rss)
        
        info = {
            'timestamp': current_time,
            'relative_time': current_time - self.start_time,
            'rss_mb': current_rss,
            'memory_percent': memory_percent,
            'python_objects': len(gc.get_objects()),
            'gc_collections': total_objects,
            'total_growth_mb': total_growth,
            'phase_growth_mb': phase_growth,
            'memory_type': memory_type
        }
        
        return info
    
    def _calculate_phase_growth(self, current_rss: float) -> float:
        """计算当前阶段的内存增长"""
        phase_key = self.current_phase.split('_')[0]  # 去掉'_连续监控'后缀
        
        if phase_key in self.memory_phases:
            phase_start_memory = self.memory_phases[phase_key]['start_memory']
            return current_rss - phase_start_memory
        
        return 0
    
    def _classify_memory_usage(self, current_rss: float) -> str:
        """根据当前阶段和内存使用情况分类内存类型"""
        if current_rss < self.baseline_memory + 10:
            return "基线内存"
        elif self.pre_index_memory > 0 and current_rss < self.pre_index_memory + 20:
            return "程序内存"
        elif self.post_index_memory > 0 and current_rss >= self.post_index_memory * 0.9:
            return "索引+程序"
        else:
            return "动态内存"
    
    def mark_phase_start(self, phase_name: str):
        """标记阶段开始"""
        current_rss = self._get_current_rss()
        self.current_phase = phase_name
        
        # 记录阶段开始的内存状态
        self.memory_phases[phase_name] = {
            'start_time': time.time() - self.start_time,
            'start_memory': current_rss,
            'end_memory': None,
            'end_time': None,
            'peak_memory': current_rss
        }
        
        print(f"[阶段开始] {phase_name}: RSS={current_rss:.2f}MB")
    
    def mark_phase_end(self, phase_name: str):
        """标记阶段结束"""
        current_rss = self._get_current_rss()
        
        if phase_name in self.memory_phases:
            self.memory_phases[phase_name]['end_memory'] = current_rss
            self.memory_phases[phase_name]['end_time'] = time.time() - self.start_time
            
            growth = current_rss - self.memory_phases[phase_name]['start_memory']
            duration = self.memory_phases[phase_name]['end_time'] - self.memory_phases[phase_name]['start_time']
            
            print(f"[阶段结束] {phase_name}: RSS={current_rss:.2f}MB, 增长={growth:+.2f}MB, 耗时={duration:.2f}s")
    
    def mark_index_loading(self, before: bool = True):
        """标记索引加载前后的内存状态"""
        current_rss = self._get_current_rss()
        
        if before:
            self.pre_index_memory = current_rss
            print(f"[索引加载前] RSS={current_rss:.2f}MB")
        else:
            self.post_index_memory = current_rss
            index_memory_impact = current_rss - self.pre_index_memory
            print(f"[索引加载后] RSS={current_rss:.2f}MB, 索引影响={index_memory_impact:+.2f}MB")
    
    def estimate_mmap_index_memory(self) -> Dict:
        """估算mmap模式下索引的内存影响"""
        if self.pre_index_memory == 0 or self.post_index_memory == 0:
            return {
                'direct_impact_mb': 0,
                'estimated_mmap_overhead_mb': 0,
                'total_program_memory_mb': self._get_current_rss(),
                'estimation_confidence': 'low'
            }
        
        # mmap模式下的索引内存影响分析
        direct_impact = self.post_index_memory - self.pre_index_memory
        current_rss = self._get_current_rss()
        
        # 在mmap模式下，直接影响通常很小（主要是索引元数据）
        # 实际的索引数据通过页面缓存按需加载
        estimation = {
            'direct_impact_mb': direct_impact,  # 索引加载的直接影响
            'estimated_mmap_overhead_mb': max(0, direct_impact),  # mmap开销
            'total_program_memory_mb': current_rss,  # 总程序内存
            'baseline_memory_mb': self.baseline_memory,  # 基线内存
            'estimation_confidence': 'medium' if direct_impact > 0 else 'low'
        }
        
        return estimation
    
    def start_continuous_monitoring(self):
        """启动连续内存监控线程"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._continuous_monitoring_worker, daemon=True)
        self.monitoring_thread.start()
        print(f"[连续监控] 已启动，采样间隔: {self.sampling_interval}秒，变化阈值: {self.change_threshold}MB")
    
    def stop_continuous_monitoring(self):
        """停止连续内存监控"""
        if not self.monitoring_active:
            return
            
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        self.monitoring_active = False
        print("[连续监控] 已停止")
    
    def _continuous_monitoring_worker(self):
        """连续监控的工作线程"""
        while not self.stop_monitoring.is_set():
            try:
                info = self.get_memory_info()
                current_rss = info['rss_mb']
                
                # 检查是否需要记录
                should_record = False
                monitoring_type = "定时采样"
                
                # 内存显著变化时立即记录
                if abs(current_rss - self.last_rss_memory) >= self.change_threshold:
                    should_record = True
                    monitoring_type = "变化触发"
                    
                # 定时采样
                elif len(self.memory_snapshots) == 0 or \
                     (info['relative_time'] - self.memory_snapshots[-1]['relative_time']) >= self.sampling_interval:
                    should_record = True
                    monitoring_type = "定时采样"
                
                if should_record:
                    info['phase'] = f"{self.current_phase}_连续监控"
                    info['monitoring_type'] = monitoring_type
                    self.memory_snapshots.append(info)
                    self.last_rss_memory = current_rss
                    
                    # 更新阶段峰值内存
                    phase_key = self.current_phase.split('_')[0]
                    if phase_key in self.memory_phases:
                        self.memory_phases[phase_key]['peak_memory'] = max(
                            self.memory_phases[phase_key]['peak_memory'], 
                            current_rss
                        )
                    
                    # 写入日志文件
                    if self.log_file:
                        with open(self.log_file, 'a') as f:
                            f.write(f"{info['timestamp']:.2f},{info['relative_time']:.2f},"
                                   f"{info['phase']},{info['rss_mb']:.2f},{info['memory_percent']:.2f},"
                                   f"{info['python_objects']},{info['gc_collections']},"
                                   f"{info['phase_growth_mb']:.2f},{info['total_growth_mb']:.2f},"
                                   f"{info['memory_type']},{monitoring_type}\n")
                
                # 等待下一次采样
                self.stop_monitoring.wait(min(self.sampling_interval, 0.5))
                
            except Exception as e:
                print(f"[连续监控] 错误: {e}")
                break
    
    def log_memory_snapshot(self, phase: str, monitoring_type: str = "手动"):
        """记录内存快照"""
        info = self.get_memory_info()
        info['phase'] = phase
        info['monitoring_type'] = monitoring_type
        self.memory_snapshots.append(info)
        
        # 更新当前阶段
        if not phase.endswith("_连续监控"):
            self.current_phase = phase
        
        # 添加阶段标记
        self.add_phase_marker(phase)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{info['timestamp']:.2f},{info['relative_time']:.2f},{phase},"
                       f"{info['rss_mb']:.2f},{info['memory_percent']:.2f},"
                       f"{info['python_objects']},{info['gc_collections']},"
                       f"{info['phase_growth_mb']:.2f},{info['total_growth_mb']:.2f},"
                       f"{info['memory_type']},{monitoring_type}\n")
        
        print(f"[内存监控] {phase}: RSS={info['rss_mb']:.2f}MB, "
              f"总增长={info['total_growth_mb']:+.2f}MB, "
              f"阶段增长={info['phase_growth_mb']:+.2f}MB, "
              f"类型={info['memory_type']}")
    
    def add_phase_marker(self, phase_name: str, phase_type: str = "阶段"):
        """添加阶段标记，用于在图表中显示"""
        current_time = time.time()
        self.phase_markers.append({
            'time': current_time,
            'relative_time': current_time - self.start_time,
            'phase_name': phase_name,
            'phase_type': phase_type
        })
    
    @contextmanager
    def monitor_phase(self, phase_name: str):
        """上下文管理器，用于监控特定代码段的内存使用"""
        print(f"\n[内存监控] 开始阶段: {phase_name}")
        
        # 标记阶段开始
        self.mark_phase_start(phase_name)
        self.log_memory_snapshot(f"{phase_name}_开始")
        
        start_info = self.get_memory_info()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_info = self.get_memory_info()
            
            # 标记阶段结束
            self.mark_phase_end(phase_name)
            
            # 计算阶段统计
            rss_diff = end_info['rss_mb'] - start_info['rss_mb']
            objects_diff = end_info['python_objects'] - start_info['python_objects']
            
            print(f"[内存监控] 结束阶段: {phase_name}")
            print(f"  耗时: {end_time - start_time:.2f}秒")
            print(f"  RSS变化: {rss_diff:+.2f}MB")
            print(f"  Python对象变化: {objects_diff:+d}")
            print(f"  内存类型变化: {start_info['memory_type']} -> {end_info['memory_type']}")
            
            self.log_memory_snapshot(f"{phase_name}_结束")
    
    def get_phase_summary(self) -> Dict:
        """获取各阶段的内存使用摘要"""
        summary = {}
        
        for phase_name, phase_data in self.memory_phases.items():
            if phase_data['end_memory'] is not None:
                summary[phase_name] = {
                    'duration_s': phase_data['end_time'] - phase_data['start_time'],
                    'memory_growth_mb': phase_data['end_memory'] - phase_data['start_memory'],
                    'peak_memory_mb': phase_data['peak_memory'],
                    'start_memory_mb': phase_data['start_memory'],
                    'end_memory_mb': phase_data['end_memory']
                }
        
        return summary
    
    def get_memory_summary(self) -> Dict:
        """获取内存使用摘要"""
        if not self.memory_snapshots:
            return {}
        
        rss_values = [s['rss_mb'] for s in self.memory_snapshots]
        
        # mmap索引内存估算
        mmap_analysis = self.estimate_mmap_index_memory()
        
        return {
            'peak_rss_mb': max(rss_values),
            'avg_rss_mb': sum(rss_values) / len(rss_values),
            'total_snapshots': len(self.memory_snapshots),
            'total_duration': self.memory_snapshots[-1]['relative_time'] if self.memory_snapshots else 0,
            'baseline_memory_mb': self.baseline_memory,
            'total_growth_mb': rss_values[-1] - self.baseline_memory if rss_values else 0,
            'mmap_analysis': mmap_analysis,
            'phase_summary': self.get_phase_summary()
        }
    
    def cleanup(self):
        """清理资源，停止监控线程"""
        if self.enable_continuous:
            self.stop_continuous_monitoring()
        
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()


if __name__ == "__main__":
    # 简单测试
    monitor = ImprovedMemoryMonitor(enable_continuous=True, sampling_interval=0.5)
    
    with monitor.monitor_phase("测试阶段"):
        import numpy as np
        data = np.random.rand(1000, 100)
        time.sleep(2)
    
    print("\n内存摘要:")
    summary = monitor.get_memory_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    monitor.cleanup()
