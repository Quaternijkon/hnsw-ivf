#!/usr/bin/env python3
"""
mmap模式专用内存监控器
移除索引内存基线概念，采用更准确的内存分析方法
"""

import os
import time
import psutil
import tracemalloc
import gc
import threading
import queue
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt


class MmapMemoryMonitor:
    """专门针对mmap模式的内存监控器"""
    
    def __init__(self, enable_tracemalloc: bool = True, log_file: Optional[str] = None, 
                 enable_continuous: bool = False, sampling_interval: float = 1.0, 
                 change_threshold: float = 5.0):
        self.enable_tracemalloc = enable_tracemalloc
        self.log_file = log_file
        self.memory_snapshots: List[Dict] = []
        self.process = psutil.Process()
        self.start_time = time.time()
        
        # mmap特定的内存追踪
        self.phase_memory_tracking = {}  # 各阶段内存追踪
        self.mmap_files = {}  # 记录mmap文件信息
        self.baseline_memory = self._get_current_memory_info()
        
        # 连续监控相关
        self.enable_continuous = enable_continuous
        self.sampling_interval = sampling_interval
        self.change_threshold = change_threshold
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.current_phase = "初始化"
        self.monitoring_active = False
        
        if self.enable_tracemalloc:
            tracemalloc.start()
        
        if self.log_file:
            self._initialize_log_file()
        
        # 启动连续监控
        if self.enable_continuous:
            self.start_continuous_monitoring()
        
        print(f"[mmap内存监控] 已初始化，基线内存: RSS={self.baseline_memory['rss_mb']:.2f}MB, "
              f"VMS={self.baseline_memory['vms_mb']:.2f}MB")
    
    def _initialize_log_file(self):
        """初始化日志文件"""
        with open(self.log_file, 'w') as f:
            f.write("时间戳,相对时间,阶段,RSS_MB,VMS_MB,共享内存_MB,私有内存_MB,页面缓存_MB,"
                   f"内存百分比,Python对象数,mmap文件数,监控类型\n")
    
    def _get_current_memory_info(self) -> Dict:
        """获取当前详细内存信息"""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        # 尝试获取更详细的内存信息
        try:
            # 在Linux系统上获取更详细的内存信息
            memory_info_ex = self.process.memory_info_ex()
            shared_mb = getattr(memory_info_ex, 'shared', 0) / (1024 * 1024)
            private_mb = (memory_info.rss - getattr(memory_info_ex, 'shared', 0)) / (1024 * 1024)
        except (AttributeError, psutil.NoSuchProcess):
            shared_mb = 0
            private_mb = memory_info.rss / (1024 * 1024)
        
        # 尝试获取页面缓存信息（Linux特有）
        page_cache_mb = self._estimate_page_cache_usage()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),  # 物理内存
            'vms_mb': memory_info.vms / (1024 * 1024),  # 虚拟内存
            'shared_mb': shared_mb,  # 共享内存
            'private_mb': private_mb,  # 私有内存
            'page_cache_mb': page_cache_mb,  # 页面缓存估算
            'memory_percent': memory_percent,
            'python_objects': len(gc.get_objects()),
            'mmap_files_count': len(self.mmap_files)
        }
    
    def _estimate_page_cache_usage(self) -> float:
        """估算页面缓存使用量（Linux特有）"""
        try:
            # 读取进程的内存映射信息
            maps_path = f"/proc/{self.process.pid}/smaps"
            if os.path.exists(maps_path):
                total_cached = 0
                with open(maps_path, 'r') as f:
                    for line in f:
                        if line.startswith('Cached:'):
                            # 解析缓存大小 (格式: "Cached:    1234 kB")
                            cached_kb = int(line.split()[1])
                            total_cached += cached_kb
                return total_cached / 1024  # 转换为MB
        except (FileNotFoundError, PermissionError, ValueError):
            pass
        
        return 0
    
    def register_mmap_file(self, file_path: str, file_purpose: str = "索引文件"):
        """注册mmap文件，用于追踪其内存影响"""
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            self.mmap_files[file_path] = {
                'size_mb': file_size / (1024 * 1024),
                'purpose': file_purpose,
                'register_time': time.time() - self.start_time
            }
            print(f"[mmap追踪] 注册文件: {file_path} ({file_size/1024/1024:.2f}MB, {file_purpose})")
    
    def mark_phase_start(self, phase_name: str):
        """标记阶段开始"""
        memory_info = self._get_current_memory_info()
        memory_info['timestamp'] = time.time()
        memory_info['relative_time'] = memory_info['timestamp'] - self.start_time
        
        self.current_phase = phase_name
        self.phase_memory_tracking[phase_name] = {
            'start_memory': memory_info,
            'end_memory': None,
            'peak_rss': memory_info['rss_mb'],
            'peak_vms': memory_info['vms_mb']
        }
        
        print(f"[阶段开始] {phase_name}: RSS={memory_info['rss_mb']:.2f}MB, "
              f"VMS={memory_info['vms_mb']:.2f}MB")
    
    def mark_phase_end(self, phase_name: str):
        """标记阶段结束"""
        memory_info = self._get_current_memory_info()
        memory_info['timestamp'] = time.time()
        memory_info['relative_time'] = memory_info['timestamp'] - self.start_time
        
        if phase_name in self.phase_memory_tracking:
            self.phase_memory_tracking[phase_name]['end_memory'] = memory_info
            
            start_mem = self.phase_memory_tracking[phase_name]['start_memory']
            rss_growth = memory_info['rss_mb'] - start_mem['rss_mb']
            vms_growth = memory_info['vms_mb'] - start_mem['vms_mb']
            duration = memory_info['relative_time'] - start_mem['relative_time']
            
            print(f"[阶段结束] {phase_name}: RSS={memory_info['rss_mb']:.2f}MB (+{rss_growth:+.2f}MB), "
                  f"VMS={memory_info['vms_mb']:.2f}MB (+{vms_growth:+.2f}MB), 耗时={duration:.2f}s")
    
    def get_mmap_memory_analysis(self) -> Dict:
        """分析mmap模式下的内存使用情况"""
        current_memory = self._get_current_memory_info()
        
        # 计算各种内存增长
        rss_growth = current_memory['rss_mb'] - self.baseline_memory['rss_mb']
        vms_growth = current_memory['vms_mb'] - self.baseline_memory['vms_mb']
        
        # 分析mmap文件的潜在影响
        total_mmap_file_size = sum(info['size_mb'] for info in self.mmap_files.values())
        
        # 估算实际被加载到内存的mmap数据
        estimated_loaded_mmap_mb = min(rss_growth, total_mmap_file_size * 0.3)  # 假设30%被实际加载
        
        # 程序自身的内存使用（排除mmap影响）
        program_memory_mb = max(0, rss_growth - estimated_loaded_mmap_mb)
        
        analysis = {
            'current_rss_mb': current_memory['rss_mb'],
            'current_vms_mb': current_memory['vms_mb'],
            'rss_growth_mb': rss_growth,
            'vms_growth_mb': vms_growth,
            'total_mmap_file_size_mb': total_mmap_file_size,
            'estimated_loaded_mmap_mb': estimated_loaded_mmap_mb,
            'program_memory_mb': program_memory_mb,
            'page_cache_mb': current_memory['page_cache_mb'],
            'mmap_efficiency': estimated_loaded_mmap_mb / total_mmap_file_size if total_mmap_file_size > 0 else 0,
            'memory_breakdown': {
                'baseline_mb': self.baseline_memory['rss_mb'],
                'program_growth_mb': program_memory_mb,
                'mmap_loaded_mb': estimated_loaded_mmap_mb,
                'total_mb': current_memory['rss_mb']
            }
        }
        
        return analysis
    
    def log_memory_snapshot(self, phase: str, monitoring_type: str = "手动"):
        """记录内存快照"""
        memory_info = self._get_current_memory_info()
        memory_info['timestamp'] = time.time()
        memory_info['relative_time'] = memory_info['timestamp'] - self.start_time
        memory_info['phase'] = phase
        memory_info['monitoring_type'] = monitoring_type
        
        self.memory_snapshots.append(memory_info)
        
        # 更新当前阶段
        if not phase.endswith("_连续监控"):
            self.current_phase = phase
        
        # 更新阶段峰值
        phase_key = phase.split('_')[0]
        if phase_key in self.phase_memory_tracking:
            self.phase_memory_tracking[phase_key]['peak_rss'] = max(
                self.phase_memory_tracking[phase_key]['peak_rss'], 
                memory_info['rss_mb']
            )
            self.phase_memory_tracking[phase_key]['peak_vms'] = max(
                self.phase_memory_tracking[phase_key]['peak_vms'], 
                memory_info['vms_mb']
            )
        
        # 写入日志文件
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{memory_info['timestamp']:.2f},{memory_info['relative_time']:.2f},{phase},"
                       f"{memory_info['rss_mb']:.2f},{memory_info['vms_mb']:.2f},"
                       f"{memory_info['shared_mb']:.2f},{memory_info['private_mb']:.2f},"
                       f"{memory_info['page_cache_mb']:.2f},{memory_info['memory_percent']:.2f},"
                       f"{memory_info['python_objects']},{memory_info['mmap_files_count']},{monitoring_type}\n")
        
        print(f"[mmap内存监控] {phase}: RSS={memory_info['rss_mb']:.2f}MB, "
              f"VMS={memory_info['vms_mb']:.2f}MB, 缓存={memory_info['page_cache_mb']:.2f}MB")
    
    @contextmanager
    def monitor_phase(self, phase_name: str):
        """上下文管理器，用于监控特定代码段的内存使用"""
        print(f"\n[mmap内存监控] 开始阶段: {phase_name}")
        
        # 标记阶段开始
        self.mark_phase_start(phase_name)
        self.log_memory_snapshot(f"{phase_name}_开始")
        
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            
            # 标记阶段结束
            self.mark_phase_end(phase_name)
            self.log_memory_snapshot(f"{phase_name}_结束")
            
            print(f"[mmap内存监控] 阶段 {phase_name} 完成，耗时: {end_time - start_time:.2f}秒")
    
    def get_phase_summary(self) -> Dict:
        """获取各阶段的内存使用摘要"""
        summary = {}
        
        for phase_name, phase_data in self.phase_memory_tracking.items():
            if phase_data['end_memory'] is not None:
                summary[phase_name] = {
                    'duration_s': phase_data['end_memory']['relative_time'] - phase_data['start_memory']['relative_time'],
                    'memory_growth_mb': phase_data['end_memory']['rss_mb'] - phase_data['start_memory']['rss_mb'],
                    'peak_rss': phase_data['peak_rss'],
                    'peak_vms': phase_data['peak_vms'],
                    'start_memory_mb': phase_data['start_memory']['rss_mb'],
                    'end_memory_mb': phase_data['end_memory']['rss_mb']
                }
        
        return summary
    
    def start_continuous_monitoring(self):
        """启动连续内存监控线程"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._continuous_monitoring_worker, daemon=True)
        self.monitoring_thread.start()
        print(f"[连续监控] 已启动，采样间隔: {self.sampling_interval}秒")
    
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
        last_rss = 0
        
        while not self.stop_monitoring.is_set():
            try:
                memory_info = self._get_current_memory_info()
                current_rss = memory_info['rss_mb']
                
                # 检查是否需要记录
                should_record = False
                monitoring_type = "定时采样"
                
                # 内存显著变化时立即记录
                if abs(current_rss - last_rss) >= self.change_threshold:
                    should_record = True
                    monitoring_type = "变化触发"
                    
                # 定时采样
                elif len(self.memory_snapshots) == 0 or \
                     (time.time() - self.memory_snapshots[-1]['timestamp']) >= self.sampling_interval:
                    should_record = True
                    monitoring_type = "定时采样"
                
                if should_record:
                    self.log_memory_snapshot(f"{self.current_phase}_连续监控", monitoring_type)
                    last_rss = current_rss
                
                # 等待下一次采样
                self.stop_monitoring.wait(min(self.sampling_interval, 0.5))
                
            except Exception as e:
                print(f"[连续监控] 错误: {e}")
                break
    
    def generate_mmap_memory_report(self) -> str:
        """生成mmap内存使用报告"""
        analysis = self.get_mmap_memory_analysis()
        
        report = f"""
=== mmap模式内存使用分析报告 ===

基本信息:
• 程序运行时长: {time.time() - self.start_time:.1f}秒
• 注册的mmap文件数: {len(self.mmap_files)}
• 内存快照数: {len(self.memory_snapshots)}

内存使用情况:
• 当前RSS内存: {analysis['current_rss_mb']:.2f} MB
• 当前VMS内存: {analysis['current_vms_mb']:.2f} MB
• RSS增长: {analysis['rss_growth_mb']:+.2f} MB
• VMS增长: {analysis['vms_growth_mb']:+.2f} MB

mmap文件分析:
• 总mmap文件大小: {analysis['total_mmap_file_size_mb']:.2f} MB
• 估算实际加载: {analysis['estimated_loaded_mmap_mb']:.2f} MB
• mmap加载效率: {analysis['mmap_efficiency']:.1%}

内存分解:
• 基线内存: {analysis['memory_breakdown']['baseline_mb']:.2f} MB
• 程序增长: {analysis['memory_breakdown']['program_growth_mb']:.2f} MB
• mmap加载: {analysis['memory_breakdown']['mmap_loaded_mb']:.2f} MB
• 页面缓存: {analysis['page_cache_mb']:.2f} MB

"""
        
        # 添加mmap文件详情
        if self.mmap_files:
            report += "mmap文件详情:\n"
            for file_path, info in self.mmap_files.items():
                file_name = os.path.basename(file_path)
                report += f"• {file_name}: {info['size_mb']:.2f} MB ({info['purpose']})\n"
        
        # 添加阶段分析
        if self.phase_memory_tracking:
            report += "\n阶段内存分析:\n"
            for phase_name, phase_data in self.phase_memory_tracking.items():
                if phase_data['end_memory']:
                    start_rss = phase_data['start_memory']['rss_mb']
                    end_rss = phase_data['end_memory']['rss_mb']
                    peak_rss = phase_data['peak_rss']
                    growth = end_rss - start_rss
                    duration = phase_data['end_memory']['relative_time'] - phase_data['start_memory']['relative_time']
                    
                    report += f"• {phase_name}: RSS {start_rss:.1f}→{end_rss:.1f}MB (+{growth:+.1f}MB), "
                    report += f"峰值{peak_rss:.1f}MB, 耗时{duration:.1f}s\n"
        
        report += "\n=== 报告结束 ==="
        
        return report
    
    def cleanup(self):
        """清理资源，停止监控线程"""
        if self.enable_continuous:
            self.stop_continuous_monitoring()
        
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()


if __name__ == "__main__":
    # 简单测试
    monitor = MmapMemoryMonitor(enable_continuous=True, sampling_interval=0.5)
    
    # 模拟mmap文件注册
    monitor.register_mmap_file("/tmp/test_index.bin", "测试索引")
    
    with monitor.monitor_phase("测试阶段"):
        import numpy as np
        data = np.random.rand(1000, 100)
        time.sleep(2)
    
    print(monitor.generate_mmap_memory_report())
    monitor.cleanup()
