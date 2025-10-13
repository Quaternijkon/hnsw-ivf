import numpy as np
import faiss
import time
import os
import platform
import resource
import struct
import re
import psutil
import matplotlib.pyplot as plt
import threading
from typing import List, Dict, Tuple

# ==============================================================================
# 0. 路径和文件名配置 & 调试开关
# ==============================================================================
DATA_DIR = "./sift"
LEARN_FILE = os.path.join(DATA_DIR, "learn.fbin")
BASE_FILE = os.path.join(DATA_DIR, "base.fbin")
QUERY_FILE = os.path.join(DATA_DIR, "query.fbin")
GROUNDTRUTH_FILE = os.path.join(DATA_DIR, "groundtruth.ivecs")

# 内存监控配置
ENABLE_MEMORY_MONITORING = True
MEMORY_PLOT_FILENAME = os.path.join(DATA_DIR, "lightweight_memory_usage.png")

# 监测频率配置
ENABLE_CONTINUOUS_MONITORING = True  # 启用连续监测
MONITORING_INTERVAL = 0.1  # 监测间隔（秒），建议范围：0.1-2.0
MEMORY_CHANGE_THRESHOLD = 10.0  # 控制台输出阈值（MB），超过此值才打印到控制台

# 监测频率调整说明：
# 1. 高频监测：MONITORING_INTERVAL = 0.1 (每0.1秒记录一次，获得最平滑的曲线)
# 2. 中频监测：MONITORING_INTERVAL = 0.5 (每0.5秒记录一次，平衡精度和开销)
# 3. 低频监测：MONITORING_INTERVAL = 1.0 (每1秒记录一次，减少开销)
# 4. 仅手动监测：ENABLE_CONTINUOUS_MONITORING = False
# 
# 注意：MEMORY_CHANGE_THRESHOLD现在只控制控制台输出，不影响数据记录
# 所有监测到的内存数据都会被记录，确保图表的连续性和平滑性

# ==============================================================================
# 1. 轻量级内存监控类
# ==============================================================================
class LightweightMemoryMonitor:
    """轻量级内存监控类，支持连续监测和变化触发监测"""
    
    def __init__(self, enable_continuous: bool = True, interval: float = 0.5, 
                 change_threshold: float = 10.0):
        self.process = psutil.Process()
        self.memory_records: List[Dict] = []
        self.start_time = time.time()
        self.current_phase = "初始化"
        self.phase_boundaries: List[Tuple[float, str]] = []  # (时间, 阶段名)
        
        # 连续监测相关
        self.enable_continuous = enable_continuous
        self.interval = interval
        self.change_threshold = change_threshold
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.last_rss_mb = 0
        self.monitoring_active = False
    
    def record_memory(self, phase: str = None, monitoring_type: str = "手动"):
        """记录当前内存使用情况"""
        if not ENABLE_MEMORY_MONITORING:
            return
            
        current_time = time.time()
        relative_time = current_time - self.start_time
        
        # 只获取RSS内存，避免昂贵的操作
        memory_info = self.process.memory_info()
        rss_mb = memory_info.rss / (1024 * 1024)
        
        # 记录阶段边界
        if phase and phase != self.current_phase:
            self.phase_boundaries.append((relative_time, phase))
            self.current_phase = phase
        
        # 记录内存数据
        self.memory_records.append({
            'time': relative_time,
            'rss_mb': rss_mb,
            'phase': phase or self.current_phase,
            'monitoring_type': monitoring_type
        })
        
        # 更新最后记录的内存值
        self.last_rss_mb = rss_mb
        
        # 只在手动记录或变化触发时打印
        if monitoring_type in ["手动", "变化触发"]:
            print(f"[内存监控] {phase or self.current_phase}: {rss_mb:.2f} MB ({monitoring_type})")
    
    def start_continuous_monitoring(self):
        """启动连续监测"""
        if not self.enable_continuous or self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._continuous_monitoring_worker, daemon=True)
        self.monitoring_thread.start()
        print(f"[连续监控] 已启动，监测间隔: {self.interval}秒，变化阈值: {self.change_threshold}MB")
    
    def stop_continuous_monitoring(self):
        """停止连续监测"""
        if not self.monitoring_active:
            return
        
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        self.monitoring_active = False
        print("[连续监控] 已停止")
    
    def _continuous_monitoring_worker(self):
        """连续监测工作线程 - 连续记录所有内存数据"""
        while not self.stop_monitoring.is_set():
            try:
                current_time = time.time()
                relative_time = current_time - self.start_time
                
                # 获取当前内存使用
                memory_info = self.process.memory_info()
                rss_mb = memory_info.rss / (1024 * 1024)
                
                # 连续记录所有内存数据（无判断条件）
                self.memory_records.append({
                    'time': relative_time,
                    'rss_mb': rss_mb,
                    'phase': f"{self.current_phase}_连续监控",
                    'monitoring_type': "连续监测"
                })
                self.last_rss_mb = rss_mb
                
                # 只在内存变化显著时打印（减少控制台输出）
                if len(self.memory_records) > 1:
                    prev_rss = self.memory_records[-2]['rss_mb']
                    if abs(rss_mb - prev_rss) >= self.change_threshold:
                        print(f"[内存监控] {self.current_phase}_连续监控: {rss_mb:.2f} MB (变化: {rss_mb - prev_rss:+.1f} MB)")
                
                # 等待下一次监测
                self.stop_monitoring.wait(self.interval)
                
            except Exception as e:
                print(f"[连续监控] 错误: {e}")
                break
    
    def generate_memory_plot(self, output_file: str):
        """生成内存使用图表"""
        if not self.memory_records:
            print("没有内存记录数据，无法生成图表")
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 准备数据
        times = [record['time'] for record in self.memory_records]
        rss_values = [record['rss_mb'] for record in self.memory_records]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 分离不同类型的监测点
        manual_times = [record['time'] for record in self.memory_records if record.get('monitoring_type') == '手动']
        manual_values = [record['rss_mb'] for record in self.memory_records if record.get('monitoring_type') == '手动']
        continuous_times = [record['time'] for record in self.memory_records if record.get('monitoring_type') == '连续监测']
        continuous_values = [record['rss_mb'] for record in self.memory_records if record.get('monitoring_type') == '连续监测']
        
        # 绘制连续监测的平滑曲线
        if continuous_times:
            ax.plot(continuous_times, continuous_values, 'b-', linewidth=1, label='连续监测 (RSS)', alpha=0.8)
            ax.fill_between(continuous_times, continuous_values, alpha=0.2, color='blue')
        else:
            # 回退到所有数据
            ax.plot(times, rss_values, 'b-', linewidth=1, label='物理内存使用 (RSS)', alpha=0.8)
            ax.fill_between(times, rss_values, alpha=0.2, color='blue')
        
        # 不再绘制关键节点
        
        # 添加阶段背景色
        self._add_phase_backgrounds(ax, times, rss_values)
        
        # 不再添加阶段分隔线，背景颜色已能区分阶段
        
        # 设置图表属性
        ax.set_xlabel('运行时间 (秒)', fontsize=12)
        ax.set_ylabel('内存使用量 (MB)', fontsize=12)
        ax.set_title('Faiss程序运行期间内存使用情况', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # 设置坐标轴间距
        self._set_axis_intervals(ax, times, rss_values)
        
        # 添加各阶段峰值内存标注
        self._add_phase_peak_annotations(ax, times, rss_values)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"内存使用图表已保存到: {output_file}")
    
    def _add_phase_backgrounds(self, ax, times, rss_values):
        """添加阶段背景色"""
        if not self.phase_boundaries:
            return
        
        # 定义阶段颜色（加深颜色并添加斜条纹）
        phase_colors = {
            '训练': '#B3D9FF',      # 深蓝色
            '构建': '#B3FFB3',      # 深绿色
            '搜索': '#FFD9B3',      # 深橙色
            '其他': '#D9D9D9'       # 深灰色
        }
        
        # 记录已添加的阶段类型，避免重复图例
        added_phases = set()
        
        # 添加背景色
        for i, (time_point, phase) in enumerate(self.phase_boundaries):
            # 计算阶段结束时间
            if i + 1 < len(self.phase_boundaries):
                end_time = self.phase_boundaries[i + 1][0]
            else:
                end_time = max(times) if times else time_point + 1
            
            # 确定阶段类型
            phase_type = self._get_phase_type(phase)
            color = phase_colors.get(phase_type, '#F0F0F0')
            
            # 只在第一次遇到该阶段类型时添加图例
            label = f'{phase_type}阶段' if phase_type not in added_phases else ""
            if phase_type not in added_phases:
                added_phases.add(phase_type)
            
            # 添加带斜条纹的背景
            ax.axvspan(time_point, end_time, alpha=0.4, color=color, label=label, 
                      hatch='///' if phase_type != '其他' else None)
    
    def _get_phase_type(self, phase: str) -> str:
        """根据阶段名称确定阶段类型"""
        if '训练' in phase or 'train' in phase.lower():
            return '训练'
        elif '构建' in phase or '添加' in phase or 'build' in phase.lower():
            return '构建'
        elif '搜索' in phase or 'search' in phase.lower():
            return '搜索'
        else:
            return '其他'
    
    def _add_phase_peak_annotations(self, ax, times, rss_values):
        """添加各阶段峰值内存标注"""
        import numpy as np
        
        # 定义各阶段的关键词
        phase_keywords = {
            '训练': ['训练', 'train'],
            '构建': ['构建', '添加', 'build', 'add'],
            '搜索': ['搜索', 'search']
        }
        phase_colors = ['blue', 'green', 'orange']
        
        for phase, keywords in phase_keywords.items():
            # 找到该阶段的数据点
            phase_indices = []
            for i, record in enumerate(self.memory_records):
                phase_name = record.get('phase', '').lower()
                if any(keyword.lower() in phase_name for keyword in keywords):
                    phase_indices.append(i)
            
            if phase_indices:
                # 获取该阶段的时间和内存数据
                phase_times = [times[i] for i in phase_indices]
                phase_memories = [rss_values[i] for i in phase_indices]
                
                # 找到该阶段的峰值
                if phase_memories:
                    peak_idx = np.argmax(phase_memories)
                    peak_time = phase_times[peak_idx]
                    peak_memory = phase_memories[peak_idx]
                    
                    # 添加峰值标注
                    ax.annotate(f'{phase}峰值: {peak_memory:.1f}MB', 
                               xy=(peak_time, peak_memory), 
                               xytext=(peak_time + max(times)*0.05, peak_memory + max(rss_values)*0.05),
                               arrowprops=dict(arrowstyle='->', color=phase_colors[list(phase_keywords.keys()).index(phase)], lw=1.5),
                               fontsize=9, color=phase_colors[list(phase_keywords.keys()).index(phase)], fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor=phase_colors[list(phase_keywords.keys()).index(phase)]))
    
    def _set_axis_intervals(self, ax, times, rss_values):
        """设置坐标轴间距"""
        import matplotlib.ticker as ticker
        import numpy as np
        
        # 计算时间轴间距
        total_time = max(times) - min(times)
        if total_time > 0:
            # 时间精确到总用时的小一位数量级
            time_interval = 10 ** (int(np.log10(total_time)) - 1)
            # 确保最小间隔不小于0.1秒
            time_interval = max(time_interval, 0.1)
            # 设置x轴刻度：每10格标注一次
            ax.xaxis.set_major_locator(ticker.MultipleLocator(time_interval * 10))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(time_interval))
        
        # 计算内存轴间距
        max_memory = max(rss_values)
        if max_memory > 0:
            # 内存精确到峰值内存最高位四舍五入后的百分之一的数量级
            # 例如：峰值900MB -> 四舍五入到1000 -> 百分之一 -> 10MB
            rounded_peak = round(max_memory, -int(np.log10(max_memory)))
            memory_interval = rounded_peak / 100
            # 确保最小间隔不小于1MB
            memory_interval = max(memory_interval, 1.0)
            # 设置y轴刻度：每10格标注一次
            ax.yaxis.set_major_locator(ticker.MultipleLocator(memory_interval * 10))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(memory_interval))
        
        # 设置坐标轴从0开始
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    
    def get_memory_summary(self) -> Dict:
        """获取内存使用摘要"""
        if not self.memory_records:
            return {}
        
        rss_values = [record['rss_mb'] for record in self.memory_records]
        
        return {
            'peak_rss_mb': max(rss_values),
            'avg_rss_mb': sum(rss_values) / len(rss_values),
            'total_records': len(self.memory_records),
            'total_duration': self.memory_records[-1]['time'] if self.memory_records else 0
        }

# 创建全局内存监控器
memory_monitor = LightweightMemoryMonitor(
    enable_continuous=ENABLE_CONTINUOUS_MONITORING,
    interval=MONITORING_INTERVAL,
    change_threshold=MEMORY_CHANGE_THRESHOLD
)


# ==============================================================================
# 1. 辅助函数：读取.fbin文件
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
            data = np.fromfile(f, dtype=np.float32, count=nvecs*dim)
            data = data.reshape(nvecs, dim)
            return data
        else:
            # 读取指定块
            end_idx = min(start_idx + chunk_size, nvecs)
            num_vectors_in_chunk = end_idx - start_idx
            offset = start_idx * dim * 4  # 每个float32占4字节
            f.seek(offset, os.SEEK_CUR)
            data = np.fromfile(f, dtype=np.float32, count=num_vectors_in_chunk*dim)
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
efconstruction = 40 # 默认40
efsearch = 16       # 默认16

# ==============================================================================
# 【重构点】: 在索引文件名中同时体现 M 和 efConstruction 的值
# ==============================================================================
base_name = os.path.splitext(os.path.basename(BASE_FILE))[0]
# 清理文件名中的特殊字符
clean_base_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name)
# 在文件名中添加 M 和 efc 参数，以区分不同参数构建的索引
INDEX_FILE = os.path.join(DATA_DIR, f"{clean_base_name}_d{d_train}_nlist{nlist}_HNSWM{M}_efc{efconstruction}_IVFFlat.index")
# ==============================================================================

print("="*60)
print("Phase 0: 环境设置")
print(f"向量维度 (d): {d_train}")
print(f"基础集大小 (nb): {nb}, 训练集大小 (ntrain): {nt}")
print(f"查询集大小 (nq): {nq}, 分块大小 (chunk_size): {chunk_size}")
print(f"HNSW M (构建参数): {M}")
print(f"HNSW efConstruction (构建参数): {efconstruction}")
print(f"索引将保存在磁盘文件: {INDEX_FILE}")
print("="*60)

# 记录程序开始时的内存状态
memory_monitor.record_memory("程序开始")

# 启动连续监测
if ENABLE_CONTINUOUS_MONITORING:
    memory_monitor.start_continuous_monitoring()

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
    print("\nPhase 1: 训练 HNSW 粗量化器 (in-memory)")
    memory_monitor.record_memory("训练开始")
    
    coarse_quantizer = faiss.IndexHNSWFlat(d_train, M, faiss.METRIC_L2)
    coarse_quantizer.hnsw.efConstruction = efconstruction
    coarse_quantizer.hnsw.efSearch = efsearch
    print(f"efconstruction: {coarse_quantizer.hnsw.efConstruction}, efSearch: {coarse_quantizer.hnsw.efSearch}")
    # coarse_quantizer = faiss.IndexFlatL2(d_train)
    index_for_training = faiss.IndexIVFFlat(coarse_quantizer, d_train, nlist, faiss.METRIC_L2)
    index_for_training.verbose = True

    xt = read_fbin(LEARN_FILE)
    memory_monitor.record_memory("训练数据加载完成")

    print("训练聚类中心并构建 HNSW 量化器...")
    start_time = time.time()
    index_for_training.train(xt)
    end_time = time.time()

    print(f"量化器训练完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"粗量化器中的质心数量: {coarse_quantizer.ntotal}")
    memory_monitor.record_memory("训练完成")
    del xt
    del index_for_training

    # ==============================================================================
    # 5. 创建一个空的、基于磁盘的索引框架
    # ==============================================================================
    print("\nPhase 2: 创建空的磁盘索引框架")
    memory_monitor.record_memory("构建开始")
    index_shell = faiss.IndexIVFFlat(coarse_quantizer, d_train, nlist, faiss.METRIC_L2)
    print(f"将空的索引框架写入磁盘: {INDEX_FILE}")
    faiss.write_index(index_shell, INDEX_FILE)
    del index_shell

    # ==============================================================================
    # 6. 分块向磁盘索引中添加数据 (从base.fbin)
    # ==============================================================================
    print("\nPhase 3: 分块添加数据到磁盘索引")
    memory_monitor.record_memory("开始分块添加数据")

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
        print(f"       -> 正在处理块 {chunk_idx}/{num_chunks}: 向量 {i} 到 {min(i+chunk_size, nb)-1}")
        
        # 从base.fbin中读取数据块
        xb_chunk, _, _ = read_fbin(BASE_FILE, i, chunk_size)
        
        index_ondisk.add(xb_chunk)
        del xb_chunk

    print(f"\n所有数据块添加完成，总耗时: {time.time() - start_time:.2f} 秒")
    print(f"磁盘索引中的向量总数 (ntotal): {index_ondisk.ntotal}")
    memory_monitor.record_memory("分块添加完成")
    
    # 保存索引到磁盘
    print(f"正在将最终索引写回磁盘: {INDEX_FILE}")
    faiss.write_index(index_ondisk, INDEX_FILE)
    del index_ondisk
    memory_monitor.record_memory("构建完成")
    
    # 清理内存状态，模拟"干净"的搜索环境
    print("\n清理内存状态，模拟干净搜索环境...")
    import gc
    import sys
    
    # 记录清理前的内存状态
    process = psutil.Process()
    memory_before = process.memory_info().rss / (1024*1024)
    print(f"清理前内存使用: {memory_before:.2f} MB")
    
    # 清理所有可能残留的变量
    variables_to_clean = [
        'coarse_quantizer', 'index_shell', 'index_ondisk', 
        'index_for_training', 'xt', 'xb_chunk'
    ]
    
    for var_name in variables_to_clean:
        if var_name in locals():
            del locals()[var_name]
        if var_name in globals():
            del globals()[var_name]
    
    # 强制垃圾回收
    gc.collect()
    
    # 清理Python内部缓存
    if hasattr(sys, '_clear_type_cache'):
        sys._clear_type_cache()
    
    # 清理更多Python内部结构
    try:
        import ctypes
        # 尝试清理内存碎片
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass
    
    # 再次强制垃圾回收
    gc.collect()
    
    # 设置垃圾回收阈值，强制更频繁的回收
    old_thresholds = gc.get_threshold()
    gc.set_threshold(0, 0, 0)  # 立即回收
    gc.collect()
    gc.set_threshold(*old_thresholds)  # 恢复原设置
    
    # 记录清理后的内存状态
    memory_after = process.memory_info().rss / (1024*1024)
    print(f"清理后内存使用: {memory_after:.2f} MB")
    print(f"释放内存: {memory_before - memory_after:.2f} MB")
    
    memory_monitor.record_memory("内存清理完成")
    print("内存清理完成，准备进入搜索阶段")

# ==============================================================================
# 8. 使用内存映射 (mmap) 进行搜索 (使用query.fbin)
# ==============================================================================
print("\nPhase 4: 使用内存映射模式进行搜索")

# 如果是首次运行（包含构建），进行最终的内存清理
if not skip_index_building:
    print("首次运行：进行最终内存清理...")
    import gc
    import sys
    import os
    
    # 记录清理前的内存状态
    process = psutil.Process()
    memory_before_final = process.memory_info().rss / (1024*1024)
    print(f"最终清理前内存使用: {memory_before_final:.2f} MB")
    
    # 最终清理
    gc.collect()
    if hasattr(sys, '_clear_type_cache'):
        sys._clear_type_cache()
    
    # 尝试清理系统内存碎片
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass
    
    # 强制垃圾回收
    gc.collect()
    
    # 尝试释放更多内存
    try:
        # 清理Python内部结构
        if hasattr(sys, 'intern'):
            sys.intern.__dict__.clear()
    except:
        pass
    
    # 记录清理后的内存状态
    memory_after_final = process.memory_info().rss / (1024*1024)
    print(f"最终清理后内存使用: {memory_after_final:.2f} MB")
    print(f"最终释放内存: {memory_before_final - memory_after_final:.2f} MB")
    print("最终内存清理完成")

print(f"以 mmap 模式打开磁盘索引: {INDEX_FILE}")
memory_monitor.record_memory("搜索准备")

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

print("从 query.fbin 加载查询向量...")
xq = read_fbin(QUERY_FILE)
memory_monitor.record_memory("查询数据加载完成")

print("\n执行搜索...")
memory_monitor.record_memory("搜索开始")
start_time = time.time()
D, I = index_final.search(xq, k)
end_time = time.time()
memory_monitor.record_memory("搜索完成")

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
print("\n" + "="*60)
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

print("="*60)


# ==============================================================================
# 10. 报告峰值内存和生成图表
# ==============================================================================
print("\n" + "="*60)
print("内存使用情况报告")
print("="*60)

# 传统方法（resource.getrusage）
if platform.system() in ["Linux", "Darwin"]:
    peak_memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Linux":
        peak_memory_bytes *= 1024
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    print(f"传统方法 - 整个程序运行期间的峰值内存占用: {peak_memory_mb:.2f} MB")

# 轻量级监控方法
if ENABLE_MEMORY_MONITORING:
    print("\n轻量级监控方法:")
    summary = memory_monitor.get_memory_summary()
    if summary:
        print(f"  峰值RSS内存: {summary['peak_rss_mb']:.2f} MB")
        print(f"  平均RSS内存: {summary['avg_rss_mb']:.2f} MB")
        print(f"  总监控记录数: {summary['total_records']}")
        print(f"  总运行时长: {summary['total_duration']:.2f} 秒")
    
    # 停止连续监测
    if ENABLE_CONTINUOUS_MONITORING:
        memory_monitor.stop_continuous_monitoring()
    
    # 生成内存使用图表
    print("\n生成内存使用图表...")
    memory_monitor.generate_memory_plot(MEMORY_PLOT_FILENAME)
else:
    print("内存监控已禁用")

print("="*60)