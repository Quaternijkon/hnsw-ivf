# ==============================================================================
# 导入: 新增 threading
# ==============================================================================
import numpy as np
import faiss
import time
import os
import platform
import resource
import struct
import re
import threading

# ==============================================================================
# [新] 多线程实时内存监控器
# ==============================================================================
# --- 1. 开关配置 ---
USE_PSUTIL = True         # 开关：是否使用 psutil 库来监控
USE_PROC_FS = True        # 开关：是否使用读取 /proc 文件系统的方式来监控
# 注意: 绘图开关现在由 MemoryMonitor 的构造函数控制

# --- 2. 模块导入与检查 (与之前相同) ---
PSUTIL_AVAILABLE = False
# ... (省略检查代码, 已包含在下方类中) ...
MATPLOTLIB_AVAILABLE = False
# ... (省略检查代码, 已包含在下方类中) ...


# --- 3. 重构后的 MemoryMonitor 类 ---
class MemoryMonitor:
    """
    一个使用后台线程进行高频实时内存监控的类。
    通过 'with' 语句进行管理，确保监控的自动启停。
    """
    def __init__(self, interval=0.1, plot=True, use_psutil=True, use_proc_fs=True):
        self.interval = interval  # 检测频率（秒）
        self.plot_enabled = plot
        self.use_psutil = use_psutil
        self.use_proc_fs = use_proc_fs
        
        # 检查依赖
        self._check_dependencies()

        # 只有在 psutil 可用时才创建进程对象
        if self.use_psutil and PSUTIL_AVAILABLE:
            import psutil
            self.process = psutil.Process(os.getpid())
        else:
            self.process = None
        
        # 存储数据
        self.timestamps = []
        self.psutil_data = []
        self.procfs_data = []

        # 线程控制
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self._start_time = None

    def _check_dependencies(self):
        global PSUTIL_AVAILABLE, MATPLOTLIB_AVAILABLE
        if self.use_psutil and not PSUTIL_AVAILABLE:
            try:
                import psutil
                PSUTIL_AVAILABLE = True
                self.use_psutil = True
            except ImportError:
                print("警告: 'psutil' 未安装，其监控功能已禁用。请运行 'pip install psutil'。")
                self.use_psutil = False
        
        # 确保 psutil 在全局范围内可用
        if self.use_psutil and PSUTIL_AVAILABLE:
            import psutil
        
        if self.plot_enabled and not MATPLOTLIB_AVAILABLE:
            try:
                import matplotlib.pyplot as plt
                MATPLOTLIB_AVAILABLE = True
                self.plot_enabled = True
            except ImportError:
                print("警告: 'matplotlib' 未安装，绘图功能已禁用。请运行 'pip install matplotlib'。")
                self.plot_enabled = False

    def _get_memory_psutil(self):
        return self.process.memory_info().rss if self.process else 0

    def _get_memory_procfs(self):
        try:
            with open('/proc/self/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) * 1024
        except:
            return 0
            
    def _worker(self):
        """后台线程执行的函数"""
        while not self._stop_event.is_set():
            elapsed_time = time.time() - self._start_time
            self.timestamps.append(elapsed_time)

            if self.use_psutil:
                self.psutil_data.append(self._get_memory_psutil() / 1024 / 1024) # MB
            if self.use_proc_fs:
                self.procfs_data.append(self._get_memory_procfs() / 1024 / 1024) # MB
            
            time.sleep(self.interval)

    def start(self):
        print("=" * 20, "实时内存监控已启动", "=" * 20)
        print(f"检测频率: {self.interval} 秒")
        self._start_time = time.time()
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._worker, daemon=True)
        self._monitor_thread.start()

    def stop(self):
        self._stop_event.set()
        self._monitor_thread.join()
        print("=" * 20, "实时内存监控已停止", "=" * 20)
        if self.plot_enabled:
            self.plot()

    def plot(self):
        if not self.timestamps:
            print("[内存监控] 没有记录到任何数据点，无法生成图表。")
            return
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 7))
        
        if self.psutil_data:
            plt.plot(self.timestamps, self.psutil_data, label='psutil Method', linestyle='-')
        if self.procfs_data:
            plt.plot(self.timestamps, self.procfs_data, label='/proc FS Method', linestyle='--')

        plt.title('Real-time Memory Usage During Faiss Workflow')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (MB)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        filename = 'faiss_realtime_memory.png'
        plt.savefig(filename)
        print(f"\n[内存监控] 实时曲线图已生成并保存为 '{filename}'")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ==============================================================================
# Faiss 脚本主体 (几乎无需改动)
# ==============================================================================

# 0. 路径配置
DATA_DIR = "./sift"
# ... (省略其他路径配置, 与之前相同) ...
LEARN_FILE = os.path.join(DATA_DIR, "learn.fbin")
BASE_FILE = os.path.join(DATA_DIR, "base.fbin")
QUERY_FILE = os.path.join(DATA_DIR, "query.fbin")
GROUNDTRUTH_FILE = os.path.join(DATA_DIR, "groundtruth.ivecs")

# 1. 辅助函数
# ... (read_fbin, read_ivecs 函数与之前相同) ...
def read_fbin(filename, start_idx=0, chunk_size=None):
    with open(filename, 'rb') as f:
        nvecs, dim = struct.unpack('ii', f.read(8))
        if chunk_size is None:
            data = np.fromfile(f, dtype=np.float32, count=nvecs * dim)
            return data.reshape(nvecs, dim)
        else:
            f.seek(8 + start_idx * dim * 4)
            num_vectors_in_chunk = min(chunk_size, nvecs - start_idx)
            data = np.fromfile(f, dtype=np.float32, count=num_vectors_in_chunk * dim)
            return data.reshape(num_vectors_in_chunk, dim), nvecs, dim

# --- 主程序入口 ---
if __name__ == "__main__":
    # 使用 'with' 语句来自动管理监控的启停
    # 您可以在这里调整检测频率，例如 interval=0.05 会更平滑但开销稍大
    with MemoryMonitor(interval=0.1, plot=True, use_psutil=USE_PSUTIL, use_proc_fs=USE_PROC_FS):
        
        # =================================================================
        # 您所有的 Faiss 核心代码都放在这个 with 代码块内
        # =================================================================
        
        # 2. 设置参数与环境
        _, nt, d_train = read_fbin(LEARN_FILE, chunk_size=1)
        _, nb, d_base = read_fbin(BASE_FILE, chunk_size=1)
        _, nq, d_query = read_fbin(QUERY_FILE, chunk_size=1)

        # ... (省略大量参数设置和打印代码，与之前完全相同) ...
        nlist = nb // 1000
        chunk_size = 100000
        k = 10
        M = 32
        efconstruction = 40
        efsearch = 16
        nprobe = 32
        
        INDEX_FILE = os.path.join(DATA_DIR, f"sift_base_d{d_train}_nlist{nlist}_HNSWM{M}_efc{efconstruction}.index")
        print("="*60)
        print("Phase 0: 环境设置")
        print(f"索引文件: {INDEX_FILE}")
        print("="*60)
        
        # 3. 检查索引文件
        skip_index_building = os.path.exists(INDEX_FILE)

        # 4. 训练
        if not skip_index_building:
            print("\nPhase 1: 训练 HNSW 粗量化器")
            coarse_quantizer = faiss.IndexHNSWFlat(d_train, M, faiss.METRIC_L2)
            # ... (其他训练代码与之前相同) ...
            index_for_training = faiss.IndexIVFFlat(coarse_quantizer, d_train, nlist, faiss.METRIC_L2)
            xt = read_fbin(LEARN_FILE)
            print("训练开始...")
            index_for_training.train(xt)
            print("训练结束。")
            del xt
            
            # 5. 创建空索引
            print("\nPhase 2: 创建空的磁盘索引框架")
            faiss.write_index(index_for_training, INDEX_FILE)
            del index_for_training
            
            # 6. 分块添加数据
            print("\nPhase 3: 分块添加数据到磁盘索引")
            
            # 兼容不同Faiss版本的IO标志处理
            try:
                IO_FLAG_READ_WRITE = faiss.IO_FLAG_READ_WRITE
            except AttributeError:
                try:
                    IO_FLAG_READ_WRITE = faiss.index_io.IO_FLAG_READ_WRITE
                except AttributeError:
                    IO_FLAG_READ_WRITE = 0  # 默认值，表示读写模式
            
            index_ondisk = faiss.read_index(INDEX_FILE, IO_FLAG_READ_WRITE)
            for i in range(0, nb, chunk_size):
                xb_chunk, _, _ = read_fbin(BASE_FILE, i, chunk_size)
                index_ondisk.add(xb_chunk)
                del xb_chunk
            faiss.write_index(index_ondisk, INDEX_FILE)
            del index_ondisk
            
        # 8. 搜索
        print("\nPhase 4: 使用内存映射模式进行搜索")
        
        # 兼容不同Faiss版本的IO标志处理
        try:
            IO_FLAG_MMAP = faiss.IO_FLAG_MMAP
        except AttributeError:
            try:
                IO_FLAG_MMAP = faiss.index_io.IO_FLAG_MMAP
            except AttributeError:
                IO_FLAG_MMAP = 0x646f0000  # 默认的MMAP标志值
        
        index_final = faiss.read_index(INDEX_FILE, IO_FLAG_MMAP)
        # ... (其他搜索设置与之前相同) ...
        index_final.nprobe = nprobe
        xq = read_fbin(QUERY_FILE)
        print("搜索开始...")
        D, I = index_final.search(xq, k)
        print("搜索结束。")
        
        # 9. 召回率计算 (如果需要)
        # ...

    # 'with' 代码块结束，监控自动停止，图表自动生成
    
    # 10. 原始的峰值内存报告 (仍然可以保留作为参考)
    print("\n" + "="*60)
    # ... (与之前相同的峰值内存报告代码) ...
    if platform.system() in ["Linux", "Darwin"]:
        peak_memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Linux": peak_memory_bytes *= 1024
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        print(f"整个程序运行期间的峰值内存占用 (来自resource): {peak_memory_mb:.2f} MB")
    print("="*60)