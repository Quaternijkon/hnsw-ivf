#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import numpy as np
import faiss
from faiss.contrib.ondisk import merge_ondisk
import time
import gc
import os # <-- 需要导入 os 模块

#################################################################
# Memory Monitoring
#################################################################

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    import resource


class MemoryMonitor:
    """Simple memory monitor for tracking memory usage across stages"""
    
    def __init__(self):
        self.stages = {}
        self.current_stage = None
        self.stage_start_memory = None
        self.stage_start_time = None
        self.peak_memory = None
        
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
        else:
            print("[警告] psutil 不可用，将使用 resource 模块（精度较低）")
            # Check if we're on Linux (KB) or macOS (bytes)
            import platform
            self.is_linux = platform.system() == 'Linux'
    
    def get_memory_mb(self):
        """Get current memory usage in MB"""
        if PSUTIL_AVAILABLE:
            return self.process.memory_info().rss / (1024 * 1024)
        else:
            # Use resource module as fallback
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux returns KB, macOS returns bytes
            if self.is_linux:
                return rss / 1024
            else:
                return rss / (1024 * 1024)
    
    def update_peak(self):
        """Update peak memory for current stage"""
        current = self.get_memory_mb()
        if self.peak_memory is None or current > self.peak_memory:
            self.peak_memory = current
    
    def start_stage(self, stage_name):
        """Start monitoring a stage"""
        # Force garbage collection before starting new stage
        gc.collect()
        
        self.current_stage = stage_name
        self.stage_start_memory = self.get_memory_mb()
        self.stage_start_time = time.time()
        self.peak_memory = self.stage_start_memory
        
        print(f"\n{'='*60}")
        print(f"开始阶段: {stage_name}")
        print(f"初始内存: {self.stage_start_memory:.2f} MB")
        print(f"{'='*60}")
    
    def end_stage(self):
        """End monitoring current stage"""
        if self.current_stage is None:
            return
        
        # Update peak one more time before GC
        self.update_peak()
        
        # Force garbage collection before measuring
        gc.collect()
        
        end_memory = self.get_memory_mb()
        elapsed_time = time.time() - self.stage_start_time
        
        memory_delta = end_memory - self.stage_start_memory
        peak_delta = self.peak_memory - self.stage_start_memory
        
        self.stages[self.current_stage] = {
            'start_memory': self.stage_start_memory,
            'end_memory': end_memory,
            'peak_memory': self.peak_memory,
            'delta_memory': memory_delta,
            'peak_delta': peak_delta,
            'elapsed_time': elapsed_time
        }
        
        print(f"\n{'='*60}")
        print(f"阶段完成: {self.current_stage}")
        print(f"初始内存: {self.stage_start_memory:.2f} MB")
        print(f"结束内存: {end_memory:.2f} MB")
        print(f"峰值内存: {self.peak_memory:.2f} MB")
        print(f"内存增长: {memory_delta:+.2f} MB")
        print(f"峰值增长: {peak_delta:+.2f} MB")
        print(f"耗时: {elapsed_time:.2f} 秒")
        print(f"{'='*60}\n")
        
        self.current_stage = None
        self.peak_memory = None
    
    def print_summary(self):
        """Print summary of all stages"""
        print(f"\n{'='*60}")
        print("内存使用总结")
        print(f"{'='*60}")
        print(f"{'阶段':<25} {'初始(MB)':<12} {'结束(MB)':<12} {'峰值(MB)':<12} {'增长(MB)':<12} {'耗时(秒)':<10}")
        print("-" * 95)
        
        overall_peak = 0
        for stage_name, info in self.stages.items():
            print(f"{stage_name:<25} {info['start_memory']:<12.2f} {info['end_memory']:<12.2f} "
                  f"{info['peak_memory']:<12.2f} {info['delta_memory']:<12.2f} {info['elapsed_time']:<10.2f}")
            if info['peak_memory'] > overall_peak:
                overall_peak = info['peak_memory']
        
        print("-" * 95)
        print(f"各阶段峰值内存: {overall_peak:.2f} MB")
        print(f"{'='*60}\n")


#################################################################
# Small I/O functions
#################################################################


def ivecs_read(fname):
    """ 读取完整的 .ivecs 文件 """
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    """ 读取完整的 .fvecs 文件 """
    return ivecs_read(fname).view('float32')

# ----------------- 新增的辅助函数 -----------------

def get_fvecs_shape(fname):
    """
    不加载数据，只读取文件大小和第一个整数来获取向量形状 (n, d)
    """
    fsize = os.path.getsize(fname)
    # 读取第一个 int32 (4 字节) 来获取维度 d
    d_array = np.memmap(fname, dtype='int32', mode='r', shape=(1,))
    d = int(d_array[0])
    del d_array # 释放 memmap

    # 每个向量 = 1 个 int32 (维度) + d 个 float32 (向量)
    # (1 + d) * 4 字节
    vector_size_bytes = (d + 1) * 4
    
    if fsize % vector_size_bytes != 0:
        raise IOError(f"文件大小 {fsize} 不是向量大小 {vector_size_bytes} 的整数倍")
        
    n = fsize // vector_size_bytes
    return n, d


def fvecs_read_chunk(fname, i0, i1):
    """
    使用 memmap 高效读取 .fvecs 文件的 [i0, i1) 分块
    """
    n, d = get_fvecs_shape(fname)
    
    if i0 < 0 or i1 > n or i0 >= i1:
        raise ValueError(f"请求的范围 [{i0}, {i1}) 超出总范围 [0, {n})")

    vector_size_elements = d + 1  # d 个 float32 + 1 个 int32
    vector_size_bytes = vector_size_elements * 4
    
    num_vectors_to_read = i1 - i0
    shape_to_read = (num_vectors_to_read, vector_size_elements)
    
    # 计算起始字节偏移量
    start_byte_offset = i0 * vector_size_bytes
    
    # 使用 memmap 映射文件的特定部分
    chunk_data_int32 = np.memmap(
        fname,
        dtype='int32',
        mode='r',
        offset=start_byte_offset,
        shape=shape_to_read
    )
    
    # 从 memmap 中复制所需数据到内存
    # [:, 1:] 切片操作去掉每行开头的维度 d
    chunk_vectors = chunk_data_int32[:, 1:].copy()
    del chunk_data_int32 # 释放 memmap
    
    # 将 int32 视图转换为 float32
    return chunk_vectors.view('float32')

# ----------------- 辅助函数结束 -----------------


#################################################################
# Main program
#################################################################

stage = int(sys.argv[1])
tmpdir = '/tmp/'

# Initialize memory monitor
monitor = MemoryMonitor()

if stage == 0:
    monitor.start_stage("Stage 0: 训练索引")
    # train the index
    xt = fvecs_read("sift1M/sift_learn.fvecs")
    monitor.update_peak()
    index = faiss.index_factory(xt.shape[1], "IVF4096,Flat")
    print("training index")
    index.train(xt)
    monitor.update_peak()
    print("write " + tmpdir + "trained.index")
    faiss.write_index(index, tmpdir + "trained.index")
    monitor.update_peak()
    # Clean up
    del xt, index
    monitor.end_stage()


if 1 <= stage <= 4:
    monitor.start_stage(f"Stage {stage}: 添加向量块 {stage-1}")
    
    bno = stage - 1
    db_fname = "sift1M/sift_base.fvecs"
    
    # --- 这是关键的修复 ---
    # 1. 获取数据库形状，而不加载它
    n_total, d = get_fvecs_shape(db_fname)
    monitor.update_peak()

    # 2. 计算当前块的边界
    i0 = int(bno * n_total / 4)
    i1 = int((bno + 1) * n_total / 4)

    # 3. 只从磁盘读取需要的分块！
    print(f"从 {db_fname} 读取分块 {bno} (向量 {i0}:{i1})...")
    xb_chunk = fvecs_read_chunk(db_fname, i0, i1)
    print(f"分块加载完毕，形状: {xb_chunk.shape}")
    monitor.update_peak()
    # --- 修复结束 ---
    
    index = faiss.read_index(tmpdir + "trained.index")
    monitor.update_peak()
    
    print("adding vectors %d:%d" % (i0, i1))
    # 4. 添加分块数据
    index.add_with_ids(xb_chunk, np.arange(i0, i1))
    monitor.update_peak()
    
    print("write " + tmpdir + "block_%d.index" % bno)
    faiss.write_index(index, tmpdir + "block_%d.index" % bno)
    monitor.update_peak()
    
    # Clean up
    del xb_chunk, index # <-- 现在只删除内存中的分块
    monitor.end_stage()

if stage == 5:
    monitor.start_stage("Stage 5: 合并磁盘索引")
    print('loading trained index')
    # construct the output index
    index = faiss.read_index(tmpdir + "trained.index")
    monitor.update_peak()

    block_fnames = [
        tmpdir + "block_%d.index" % bno
        for bno in range(4)
    ]

    merge_ondisk(index, block_fnames, tmpdir + "merged_index.ivfdata")
    monitor.update_peak()

    print("write " + tmpdir + "populated.index")
    faiss.write_index(index, tmpdir + "populated.index")
    monitor.update_peak()
    # Clean up
    del index
    monitor.end_stage()


if stage == 6:
    monitor.start_stage("Stage 6: 搜索")
    # perform a search from disk
    print("read " + tmpdir + "populated.index")
    index = faiss.read_index(tmpdir + "populated.index")
    monitor.update_peak()
    index.nprobe = 16

    # load query vectors and ground-truth
    xq = fvecs_read("sift1M/sift_query.fvecs")
    monitor.update_peak()
    gt = ivecs_read("sift1M/sift_groundtruth.ivecs")
    monitor.update_peak()

    D, I = index.search(xq, 5)
    monitor.update_peak()

    recall_at_1 = (I[:, :1] == gt[:, :1]).sum() / float(xq.shape[0])
    print("recall@1: %.3f" % recall_at_1)
    # Clean up
    del xq, gt, D, I, index
    monitor.end_stage()

# 在所有阶段都执行完毕后，打印一个总结
# 注意：你需要手动运行所有 7 个阶段才能收集到所有数据
# 这里只打印当前运行阶段的数据
monitor.print_summary()