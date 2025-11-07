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
            overall_peak = max(overall_peak, info['peak_memory'])
        
        print("-" * 95)
        print(f"各阶段峰值内存: {overall_peak:.2f} MB")
        print(f"{'='*60}\n")


#################################################################
# Small I/O functions
#################################################################


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


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
    # add 1/4 of the database to 4 independent indexes
    bno = stage - 1
    xb = fvecs_read("sift1M/sift_base.fvecs")
    monitor.update_peak()
    i0, i1 = int(bno * xb.shape[0] / 4), int((bno + 1) * xb.shape[0] / 4)
    index = faiss.read_index(tmpdir + "trained.index")
    monitor.update_peak()
    print("adding vectors %d:%d" % (i0, i1))
    index.add_with_ids(xb[i0:i1], np.arange(i0, i1))
    monitor.update_peak()
    print("write " + tmpdir + "block_%d.index" % bno)
    faiss.write_index(index, tmpdir + "block_%d.index" % bno)
    monitor.update_peak()
    # Clean up
    del xb, index
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

# Print summary
monitor.print_summary()

