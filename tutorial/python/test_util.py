'''
Author: Quaternijkon quaternijkon@mail.ustc.edu.cn
Date: 2025-07-17 03:30:18
LastEditors: Quaternijkon quaternijkon@mail.ustc.edu.cn
LastEditTime: 2025-07-17 03:30:36
FilePath: /faiss/tutorial/python/test_util.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import struct
import numpy as np
import platform
import resource

# ==============================================================================
# 文件读写工具
# ==============================================================================

def read_fbin(filename):
    """
    读取.fbin格式的文件 (一次性读入内存)
    格式: [nvecs: int32, dim: int32, data: float32[nvecs*dim]]
    """
    if not os.path.exists(filename):
        print(f"错误: 文件未找到 {filename}")
        exit(1)
        
    with open(filename, 'rb') as f:
        nvecs, dim = struct.unpack('ii', f.read(8))
        data = np.fromfile(f, dtype=np.float32, count=nvecs * dim).reshape(nvecs, dim)
    return data, nvecs, dim

def read_ivecs(filename):
    """
    读取.ivecs格式的二进制文件 (一次性读入内存)
    格式: 向量循环 [dim: int32, data: int32[dim]]
    """
    if not os.path.exists(filename):
        print(f"错误: 文件未找到 {filename}")
        exit(1)

    a = np.fromfile(filename, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

# ==============================================================================
# 性能评估工具
# ==============================================================================

def calculate_recall(search_results_I, groundtruth_path, k):
    """
    计算并返回召回率。

    返回:
    float: Recall@k 的值，如果文件不存在则返回 -1.0。
    """
    print("\n" + "="*60); print("Phase: 计算召回率")
    if search_results_I.shape[1] != k:
        raise ValueError(f"搜索结果的列数 ({search_results_I.shape[1]}) 与指定的 k ({k}) 不匹配。")

    print(f"  -> 正在从 {groundtruth_path} 加载Groundtruth数据...")
    if not os.path.exists(groundtruth_path):
        print(f"  -> 错误: Groundtruth文件未找到!"); return -1.0
        
    ground_truth_I = read_ivecs(groundtruth_path)
    nq = search_results_I.shape[0]
    
    if nq != ground_truth_I.shape[0]:
        print(f"  -> 警告: 结果数({nq})与GT数({ground_truth_I.shape[0]})不匹配!")
        min_nq = min(nq, ground_truth_I.shape[0])
        search_results_I, ground_truth_I, nq = search_results_I[:min_nq], ground_truth_I[:min_nq], min_nq

    gt_k = ground_truth_I[:, :k]
    found_count = sum(len(set(search_results_I[i]).intersection(set(gt_k[i]))) for i in range(nq))
    total_possible = nq * k
    recall = found_count / total_possible if total_possible > 0 else 0
    
    print(f"\n查询了 {nq} 个向量, k={k}")
    print(f"在top-{k}的结果中，总共找到了 {found_count} 个真实的近邻。")
    return recall

def report_peak_memory():
    """报告程序运行期间的峰值内存占用。"""
    print("\n" + "="*60); print("Phase: 性能报告")
    if platform.system() in ["Linux", "Darwin"]:
        peak_memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Linux": peak_memory_bytes *= 1024
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        print(f"整个程序运行期间的峰值内存占用: {peak_memory_mb:.2f} MB")
    else:
        print("无法在当前操作系统上自动获取峰值内存。")
    print("="*60)