#!/bin/bash
# 运行 2-IVFFlat 程序的脚本
# 设置 OpenBLAS 线程数以避免内存分配错误

# 设置 OpenBLAS 使用单线程（避免内存分配问题）
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

# 运行程序
./2-IVFFlat

