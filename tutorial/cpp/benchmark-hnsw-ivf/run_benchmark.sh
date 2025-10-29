#!/bin/bash

# HNSW-IVF Benchmark 运行脚本

echo "=== HNSW-IVF Benchmark (内存优化版) ==="
echo ""

# 检查是否存在配置文件
if [ ! -f "benchmark.config" ]; then
    echo "错误: 找不到配置文件 benchmark.config"
    exit 1
fi

# 检查是否已编译
if [ ! -f "benchmark_hnsw_ivf" ]; then
    echo "程序未编译，开始编译..."
    make
    if [ $? -ne 0 ]; then
        echo "编译失败!"
        exit 1
    fi
    echo "编译成功!"
    echo ""
fi

# 检查数据集是否存在
DATA_DIR="../sift"
if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 找不到数据集目录 $DATA_DIR"
    exit 1
fi

if [ ! -f "$DATA_DIR/learn.fbin" ]; then
    echo "错误: 找不到训练数据 $DATA_DIR/learn.fbin"
    exit 1
fi

if [ ! -f "$DATA_DIR/base.fbin" ]; then
    echo "错误: 找不到基础数据 $DATA_DIR/base.fbin"
    exit 1
fi

if [ ! -f "$DATA_DIR/query.fbin" ]; then
    echo "错误: 找不到查询数据 $DATA_DIR/query.fbin"
    exit 1
fi

# 创建索引目录
mkdir -p indices

# 运行benchmark
echo "开始运行 benchmark..."
echo ""

./benchmark_hnsw_ivf benchmark.config

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Benchmark 完成 ==="
    echo ""
    echo "结果文件："
    ls -lh benchmark_*.csv 2>/dev/null
    echo ""
    echo "索引文件："
    ls -lh indices/*.index 2>/dev/null
else
    echo ""
    echo "Benchmark 运行失败!"
    exit 1
fi

