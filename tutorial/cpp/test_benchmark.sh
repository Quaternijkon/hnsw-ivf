#!/bin/bash

# 测试benchmark程序的脚本

echo "=== Faiss Benchmark测试脚本 ==="

# 检查必要的数据文件是否存在
echo "检查数据文件..."
DATA_DIR="./sift"
REQUIRED_FILES=("learn.fbin" "base.fbin" "query.fbin")

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$DATA_DIR/$file" ]; then
        echo "错误: 缺少数据文件 $DATA_DIR/$file"
        echo "请确保数据文件存在于 $DATA_DIR/ 目录下"
        exit 1
    fi
done

echo "所有必要的数据文件都存在"

# 检查可执行文件
if [ ! -f "./benchmark_advanced" ]; then
    echo "错误: 可执行文件 benchmark_advanced 不存在"
    echo "请先运行 'make benchmark_advanced' 编译程序"
    exit 1
fi

echo "可执行文件存在"

# 创建测试用的配置文件（较小的参数集）
echo "创建测试配置文件..."
cat > test_benchmark.config << EOF
build
  param
    nlist:2000,4000,6000,8000,10000,14000,18000,22000,26000,30000,35000,40000
    efconstruction:10,20,40,100,200
  metric
    training_memory
    add_memory
    training_time
    total_time
search
  param
    nprobe_ratio:0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.012,0.014,0.016,0.018,0.02,0.025,0.03,0.035,0.04,0.045,0.05
    efsearch_ratio:0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.4,1.6,1.8,2.0
  metric
    recall
    QPS
    mSPQ
    search_memory
    search_time
    mean_latency
    P50_latency
    P99_latency
EOF

echo "测试配置文件已创建"

# 运行测试
echo "开始运行benchmark测试..."
echo "注意: 这可能需要几分钟时间，请耐心等待..."

./benchmark_advanced test_benchmark.config

# 检查结果文件
if [ $? -eq 0 ]; then
    echo "测试完成！"
    
    # 查找生成的CSV文件
    CSV_FILES=$(ls benchmark_results_*.csv 2>/dev/null)
    if [ -n "$CSV_FILES" ]; then
        echo "结果文件:"
        ls -la benchmark_results_*.csv
        echo ""
        echo "CSV文件内容预览:"
        head -5 benchmark_results_*.csv
    else
        echo "警告: 未找到结果CSV文件"
    fi
else
    echo "测试失败，退出码: $?"
fi

# 清理测试配置文件
rm -f test_benchmark.config

echo "测试脚本执行完成"
