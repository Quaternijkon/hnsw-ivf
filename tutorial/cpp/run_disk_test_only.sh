#!/bin/bash

# 仅运行普通磁盘测试（不需要sudo权限）

echo "=== Faiss 磁盘性能测试（无需sudo） ==="
echo ""

# 检查必要文件
if [ ! -f "./benchmark-thread" ]; then
    echo "❌ 错误: 可执行文件 benchmark-thread 不存在"
    echo "正在编译..."
    g++ -std=c++17 -O3 -o benchmark-thread benchmark-thread.cpp \
        -I ../.. -L ../../build/faiss \
        -Wl,-rpath,../../build/faiss -lfaiss -lopenblas -fopenmp
    
    if [ $? -ne 0 ]; then
        echo "❌ 编译失败"
        exit 1
    fi
    echo "✅ 编译成功"
fi

if [ ! -d "./sift" ]; then
    echo "❌ 错误: 数据目录 ./sift 不存在"
    exit 1
fi

echo "✅ 所有必要文件都存在"
echo ""

# 显示系统信息
echo "=== 系统信息 ==="
echo "CPU核心数: $(nproc)"
echo "可用内存: $(free -h | grep Mem | awk '{print $7}')"
echo "数据目录大小: $(du -sh ./sift | cut -f1)"
echo ""

# 运行测试
echo "=== 开始性能测试 ==="
echo "这可能需要几分钟时间..."
echo ""

./benchmark-thread 2>&1 | tee disk_performance_test.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 测试完成！"
    echo "=========================================="
    echo ""
    
    if [ -f "benchmark_results.txt" ]; then
        echo "📊 性能结果预览:"
        echo "----------------------------------------"
        head -5 benchmark_results.txt
        echo "..."
        echo "----------------------------------------"
        echo ""
        echo "完整结果已保存到以下文件:"
        echo "  - benchmark_results.txt (CSV格式)"
        echo "  - disk_performance_test.log (详细日志)"
    fi
else
    echo ""
    echo "❌ 测试失败，请检查 disk_performance_test.log 查看错误详情"
    exit 1
fi

echo ""
echo "💡 提示: 如需对比内存磁盘性能，请运行: ./simple_disk_memory_test.sh"

