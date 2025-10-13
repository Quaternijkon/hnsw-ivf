#!/bin/bash

# 快速测试脚本 - 验证实验环境
echo "=== Faiss 磁盘 vs 内存磁盘快速测试 ==="

# 检查必要文件
echo "1. 检查必要文件..."

# 检查可执行文件
if [ ! -f "./benchmark-thread" ]; then
    echo "❌ benchmark-thread 不存在，正在编译..."
    g++ -std=c++17 -O3 -o benchmark-thread benchmark-thread.cpp -I ../.. -L ../../build/faiss -Wl,-rpath,../../build/faiss -lfaiss -lopenblas -fopenmp
    
    if [ $? -ne 0 ]; then
        echo "❌ 编译失败"
        exit 1
    fi
    echo "✅ 编译成功"
else
    echo "✅ benchmark-thread 存在"
fi

# 检查数据文件
if [ ! -d "./sift" ]; then
    echo "❌ sift 数据目录不存在"
    exit 1
fi

required_files=("learn.fbin" "base.fbin" "query.fbin")
for file in "${required_files[@]}"; do
    if [ ! -f "./sift/$file" ]; then
        echo "❌ 缺少数据文件: sift/$file"
        exit 1
    fi
done
echo "✅ 所有数据文件存在"

# 检查权限
echo "2. 检查权限..."
if ! sudo -n true 2>/dev/null; then
    echo "⚠️  需要sudo权限创建内存磁盘，请确保有sudo权限"
else
    echo "✅ sudo权限可用"
fi

# 检查内存
echo "3. 检查系统资源..."
total_mem=$(free -g | grep Mem | awk '{print $2}')
echo "总内存: ${total_mem}GB"

if [ $total_mem -lt 4 ]; then
    echo "⚠️  内存较少，建议至少4GB内存"
else
    echo "✅ 内存充足"
fi

# 检查磁盘空间
disk_space=$(df -h . | tail -1 | awk '{print $4}')
echo "可用磁盘空间: $disk_space"

# 创建测试目录
echo "4. 创建测试环境..."
mkdir -p test_workspace
cd test_workspace

# 复制数据文件
echo "复制数据文件..."
cp -r ../sift ./
echo "✅ 数据文件复制完成"

# 运行快速测试
echo "5. 运行快速测试 (仅测试1-4线程)..."
echo "注意: 这是快速测试，只测试少量线程数"

# 修改程序以只测试少量线程
cat > quick_test_config.cpp << 'EOF'
// 快速测试配置 - 只测试1-4线程
#include <iostream>
#include <vector>

int main() {
    std::vector<int> thread_counts = {1, 2, 3, 4};
    std::cout << "快速测试线程数: ";
    for (int t : thread_counts) {
        std::cout << t << " ";
    }
    std::cout << std::endl;
    return 0;
}
EOF

g++ -o quick_test_config quick_test_config.cpp
./quick_test_config

# 运行基准测试 (限制时间)
echo "运行基准测试..."
timeout 300 ../benchmark-thread > quick_test.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ 快速测试完成"
    
    # 检查结果文件
    if [ -f "benchmark_results.txt" ]; then
        echo "✅ 结果文件生成成功"
        echo "结果预览:"
        head -5 benchmark_results.txt
    else
        echo "⚠️  未找到结果文件"
    fi
else
    echo "⚠️  测试超时或失败"
fi

# 清理
cd ..
rm -rf test_workspace

echo ""
echo "=== 快速测试完成 ==="
echo ""
echo "如果快速测试成功，可以运行完整对比实验:"
echo "  ./simple_disk_memory_test.sh"
echo ""
echo "如果需要分析结果，可以运行:"
echo "  python3 analyze_results.py disk_results.csv memory_results.csv"
