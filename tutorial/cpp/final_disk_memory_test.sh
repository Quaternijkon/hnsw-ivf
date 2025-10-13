#!/bin/bash

# 最终版磁盘 vs 内存磁盘对比实验
# 修复结果文件提取问题

echo "=== Faiss 磁盘 vs 内存磁盘性能对比实验 (最终版) ==="

# 检查必要文件
if [ ! -f "./benchmark-thread" ]; then
    echo "错误: 可执行文件 benchmark-thread 不存在"
    echo "请先编译程序: g++ -std=c++17 -O3 -o benchmark-thread benchmark-thread.cpp -I ../.. -L ../../build/faiss -Wl,-rpath,../../build/faiss -lfaiss -lopenblas -fopenmp"
    exit 1
fi

if [ ! -d "./sift" ]; then
    echo "错误: 数据目录 ./sift 不存在"
    exit 1
fi

echo "✅ 所有必要文件都存在"

# 创建内存磁盘
MEMORY_DIR="./memory_workspace"
DISK_DIR="./disk_workspace"

echo "创建测试工作空间..."
mkdir -p "$MEMORY_DIR"
mkdir -p "$DISK_DIR"

# 创建内存磁盘 (tmpfs)
echo "创建内存磁盘 (tmpfs)..."
sudo mount -t tmpfs -o size=2G tmpfs "$MEMORY_DIR"

if [ $? -ne 0 ]; then
    echo "❌ 内存磁盘创建失败，请检查sudo权限"
    exit 1
fi

echo "✅ 内存磁盘创建成功"
echo "内存磁盘信息:"
df -h "$MEMORY_DIR"

# 复制数据到两个工作空间
echo ""
echo "复制数据文件..."
cp -r ./sift "$DISK_DIR/"
cp -r ./sift "$MEMORY_DIR/"

echo "✅ 数据文件复制完成"

# 清理可能存在的索引文件和结果文件
echo ""
echo "清理旧的索引文件和结果文件..."
rm -f "$DISK_DIR/sift"/*.index
rm -f "$MEMORY_DIR/sift"/*.index
rm -f ./sift/*.index
rm -f benchmark_results.txt
rm -f disk_results.csv memory_results.csv

echo "✅ 清理完成"

# 函数：运行测试并收集结果
run_test() {
    local test_name="$1"
    local workspace="$2"
    local output_prefix="$3"
    
    echo ""
    echo "=========================================="
    echo "运行 $test_name 测试"
    echo "工作目录: $workspace"
    echo "=========================================="
    
    # 进入工作目录运行测试
    cd "$workspace"
    
    echo "当前目录: $(pwd)"
    echo "数据文件:"
    ls -la sift/
    
    echo ""
    echo "开始运行基准测试..."
    echo "注意: 这可能需要几分钟时间..."
    
    # 运行基准测试，重定向输出到文件
    ../benchmark-thread > "../${output_prefix}_test.log" 2>&1
    
    local exit_code=$?
    
    # 检查结果文件是否在工作目录中生成
    if [ -f "benchmark_results.txt" ]; then
        echo "✅ 在工作目录中找到结果文件"
        cp benchmark_results.txt "../${output_prefix}_results.csv"
        echo "结果已保存到: ${output_prefix}_results.csv"
        
        # 显示结果预览
        echo "结果预览:"
        head -5 "../${output_prefix}_results.csv"
    else
        echo "⚠️  在工作目录中未找到结果文件"
    fi
    
    # 返回上级目录
    cd ..
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $test_name 测试完成"
        
        # 检查是否有结果文件
        if [ -f "${output_prefix}_results.csv" ]; then
            echo "✅ 结果文件已保存: ${output_prefix}_results.csv"
        else
            echo "⚠️  未找到结果文件"
        fi
        
        # 检查索引文件是否生成
        if [ -f "$workspace/sift"/*.index ]; then
            echo "✅ 索引文件已生成"
            ls -la "$workspace/sift"/*.index
        else
            echo "⚠️  未找到索引文件"
        fi
    else
        echo "❌ $test_name 测试失败 (退出码: $exit_code)"
    fi
}

# 运行普通磁盘测试
run_test "普通磁盘" "$DISK_DIR" "disk"

# 运行内存磁盘测试  
run_test "内存磁盘" "$MEMORY_DIR" "memory"

# 检查索引文件位置
echo ""
echo "=========================================="
echo "检查索引文件位置"
echo "=========================================="

echo "普通磁盘索引文件:"
find "$DISK_DIR" -name "*.index" -exec ls -la {} \;

echo "内存磁盘索引文件:"
find "$MEMORY_DIR" -name "*.index" -exec ls -la {} \;

# 验证结果文件是否不同
echo ""
echo "=========================================="
echo "验证结果文件差异"
echo "=========================================="

if [ -f "disk_results.csv" ] && [ -f "memory_results.csv" ]; then
    echo "检查文件差异..."
    
    if diff disk_results.csv memory_results.csv > /dev/null; then
        echo "⚠️  警告: 两个结果文件完全相同！"
    else
        echo "✅ 两个结果文件不同，存在性能差异"
        echo "差异统计:"
        diff disk_results.csv memory_results.csv | head -10
    fi
    
    # 计算文件大小
    disk_size=$(stat -c%s disk_results.csv)
    memory_size=$(stat -c%s memory_results.csv)
    echo "普通磁盘结果文件大小: $disk_size 字节"
    echo "内存磁盘结果文件大小: $memory_size 字节"
    
    # 显示关键性能指标对比
    echo ""
    echo "关键性能指标对比:"
    echo "普通磁盘测试结果:"
    grep -E "Threads.*QPS.*Avg Lat" disk_results.csv
    grep -E "^[0-9]" disk_results.csv | head -3
    
    echo "内存磁盘测试结果:"
    grep -E "Threads.*QPS.*Avg Lat" memory_results.csv
    grep -E "^[0-9]" memory_results.csv | head -3
    
else
    echo "❌ 结果文件缺失"
fi

# 生成对比分析
echo ""
echo "=========================================="
echo "生成对比分析报告"
echo "=========================================="

# 创建对比报告
cat > performance_comparison_final.md << EOF
# Faiss 性能对比实验报告 (最终版)

## 实验目的
通过对比普通磁盘和内存磁盘的性能表现，分析磁盘I/O是否在高线程数下成为性能瓶颈。

## 测试环境
- 测试时间: $(date)
- 系统: $(uname -s) $(uname -r)
- 内存: $(free -h | grep Mem | awk '{print $2}')
- 磁盘信息: $(df -h | grep -E "(tmpfs|/dev/)")

## 存储介质信息
- 普通磁盘: 本地文件系统
- 内存磁盘: tmpfs (2GB)

## 测试结果

### 普通磁盘测试结果
EOF

if [ -f "disk_results.csv" ]; then
    echo "```csv" >> performance_comparison_final.md
    head -10 disk_results.csv >> performance_comparison_final.md
    echo "```" >> performance_comparison_final.md
else
    echo "普通磁盘测试结果文件不存在" >> performance_comparison_final.md
fi

cat >> performance_comparison_final.md << EOF

### 内存磁盘测试结果
EOF

if [ -f "memory_results.csv" ]; then
    echo "```csv" >> performance_comparison_final.md
    head -10 memory_results.csv >> performance_comparison_final.md
    echo "```" >> performance_comparison_final.md
else
    echo "内存磁盘测试结果文件不存在" >> performance_comparison_final.md
fi

cat >> performance_comparison_final.md << EOF

## 性能对比分析

### 关键指标对比
EOF

if [ -f "disk_results.csv" ] && [ -f "memory_results.csv" ]; then
    echo "| 指标 | 普通磁盘 | 内存磁盘 | 性能差异 |" >> performance_comparison_final.md
    echo "|------|----------|----------|----------|" >> performance_comparison_final.md
    
    # 提取关键数据
    disk_qps=$(grep -E "^20" disk_results.csv | cut -d',' -f3 | head -1)
    memory_qps=$(grep -E "^20" memory_results.csv | cut -d',' -f3 | head -1)
    disk_latency=$(grep -E "^20" disk_results.csv | cut -d',' -f9 | head -1)
    memory_latency=$(grep -E "^20" memory_results.csv | cut -d',' -f9 | head -1)
    
    if [ -n "$disk_qps" ] && [ -n "$memory_qps" ]; then
        qps_diff=$(echo "scale=2; ($memory_qps - $disk_qps) / $disk_qps * 100" | bc -l 2>/dev/null || echo "N/A")
        echo "| QPS (20线程) | $disk_qps | $memory_qps | $qps_diff% |" >> performance_comparison_final.md
    fi
    
    if [ -n "$disk_latency" ] && [ -n "$memory_latency" ]; then
        latency_diff=$(echo "scale=2; ($disk_latency - $memory_latency) / $disk_latency * 100" | bc -l 2>/dev/null || echo "N/A")
        echo "| 延迟 (20线程) | $disk_latency ms | $memory_latency ms | $latency_diff% |" >> performance_comparison_final.md
    fi
fi

cat >> performance_comparison_final.md << EOF

## 分析结论

### 磁盘I/O瓶颈分析
通过对比两种存储介质的性能表现，可以得出以下结论：

1. **高线程数下的性能差异**: 如果内存磁盘在高线程数下表现明显优于普通磁盘，说明存在磁盘I/O瓶颈
2. **最优线程数**: 找出两种存储介质下的最优线程配置
3. **性能提升幅度**: 量化磁盘I/O瓶颈对整体性能的影响

### 优化建议
- 如果存在明显性能差异，建议考虑使用SSD或增加内存缓存
- 根据测试结果调整线程数配置
- 考虑使用内存映射文件优化I/O性能

## 文件说明
- \`disk_results.csv\`: 普通磁盘测试结果
- \`memory_results.csv\`: 内存磁盘测试结果  
- \`disk_test.log\`: 普通磁盘测试日志
- \`memory_test.log\`: 内存磁盘测试日志
EOF

echo "✅ 对比分析报告已生成: performance_comparison_final.md"

# 显示结果
echo ""
echo "=========================================="
echo "实验结果文件"
echo "=========================================="

echo "生成的文件:"
ls -la *results.csv *test.log performance_comparison_final.md 2>/dev/null

echo ""
echo "结果预览:"
if [ -f "disk_results.csv" ]; then
    echo "--- 普通磁盘测试结果 ---"
    head -3 disk_results.csv
fi

if [ -f "memory_results.csv" ]; then
    echo "--- 内存磁盘测试结果 ---"
    head -3 memory_results.csv
fi

# 清理
echo ""
echo "清理测试环境..."
sudo umount "$MEMORY_DIR" 2>/dev/null
rm -rf "$MEMORY_DIR" "$DISK_DIR"

echo ""
echo "✅ 最终版实验完成！"
echo ""
echo "请查看以下文件获取详细结果:"
echo "- performance_comparison_final.md: 对比分析报告"
echo "- disk_results.csv: 普通磁盘测试结果"
echo "- memory_results.csv: 内存磁盘测试结果"
