#!/bin/bash

# 磁盘 vs 内存磁盘对比实验脚本
# 测试在不同存储介质下的性能表现

echo "=== Faiss 磁盘 vs 内存磁盘性能对比实验 ==="

# 检查必要的数据文件
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
if [ ! -f "./benchmark-thread" ]; then
    echo "错误: 可执行文件 benchmark-thread 不存在"
    echo "请先编译程序"
    exit 1
fi

echo "可执行文件存在"

# 创建内存磁盘挂载点
MEMORY_DISK_DIR="./memory_disk"
DISK_DIR="./disk"

echo "创建测试目录..."
mkdir -p "$MEMORY_DISK_DIR"
mkdir -p "$DISK_DIR"

# 创建内存磁盘 (tmpfs)
echo "创建内存磁盘 (tmpfs)..."
sudo mount -t tmpfs -o size=2G tmpfs "$MEMORY_DISK_DIR"

if [ $? -eq 0 ]; then
    echo "✅ 内存磁盘创建成功"
    echo "内存磁盘大小: $(df -h $MEMORY_DISK_DIR | tail -1 | awk '{print $2}')"
    echo "可用空间: $(df -h $MEMORY_DISK_DIR | tail -1 | awk '{print $4}')"
else
    echo "❌ 内存磁盘创建失败"
    exit 1
fi

# 复制数据文件到两个目录
echo "复制数据文件到测试目录..."

# 复制到普通磁盘目录
echo "复制到普通磁盘目录..."
cp -r "$DATA_DIR" "$DISK_DIR/"
cp -r "$DATA_DIR" "$MEMORY_DISK_DIR/"

echo "数据文件复制完成"

# 函数：运行基准测试
run_benchmark() {
    local test_name="$1"
    local data_path="$2"
    local output_file="$3"
    
    echo ""
    echo "=== 运行 $test_name 测试 ==="
    echo "数据路径: $data_path"
    echo "输出文件: $output_file"
    
    # 修改程序中的数据目录路径
    sed "s|const string DATA_DIR = \"./sift\";|const string DATA_DIR = \"$data_path/sift\";|g" benchmark-thread.cpp > benchmark-thread-temp.cpp
    
    # 编译临时版本
    echo "编译临时版本..."
    g++ -std=c++17 -O3 -o benchmark-temp benchmark-thread-temp.cpp -I ../.. -L ../../build/faiss -Wl,-rpath,../../build/faiss -lfaiss -lopenblas -fopenmp
    
    if [ $? -ne 0 ]; then
        echo "❌ 编译失败"
        return 1
    fi
    
    # 运行测试
    echo "开始运行测试..."
    ./benchmark-temp > "$output_file" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ $test_name 测试完成"
        
        # 提取CSV结果
        if [ -f "benchmark_results.txt" ]; then
            cp benchmark_results.txt "${output_file%.log}.csv"
            echo "结果已保存到: ${output_file%.log}.csv"
        fi
    else
        echo "❌ $test_name 测试失败"
    fi
    
    # 清理临时文件
    rm -f benchmark-thread-temp.cpp benchmark-temp
}

# 运行普通磁盘测试
echo ""
echo "=========================================="
echo "开始普通磁盘测试"
echo "=========================================="

run_benchmark "普通磁盘" "$DISK_DIR" "disk_test.log"

# 运行内存磁盘测试
echo ""
echo "=========================================="
echo "开始内存磁盘测试"
echo "=========================================="

run_benchmark "内存磁盘" "$MEMORY_DISK_DIR" "memory_test.log"

# 生成对比报告
echo ""
echo "=========================================="
echo "生成对比报告"
echo "=========================================="

# 创建对比报告
cat > comparison_report.md << EOF
# Faiss 磁盘 vs 内存磁盘性能对比报告

## 测试环境
- 测试时间: $(date)
- 系统信息: $(uname -a)
- 内存信息: $(free -h | head -2)
- 磁盘信息: $(df -h | grep -E "(tmpfs|/dev/)")

## 测试结果

### 普通磁盘测试结果
EOF

if [ -f "disk_test.csv" ]; then
    echo "```csv" >> comparison_report.md
    head -10 disk_test.csv >> comparison_report.md
    echo "```" >> comparison_report.md
else
    echo "普通磁盘测试结果文件不存在" >> comparison_report.md
fi

cat >> comparison_report.md << EOF

### 内存磁盘测试结果
EOF

if [ -f "memory_test.csv" ]; then
    echo "```csv" >> comparison_report.md
    head -10 memory_test.csv >> comparison_report.md
    echo "```" >> comparison_report.md
else
    echo "内存磁盘测试结果文件不存在" >> comparison_report.md
fi

cat >> comparison_report.md << EOF

## 性能对比分析

### 关键指标对比
EOF

# 如果两个CSV文件都存在，进行简单对比
if [ -f "disk_test.csv" ] && [ -f "memory_test.csv" ]; then
    echo "正在分析性能差异..."
    
    # 提取关键指标进行对比
    echo "| 指标 | 普通磁盘 | 内存磁盘 | 性能提升 |" >> comparison_report.md
    echo "|------|----------|----------|----------|" >> comparison_report.md
    
    # 这里可以添加更详细的分析逻辑
    echo "| QPS | 待分析 | 待分析 | 待计算 |" >> comparison_report.md
    echo "| 延迟 | 待分析 | 待分析 | 待计算 |" >> comparison_report.md
    echo "| 功耗 | 待分析 | 待分析 | 待计算 |" >> comparison_report.md
fi

cat >> comparison_report.md << EOF

## 结论

通过对比普通磁盘和内存磁盘的性能测试，可以分析：

1. **磁盘I/O瓶颈影响**: 在高线程数下，磁盘I/O是否成为性能瓶颈
2. **内存vs磁盘性能差异**: 不同存储介质的性能表现
3. **线程数优化**: 找出最优的线程配置

## 文件说明
- \`disk_test.csv\`: 普通磁盘测试结果
- \`memory_test.csv\`: 内存磁盘测试结果
- \`disk_test.log\`: 普通磁盘测试日志
- \`memory_test.log\`: 内存磁盘测试日志
EOF

echo "对比报告已生成: comparison_report.md"

# 显示结果文件
echo ""
echo "=========================================="
echo "测试结果文件"
echo "=========================================="

echo "生成的文件:"
ls -la *.csv *.log comparison_report.md 2>/dev/null

echo ""
echo "CSV文件预览:"
if [ -f "disk_test.csv" ]; then
    echo "--- 普通磁盘测试结果 ---"
    head -5 disk_test.csv
fi

if [ -f "memory_test.csv" ]; then
    echo "--- 内存磁盘测试结果 ---"
    head -5 memory_test.csv
fi

# 清理内存磁盘
echo ""
echo "清理测试环境..."
sudo umount "$MEMORY_DISK_DIR" 2>/dev/null
rm -rf "$MEMORY_DISK_DIR" "$DISK_DIR"

echo "✅ 测试完成！"
echo ""
echo "请查看以下文件获取详细结果:"
echo "- comparison_report.md: 对比报告"
echo "- disk_test.csv: 普通磁盘测试结果"
echo "- memory_test.csv: 内存磁盘测试结果"
