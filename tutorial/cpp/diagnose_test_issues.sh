#!/bin/bash

# 诊断测试问题脚本
echo "=== Faiss 测试问题诊断 ==="

# 1. 检查结果文件
echo "1. 检查结果文件..."
if [ -f "disk_results.csv" ] && [ -f "memory_results.csv" ]; then
    echo "✅ 结果文件存在"
    
    # 检查文件是否相同
    if diff disk_results.csv memory_results.csv > /dev/null; then
        echo "❌ 问题: 两个结果文件完全相同！"
        echo "MD5校验:"
        md5sum disk_results.csv memory_results.csv
    else
        echo "✅ 两个结果文件不同"
    fi
    
    # 检查文件大小
    disk_size=$(stat -c%s disk_results.csv)
    memory_size=$(stat -c%s memory_results.csv)
    echo "普通磁盘结果文件大小: $disk_size 字节"
    echo "内存磁盘结果文件大小: $memory_size 字节"
    
else
    echo "❌ 结果文件不存在"
fi

# 2. 检查测试日志
echo ""
echo "2. 检查测试日志..."
if [ -f "disk_test.log" ] && [ -f "memory_test.log" ]; then
    echo "✅ 日志文件存在"
    
    # 检查日志中的关键信息
    echo "普通磁盘测试关键信息:"
    grep -E "(QPS|延迟|Time|Threads)" disk_test.log | tail -5
    
    echo "内存磁盘测试关键信息:"
    grep -E "(QPS|延迟|Time|Threads)" memory_test.log | tail -5
    
    # 检查是否有错误
    echo "普通磁盘测试错误:"
    grep -i error disk_test.log | head -3
    
    echo "内存磁盘测试错误:"
    grep -i error memory_test.log | head -3
    
else
    echo "❌ 日志文件不存在"
fi

# 3. 检查索引文件
echo ""
echo "3. 检查索引文件..."
index_files=$(find . -name "*.index" -type f)
if [ -n "$index_files" ]; then
    echo "找到索引文件:"
    echo "$index_files"
    
    # 检查索引文件位置
    for file in $index_files; do
        echo "文件: $file"
        ls -la "$file"
        echo "位置: $(dirname "$file")"
    done
else
    echo "❌ 未找到索引文件"
fi

# 4. 检查工作空间
echo ""
echo "4. 检查工作空间..."
if [ -d "disk_workspace" ]; then
    echo "普通磁盘工作空间:"
    ls -la disk_workspace/
    echo "索引文件:"
    find disk_workspace -name "*.index" -exec ls -la {} \;
else
    echo "❌ 普通磁盘工作空间不存在"
fi

if [ -d "memory_workspace" ]; then
    echo "内存磁盘工作空间:"
    ls -la memory_workspace/
    echo "索引文件:"
    find memory_workspace -name "*.index" -exec ls -la {} \;
else
    echo "❌ 内存磁盘工作空间不存在"
fi

# 5. 检查程序行为
echo ""
echo "5. 检查程序行为..."
if [ -f "benchmark-thread" ]; then
    echo "✅ 可执行文件存在"
    
    # 检查程序是否使用固定种子
    echo "检查程序是否使用固定随机种子..."
    if strings benchmark-thread | grep -i "seed\|random" | head -3; then
        echo "⚠️  程序可能使用固定随机种子"
    else
        echo "未发现明显的随机种子设置"
    fi
    
    # 检查程序是否使用缓存
    echo "检查程序是否使用缓存..."
    if strings benchmark-thread | grep -i "cache\|mmap" | head -3; then
        echo "⚠️  程序可能使用缓存或内存映射"
    else
        echo "未发现明显的缓存设置"
    fi
    
else
    echo "❌ 可执行文件不存在"
fi

# 6. 检查系统资源
echo ""
echo "6. 检查系统资源..."
echo "内存使用:"
free -h

echo "磁盘使用:"
df -h | grep -E "(tmpfs|/dev/)"

echo "CPU信息:"
nproc

# 7. 分析可能的原因
echo ""
echo "7. 可能的原因分析..."
echo "=========================================="

if [ -f "disk_results.csv" ] && [ -f "memory_results.csv" ]; then
    if diff disk_results.csv memory_results.csv > /dev/null; then
        echo "🔍 结果完全相同的原因可能是:"
        echo ""
        echo "1. **结果文件覆盖**: 两个测试都生成 benchmark_results.txt，第二个覆盖了第一个"
        echo "2. **共享索引文件**: 两个测试使用了同一个索引文件"
        echo "3. **程序确定性**: 程序使用固定随机种子，结果完全确定"
        echo "4. **缓存影响**: 程序使用内存映射，数据已在内存中"
        echo "5. **测试环境相同**: 两个测试实际上在相同的环境中运行"
        echo ""
        echo "💡 建议解决方案:"
        echo "1. 使用不同的工作目录"
        echo "2. 清理旧的索引文件"
        echo "3. 检查程序是否使用固定种子"
        echo "4. 验证索引文件确实存储在不同位置"
    else
        echo "✅ 结果不同，测试正常"
    fi
fi

echo ""
echo "=== 诊断完成 ==="
