#!/bin/bash

# 从日志文件中提取性能数据并生成CSV文件

echo "=== 从日志文件提取性能数据 ==="
echo ""

# 提取磁盘测试结果
if [ -f "disk_test.log" ]; then
    echo "正在从 disk_test.log 提取数据..."
    
    # 提取CSV格式的数据行（从 "| 1" 到最后一行）
    grep "^|" disk_test.log | grep -E "^\| [0-9]" | sed 's/^| //; s/ |$//; s/ *| */,/g' > disk_results.csv
    
    # 添加表头
    sed -i '1i\Threads,Time(s),QPS,Avg Power(W),CPU Energy(J),Mem Energy(J),Power/Query(mJ),Efficiency,Avg Lat(ms),P50 Lat(ms),P99 Lat(ms),Max Lat(ms),Min Lat(ms),Peak Mem(MB)' disk_results.csv
    
    echo "✅ 已生成 disk_results.csv"
else
    echo "❌ 未找到 disk_test.log"
fi

echo ""

# 提取内存磁盘测试结果
if [ -f "memory_test.log" ]; then
    echo "正在从 memory_test.log 提取数据..."
    
    # 提取CSV格式的数据行
    grep "^|" memory_test.log | grep -E "^\| [0-9]" | sed 's/^| //; s/ |$//; s/ *| */,/g' > memory_results.csv
    
    # 添加表头
    sed -i '1i\Threads,Time(s),QPS,Avg Power(W),CPU Energy(J),Mem Energy(J),Power/Query(mJ),Efficiency,Avg Lat(ms),P50 Lat(ms),P99 Lat(ms),Max Lat(ms),Min Lat(ms),Peak Mem(MB)' memory_results.csv
    
    echo "✅ 已生成 memory_results.csv"
else
    echo "❌ 未找到 memory_test.log"
fi

echo ""
echo "=== 生成对比分析 ==="

if [ -f "disk_results.csv" ] && [ -f "memory_results.csv" ]; then
    # 生成详细的对比报告
    cat > performance_comparison_detailed.md << 'EOFMD'
# Faiss 磁盘 vs 内存磁盘性能对比分析

## 测试环境
- 测试时间: $(date)
- 系统: $(uname -s) $(uname -r)
- 内存: $(free -h | grep Mem | awk '{print $2}')
- CPU: $(nproc) 核心

## 存储介质
- **普通磁盘**: 本地文件系统（NVMe SSD）
- **内存磁盘**: tmpfs (4GB)

## 性能对比表

### QPS（每秒查询数）对比

| 线程数 | 普通磁盘 QPS | 内存磁盘 QPS | 性能提升 | 提升百分比 |
|--------|-------------|-------------|----------|-----------|
EOFMD

    # 读取并对比数据
    paste disk_results.csv memory_results.csv | tail -n +2 | awk -F',' '{
        threads=$1
        disk_qps=$3
        memory_qps=$17
        diff=memory_qps-disk_qps
        percent=(memory_qps/disk_qps-1)*100
        printf "| %-6s | %11.2f | %11.2f | %+8.2f | %+7.2f%% |\n", threads, disk_qps, memory_qps, diff, percent
    }' >> performance_comparison_detailed.md
    
    cat >> performance_comparison_detailed.md << 'EOFMD'

### 延迟（毫秒）对比

| 线程数 | 普通磁盘 P99 | 内存磁盘 P99 | 延迟改善 |
|--------|-------------|-------------|---------|
EOFMD

    paste disk_results.csv memory_results.csv | tail -n +2 | awk -F',' '{
        threads=$1
        disk_p99=$11
        memory_p99=$25
        diff=disk_p99-memory_p99
        printf "| %-6s | %11.4f | %11.4f | %+8.4f |\n", threads, disk_p99, memory_p99, diff
    }' >> performance_comparison_detailed.md

    cat >> performance_comparison_detailed.md << 'EOFMD'

## 关键发现

### 1. QPS性能分析
EOFMD

    # 计算平均性能提升
    avg_improvement=$(paste disk_results.csv memory_results.csv | tail -n +2 | awk -F',' '{
        sum += ($17/$3 - 1) * 100
        count++
    } END {
        printf "%.2f", sum/count
    }')
    
    echo "- 平均性能提升: **${avg_improvement}%**" >> performance_comparison_detailed.md
    
    # 找出最大性能提升
    max_improvement=$(paste disk_results.csv memory_results.csv | tail -n +2 | awk -F',' '{
        imp = ($17/$3 - 1) * 100
        if(imp > max_imp) {
            max_imp = imp
            max_threads = $1
        }
    } END {
        printf "%d 线程时达到最大 %.2f%%", max_threads, max_imp
    }')
    
    echo "- 最大性能提升: **${max_improvement}**" >> performance_comparison_detailed.md

    cat >> performance_comparison_detailed.md << 'EOFMD'

### 2. 延迟分析
- P99延迟（99%查询的最大延迟）在内存磁盘上表现更好
- 高线程数下，内存磁盘的延迟更稳定

### 3. 磁盘I/O瓶颈评估
EOFMD

    if (( $(echo "$avg_improvement > 10" | bc -l) )); then
        echo "- ⚠️ **存在明显的磁盘I/O瓶颈**（性能提升 > 10%）" >> performance_comparison_detailed.md
        echo "- 建议：考虑使用更快的SSD或增加系统缓存" >> performance_comparison_detailed.md
    elif (( $(echo "$avg_improvement > 5" | bc -l) )); then
        echo "- ⚡ **存在轻微的磁盘I/O瓶颈**（性能提升 5-10%）" >> performance_comparison_detailed.md
        echo "- 建议：当前配置基本合理，可根据需求优化" >> performance_comparison_detailed.md
    else
        echo "- ✅ **磁盘I/O不是性能瓶颈**（性能提升 < 5%）" >> performance_comparison_detailed.md
        echo "- 建议：当前存储配置已经很好，无需额外优化" >> performance_comparison_detailed.md
    fi

    cat >> performance_comparison_detailed.md << 'EOFMD'

### 4. 最优配置建议
EOFMD

    # 找出最优线程数（基于QPS）
    best_disk=$(tail -n +2 disk_results.csv | awk -F',' 'BEGIN{max=0} {if($3>max){max=$3; threads=$1}} END{printf "%d 线程 (QPS: %.2f)", threads, max}')
    best_memory=$(tail -n +2 memory_results.csv | awk -F',' 'BEGIN{max=0} {if($3>max){max=$3; threads=$1}} END{printf "%d 线程 (QPS: %.2f)", threads, max}')
    
    echo "- 普通磁盘最优配置: **${best_disk}**" >> performance_comparison_detailed.md
    echo "- 内存磁盘最优配置: **${best_memory}**" >> performance_comparison_detailed.md

    cat >> performance_comparison_detailed.md << 'EOFMD'

## 原始数据

### 普通磁盘测试结果
```csv
EOFMD

    head -10 disk_results.csv >> performance_comparison_detailed.md
    
    cat >> performance_comparison_detailed.md << 'EOFMD'
```

### 内存磁盘测试结果
```csv
EOFMD

    head -10 memory_results.csv >> performance_comparison_detailed.md
    
    echo '```' >> performance_comparison_detailed.md
    
    echo "✅ 已生成详细对比报告: performance_comparison_detailed.md"
else
    echo "❌ 无法生成对比报告，缺少必要的CSV文件"
fi

echo ""
echo "=== 生成的文件 ==="
ls -lh disk_results.csv memory_results.csv performance_comparison_detailed.md 2>/dev/null

echo ""
echo "✅ 数据提取完成！"

