#!/bin/bash

# 验证Page Cache对性能的影响

echo "=== 验证Linux Page Cache对磁盘性能的影响 ==="
echo ""

# 检查是否有root权限
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  此脚本需要root权限来清空Page Cache"
    echo "请使用: sudo ./verify_page_cache_effect.sh"
    exit 1
fi

# 检查必要文件
if [ ! -f "./benchmark-thread" ]; then
    echo "❌ 错误: benchmark-thread不存在"
    exit 1
fi

if [ ! -d "./sift" ]; then
    echo "❌ 错误: ./sift目录不存在"
    exit 1
fi

echo "✅ 环境检查通过"
echo ""

# 创建结果目录
RESULT_DIR="./page_cache_test_results"
mkdir -p "$RESULT_DIR"

echo "=== 测试计划 ==="
echo "1. 测试1: 清空Page Cache前测试（热缓存）"
echo "2. 清空Page Cache"
echo "3. 测试2: 清空Page Cache后立即测试（冷启动）"
echo "4. 测试3: 再次测试（缓存已重建）"
echo ""

# ============================================
# 测试1: 热缓存状态
# ============================================
echo "=========================================="
echo "测试1: 热缓存状态（不清空Page Cache）"
echo "=========================================="
echo "当前Page Cache状态:"
free -h | grep -E "Mem:|buff/cache"
echo ""

# 运行一次基准测试（只测试5线程，快速验证）
cd sift 2>/dev/null || {
    echo "正在sift目录外运行..."
}

echo "运行测试（5线程配置）..."
START_TIME=$(date +%s)

# 简化的测试：只运行一次，快速验证
../benchmark-thread 2>&1 | grep -A 50 "Phase 6" | tee "$RESULT_DIR/test1_hot_cache.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "✅ 测试1完成，耗时: ${DURATION}秒"
echo ""

cd ..

# ============================================
# 清空Page Cache
# ============================================
echo "=========================================="
echo "清空Page Cache"
echo "=========================================="
echo "清空前的内存状态:"
free -h | grep -E "Mem:|buff/cache"
echo ""

echo "正在清空Page Cache..."
sync
echo 3 > /proc/sys/vm/drop_caches
sleep 2

echo "清空后的内存状态:"
free -h | grep -E "Mem:|buff/cache"
echo ""
echo "✅ Page Cache已清空（注意buff/cache减少了）"
echo ""

# ============================================
# 测试2: 冷启动
# ============================================
echo "=========================================="
echo "测试2: 冷启动状态（Page Cache已清空）"
echo "=========================================="
echo "这次测试会体验到真实的磁盘I/O..."
echo ""

cd sift 2>/dev/null || true

echo "运行测试（5线程配置）..."
START_TIME=$(date +%s)

../benchmark-thread 2>&1 | grep -A 50 "Phase 6" | tee "$RESULT_DIR/test2_cold_start.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "✅ 测试2完成，耗时: ${DURATION}秒"
echo ""

cd ..

# ============================================
# 测试3: 缓存重建后
# ============================================
echo "=========================================="
echo "测试3: 缓存重建后（再次热缓存）"
echo "=========================================="
echo "当前Page Cache状态:"
free -h | grep -E "Mem:|buff/cache"
echo ""

cd sift 2>/dev/null || true

echo "运行测试（5线程配置）..."
START_TIME=$(date +%s)

../benchmark-thread 2>&1 | grep -A 50 "Phase 6" | tee "$RESULT_DIR/test3_warm_cache.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "✅ 测试3完成，耗时: ${DURATION}秒"
echo ""

cd ..

# ============================================
# 分析结果
# ============================================
echo "=========================================="
echo "结果分析"
echo "=========================================="
echo ""

# 提取QPS数据
echo "提取QPS数据..."

QPS1=$(grep "Run 1/1.*QPS" "$RESULT_DIR/test1_hot_cache.log" | head -1 | grep -oP "QPS = \K[0-9.]+")
QPS2=$(grep "Run 1/1.*QPS" "$RESULT_DIR/test2_cold_start.log" | head -1 | grep -oP "QPS = \K[0-9.]+")
QPS3=$(grep "Run 1/1.*QPS" "$RESULT_DIR/test3_warm_cache.log" | head -1 | grep -oP "QPS = \K[0-9.]+")

echo "| 测试 | 状态 | QPS |"
echo "|------|------|-----|"
echo "| 测试1 | 热缓存（Page Cache满） | ${QPS1:-N/A} |"
echo "| 测试2 | 冷启动（Page Cache清空） | ${QPS2:-N/A} |"
echo "| 测试3 | 重建缓存（Page Cache恢复） | ${QPS3:-N/A} |"
echo ""

if [ -n "$QPS1" ] && [ -n "$QPS2" ]; then
    DIFF=$(echo "scale=2; ($QPS1 - $QPS2) / $QPS2 * 100" | bc)
    echo "📊 关键发现:"
    echo "   热缓存相比冷启动性能提升: ${DIFF}%"
    echo ""
    
    if (( $(echo "$DIFF > 10" | bc -l) )); then
        echo "✅ 验证成功！Page Cache有显著影响（提升 > 10%）"
        echo "   这证明了：普通磁盘在Page Cache帮助下，性能接近纯内存访问"
    else
        echo "⚠️  性能差异小于预期（< 10%）"
        echo "   可能原因："
        echo "   1. NVMe SSD非常快，即使冷启动也很快"
        echo "   2. 系统预读（readahead）机制发挥作用"
        echo "   3. 数据集较小，加载时间短"
    fi
fi

echo ""
echo "=========================================="
echo "详细日志保存在:"
echo "- $RESULT_DIR/test1_hot_cache.log"
echo "- $RESULT_DIR/test2_cold_start.log"
echo "- $RESULT_DIR/test3_warm_cache.log"
echo "=========================================="
echo ""

echo "💡 结论："
echo "   Page Cache是Linux的自动优化机制，它会："
echo "   1. 自动将频繁访问的文件缓存到内存"
echo "   2. 使得普通磁盘文件的访问速度接近内存"
echo "   3. 这就是为什么您的磁盘vs内存测试差异很小！"
echo ""
echo "✅ 验证完成！"




