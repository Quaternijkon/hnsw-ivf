#!/bin/bash

# 检查数据目录大小和系统资源

echo "=== 数据目录检查 ==="
echo ""

if [ ! -d "./sift" ]; then
    echo "❌ 错误: ./sift 目录不存在"
    exit 1
fi

echo "📁 数据文件列表和大小:"
echo "----------------------------------------"
ls -lh ./sift/ | grep -v "^d" | awk '{print $9, "\t", $5}'
echo "----------------------------------------"

echo ""
echo "📊 数据目录总大小:"
TOTAL_SIZE_BYTES=$(du -sb ./sift | cut -f1)
TOTAL_SIZE_MB=$(echo "scale=2; $TOTAL_SIZE_BYTES / 1024 / 1024" | bc)
TOTAL_SIZE_GB=$(echo "scale=2; $TOTAL_SIZE_BYTES / 1024 / 1024 / 1024" | bc)

echo "  - 字节: $TOTAL_SIZE_BYTES"
echo "  - MB: $TOTAL_SIZE_MB"
echo "  - GB: $TOTAL_SIZE_GB"

echo ""
echo "💾 推荐的内存磁盘大小:"
RECOMMENDED_SIZE=$(echo "scale=0; $TOTAL_SIZE_GB * 1.3 / 1" | bc)
if [ "$RECOMMENDED_SIZE" -lt 3 ]; then
    RECOMMENDED_SIZE=3
fi
echo "  - 至少: ${RECOMMENDED_SIZE}GB (包含20-30%缓冲空间)"

echo ""
echo "🖥️  当前系统资源:"
echo "----------------------------------------"
echo "可用内存:"
free -h | grep "Mem:"
echo ""
echo "当前目录磁盘空间:"
df -h . | grep -v "Filesystem"
echo "----------------------------------------"

echo ""
echo "✅ 检查完成"
echo ""
echo "注意事项:"
echo "  1. 内存磁盘需要的内存 ≥ 数据目录大小 + 30%缓冲"
echo "  2. 系统需要保留至少2GB内存用于正常运行"
echo "  3. 如果内存不足，可以考虑减小测试数据集或使用磁盘测试"

