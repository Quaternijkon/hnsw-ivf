#!/bin/bash

# 从日志中提取文件大小信息
echo "=== 普通磁盘文件大小 ==="
grep "^-rw" disk_test.log | head -20

echo ""
echo "=== 内存磁盘文件大小 ==="
grep "^-rw" memory_test.log | head -20

