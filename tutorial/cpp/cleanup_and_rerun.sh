#!/bin/bash

# 清理和重新运行测试的便捷脚本

echo "=== 清理旧的测试环境 ==="

# 清理残留的挂载点
if mount | grep -q "memory_workspace"; then
    echo "发现残留的内存磁盘挂载，正在卸载..."
    # 可能有重复挂载，尝试多次卸载
    sudo umount memory_workspace 2>/dev/null
    sudo umount memory_workspace 2>/dev/null
    sudo umount memory_workspace 2>/dev/null
    
    # 检查是否还有挂载
    if mount | grep -q "memory_workspace"; then
        echo "⚠️  警告: 仍然有进程占用内存磁盘"
        echo "正在尝试强制卸载..."
        sudo fuser -km memory_workspace 2>/dev/null
        sleep 1
        sudo umount -f memory_workspace 2>/dev/null || sudo umount -l memory_workspace 2>/dev/null
    fi
fi

# 清理旧的日志和结果文件
echo "清理旧的测试文件..."
rm -f disk_test.log memory_test.log
rm -f disk_results.csv memory_results.csv
rm -f performance_comparison.md
rm -f benchmark_results.txt

# 清理工作目录
echo "清理工作目录..."
rm -rf disk_workspace 2>/dev/null

# 清理memory_workspace（可能是空目录）
if [ -d "memory_workspace" ]; then
    echo "正在清理 memory_workspace..."
    # 尝试删除内容
    rm -rf memory_workspace/* 2>/dev/null
    rm -rf memory_workspace/.* 2>/dev/null
    # 删除目录本身
    rmdir memory_workspace 2>/dev/null
    
    # 检查是否删除成功
    if [ -d "memory_workspace" ]; then
        echo "⚠️  memory_workspace 仍然存在，尝试强制删除..."
        sudo rm -rf memory_workspace 2>/dev/null
        
        if [ -d "memory_workspace" ]; then
            echo "❌ memory_workspace 无法删除"
            echo "请检查是否有进程占用: lsof +D memory_workspace"
            exit 1
        fi
    fi
    echo "✅ memory_workspace 已清理"
fi

echo "✅ 清理完成"
echo ""

# 检查系统资源
echo "=== 检查系统资源 ==="
echo "可用内存:"
free -h | grep Mem
echo ""
echo "磁盘空间:"
df -h . | grep -v "Filesystem"
echo ""
echo "数据目录大小:"
if [ -d "./sift" ]; then
    du -sh ./sift
else
    echo "⚠️  警告: ./sift 目录不存在"
fi
echo ""

# 检查可执行文件
if [ ! -f "./benchmark-thread" ]; then
    echo "❌ 可执行文件 benchmark-thread 不存在"
    echo "正在重新编译..."
    g++ -std=c++17 -O3 -o benchmark-thread benchmark-thread.cpp \
        -I ../.. -L ../../build/faiss \
        -Wl,-rpath,../../build/faiss -lfaiss -lopenblas -fopenmp
    
    if [ $? -ne 0 ]; then
        echo "❌ 编译失败"
        exit 1
    fi
    echo "✅ 编译成功"
fi

# 询问是否运行测试
echo ""
read -p "是否立即运行测试? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "=== 开始运行测试 ==="
    ./simple_disk_memory_test.sh
else
    echo "已取消测试。稍后可以手动运行: ./simple_disk_memory_test.sh"
fi

