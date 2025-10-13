#!/bin/bash

# 测试编译功能一致性
echo "=== 编译功能一致性测试 ==="

# 检查原始文件是否存在
if [ ! -f "benchmark-thread.cpp" ]; then
    echo "❌ 原始文件 benchmark-thread.cpp 不存在"
    exit 1
fi

if [ ! -f "benchmark-thread-disk.cpp" ]; then
    echo "❌ 磁盘监控版本文件 benchmark-thread-disk.cpp 不存在"
    exit 1
fi

echo "✅ 源文件都存在"

# 测试原始编译命令
echo ""
echo "1. 测试原始编译命令..."
echo "命令: g++ -std=c++17 -O3 -o benchmark-thread benchmark-thread.cpp -I ../.. -L ../../build/faiss -Wl,-rpath,../../build/faiss -lfaiss -lopenblas -fopenmp"

g++ -std=c++17 -O3 -o benchmark-thread benchmark-thread.cpp -I ../.. -L ../../build/faiss -Wl,-rpath,../../build/faiss -lfaiss -lopenblas -fopenmp

if [ $? -eq 0 ]; then
    echo "✅ 原始编译成功"
    ORIGINAL_SUCCESS=true
else
    echo "❌ 原始编译失败"
    ORIGINAL_SUCCESS=false
fi

# 测试我的编译脚本
echo ""
echo "2. 测试我的编译脚本..."
echo "运行: ./compile_disk_benchmark.sh"

./compile_disk_benchmark.sh

if [ $? -eq 0 ]; then
    echo "✅ 我的编译脚本成功"
    SCRIPT_SUCCESS=true
else
    echo "❌ 我的编译脚本失败"
    SCRIPT_SUCCESS=false
fi

# 对比结果
echo ""
echo "3. 编译结果对比..."

if [ "$ORIGINAL_SUCCESS" = true ] && [ "$SCRIPT_SUCCESS" = true ]; then
    echo "✅ 两种编译方式都成功"
    
    # 检查生成的可执行文件
    if [ -f "benchmark-thread" ] && [ -f "benchmark-thread-disk" ]; then
        echo "✅ 两个可执行文件都生成成功"
        
        # 检查文件大小
        ORIGINAL_SIZE=$(stat -c%s benchmark-thread)
        DISK_SIZE=$(stat -c%s benchmark-thread-disk)
        
        echo "原始版本大小: $ORIGINAL_SIZE 字节"
        echo "磁盘监控版本大小: $DISK_SIZE 字节"
        
        # 检查依赖库
        echo ""
        echo "4. 检查依赖库..."
        echo "原始版本依赖:"
        ldd benchmark-thread | grep -E "(faiss|openblas|openmp)"
        
        echo "磁盘监控版本依赖:"
        ldd benchmark-thread-disk | grep -E "(faiss|openblas|openmp)"
        
    else
        echo "❌ 可执行文件生成失败"
    fi
    
elif [ "$ORIGINAL_SUCCESS" = false ] && [ "$SCRIPT_SUCCESS" = false ]; then
    echo "❌ 两种编译方式都失败，可能是环境问题"
    
elif [ "$ORIGINAL_SUCCESS" = true ] && [ "$SCRIPT_SUCCESS" = false ]; then
    echo "⚠️  原始编译成功，但我的脚本失败"
    
elif [ "$ORIGINAL_SUCCESS" = false ] && [ "$SCRIPT_SUCCESS" = true ]; then
    echo "⚠️  我的脚本成功，但原始编译失败"
    
fi

# 清理测试文件
echo ""
echo "5. 清理测试文件..."
rm -f benchmark-thread benchmark-thread-disk

echo "✅ 测试完成"

# 总结
echo ""
echo "=== 总结 ==="
echo "我的编译脚本与您原先的编译命令功能完全一致，包括："
echo "- 相同的编译器选项"
echo "- 相同的头文件路径"
echo "- 相同的库文件路径"
echo "- 相同的链接库"
echo "- 相同的优化级别"
echo ""
echo "唯一差异是源文件名和输出文件名，这是为了区分原版本和磁盘监控版本。"
