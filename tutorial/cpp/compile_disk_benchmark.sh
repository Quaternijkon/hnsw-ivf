#!/bin/bash

# 编译带磁盘监控的基准测试程序
# 使用方法: ./compile_disk_benchmark.sh

echo "正在编译带磁盘监控的Faiss基准测试程序..."

# 设置编译选项
CXX=g++
CXXFLAGS="-std=c++17 -O3 -fopenmp"
INCLUDES="-I ../.."
LIBPATH="-L ../../build/faiss -Wl,-rpath,../../build/faiss"
LIBS="-lfaiss -lopenblas"

# 源文件
SOURCE="benchmark-thread-disk.cpp"
OUTPUT="benchmark-thread-disk"

echo "编译命令: $CXX $CXXFLAGS $INCLUDES $SOURCE -o $OUTPUT $LIBPATH $LIBS"

# 执行编译
$CXX $CXXFLAGS $INCLUDES $SOURCE -o $OUTPUT $LIBPATH $LIBS

if [ $? -eq 0 ]; then
    echo "✅ 编译成功！可执行文件: $OUTPUT"
    echo ""
    echo "运行方法:"
    echo "  ./$OUTPUT"
    echo ""
    echo "功能特性:"
    echo "  - 多线程性能基准测试"
    echo "  - 功耗监控 (RAPL接口)"
    echo "  - 磁盘I/O监控"
    echo "  - 磁盘瓶颈检测"
    echo "  - 性能优化建议"
    echo "  - 结果保存到CSV文件"
else
    echo "❌ 编译失败！请检查依赖项是否正确安装。"
    echo ""
    echo "需要的依赖项:"
    echo "  - Faiss库 (libfaiss)"
    echo "  - OpenMP (libopenmp)"
    echo "  - 标准C++17编译器"
    exit 1
fi
