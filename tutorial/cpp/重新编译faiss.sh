#!/bin/bash
# 重新编译 faiss 库的脚本
# 由于修改了源代码（IndexIVF.h 和 IndexIVF.cpp），需要重新编译

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始重新编译 faiss 库"
echo "=========================================="

# 进入项目根目录
cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)
echo "项目根目录: $PROJECT_ROOT"

# 检查是否存在 build 目录
if [ -d "build" ]; then
    echo "发现现有的 build 目录，将清理并重新配置..."
    rm -rf build
fi

# 创建 build 目录
mkdir -p build
cd build

echo ""
echo "步骤 1: 配置 CMake..."
echo "----------------------------------------"

# 配置 CMake（根据你的需求调整选项）
# 基本配置（CPU 版本）
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTING=OFF \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DFAISS_OPT_LEVEL=avx2 \
    -DCMAKE_INSTALL_PREFIX=./install

# 如果需要 GPU 支持，取消下面的注释并注释掉上面的配置
# cmake .. \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DBUILD_SHARED_LIBS=ON \
#     -DBUILD_TESTING=OFF \
#     -DFAISS_ENABLE_GPU=ON \
#     -DFAISS_ENABLE_PYTHON=OFF \
#     -DFAISS_OPT_LEVEL=avx2 \
#     -DCMAKE_CUDA_ARCHITECTURES="75;80" \
#     -DCMAKE_INSTALL_PREFIX=./install

if [ $? -ne 0 ]; then
    echo "错误: CMake 配置失败！"
    exit 1
fi

echo ""
echo "步骤 2: 编译 faiss 库..."
echo "----------------------------------------"

# 编译（使用多线程加速，根据你的 CPU 核心数调整 -j 后面的数字）
# 获取 CPU 核心数
CPU_CORES=$(nproc)
echo "使用 $CPU_CORES 个线程进行编译..."

make -j$CPU_CORES faiss

if [ $? -ne 0 ]; then
    echo "错误: 编译失败！"
    exit 1
fi

echo ""
echo "步骤 3: 编译优化版本（可选）..."
echo "----------------------------------------"

# 如果需要 AVX2 优化版本
make -j$CPU_CORES faiss_avx2

# 如果需要 AVX512 优化版本（如果你的 CPU 支持）
# make -j$CPU_CORES faiss_avx512

echo ""
echo "=========================================="
echo "编译完成！"
echo "=========================================="
echo ""
echo "编译产物位置:"
echo "  - 静态库: build/lib/libfaiss.a"
echo "  - 动态库: build/lib/libfaiss.so (如果 BUILD_SHARED_LIBS=ON)"
echo "  - AVX2 版本: build/lib/libfaiss_avx2.so"
echo ""
echo "下一步："
echo "  1. 编译示例程序:"
echo "     cd tutorial/cpp"
echo "     g++ -o 2-IVFFlat 2-IVFFlat.cpp \\"
echo "         -I../../ \\"
echo "         -L../../build/lib -lfaiss \\"
echo "         -fopenmp -O3"
echo ""
echo "  2. 运行示例:"
echo "     ./run_2-IVFFlat.sh"
echo ""

