#!/bin/bash
# 运行 2-IVFFlat 程序的脚本
# 设置 OpenBLAS 线程数以避免内存分配错误

# 设置 OpenBLAS 使用单线程（避免内存分配问题）
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

# 设置库路径（指向重新编译后的 faiss 库）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$SCRIPT_DIR/../../build/lib:$LD_LIBRARY_PATH"

# 检查库文件是否存在
if [ ! -f "$SCRIPT_DIR/../../build/lib/libfaiss.so" ] && [ ! -f "$SCRIPT_DIR/../../build/lib/libfaiss.a" ]; then
    echo "警告: 未找到编译后的 faiss 库文件！"
    echo "请先运行: ./重新编译faiss.sh"
    echo ""
    echo "或者手动编译:"
    echo "  cd ../../"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF"
    echo "  make -j\$(nproc) faiss"
    exit 1
fi

# 检查可执行文件是否存在
if [ ! -f "$SCRIPT_DIR/2-IVFFlat" ]; then
    echo "错误: 未找到可执行文件 2-IVFFlat！"
    echo "请先编译示例程序:"
    echo "  g++ -o 2-IVFFlat 2-IVFFlat.cpp -I../../ -L../../build/lib -lfaiss -fopenmp -O3"
    exit 1
fi

# 运行程序
echo "库路径: $LD_LIBRARY_PATH"
echo "运行程序..."
echo ""
./2-IVFFlat

