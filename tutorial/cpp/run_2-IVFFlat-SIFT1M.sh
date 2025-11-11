#!/bin/bash
# 运行 2-IVFFlat-SIFT1M 程序的脚本
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
    exit 1
fi

# 检查可执行文件是否存在
if [ ! -f "$SCRIPT_DIR/2-IVFFlat-SIFT1M" ]; then
    echo "错误: 未找到可执行文件 2-IVFFlat-SIFT1M！"
    echo "请先编译示例程序:"
    echo "  g++ -o 2-IVFFlat-SIFT1M 2-IVFFlat-SIFT1M.cpp -I../../ -L../../build/lib -lfaiss -fopenmp -O3"
    exit 1
fi

# SIFT1M 数据集路径（可以通过命令行参数指定）
SIFT1M_DIR="${1:-sift1M}"

# 检查数据集目录是否存在
if [ ! -d "$SIFT1M_DIR" ]; then
    echo "错误: 未找到 SIFT1M 数据集目录: $SIFT1M_DIR"
    echo ""
    echo "请下载 SIFT1M 数据集："
    echo "  1. 访问: http://corpus-texmex.irisa.fr/"
    echo "  2. 下载 ANN_SIFT1M 数据集"
    echo "  3. 解压到当前目录的 sift1M 文件夹"
    echo ""
    echo "或者指定数据集路径："
    echo "  ./run_2-IVFFlat-SIFT1M.sh /path/to/sift1M"
    exit 1
fi

# 检查必需的数据文件
REQUIRED_FILES=(
    "$SIFT1M_DIR/sift_learn.fvecs"
    "$SIFT1M_DIR/sift_base.fvecs"
    "$SIFT1M_DIR/sift_query.fvecs"
    "$SIFT1M_DIR/sift_groundtruth.ivecs"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "错误: 未找到必需的数据文件: $file"
        exit 1
    fi
done

# 运行程序
echo "使用 SIFT1M 数据集目录: $SIFT1M_DIR"
echo "库路径: $LD_LIBRARY_PATH"
echo "运行程序..."
echo ""
./2-IVFFlat-SIFT1M "$SIFT1M_DIR"

