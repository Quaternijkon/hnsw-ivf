#!/bin/bash
# 编译并运行 2-IVFFlat-SIFT1M 程序

# SIFT1M 数据集路径（可以通过命令行参数指定）
SIFT1M_DIR="${1:-sift1M}"

# 编译命令
g++ -std=c++17 -O3 -march=native -mavx2 -mfma -fopenmp -o 2-IVFFlat-SIFT1M 2-IVFFlat-SIFT1M.cpp \
    -I ../.. \
    -L ../../build/faiss \
    -Wl,-rpath,../../build/faiss \
    -lfaiss -lopenblas

# 检查编译是否成功
if [ $? -eq 0 ]; then
    echo "编译成功！"
    echo ""
    
    # 检查数据集目录是否存在
    if [ ! -d "$SIFT1M_DIR" ]; then
        echo "警告: 未找到 SIFT1M 数据集目录: $SIFT1M_DIR"
        echo ""
        echo "请下载 SIFT1M 数据集："
        echo "  1. 访问: http://corpus-texmex.irisa.fr/"
        echo "  2. 下载 ANN_SIFT1M 数据集"
        echo "  3. 解压到当前目录的 sift1M 文件夹"
        echo ""
        echo "或者指定数据集路径："
        echo "  ./compile_2-IVFFlat-SIFT1M.sh /path/to/sift1M"
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
    
    echo "使用 SIFT1M 数据集目录: $SIFT1M_DIR"
    echo "运行程序..."
    echo ""
    
    # 运行程序
    ./2-IVFFlat-SIFT1M "$SIFT1M_DIR"
else
    echo "编译失败！"
    exit 1
fi

