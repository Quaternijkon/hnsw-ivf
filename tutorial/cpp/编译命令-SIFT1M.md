# 编译 2-IVFFlat-SIFT1M 程序

## 基本编译命令

根据你常用的编译命令格式，编译 `2-IVFFlat-SIFT1M.cpp` 的命令如下：

```bash
g++ -std=c++17 -O3 -o 2-IVFFlat-SIFT1M 2-IVFFlat-SIFT1M.cpp \
    -I ../.. \
    -L ../../build/faiss \
    -Wl,-rpath,../../build/faiss \
    -lfaiss -lopenblas -fopenmp
```

## 编译并运行（一条命令）

```bash
g++ -std=c++17 -O3 -o 2-IVFFlat-SIFT1M 2-IVFFlat-SIFT1M.cpp \
    -I ../.. \
    -L ../../build/faiss \
    -Wl,-rpath,../../build/faiss \
    -lfaiss -lopenblas -fopenmp && ./2-IVFFlat-SIFT1M sift1M
```

## 使用编译脚本（推荐）

```bash
# 使用默认路径 sift1M
./compile_2-IVFFlat-SIFT1M.sh

# 或指定数据集路径
./compile_2-IVFFlat-SIFT1M.sh /path/to/sift1M
```

## 编译选项说明

- `-std=c++17`: 使用 C++17 标准
- `-O3`: 最高优化级别
- `-o 2-IVFFlat-SIFT1M`: 输出可执行文件名
- `-I ../..`: 头文件搜索路径（指向项目根目录）
- `-L ../../build/faiss`: 库文件搜索路径
- `-Wl,-rpath,../../build/faiss`: 设置运行时库搜索路径（避免设置 LD_LIBRARY_PATH）
- `-lfaiss`: 链接 faiss 库
- `-lopenblas`: 链接 OpenBLAS 库
- `-fopenmp`: 启用 OpenMP 支持

## 注意事项

1. **库路径**: 如果 faiss 库在 `../../build/lib` 而不是 `../../build/faiss`，需要修改 `-L` 和 `-Wl,-rpath` 参数：
   ```bash
   -L ../../build/lib -Wl,-rpath,../../build/lib
   ```

2. **数据集路径**: 程序接受一个可选参数指定 SIFT1M 数据集路径，默认为 `sift1M`

3. **环境变量**: 运行前建议设置：
   ```bash
   export OPENBLAS_NUM_THREADS=1
   export OMP_NUM_THREADS=4
   ```

## 验证编译

编译成功后，可以运行：

```bash
./2-IVFFlat-SIFT1M sift1M
```

如果看到程序开始加载数据集，说明编译成功！

