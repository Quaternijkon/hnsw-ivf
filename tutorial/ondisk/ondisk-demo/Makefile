# 编译器
CXX = g++

# 编译标志
# -std=c++17: 使用 C++17 标准
# -O3: 优化
# -Wall: 开启所有警告
# -g: 包含调试信息
# -fopenmp: 启用 OpenMP (Faiss 内部使用)
# -I ../.. : (新) 假设在faiss根目录的子目录中，用于查找头文件
CXXFLAGS = -std=c++17 -O3 -Wall -g -fopenmp -I ../../..

# 链接标志
# -L ../../build/faiss : (新) 指定库搜索路径
# -Wl,-rpath,... : (新) 指定运行时库路径，避免设置 LD_LIBRARY_PATH
# -lfaiss : 链接 Faiss 库
# -lopenblas : 链接 BLAS
# -fopenmp : 链接 OpenMP
LDFLAGS = -L ../../../build/faiss -Wl,-rpath,../../../build/faiss -lfaiss -lopenblas -fopenmp

# 目标可执行文件
TARGET_PQ = ondisk_demo
TARGET_FLAT = ondisk_flat

# 源文件
SOURCES_PQ = ondisk_demo.cpp
SOURCES_FLAT = ondisk_flat.cpp

.PHONY: all clean run run-pq run-flat run-all help

# 默认目标：构建所有版本
all: $(TARGET_PQ) $(TARGET_FLAT)
	@echo "Build complete. Available targets:"
	@echo "  make run      - Run PQ version (default)"
	@echo "  make run-pq   - Run PQ version"
	@echo "  make run-flat - Run Flat version"
	@echo "  make run-all  - Run both versions"

# 构建PQ版本（IndexIVFPQ）
$(TARGET_PQ): $(SOURCES_PQ)
	$(CXX) $(CXXFLAGS) -o $(TARGET_PQ) $(SOURCES_PQ) $(LDFLAGS)
	@echo "Built $(TARGET_PQ) (PQ version)"

# 构建Flat版本（IndexIVFFlat）
$(TARGET_FLAT): $(SOURCES_FLAT)
	$(CXX) $(CXXFLAGS) -o $(TARGET_FLAT) $(SOURCES_FLAT) $(LDFLAGS)
	@echo "Built $(TARGET_FLAT) (Flat version)"

# 默认运行PQ版本
run: $(TARGET_PQ)
	@echo "=========================================="
	@echo "Running PQ version (IndexIVFPQ)"
	@echo "=========================================="
	./$(TARGET_PQ)
	@echo "=========================================="
	@echo "PQ version finished"
	@echo "=========================================="

# 运行PQ版本
run-pq: $(TARGET_PQ)
	@echo "=========================================="
	@echo "Running PQ version (IndexIVFPQ)"
	@echo "=========================================="
	./$(TARGET_PQ)
	@echo "=========================================="
	@echo "PQ version finished"
	@echo "=========================================="

# 运行Flat版本
run-flat: $(TARGET_FLAT)
	@echo "=========================================="
	@echo "Running Flat version (IndexIVFFlat)"
	@echo "=========================================="
	./$(TARGET_FLAT)
	@echo "=========================================="
	@echo "Flat version finished"
	@echo "=========================================="

# 运行两个版本（用于对比）
run-all: $(TARGET_PQ) $(TARGET_FLAT)
	@echo "=========================================="
	@echo "Running PQ version (IndexIVFPQ)"
	@echo "=========================================="
	./$(TARGET_PQ)
	@echo ""
	@echo "=========================================="
	@echo "Running Flat version (IndexIVFFlat)"
	@echo "=========================================="
	./$(TARGET_FLAT)
	@echo "=========================================="
	@echo "All versions finished"
	@echo "=========================================="

# 帮助信息
help:
	@echo "Available targets:"
	@echo "  make all      - Build both PQ and Flat versions"
	@echo "  make run       - Run PQ version (default)"
	@echo "  make run-pq    - Run PQ version (IndexIVFPQ)"
	@echo "  make run-flat  - Run Flat version (IndexIVFFlat)"
	@echo "  make run-all   - Run both versions sequentially"
	@echo "  make clean     - Remove all executables and index files"
	@echo "  make help      - Show this help message"

# 清理目标：移除所有可执行文件和生成的索引文件
clean:
	rm -f $(TARGET_PQ) $(TARGET_FLAT)
	rm -f ondisk_demo.faiss ondisk_demo.ivf
	rm -f ondisk_demo_flat.faiss ondisk_demo_flat.ivf
	@echo "Cleaned up all executables and index files."