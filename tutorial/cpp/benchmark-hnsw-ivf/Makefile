CXX = g++
CXXFLAGS = -std=c++17 -O3 -fopenmp -march=native -I../../.. -I..
LDFLAGS = -L../../../build/faiss -Wl,-rpath,../../../build/faiss -lfaiss -lopenblas -fopenmp

# 标准版本（简洁输出）
TARGET = benchmark_hnsw_ivf
SOURCES = benchmark_hnsw_ivf.cpp

# 详细内存分析版本
TARGET_DETAILED = benchmark_hnsw_ivf_detailed
SOURCES_DETAILED = benchmark_hnsw_ivf_detailed.cpp

all: $(TARGET) $(TARGET_DETAILED)

# 构建标准版本
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

# 构建详细分析版本
$(TARGET_DETAILED): $(SOURCES_DETAILED)
	$(CXX) $(CXXFLAGS) -o $(TARGET_DETAILED) $(SOURCES_DETAILED) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

run-detailed: $(TARGET_DETAILED)
	./$(TARGET_DETAILED)

clean:
	rm -f $(TARGET) $(TARGET_DETAILED)
	rm -f *.csv
	rm -rf indices

.PHONY: all run run-detailed clean

