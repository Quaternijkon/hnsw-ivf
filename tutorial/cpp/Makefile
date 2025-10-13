CXX = g++
CXXFLAGS = -std=c++17 -O3 -fopenmp -Wall -Wextra
INCLUDES = -I/home/gpu/dry/faiss -I/usr/local/include -I/usr/include
LIBS = -L/home/gpu/dry/faiss/build/faiss -lfaiss -lopenblas -lgomp

# 目标文件
TARGETS = benchmark_advanced

# 默认目标
all: $(TARGETS)

# 编译高级benchmark程序
benchmark_advanced: benchmark_advanced.cpp config_parser.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

# 清理
clean:
	rm -f $(TARGETS) *.o

# 运行测试
run: benchmark_advanced
	./benchmark_advanced

# 运行测试（指定配置文件）
run-config: benchmark_advanced
	./benchmark_advanced benchmark.config

.PHONY: all clean run run-config
