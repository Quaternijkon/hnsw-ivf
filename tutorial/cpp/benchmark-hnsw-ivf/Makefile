CXX = g++
CXXFLAGS = -std=c++17 -O3 -fopenmp -march=native -I../../.. -I..
LDFLAGS = -L../../../build/faiss -Wl,-rpath,../../../build/faiss -lfaiss -lopenblas -fopenmp

TARGET = benchmark_hnsw_ivf
SOURCES = benchmark_hnsw_ivf.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
	rm -f *.csv
	rm -rf indices

.PHONY: all run clean

