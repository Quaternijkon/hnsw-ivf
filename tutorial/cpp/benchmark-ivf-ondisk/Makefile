CXX = g++
CXXFLAGS = -std=c++17 -O3 -fopenmp -march=native -I../../.. -I..
LDFLAGS = -L../../../build/faiss -Wl,-rpath,../../../build/faiss -lfaiss -lopenblas -fopenmp

TARGET = benchmark_ivf_ondisk
SOURCES = benchmark_ivf_ondisk.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
	rm -f *.csv

.PHONY: all run clean

