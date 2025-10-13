#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>

using namespace std;

// 获取当前实际内存使用量（通过/proc/self/status）
long getCurrentMemoryMB() {
    ifstream status_file("/proc/self/status");
    string line;
    while (getline(status_file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            istringstream iss(line);
            string key, value, unit;
            iss >> key >> value >> unit;
            return stol(value) / 1024; // 转换为MB
        }
    }
    return 0;
}

// 监控峰值内存的类
class PeakMemoryMonitor {
private:
    long start_memory_mb;
    long peak_memory_mb;
    bool monitoring;
    
public:
    PeakMemoryMonitor() : start_memory_mb(0), peak_memory_mb(0), monitoring(false) {}
    
    void start() {
        start_memory_mb = getCurrentMemoryMB();
        peak_memory_mb = start_memory_mb;
        monitoring = true;
        cout << "开始监控，初始内存: " << start_memory_mb << "MB" << endl;
    }
    
    void update() {
        if (monitoring) {
            long current_memory = getCurrentMemoryMB();
            if (current_memory > peak_memory_mb) {
                peak_memory_mb = current_memory;
                cout << "更新峰值内存: " << peak_memory_mb << "MB" << endl;
            }
        }
    }
    
    long getPeakMemoryMB() {
        return peak_memory_mb;
    }
    
    long getMemoryIncrease() {
        return peak_memory_mb - start_memory_mb;
    }
    
    void stop() {
        monitoring = false;
        cout << "停止监控，最终峰值内存: " << peak_memory_mb << "MB" << endl;
    }
};

int main() {
    cout << "=== 内存监控测试程序 ===" << endl;
    
    // 测试1: 分配大量内存
    cout << "\n测试1: 分配大量内存" << endl;
    PeakMemoryMonitor monitor1;
    monitor1.start();
    
    // 分配100MB内存
    vector<float> large_vector(100 * 1024 * 1024 / sizeof(float));
    monitor1.update();
    
    // 再分配50MB内存
    vector<float> another_vector(50 * 1024 * 1024 / sizeof(float));
    monitor1.update();
    
    monitor1.stop();
    cout << "峰值内存: " << monitor1.getPeakMemoryMB() << "MB" << endl;
    cout << "内存增长: " << monitor1.getMemoryIncrease() << "MB" << endl;
    
    // 释放内存
    large_vector.clear();
    another_vector.clear();
    vector<float>().swap(large_vector);
    vector<float>().swap(another_vector);
    
    // 等待一下让系统回收内存
    this_thread::sleep_for(chrono::milliseconds(100));
    
    // 测试2: 模拟搜索过程
    cout << "\n测试2: 模拟搜索过程" << endl;
    PeakMemoryMonitor monitor2;
    monitor2.start();
    
    // 模拟加载查询数据
    vector<float> query_data(1000 * 128); // 1000个128维向量
    monitor2.update();
    
    // 模拟搜索结果
    vector<int> search_results(1000 * 10); // 1000个查询，每个返回10个结果
    monitor2.update();
    
    // 模拟距离计算
    vector<float> distances(1000 * 10);
    monitor2.update();
    
    monitor2.stop();
    cout << "峰值内存: " << monitor2.getPeakMemoryMB() << "MB" << endl;
    cout << "内存增长: " << monitor2.getMemoryIncrease() << "MB" << endl;
    
    return 0;
}
