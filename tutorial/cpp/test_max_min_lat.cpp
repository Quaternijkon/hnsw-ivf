#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

using namespace std;

struct TestResult {
    int threads;
    double time_s;
    double qps;
    double avg_power;
    double cpu_energy;
    double mem_energy;
    double power_per_query;
    double efficiency;
    double avg_latency;
    double p50_latency;
    double p99_latency;
    double max_latency;
    double min_latency;
    double peak_memory;
};

int main() {
    // 模拟测试数据
    vector<TestResult> results = {
        {1, 8.5, 1177.46, 121.26, 831.48, 198.37, 874.66, 9.71, 0.8487, 0.8, 1.2, 2.5, 0.3, 540.97},
        {10, 1.3, 7922.13, 154.45, 160.61, 34.36, 24.61, 51.29, 1.1392, 1.1, 1.5, 2.8, 0.5, 540.97},
        {20, 1.1, 8730.37, 174.98, 168.46, 31.97, 22.96, 49.89, 2.0735, 2.0, 3.0, 4.2, 1.2, 540.97},
        {40, 1.2, 8518.31, 194.24, 194.80, 33.37, 26.82, 43.85, 3.5807, 3.5, 4.5, 5.8, 2.1, 540.97}
    };
    
    // 输出到终端
    cout << "===== Faiss Benchmark Summary =====" << endl;
    cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
    cout << "| " << left << setw(8) << "Threads" 
         << " | " << setw(10) << "Time(s)" 
         << " | " << setw(12) << "QPS" 
         << " | " << setw(10) << "Avg Power(W)" 
         << " | " << setw(12) << "CPU Energy(J)" 
         << " | " << setw(12) << "Mem Energy(J)" 
         << " | " << setw(15) << "Power/Query(mJ)" 
         << " | " << setw(12) << "Efficiency" 
         << " | " << setw(12) << "Avg Lat(ms)" 
         << " | " << setw(10) << "P50 Lat(ms)" 
         << " | " << setw(10) << "P99 Lat(ms)" 
         << " | " << setw(10) << "Max Lat(ms)" 
         << " | " << setw(10) << "Min Lat(ms)" 
         << " | " << setw(12) << "Peak Mem(MB)" << " |" << endl;
    cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
    
    // 创建文件输出流
    ofstream output_file("test_benchmark_results.txt");
    
    // 输出表头到文件
    output_file << "Threads,Time(s),QPS,Avg Power(W),CPU Energy(J),Mem Energy(J),Power/Query(mJ),Efficiency,Avg Lat(ms),P50 Lat(ms),P99 Lat(ms),Max Lat(ms),Min Lat(ms),Peak Mem(MB)" << endl;
    
    for (const auto& result : results) {
        // 输出到终端
        cout << "| " << left << setw(8) << result.threads 
             << " | " << fixed << setprecision(4) << setw(10) << result.time_s
             << " | " << fixed << setprecision(2) << setw(12) << result.qps
             << " | " << fixed << setprecision(2) << setw(10) << result.avg_power
             << " | " << fixed << setprecision(4) << setw(12) << result.cpu_energy
             << " | " << fixed << setprecision(4) << setw(12) << result.mem_energy
             << " | " << fixed << setprecision(4) << setw(15) << result.power_per_query
             << " | " << fixed << setprecision(2) << setw(12) << result.efficiency
             << " | " << fixed << setprecision(4) << setw(12) << result.avg_latency
             << " | " << fixed << setprecision(4) << setw(10) << result.p50_latency
             << " | " << fixed << setprecision(4) << setw(10) << result.p99_latency
             << " | " << fixed << setprecision(4) << setw(10) << result.max_latency
             << " | " << fixed << setprecision(4) << setw(10) << result.min_latency
             << " | " << fixed << setprecision(2) << setw(12) << result.peak_memory << " |" << endl;
        
        // 输出到文件（CSV格式）
        output_file << result.threads << ","
                   << fixed << setprecision(4) << result.time_s << ","
                   << fixed << setprecision(2) << result.qps << ","
                   << fixed << setprecision(2) << result.avg_power << ","
                   << fixed << setprecision(4) << result.cpu_energy << ","
                   << fixed << setprecision(4) << result.mem_energy << ","
                   << fixed << setprecision(4) << result.power_per_query << ","
                   << fixed << setprecision(2) << result.efficiency << ","
                   << fixed << setprecision(4) << result.avg_latency << ","
                   << fixed << setprecision(4) << result.p50_latency << ","
                   << fixed << setprecision(4) << result.p99_latency << ","
                   << fixed << setprecision(4) << result.max_latency << ","
                   << fixed << setprecision(4) << result.min_latency << ","
                   << fixed << setprecision(2) << result.peak_memory << endl;
    }
    cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
    
    // 关闭文件
    output_file.close();
    cout << "\n结果已保存到 test_benchmark_results.txt 文件中" << endl;
    
    return 0;
}
