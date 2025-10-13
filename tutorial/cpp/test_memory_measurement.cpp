#include <iostream>
#include <vector>
#include <sys/resource.h>
#include <iomanip>

using namespace std;

double get_current_memory_mb() {
    rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    long peak_memory_bytes = usage.ru_maxrss;
    peak_memory_bytes *= 1024; // Linux系统需要乘以1024
    return peak_memory_bytes / (1024.0 * 1024.0);
}

int main() {
    cout << "=== 峰值内存测量测试 ===" << endl;
    
    // 测试1：分配少量内存
    cout << "测试1：分配10MB内存" << endl;
    double mem_before_1 = get_current_memory_mb();
    cout << "分配前内存: " << fixed << setprecision(2) << mem_before_1 << " MB" << endl;
    
    vector<int> small_vec(2.5 * 1024 * 1024); // 10MB
    for (int i = 0; i < small_vec.size(); i += 100000) {
        small_vec[i] = i;
    }
    
    double mem_after_1 = get_current_memory_mb();
    cout << "分配后内存: " << fixed << setprecision(2) << mem_after_1 << " MB" << endl;
    cout << "内存增量: " << fixed << setprecision(2) << (mem_after_1 - mem_before_1) << " MB" << endl;
    
    // 测试2：分配更多内存
    cout << "\n测试2：分配50MB内存" << endl;
    double mem_before_2 = get_current_memory_mb();
    cout << "分配前内存: " << fixed << setprecision(2) << mem_before_2 << " MB" << endl;
    
    vector<int> large_vec(12.5 * 1024 * 1024); // 50MB
    for (int i = 0; i < large_vec.size(); i += 100000) {
        large_vec[i] = i;
    }
    
    double mem_after_2 = get_current_memory_mb();
    cout << "分配后内存: " << fixed << setprecision(2) << mem_after_2 << " MB" << endl;
    cout << "内存增量: " << fixed << setprecision(2) << (mem_after_2 - mem_before_2) << " MB" << endl;
    
    // 测试3：释放内存
    cout << "\n测试3：释放内存" << endl;
    small_vec.clear();
    small_vec.shrink_to_fit();
    large_vec.clear();
    large_vec.shrink_to_fit();
    
    double mem_after_3 = get_current_memory_mb();
    cout << "释放后内存: " << fixed << setprecision(2) << mem_after_3 << " MB" << endl;
    cout << "峰值内存: " << fixed << setprecision(2) << mem_after_2 << " MB" << endl;
    
    cout << "\n测试完成！" << endl;
    return 0;
}
