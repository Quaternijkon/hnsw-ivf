#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

int main() {
    cout << "=== 单次运行测试验证 ===" << endl;
    
    // 模拟单次测试数据
    vector<double> qps = {1200.0, 1500.0, 1800.0, 2000.0};
    vector<double> latency = {1.5, 1.2, 1.0, 0.9};
    
    cout << "单次运行结果:" << endl;
    for (int i = 0; i < 4; ++i) {
        cout << "线程" << (i+1) << ": QPS=" << fixed << setprecision(0) << qps[i] 
             << ", Latency=" << fixed << setprecision(1) << latency[i] << "ms" << endl;
    }
    
    cout << "\n特点:" << endl;
    cout << "1. 每组线程配置只运行一次" << endl;
    cout << "2. 无需复杂的取样策略" << endl;
    cout << "3. 结果直接反映单次运行的真实性能" << endl;
    cout << "4. 测试速度更快" << endl;
    
    cout << "\n适用场景:" << endl;
    cout << "- 快速性能评估" << endl;
    cout << "- 参数调优" << endl;
    cout << "- 初步性能对比" << endl;
    
    return 0;
}
