#include <iostream>
#include <iomanip>

int main() {
    std::cout << "=== 精度设置测试 ===" << std::endl;
    
    // 模拟一些延迟数据
    double mspq = 0.0123456;
    double mean_latency = 0.1234567;
    double p50_latency = 0.0987654;
    double p99_latency = 0.4567890;
    
    std::cout << "原始数据:" << std::endl;
    std::cout << "mSPQ: " << mspq << "ms" << std::endl;
    std::cout << "平均延迟: " << mean_latency << "ms" << std::endl;
    std::cout << "P50延迟: " << p50_latency << "ms" << std::endl;
    std::cout << "P99延迟: " << p99_latency << "ms" << std::endl;
    
    std::cout << "\n设置4位有效数字后:" << std::endl;
    std::cout << "mSPQ: " << std::fixed << std::setprecision(4) << mspq << "ms" << std::endl;
    std::cout << "平均延迟: " << std::fixed << std::setprecision(4) << mean_latency << "ms" << std::endl;
    std::cout << "P50延迟: " << std::fixed << std::setprecision(4) << p50_latency << "ms" << std::endl;
    std::cout << "P99延迟: " << std::fixed << std::setprecision(4) << p99_latency << "ms" << std::endl;
    
    return 0;
}
