#include <iostream>
#include <iomanip>

int main() {
    std::cout << "=== mSPQ指标测试 ===" << std::endl;
    
    // 模拟一些QPS和mSPQ的计算
    double search_time_s = 0.12;
    int nq = 10000;
    
    double qps = nq / search_time_s;
    double mspq = (search_time_s * 1000.0) / nq;
    
    std::cout << "搜索时间: " << std::fixed << std::setprecision(2) << search_time_s << "s" << std::endl;
    std::cout << "查询数量: " << nq << std::endl;
    std::cout << "QPS: " << std::fixed << std::setprecision(2) << qps << std::endl;
    std::cout << "mSPQ: " << std::fixed << std::setprecision(2) << mspq << "ms" << std::endl;
    
    // 验证QPS和mSPQ的关系
    double qps_from_mspq = 1000.0 / mspq;
    std::cout << "从mSPQ计算的QPS: " << std::fixed << std::setprecision(2) << qps_from_mspq << std::endl;
    std::cout << "QPS和mSPQ互为倒数: " << (std::abs(qps - qps_from_mspq) < 0.01 ? "是" : "否") << std::endl;
    
    return 0;
}
