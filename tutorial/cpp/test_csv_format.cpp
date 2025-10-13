#include <iostream>
#include <iomanip>
#include <fstream>

int main() {
    std::cout << "=== CSV格式测试 ===" << std::endl;
    
    // 模拟一些测试数据
    struct TestData {
        int nlist = 1953;
        int efconstruction = 40;
        int nprobe = 16;
        int efsearch = 16;
        double training_memory_mb = 123.45;
        double add_memory_mb = 234.56;
        double training_time_s = 12.34;
        double total_time_s = 45.67;
        double recall = 0.9876;
        double qps = 83333.33;
        double mspq = 0.0120;
        double search_memory_mb = 12.34;
        double search_time_s = 0.12;
        double mean_latency_ms = 0.1500;
        double p50_latency_ms = 0.1200;
        double p99_latency_ms = 0.4500;
    };
    
    TestData data;
    
    // 创建测试CSV文件
    std::ofstream file("test_output.csv");
    if (!file.is_open()) {
        std::cerr << "无法创建测试文件" << std::endl;
        return 1;
    }
    
    // 写入CSV头部
    file << "nlist,efconstruction,nprobe,efsearch,training_memory_mb,add_memory_mb,training_time_s,total_time_s,"
         << "recall,qps,mspq,search_memory_mb,search_time_s,mean_latency_ms,p50_latency_ms,p99_latency_ms" << std::endl;
    
    // 写入数据
    file << data.nlist << "," << data.efconstruction << "," << data.nprobe << "," << data.efsearch << ","
         << std::fixed << std::setprecision(2) << data.training_memory_mb << ","
         << std::fixed << std::setprecision(2) << data.add_memory_mb << ","
         << std::fixed << std::setprecision(4) << data.training_time_s << ","
         << std::fixed << std::setprecision(4) << data.total_time_s << ","
         << std::fixed << std::setprecision(4) << data.recall << ","
         << std::fixed << std::setprecision(2) << data.qps << ","
         << std::fixed << std::setprecision(4) << data.mspq << ","
         << std::fixed << std::setprecision(2) << data.search_memory_mb << ","
         << std::fixed << std::setprecision(4) << data.search_time_s << ","
         << std::fixed << std::setprecision(4) << data.mean_latency_ms << ","
         << std::fixed << std::setprecision(4) << data.p50_latency_ms << ","
         << std::fixed << std::setprecision(4) << data.p99_latency_ms << std::endl;
    
    file.close();
    
    std::cout << "测试CSV文件已创建: test_output.csv" << std::endl;
    std::cout << "文件内容:" << std::endl;
    
    // 读取并显示文件内容
    std::ifstream read_file("test_output.csv");
    std::string line;
    while (std::getline(read_file, line)) {
        std::cout << line << std::endl;
    }
    
    return 0;
}
