#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::cout << "=== 比例参数计算测试 ===" << std::endl;
    
    // 模拟一些nlist值
    std::vector<int> nlist_values = {1953, 3906, 7812, 15625, 31250};
    
    // 模拟nprobe比例
    std::vector<double> nprobe_ratios = {0.004096, 0.008192, 0.016384, 0.032768, 0.065536};
    
    // 模拟efsearch比例
    std::vector<double> efsearch_ratios = {0.5, 0.9, 1.0, 1.1, 1.5, 2.0};
    
    std::cout << "参数计算示例:" << std::endl;
    std::cout << "nlist\tnprobe_ratio\tnprobe\tefsearch_ratio\tefsearch" << std::endl;
    std::cout << "-----\t------------\t------\t--------------\t-------" << std::endl;
    
    for (int nlist : nlist_values) {
        for (double nprobe_ratio : nprobe_ratios) {
            int nprobe = std::max(1, static_cast<int>(nlist * nprobe_ratio));
            
            for (double efsearch_ratio : efsearch_ratios) {
                int efsearch = std::max(1, static_cast<int>(nprobe * efsearch_ratio));
                
                std::cout << nlist << "\t" << nprobe_ratio << "\t\t" << nprobe 
                         << "\t" << efsearch_ratio << "\t\t" << efsearch << std::endl;
            }
        }
    }
    
    std::cout << "\n参数合理性分析:" << std::endl;
    std::cout << "- nprobe总是nlist的固定比例，确保搜索范围合理" << std::endl;
    std::cout << "- efsearch总是nprobe的固定比例，确保搜索深度合理" << std::endl;
    std::cout << "- 避免了固定值可能导致的参数不合理问题" << std::endl;
    
    return 0;
}
