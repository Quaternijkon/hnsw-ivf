#include <iostream>
#include <omp.h>

int main() {
    // 设置OpenMP线程数为20
    omp_set_num_threads(20);
    
    std::cout << "=== OpenMP线程数测试 ===" << std::endl;
    std::cout << "设置的线程数: 20" << std::endl;
    std::cout << "实际线程数: " << omp_get_max_threads() << std::endl;
    
    // 测试并行区域
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        
        #pragma omp critical
        {
            std::cout << "线程 " << thread_id << " / " << num_threads << " 正在运行" << std::endl;
        }
    }
    
    return 0;
}
