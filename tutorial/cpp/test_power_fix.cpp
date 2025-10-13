#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include <cstdint>
#include <iomanip>

using namespace std;

int main() {
    cout << "=== 功耗监控修复验证 ===" << endl;
    
    string rapl_package0_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj";
    string rapl_dram0_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj";
    
    // 读取初始值
    ifstream file0(rapl_package0_path);
    ifstream file1(rapl_dram0_path);
    
    if (!file0.is_open() || !file1.is_open()) {
        cout << "无法读取RAPL接口" << endl;
        return 1;
    }
    
    uint64_t cpu_start, dram_start;
    file0 >> cpu_start;
    file1 >> dram_start;
    
    cout << "初始CPU累计能耗: " << cpu_start << " 微焦耳" << endl;
    cout << "初始内存累计能耗: " << dram_start << " 微焦耳" << endl;
    
    // 等待一段时间
    cout << "等待3秒..." << endl;
    this_thread::sleep_for(chrono::seconds(3));
    
    // 读取结束值
    file0.close();
    file1.close();
    
    ifstream file0_end(rapl_package0_path);
    ifstream file1_end(rapl_dram0_path);
    
    uint64_t cpu_end, dram_end;
    file0_end >> cpu_end;
    file1_end >> dram_end;
    
    cout << "结束CPU累计能耗: " << cpu_end << " 微焦耳" << endl;
    cout << "结束内存累计能耗: " << dram_end << " 微焦耳" << endl;
    
    // 计算增量
    if (cpu_end >= cpu_start) {
        double cpu_energy_j = (cpu_end - cpu_start) / 1000000.0;
        cout << "CPU能耗增量: " << fixed << setprecision(6) << cpu_energy_j << " 焦耳" << endl;
        cout << "CPU平均功耗: " << fixed << setprecision(2) << (cpu_energy_j / 3.0) << " 瓦特" << endl;
    } else {
        cout << "警告: CPU能耗计数器可能发生了回绕" << endl;
    }
    
    if (dram_end >= dram_start) {
        double dram_energy_j = (dram_end - dram_start) / 1000000.0;
        cout << "内存能耗增量: " << fixed << setprecision(6) << dram_energy_j << " 焦耳" << endl;
        cout << "内存平均功耗: " << fixed << setprecision(2) << (dram_energy_j / 3.0) << " 瓦特" << endl;
    } else {
        cout << "警告: 内存能耗计数器可能发生了回绕" << endl;
    }
    
    cout << "\n测试完成！" << endl;
    return 0;
}
