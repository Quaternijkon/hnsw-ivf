#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>

using namespace std;

struct TestResult {
    double qps;
    double latency;
    double power;
};

void test_flexible_runs(int num_runs) {
    cout << "\n=== 测试 " << num_runs << " 次运行 ===" << endl;
    
    // 模拟测试数据
    vector<TestResult> results;
    for (int i = 0; i < num_runs; ++i) {
        TestResult result;
        result.qps = 1000.0 + (i * 100.0) + (rand() % 200 - 100); // 添加随机变化
        result.latency = 1.0 + (i * 0.1) + (rand() % 20 - 10) / 100.0;
        result.power = 50.0 + (i * 5.0) + (rand() % 20 - 10);
        results.push_back(result);
    }
    
    cout << "原始数据:" << endl;
    for (int i = 0; i < num_runs; ++i) {
        cout << "  运行" << (i+1) << ": QPS=" << fixed << setprecision(0) << results[i].qps 
             << ", Latency=" << fixed << setprecision(2) << results[i].latency 
             << "ms, Power=" << fixed << setprecision(1) << results[i].power << "W" << endl;
    }
    
    TestResult avg_result;
    
    if (num_runs == 1) {
        // 单次运行，直接使用结果
        avg_result = results[0];
        cout << "处理方式: 直接使用单次结果" << endl;
    } else if (num_runs == 2) {
        // 两次运行，直接取平均
        avg_result.qps = (results[0].qps + results[1].qps) / 2.0;
        avg_result.latency = (results[0].latency + results[1].latency) / 2.0;
        avg_result.power = (results[0].power + results[1].power) / 2.0;
        cout << "处理方式: 直接取两次结果的平均值" << endl;
    } else {
        // 3次或以上运行，按QPS排序后去掉最大最小值
        vector<pair<double, int>> qps_with_index;
        for (int i = 0; i < num_runs; ++i) {
            qps_with_index.push_back({results[i].qps, i});
        }
        
        // 按QPS排序（升序）
        sort(qps_with_index.begin(), qps_with_index.end());
        
        // 计算有效数据数量（去掉最大最小值）
        int valid_count = num_runs - 2;
        if (valid_count <= 0) valid_count = 1;
        
        // 选择有效数据的索引
        vector<int> valid_indices;
        if (valid_count == 1) {
            valid_indices.push_back(qps_with_index[num_runs / 2].second);
        } else {
            for (int i = 1; i <= valid_count; ++i) {
                valid_indices.push_back(qps_with_index[i].second);
            }
        }
        
        cout << "QPS排序后:" << endl;
        for (int i = 0; i < num_runs; ++i) {
            cout << "  排序" << (i+1) << ": QPS=" << fixed << setprecision(0) << qps_with_index[i].first 
                 << " (原运行" << (qps_with_index[i].second + 1) << ")" << endl;
        }
        
        cout << "有效数据: ";
        for (int idx : valid_indices) {
            cout << "运行" << (idx + 1) << " ";
        }
        cout << "(" << valid_indices.size() << "个)" << endl;
        
        // 计算平均值
        double sum_qps = 0.0, sum_latency = 0.0, sum_power = 0.0;
        for (int idx : valid_indices) {
            sum_qps += results[idx].qps;
            sum_latency += results[idx].latency;
            sum_power += results[idx].power;
        }
        
        int count = valid_indices.size();
        avg_result.qps = sum_qps / count;
        avg_result.latency = sum_latency / count;
        avg_result.power = sum_power / count;
        
        cout << "处理方式: 去掉最大最小值，剩余" << count << "个数据取平均" << endl;
    }
    
    cout << "最终结果: QPS=" << fixed << setprecision(0) << avg_result.qps 
         << ", Latency=" << fixed << setprecision(2) << avg_result.latency 
         << "ms, Power=" << fixed << setprecision(1) << avg_result.power << "W" << endl;
}

int main() {
    cout << "=== 灵活运行次数测试 ===" << endl;
    cout << "测试不同运行次数的处理逻辑" << endl;
    
    srand(42); // 固定随机种子，确保结果可重复
    
    // 测试不同的运行次数
    test_flexible_runs(1);
    test_flexible_runs(2);
    test_flexible_runs(3);
    test_flexible_runs(5);
    test_flexible_runs(10);
    
    cout << "\n=== 配置说明 ===" << endl;
    cout << "只需修改 num_runs_per_thread_setting 即可调整运行次数:" << endl;
    cout << "- 1次: 直接使用单次结果" << endl;
    cout << "- 2次: 直接取两次结果的平均值" << endl;
    cout << "- 3次或以上: 按QPS排序，去掉最大最小值，剩余数据取平均" << endl;
    cout << "- 取样策略始终以QPS为基准进行排序和筛选" << endl;
    
    return 0;
}

