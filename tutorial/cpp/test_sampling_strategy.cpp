#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>

using namespace std;

int main() {
    cout << "=== 取样策略测试 ===" << endl;
    
    // 模拟5次测试的QPS数据
    vector<double> test_qps = {1000.0, 1200.0, 1100.0, 1300.0, 1050.0};
    
    cout << "原始5次测试QPS: ";
    for (double qps : test_qps) {
        cout << fixed << setprecision(2) << qps << " ";
    }
    cout << endl;
    
    // 创建用于排序的向量，存储QPS值和索引
    vector<pair<double, int>> qps_with_index;
    for (int i = 0; i < test_qps.size(); ++i) {
        qps_with_index.push_back({test_qps[i], i});
    }
    
    // 按QPS排序（升序）
    sort(qps_with_index.begin(), qps_with_index.end());
    
    cout << "排序后的QPS和索引: ";
    for (const auto& pair : qps_with_index) {
        cout << "(" << fixed << setprecision(2) << pair.first << "," << pair.second << ") ";
    }
    cout << endl;
    
    // 去掉最小值（索引0）和最大值（索引4），保留中间3个值（索引1,2,3）
    vector<int> valid_indices = {qps_with_index[1].second, qps_with_index[2].second, qps_with_index[3].second};
    
    cout << "保留的索引: ";
    for (int idx : valid_indices) {
        cout << idx << " ";
    }
    cout << endl;
    
    cout << "保留的QPS值: ";
    double sum = 0.0;
    for (int idx : valid_indices) {
        cout << fixed << setprecision(2) << test_qps[idx] << " ";
        sum += test_qps[idx];
    }
    cout << endl;
    
    double average = sum / 3.0;
    cout << "平均值: " << fixed << setprecision(2) << average << endl;
    
    // 验证：手动计算
    double manual_avg = (1100.0 + 1200.0 + 1050.0) / 3.0;
    cout << "手动验证平均值: " << fixed << setprecision(2) << manual_avg << endl;
    
    cout << "\n测试完成！" << endl;
    return 0;
}
