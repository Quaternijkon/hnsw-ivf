#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>

using namespace std;

int main() {
    cout << "=== 选择逻辑验证测试 ===" << endl;
    
    // 模拟5次测试数据
    vector<double> qps = {1000.0, 1200.0, 1100.0, 1300.0, 1050.0};
    vector<double> latency = {2.0, 1.5, 3.0, 1.0, 4.0};  // 注意：latency的分布与QPS不同
    
    cout << "原始5次测试数据:" << endl;
    for (int i = 0; i < 5; ++i) {
        cout << "测试" << (i+1) << ": QPS=" << fixed << setprecision(0) << qps[i] 
             << ", Latency=" << fixed << setprecision(1) << latency[i] << "ms" << endl;
    }
    
    // 按QPS排序选择中间3组
    vector<pair<double, int>> qps_with_index;
    for (int i = 0; i < 5; ++i) {
        qps_with_index.push_back({qps[i], i});
    }
    sort(qps_with_index.begin(), qps_with_index.end());
    
    cout << "\n按QPS排序后:" << endl;
    for (int i = 0; i < 5; ++i) {
        int idx = qps_with_index[i].second;
        cout << "排序" << (i+1) << ": 测试" << (idx+1) << " QPS=" << fixed << setprecision(0) << qps[idx] 
             << ", Latency=" << fixed << setprecision(1) << latency[idx] << "ms" << endl;
    }
    
    // 选择中间3组（索引1,2,3）
    vector<int> valid_indices = {qps_with_index[1].second, qps_with_index[2].second, qps_with_index[3].second};
    
    cout << "\n保留的3组数据:" << endl;
    double qps_sum = 0, lat_sum = 0;
    for (int i = 0; i < 3; ++i) {
        int idx = valid_indices[i];
        cout << "测试" << (idx+1) << ": QPS=" << fixed << setprecision(0) << qps[idx] 
             << ", Latency=" << fixed << setprecision(1) << latency[idx] << "ms" << endl;
        qps_sum += qps[idx];
        lat_sum += latency[idx];
    }
    
    cout << "\n平均值:" << endl;
    cout << "QPS平均: " << fixed << setprecision(0) << (qps_sum / 3.0) << endl;
    cout << "Latency平均: " << fixed << setprecision(1) << (lat_sum / 3.0) << "ms" << endl;
    
    cout << "\n分析:" << endl;
    cout << "QPS范围: " << *min_element(qps.begin(), qps.end()) << " - " << *max_element(qps.begin(), qps.end()) << endl;
    cout << "保留的QPS范围: " << qps[valid_indices[0]] << " - " << qps[valid_indices[2]] << endl;
    cout << "Latency范围: " << *min_element(latency.begin(), latency.end()) << " - " << *max_element(latency.begin(), latency.end()) << endl;
    cout << "保留的Latency范围: " << latency[valid_indices[0]] << " - " << latency[valid_indices[2]] << endl;
    
    cout << "\n结论: 基于QPS选择可能保留其他指标的极值！" << endl;
    
    return 0;
}
