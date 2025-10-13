#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <regex>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sys/resource.h>
#include <unistd.h>
#include <iomanip>
#include <numeric>
#include <omp.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/utils/Heap.h>
#include <faiss/impl/io.h>
#include <faiss/impl/FaissAssert.h>
#include <sys/statvfs.h>
#include <sys/stat.h>
#include <dirent.h>
#include <fstream>
#include <sstream>

using namespace std;
using faiss::QueryLatencyStats;
using idx_t = faiss::idx_t;

// === 磁盘监控结构体 ===
struct DiskStats {
    double read_bytes = 0.0;              // 读取字节数
    double write_bytes = 0.0;             // 写入字节数
    double read_ops = 0.0;                // 读取操作数
    double write_ops = 0.0;               // 写入操作数
    double read_speed_mbps = 0.0;         // 读取速度 (MB/s)
    double write_speed_mbps = 0.0;        // 写入速度 (MB/s)
    double read_iops = 0.0;               // 读取IOPS
    double write_iops = 0.0;               // 写入IOPS
    double total_iops = 0.0;              // 总IOPS
    double avg_io_latency_ms = 0.0;        // 平均IO延迟 (ms)
    double disk_utilization = 0.0;         // 磁盘利用率 (%)
    double queue_depth = 0.0;             // 队列深度
    bool disk_bottleneck = false;         // 是否为磁盘瓶颈
    string disk_device = "";              // 磁盘设备名
    double available_space_gb = 0.0;      // 可用空间 (GB)
    double total_space_gb = 0.0;          // 总空间 (GB)
    
    // 初始磁盘IO计数（用于计算增量）
    struct {
        unsigned long read_ios = 0;
        unsigned long read_merges = 0;
        unsigned long read_sectors = 0;
        unsigned long read_ticks = 0;
        unsigned long write_ios = 0;
        unsigned long write_merges = 0;
        unsigned long write_sectors = 0;
        unsigned long write_ticks = 0;
        unsigned long in_flight = 0;
        unsigned long io_ticks = 0;
        unsigned long time_in_queue = 0;
    } initial_io_stats;
    double space_utilization = 0.0;       // 空间利用率 (%)
};

// === 功耗监控结构体 ===
struct PowerStats {
    double cpu_energy_joules = 0.0;        // CPU能耗（焦耳）
    double memory_energy_joules = 0.0;     // 内存能耗（焦耳）
    double total_energy_joules = 0.0;      // 总能耗（焦耳）
    double avg_power_watts = 0.0;          // 平均功耗（瓦特）
    double peak_power_watts = 0.0;         // 峰值功耗（瓦特）
    double power_per_query_mj = 0.0;       // 每查询功耗（毫焦耳）
    double energy_efficiency = 0.0;        // 能效比（QPS/瓦特）
    bool rapl_available = false;           // RAPL是否可用
};

// === 增强的基准测试结果结构体 ===
struct BenchmarkResult {
    double total_wall_time_s = 0.0;
    double qps = 0.0;
    double avg_latency_ms = 0.0;
    double p50_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
    double max_latency_ms = 0.0;
    double min_latency_ms = 0.0;
    double peak_memory_mb = 0.0;
    PowerStats power_stats;
    DiskStats disk_stats;
};

// ==============================================================================
// 内存监控函数
// ==============================================================================
double get_current_memory_mb() {
    rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    long peak_memory_bytes = usage.ru_maxrss;
    peak_memory_bytes *= 1024; // Linux系统需要乘以1024
    return peak_memory_bytes / (1024.0 * 1024.0);
}

// ==============================================================================
// 磁盘监控类
// ==============================================================================
class DiskMonitor {
private:
    string disk_device;
    string proc_diskstats_path;
    string proc_mounts_path;
    bool monitoring_available;
    
    struct DiskIOStats {
        unsigned long read_ios;
        unsigned long read_merges;
        unsigned long read_sectors;
        unsigned long read_ticks;
        unsigned long write_ios;
        unsigned long write_merges;
        unsigned long write_sectors;
        unsigned long write_ticks;
        unsigned long in_flight;
        unsigned long io_ticks;
        unsigned long time_in_queue;
    };
    
public:
    DiskMonitor() {
        monitoring_available = false;
        disk_device = "";
        
        // 尝试检测主要磁盘设备
        detect_disk_device();
        
        if (!disk_device.empty()) {
            proc_diskstats_path = "/proc/diskstats";
            proc_mounts_path = "/proc/mounts";
            monitoring_available = true;
            cout << "磁盘监控已启用，监控设备: " << disk_device << endl;
        } else {
            cout << "警告: 无法检测磁盘设备，磁盘监控将被禁用" << endl;
        }
    }
    
    // 检测主要磁盘设备
    void detect_disk_device() {
        // 优先使用环境变量覆盖，支持传入 /dev/<name> 或 <name>
        const char* env = getenv("DISK_DEVICE");
        if (env && *env) {
            string dev_env(env);
            if (dev_env.rfind("/dev/", 0) == 0) {
                dev_env = dev_env.substr(5);
            }
            disk_device = dev_env;
            return;
        }

        ifstream mounts_file("/proc/mounts");
        string line;
        
        // 查找根文件系统设备，保留分区名以便精确匹配
        while (getline(mounts_file, line)) {
            istringstream iss(line);
            string device, mountpoint, fstype;
            iss >> device >> mountpoint >> fstype;
            
            if (mountpoint == "/" && device.find("/dev/") == 0) {
                // 去掉 /dev/ 前缀，直接保留分区名（如 nvme0n1p2 / sda1 / dm-0）
                disk_device = device.substr(5);
                break;
            }
        }
    }
    
    // 读取磁盘统计信息
    DiskIOStats read_disk_stats() {
        DiskIOStats stats = {};
        
        if (!monitoring_available) {
            return stats;
        }
        
        ifstream diskstats_file(proc_diskstats_path);
        if (!diskstats_file) {
            return stats;
        }
        string line;
        
        while (getline(diskstats_file, line)) {
            istringstream iss(line);
            string major, minor, device;
            iss >> major >> minor >> device;
            
            if (device == disk_device) {
                iss >> stats.read_ios >> stats.read_merges >> stats.read_sectors
                    >> stats.read_ticks >> stats.write_ios >> stats.write_merges
                    >> stats.write_sectors >> stats.write_ticks >> stats.in_flight
                    >> stats.io_ticks >> stats.time_in_queue;
                break;
            }
        }

        return stats;
        
        return stats;
    }
    
    // 获取磁盘空间信息
    void get_disk_space_info(DiskStats& stats) {
        struct statvfs vfs;
        if (statvfs(".", &vfs) == 0) {
            unsigned long total_bytes = vfs.f_blocks * vfs.f_frsize;
            unsigned long available_bytes = vfs.f_bavail * vfs.f_frsize;
            
            stats.total_space_gb = total_bytes / (1024.0 * 1024.0 * 1024.0);
            stats.available_space_gb = available_bytes / (1024.0 * 1024.0 * 1024.0);
            stats.space_utilization = ((double)(total_bytes - available_bytes) / total_bytes) * 100.0;
        }
    }
    
    // 开始监控
    DiskStats start_monitoring() {
        DiskStats stats;
        stats.disk_device = disk_device;
        
        if (!monitoring_available) {
            return stats;
        }
        
        // 获取初始磁盘统计信息
        DiskIOStats initial_stats = read_disk_stats();
        // 保存初始IO计数到返回的 DiskStats
        stats.initial_io_stats.read_ios = initial_stats.read_ios;
        stats.initial_io_stats.read_merges = initial_stats.read_merges;
        stats.initial_io_stats.read_sectors = initial_stats.read_sectors;
        stats.initial_io_stats.read_ticks = initial_stats.read_ticks;
        stats.initial_io_stats.write_ios = initial_stats.write_ios;
        stats.initial_io_stats.write_merges = initial_stats.write_merges;
        stats.initial_io_stats.write_sectors = initial_stats.write_sectors;
        stats.initial_io_stats.write_ticks = initial_stats.write_ticks;
        stats.initial_io_stats.in_flight = initial_stats.in_flight;
        stats.initial_io_stats.io_ticks = initial_stats.io_ticks;
        stats.initial_io_stats.time_in_queue = initial_stats.time_in_queue;
        
        // 获取磁盘空间信息
        get_disk_space_info(stats);
        
        return stats;
    }
    
    // 结束监控并计算磁盘统计
    DiskStats end_monitoring(DiskStats start_stats, double duration_s) {
        DiskStats stats = start_stats;
        
        if (!monitoring_available) {
            return stats;
        }
        
        // 获取最终磁盘统计信息
        DiskIOStats final_stats = read_disk_stats();
        
        // 计算增量（与初始计数相比）
        unsigned long read_ios_diff = final_stats.read_ios - start_stats.initial_io_stats.read_ios;
        unsigned long write_ios_diff = final_stats.write_ios - start_stats.initial_io_stats.write_ios;
        unsigned long read_sectors_diff = final_stats.read_sectors - start_stats.initial_io_stats.read_sectors;
        unsigned long write_sectors_diff = final_stats.write_sectors - start_stats.initial_io_stats.write_sectors;
        unsigned long io_ticks_diff = final_stats.io_ticks - start_stats.initial_io_stats.io_ticks;
        
        // 转换为字节（假设扇区大小为512字节）
        stats.read_bytes = read_sectors_diff * 512.0;
        stats.write_bytes = write_sectors_diff * 512.0;
        stats.read_ops = read_ios_diff;
        stats.write_ops = write_ios_diff;
        
        // 计算速度
        if (duration_s > 0) {
            stats.read_speed_mbps = (stats.read_bytes / (1024.0 * 1024.0)) / duration_s;
            stats.write_speed_mbps = (stats.write_bytes / (1024.0 * 1024.0)) / duration_s;
            stats.read_iops = stats.read_ops / duration_s;
            stats.write_iops = stats.write_ops / duration_s;
            stats.total_iops = stats.read_iops + stats.write_iops;
            
            // 计算平均IO延迟
            if (stats.total_iops > 0) {
                // io_ticks_diff 已经是毫秒(ms)，直接除以总IO次数得到每次IO的平均毫秒
                stats.avg_io_latency_ms = io_ticks_diff / stats.total_iops;
            }
        }
        
        // 计算磁盘利用率（基于IO时间）
        if (duration_s > 0) {
            // 利用率(%) = 100 * (IO占用毫秒) / (总毫秒)
            // 等价简化: io_ticks_diff / (duration_s * 10.0)
            stats.disk_utilization = (io_ticks_diff / (duration_s * 10.0));
        }
        
        // 检测磁盘瓶颈
        detect_disk_bottleneck(stats);
        
        return stats;
    }
    
private:
    // 检测磁盘瓶颈
    void detect_disk_bottleneck(DiskStats& stats) {
        // 活动门槛：只有足够I/O活动时才评估瓶颈，避免无I/O场景误判
        bool active_io = (stats.read_bytes + stats.write_bytes) > (8 * 1024 * 1024) || stats.total_iops > 100.0;
        if (!active_io) {
            stats.disk_bottleneck = false;
            return;
        }
        
        bool high_utilization = stats.disk_utilization > 80.0;
        bool low_speed = stats.read_speed_mbps < 100.0 && stats.write_speed_mbps < 100.0;
        bool high_latency = stats.avg_io_latency_ms > 10.0;
        
        stats.disk_bottleneck = high_utilization || (low_speed && high_latency);
    }
};

// ==============================================================================
// 功耗监控类
// ==============================================================================
class PowerMonitor {
private:
    string rapl_package0_path;
    string rapl_package1_path;
    string rapl_dram0_path;
    string rapl_dram1_path;
    bool rapl_available;
    
public:
    PowerMonitor() {
        // 检测RAPL是否可用
        rapl_package0_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj";
        rapl_package1_path = "/sys/class/powercap/intel-rapl/intel-rapl:1/energy_uj";
        rapl_dram0_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj";
        rapl_dram1_path = "/sys/class/powercap/intel-rapl/intel-rapl:1/intel-rapl:1:0/energy_uj";
        
        // 检查RAPL接口是否可用
        ifstream test_file(rapl_package0_path);
        rapl_available = test_file.good();
        
        if (!rapl_available) {
            cout << "警告: RAPL不可用，将使用替代功耗估算方法" << endl;
        } else {
            cout << "RAPL功耗监控接口可用" << endl;
        }
    }
    
    // 读取当前能量值（微焦耳）
    uint64_t read_energy_uj(const string& path) {
        ifstream file(path);
        if (!file.is_open()) return 0;
        
        uint64_t energy;
        file >> energy;
        return energy;
    }
    
    // 开始监控
    PowerStats start_monitoring() {
        PowerStats stats;
        stats.rapl_available = rapl_available;
        
        if (!rapl_available) {
            return stats;
        }
        
        // 读取初始能量值（累计值）
        stats.cpu_energy_joules = read_energy_uj(rapl_package0_path) + read_energy_uj(rapl_package1_path);
        stats.memory_energy_joules = read_energy_uj(rapl_dram0_path) + read_energy_uj(rapl_dram1_path);
        stats.total_energy_joules = stats.cpu_energy_joules + stats.memory_energy_joules;
        return stats;
    }
    
    // 结束监控并计算功耗统计
    PowerStats end_monitoring(PowerStats start_stats, double duration_s, double qps) {
        PowerStats stats;
        stats.rapl_available = rapl_available;
        
        if (!rapl_available) {
            // 使用CPU使用率估算功耗（粗略估算）
            stats.avg_power_watts = estimate_power_from_cpu_usage();
            stats.total_energy_joules = stats.avg_power_watts * duration_s;
            stats.cpu_energy_joules = stats.total_energy_joules * 0.8; // 假设80%是CPU
            stats.memory_energy_joules = stats.total_energy_joules * 0.2; // 假设20%是内存
        } else {
            // 使用RAPL读取实际功耗
            uint64_t end_cpu0_uj = read_energy_uj(rapl_package0_path);
            uint64_t end_cpu1_uj = read_energy_uj(rapl_package1_path);
            uint64_t end_dram0_uj = read_energy_uj(rapl_dram0_path);
            uint64_t end_dram1_uj = read_energy_uj(rapl_dram1_path);
            
            // 计算CPU能耗增量（两个package的总和）
            uint64_t total_cpu_uj = end_cpu0_uj + end_cpu1_uj;
            uint64_t start_cpu_uj = static_cast<uint64_t>(start_stats.cpu_energy_joules);
            
            // 处理计数器回绕（假设最大值为2^32微焦耳）
            uint64_t cpu_diff;
            if (total_cpu_uj >= start_cpu_uj) {
                cpu_diff = total_cpu_uj - start_cpu_uj;
            } else {
                // 处理回绕情况
                cpu_diff = (UINT64_MAX - start_cpu_uj) + total_cpu_uj + 1;
            }
            stats.cpu_energy_joules = cpu_diff / 1000000.0; // 转换为焦耳
            
            // 计算内存能耗增量（两个DRAM的总和）
            uint64_t total_dram_uj = end_dram0_uj + end_dram1_uj;
            uint64_t start_dram_uj = static_cast<uint64_t>(start_stats.memory_energy_joules);
            
            // 处理计数器回绕
            uint64_t dram_diff;
            if (total_dram_uj >= start_dram_uj) {
                dram_diff = total_dram_uj - start_dram_uj;
            } else {
                // 处理回绕情况
                dram_diff = (UINT64_MAX - start_dram_uj) + total_dram_uj + 1;
            }
            stats.memory_energy_joules = dram_diff / 1000000.0; // 转换为焦耳
            
            // 总能耗增量
            stats.total_energy_joules = stats.cpu_energy_joules + stats.memory_energy_joules;
            
            // 计算平均功耗
            if (duration_s > 0) {
                stats.avg_power_watts = stats.total_energy_joules / duration_s;
            }
        }
        
        // 计算衍生指标
        if (qps > 0) {
            stats.power_per_query_mj = (stats.total_energy_joules * 1000) / qps;
        }
        if (stats.avg_power_watts > 0) {
            stats.energy_efficiency = qps / stats.avg_power_watts;
        }
        
        return stats;
    }
    
private:
    // 基于CPU使用率的功耗估算（粗略方法）
    double estimate_power_from_cpu_usage() {
        // 这是一个简化的估算，实际应用中需要更复杂的模型
        // 假设基础功耗为50W，每个CPU核心增加10W
        int num_cores = omp_get_max_threads();
        return 50.0 + (num_cores * 10.0);
    }
};

// ==============================================================================
// 0. 路径和文件名配置 & 调试开关
// ==============================================================================
const string DATA_DIR = "./sift";
const string LEARN_FILE = DATA_DIR + "/learn.fbin";
const string BASE_FILE = DATA_DIR + "/base.fbin";
const string QUERY_FILE = DATA_DIR + "/query.fbin";
const string GROUNDTRUTH_FILE = DATA_DIR + "/groundtruth.ivecs";

// ==============================================================================
// 1. 辅助函数：读取.fbin文件
// ==============================================================================
pair<vector<float>, pair<size_t, size_t>> read_fbin(const string& filename, size_t start_idx = 0, size_t chunk_size = 0) {
    ifstream f(filename, ios::binary);
    if (!f.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }

    int32_t nvecs_raw, dim_raw;
    f.read(reinterpret_cast<char*>(&nvecs_raw), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&dim_raw), sizeof(int32_t));
    size_t nvecs = static_cast<size_t>(nvecs_raw);
    size_t dim = static_cast<size_t>(dim_raw);

    size_t num_vectors_in_chunk = nvecs;
    if (chunk_size > 0) {
        size_t end_idx = min(start_idx + chunk_size, nvecs);
        num_vectors_in_chunk = end_idx - start_idx;
        size_t offset = 8 + start_idx * dim * sizeof(float); // Skip header if start_idx > 0, but since we read header, adjust
        f.seekg(offset, ios::beg);
    }

    vector<float> data(num_vectors_in_chunk * dim);
    f.read(reinterpret_cast<char*>(data.data()), num_vectors_in_chunk * dim * sizeof(float));

    return {data, {nvecs, dim}};
}

vector<vector<int32_t>> read_ivecs(const string& filename) {
    ifstream f(filename, ios::binary | ios::ate);
    if (!f.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }
    size_t file_size = f.tellg();
    f.seekg(0, ios::beg);

    vector<int32_t> a(file_size / sizeof(int32_t));
    f.read(reinterpret_cast<char*>(a.data()), file_size);

    int32_t d = a[0];
    vector<vector<int32_t>> result(a.size() / (d + 1));
    for (size_t i = 0, j = 0; i < result.size(); ++i) {
        result[i].resize(d);
        ++j; // skip dim
        copy(a.begin() + j, a.begin() + j + d, result[i].begin());
        j += d;
    }
    return result;
}

// ==============================================================================
// 2. 设置参数与环境
// ==============================================================================
int main() {
    // 初始化监控器
    PowerMonitor power_monitor;
    DiskMonitor disk_monitor;
    
    // 从训练文件中获取维度信息
    auto [_, meta_train] = read_fbin(LEARN_FILE, 0, 1);
    size_t nt = meta_train.first;
    size_t d_train = meta_train.second;

    // 获取数据集大小信息
    auto [__, meta_base] = read_fbin(BASE_FILE, 0, 1);
    size_t nb = meta_base.first;
    size_t d_base = meta_base.second;

    auto [___, meta_query] = read_fbin(QUERY_FILE, 0, 1);
    size_t nq = meta_query.first;
    size_t d_query = meta_query.second;

    // 验证维度一致性
    if (d_train != d_base || d_train != d_query) {
        throw runtime_error("维度不一致: 训练集" + to_string(d_train) + "维, 基础集" + to_string(d_base) + "维, 查询集" + to_string(d_query) + "维");
    }

    // 设置其他参数
    size_t cell_size = 64;
    size_t nlist = nb / cell_size;
    size_t nprobe = 128;
    size_t chunk_size = 100000; // 每次处理的数据块大小
    size_t k = 10; // 查找最近的10个邻居

    size_t M = 32; // HNSW的连接数
    size_t efconstruction = 100; // 默认40
    size_t efsearch = 200;       // 默认16

    // ==============================================================================
    // 【重构点】: 在索引文件名中同时体现 M 和 efConstruction 的值
    // ==============================================================================
    string base_name = BASE_FILE.substr(BASE_FILE.find_last_of("/") + 1);
    base_name = base_name.substr(0, base_name.find_last_of("."));
    regex non_alnum("[^a-zA-Z0-9_]");
    string clean_base_name = regex_replace(base_name, non_alnum, "_");
    string INDEX_FILE = DATA_DIR + "/" + clean_base_name + "_d" + to_string(d_train) + "_nlist" + to_string(nlist) +
                        "_HNSWM" + to_string(M) + "_efc" + to_string(efconstruction) + "_IVFFlat.index";
    // ==============================================================================

    cout << string(60, '=') << endl;
    cout << "Phase 0: 环境设置" << endl;
    cout << "向量维度 (d): " << d_train << endl;
    cout << "基础集大小 (nb): " << nb << ", 训练集大小 (ntrain): " << nt << endl;
    cout << "查询集大小 (nq): " << nq << ", 分块大小 (chunk_size): " << chunk_size << endl;
    cout << "HNSW M (构建参数): " << M << endl;
    cout << "HNSW efConstruction (构建参数): " << efconstruction << endl;
    cout << "索引将保存在磁盘文件: " << INDEX_FILE << endl;
    cout << string(60, '=') << endl;

    // ==============================================================================
    // 3. 检查索引文件是否存在
    // ==============================================================================
    bool skip_index_building = false;
    ifstream index_check(INDEX_FILE);
    if (index_check.good()) {
        cout << "索引文件 " << INDEX_FILE << " 已存在，跳过索引构建阶段" << endl;
        skip_index_building = true;
    } else {
        cout << "索引文件不存在，将构建新索引" << endl;
    }

    faiss::Index* coarse_quantizer = nullptr;

    if (!skip_index_building) {
        // ==============================================================================
        // 4. 训练量化器 
        // ==============================================================================
        cout << "\nPhase 1: 训练 HNSW 粗量化器 (in-memory)" << endl;
        coarse_quantizer = new faiss::IndexHNSWFlat(d_train, M);
        dynamic_cast<faiss::IndexHNSW*>(coarse_quantizer)->hnsw.efConstruction = efconstruction;
        dynamic_cast<faiss::IndexHNSW*>(coarse_quantizer)->hnsw.efSearch = efsearch;
        cout << "efconstruction: " << dynamic_cast<faiss::IndexHNSW*>(coarse_quantizer)->hnsw.efConstruction
             << ", efSearch: " << dynamic_cast<faiss::IndexHNSW*>(coarse_quantizer)->hnsw.efSearch << endl;

        faiss::IndexIVFFlat* index_for_training = new faiss::IndexIVFFlat(coarse_quantizer, d_train, nlist, faiss::METRIC_L2);
        index_for_training->verbose = true;

        auto [xt_data, ___] = read_fbin(LEARN_FILE);
        float* xt = xt_data.data();

        cout << "训练聚类中心并构建 HNSW 量化器..." << endl;
        auto start_time = chrono::high_resolution_clock::now();
        index_for_training->train(nt, xt);
        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end_time - start_time;

        cout << "量化器训练完成，耗时: " << fixed << setprecision(2) << duration.count() << " 秒" << endl;
        cout << "粗量化器中的质心数量: " << coarse_quantizer->ntotal << endl;

        delete index_for_training;

        // ==============================================================================
        // 5. 创建一个空的、基于磁盘的索引框架
        // ==============================================================================
        cout << "\nPhase 2: 创建空的磁盘索引框架" << endl;
        faiss::IndexIVFFlat* index_shell = new faiss::IndexIVFFlat(coarse_quantizer, d_train, nlist, faiss::METRIC_L2);
        cout << "将空的索引框架写入磁盘: " << INDEX_FILE << endl;
        faiss::write_index(index_shell, INDEX_FILE.c_str());
        delete index_shell;

        // ==============================================================================
        // 6. 分块向磁盘索引中添加数据 (从base.fbin)
        // ==============================================================================
        cout << "\nPhase 3: 分块添加数据到磁盘索引" << endl;

        int IO_FLAG_READ_WRITE = 0;
        cout << "使用IO标志: " << IO_FLAG_READ_WRITE << " (读写模式)" << endl;

        faiss::Index* index_ondisk = faiss::read_index(INDEX_FILE.c_str(), IO_FLAG_READ_WRITE);
        auto start_time_add = chrono::high_resolution_clock::now();

        size_t num_chunks = (nb + chunk_size - 1) / chunk_size;
        for (size_t i = 0; i < nb; i += chunk_size) {
            size_t chunk_idx = i / chunk_size + 1;
            cout << "       -> 正在处理块 " << chunk_idx << "/" << num_chunks << ": 向量 " << i << " 到 " << min(i + chunk_size, nb) - 1 << endl;

            auto [xb_chunk_data, ____] = read_fbin(BASE_FILE, i, chunk_size);
            float* xb_chunk = xb_chunk_data.data();

            index_ondisk->add(min(chunk_size, nb - i), xb_chunk);
        }

        auto end_time_add = chrono::high_resolution_clock::now();
        chrono::duration<double> duration_add = end_time_add - start_time_add;
        cout << "\n所有数据块添加完成，总耗时: " << fixed << setprecision(2) << duration_add.count() << " 秒" << endl;
        cout << "磁盘索引中的向量总数 (ntotal): " << index_ondisk->ntotal << endl;

        cout << "正在将最终索引写回磁盘: " << INDEX_FILE << endl;
        faiss::write_index(index_ondisk, INDEX_FILE.c_str());
        delete index_ondisk;
    }

    // ==============================================================================
    // 8. 使用内存映射 (mmap) 进行搜索 (使用query.fbin)
    // ==============================================================================
    cout << "\nPhase 4: 使用内存映射模式进行搜索" << endl;
    cout << "以 mmap 模式打开磁盘索引: " << INDEX_FILE << endl;

    int IO_FLAG_MMAP = faiss::IO_FLAG_MMAP;
    cout << "使用IO标志: " << IO_FLAG_MMAP << " (内存映射模式)" << endl;

    faiss::Index* index_final = faiss::read_index(INDEX_FILE.c_str(), IO_FLAG_MMAP);
    dynamic_cast<faiss::IndexIVF*>(index_final)->nprobe = nprobe;
    omp_set_num_threads(20);
    dynamic_cast<faiss::IndexIVF*>(index_final)->parallel_mode = 0;
    cout << "并行模式线程数: " << omp_get_max_threads() << endl;
    cout << "并行模式: " << dynamic_cast<faiss::IndexIVF*>(index_final)->parallel_mode << endl;
    cout << "索引已准备好搜索 (nprobe=" << dynamic_cast<faiss::IndexIVF*>(index_final)->nprobe << ")" << endl;

    faiss::Index* generic_quantizer = dynamic_cast<faiss::IndexIVF*>(index_final)->quantizer;
    faiss::IndexHNSW* quantizer_hnsw = dynamic_cast<faiss::IndexHNSW*>(generic_quantizer);
    quantizer_hnsw->hnsw.efSearch = efsearch;
    cout << "efConstruction: " << quantizer_hnsw->hnsw.efConstruction << ", efSearch: " << quantizer_hnsw->hnsw.efSearch << endl;

    cout << "从 query.fbin 加载查询向量..." << endl;
    auto [xq_data, _____] = read_fbin(QUERY_FILE);
    float* xq = xq_data.data();

    cout << "\n执行搜索..." << endl;
    vector<faiss::idx_t> I(nq * k);
    vector<float> D(nq * k);
    
    auto start_time_search = chrono::high_resolution_clock::now();
    index_final->search(nq, xq, k, D.data(), I.data());
    auto end_time_search = chrono::high_resolution_clock::now();
    chrono::duration<double> search_duration = end_time_search - start_time_search;

    cout << "搜索完成，耗时: " << fixed << setprecision(2) << search_duration.count() << " 秒" << endl;

    if (search_duration.count() > 0) {
        double qps = nq / search_duration.count();
        cout << "QPS (每秒查询率): " << fixed << setprecision(2) << qps << endl;
    } else {
        cout << "搜索耗时过短，无法计算QPS" << endl;
    }
    
    cout << "索引功能验证完成，详细性能分析将在Phase 6中进行。" << endl;

    // ==============================================================================
    // 9.  新增: 根据Groundtruth计算召回率 (内存优化版)
    // ==============================================================================
    cout << "\n" << string(60, '=') << endl;
    cout << "Phase 5: 计算召回率 (内存优化版)" << endl;

    ifstream gt_file_check(GROUNDTRUTH_FILE);
    if (!gt_file_check.good()) {
        cout << "Groundtruth文件未找到: " << GROUNDTRUTH_FILE << endl;
        cout << "跳过召回率计算。" << endl;
    } else {
        cout << "以流式方式从 " << GROUNDTRUTH_FILE << " 读取 groundtruth 数据进行计算..." << endl;

        ifstream f(GROUNDTRUTH_FILE, ios::binary);
        int32_t k_gt;
        f.read(reinterpret_cast<char*>(&k_gt), sizeof(int32_t));
        cout << "Groundtruth 维度 (k_gt): " << k_gt << endl;

        f.seekg(0, ios::end);
        size_t total_file_size = f.tellg();
        size_t record_size_bytes = (k_gt + 1) * sizeof(int32_t);
        size_t num_gt_vectors = total_file_size / record_size_bytes;
        if (nq != num_gt_vectors) {
            cout << "警告: 查询数量(" << nq << ")与groundtruth中的数量(" << num_gt_vectors << ")不匹配!" << endl;
        }

        cout << "正在计算 Recall@" << k << "..." << endl;

        size_t total_found = 0;
        for (size_t i = 0; i < nq; ++i) {
            size_t offset = i * record_size_bytes;
            f.seekg(offset, ios::beg);

            vector<int32_t> record_data(k_gt + 1);
            f.read(reinterpret_cast<char*>(record_data.data()), (k_gt + 1) * sizeof(int32_t));

            vector<faiss::idx_t> gt_i(k_gt);
            for (int m = 0; m < k_gt; ++m) {
                gt_i[m] = static_cast<faiss::idx_t>(record_data[m + 1]);
            }

            size_t found_count = 0;
            size_t check_size = min(k, static_cast<size_t>(k_gt));
            for (size_t j = 0; j < k; ++j) {
                faiss::idx_t neighbor = I[i * k + j];
                if (find(gt_i.begin(), gt_i.begin() + check_size, neighbor) != gt_i.begin() + check_size) {
                    ++found_count;
                }
            }
            total_found += found_count;
        }

        double recall = static_cast<double>(total_found) / (nq * k);

        cout << "\n查询了 " << nq << " 个向量, k=" << k << endl;
        cout << "在top-" << k << "的结果中，总共找到了 " << total_found << " 个真实的近邻。" << endl;
        cout << "Recall@" << k << ": " << fixed << setprecision(4) << recall << endl;
    }

    cout << string(60, '=') << endl;

    // ==============================================================================
    // Latency Benchmark Module with Disk Monitoring
    // ==============================================================================
    cout << "\n" << string(60, '=') << endl;
    cout << "Phase 6: Latency Benchmark with Disk Monitoring" << endl;
    cout << string(60, '=') << endl;

    // === 基准测试循环设置 ===
    std::vector<int> thread_counts = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    const int num_runs_per_thread_setting = 1;  // 修改此值即可调整运行次数
    
    // 用于存储最终平均结果的向量
    std::vector<std::pair<int, BenchmarkResult>> final_results;

    cout << "Starting benchmark with disk monitoring..." << endl;

    // === 外层循环：遍历不同的线程数设置 ===
    for (int threads : thread_counts) {
        omp_set_num_threads(threads);
        
        std::vector<BenchmarkResult> run_results;
        cout << "\n--- Testing with " << threads << " OMP threads ---" << endl;

        // === 内层循环：为每个线程数设置重复运行多次 ===
        for (int i = 0; i < num_runs_per_thread_setting; ++i) {
            vector<idx_t> I_bench(nq * k);
            vector<float> D_bench(nq * k);
            QueryLatencyStats* latency_stats = new QueryLatencyStats[nq];

            // 开始功耗和磁盘监控
            PowerStats start_power = power_monitor.start_monitoring();
            DiskStats start_disk = disk_monitor.start_monitoring();

            double t_start = omp_get_wtime();
            
            dynamic_cast<faiss::IndexIVFFlat*>(index_final)->search_stats(nq, xq, k, D_bench.data(), I_bench.data(), nullptr, latency_stats);

            double t_end = omp_get_wtime();
            
            // 结束监控
            double duration = t_end - t_start;
            double current_qps = nq / duration;
            PowerStats power_stats = power_monitor.end_monitoring(start_power, duration, current_qps);
            DiskStats disk_stats = disk_monitor.end_monitoring(start_disk, duration);
            
            // --- 数据收集 ---
            BenchmarkResult current_run;
            current_run.total_wall_time_s = duration;
            current_run.qps = current_qps;
            current_run.power_stats = power_stats;
            current_run.disk_stats = disk_stats;

            std::vector<double> latencies_ms;
            for (int j = 0; j < nq; ++j) {
                latencies_ms.push_back(latency_stats[j].total_us / 1000.0);
            }
            std::sort(latencies_ms.begin(), latencies_ms.end());
            
            current_run.avg_latency_ms = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) / nq;
            current_run.p50_latency_ms = latencies_ms[nq / 2];
            current_run.p99_latency_ms = latencies_ms[int(nq * 0.99)];
            current_run.max_latency_ms = latencies_ms.back(); // 最大值（已排序）
            current_run.min_latency_ms = latencies_ms.front(); // 最小值（已排序）
            
            // 记录当前运行的峰值内存
            current_run.peak_memory_mb = get_current_memory_mb();
            
            run_results.push_back(current_run);

            cout << "Run " << (i + 1) << "/" << num_runs_per_thread_setting 
                 << ": Total Time = " << fixed << setprecision(4) << current_run.total_wall_time_s << " s" 
                 << ", QPS = " << fixed << setprecision(2) << current_run.qps
                 << ", Disk Read = " << fixed << setprecision(2) << current_run.disk_stats.read_speed_mbps << " MB/s"
                 << ", Disk Write = " << fixed << setprecision(2) << current_run.disk_stats.write_speed_mbps << " MB/s"
                 << ", IOPS = " << fixed << setprecision(0) << current_run.disk_stats.total_iops << endl;

            delete[] latency_stats;
        }

        // === 计算平均值（去掉最大值和最小值，剩余数据取平均） ===
        BenchmarkResult avg_result;
        
        if (num_runs_per_thread_setting == 1) {
            // 单次运行，直接使用结果
            avg_result = run_results[0];
        } else if (num_runs_per_thread_setting == 2) {
            // 两次运行，直接取平均
            avg_result.total_wall_time_s = (run_results[0].total_wall_time_s + run_results[1].total_wall_time_s) / 2.0;
            avg_result.qps = (run_results[0].qps + run_results[1].qps) / 2.0;
            avg_result.avg_latency_ms = (run_results[0].avg_latency_ms + run_results[1].avg_latency_ms) / 2.0;
            avg_result.p50_latency_ms = (run_results[0].p50_latency_ms + run_results[1].p50_latency_ms) / 2.0;
            avg_result.p99_latency_ms = (run_results[0].p99_latency_ms + run_results[1].p99_latency_ms) / 2.0;
            avg_result.max_latency_ms = (run_results[0].max_latency_ms + run_results[1].max_latency_ms) / 2.0;
            avg_result.min_latency_ms = (run_results[0].min_latency_ms + run_results[1].min_latency_ms) / 2.0;
            avg_result.peak_memory_mb = (run_results[0].peak_memory_mb + run_results[1].peak_memory_mb) / 2.0;
            
            // 功耗统计平均
            avg_result.power_stats.cpu_energy_joules = (run_results[0].power_stats.cpu_energy_joules + run_results[1].power_stats.cpu_energy_joules) / 2.0;
            avg_result.power_stats.memory_energy_joules = (run_results[0].power_stats.memory_energy_joules + run_results[1].power_stats.memory_energy_joules) / 2.0;
            avg_result.power_stats.total_energy_joules = (run_results[0].power_stats.total_energy_joules + run_results[1].power_stats.total_energy_joules) / 2.0;
            avg_result.power_stats.avg_power_watts = (run_results[0].power_stats.avg_power_watts + run_results[1].power_stats.avg_power_watts) / 2.0;
            avg_result.power_stats.peak_power_watts = (run_results[0].power_stats.peak_power_watts + run_results[1].power_stats.peak_power_watts) / 2.0;
            avg_result.power_stats.power_per_query_mj = (run_results[0].power_stats.power_per_query_mj + run_results[1].power_stats.power_per_query_mj) / 2.0;
            avg_result.power_stats.energy_efficiency = (run_results[0].power_stats.energy_efficiency + run_results[1].power_stats.energy_efficiency) / 2.0;
            avg_result.power_stats.rapl_available = run_results[0].power_stats.rapl_available;
            
            // 磁盘统计平均
            avg_result.disk_stats.read_bytes = (run_results[0].disk_stats.read_bytes + run_results[1].disk_stats.read_bytes) / 2.0;
            avg_result.disk_stats.write_bytes = (run_results[0].disk_stats.write_bytes + run_results[1].disk_stats.write_bytes) / 2.0;
            avg_result.disk_stats.read_ops = (run_results[0].disk_stats.read_ops + run_results[1].disk_stats.read_ops) / 2.0;
            avg_result.disk_stats.write_ops = (run_results[0].disk_stats.write_ops + run_results[1].disk_stats.write_ops) / 2.0;
            avg_result.disk_stats.read_speed_mbps = (run_results[0].disk_stats.read_speed_mbps + run_results[1].disk_stats.read_speed_mbps) / 2.0;
            avg_result.disk_stats.write_speed_mbps = (run_results[0].disk_stats.write_speed_mbps + run_results[1].disk_stats.write_speed_mbps) / 2.0;
            avg_result.disk_stats.read_iops = (run_results[0].disk_stats.read_iops + run_results[1].disk_stats.read_iops) / 2.0;
            avg_result.disk_stats.write_iops = (run_results[0].disk_stats.write_iops + run_results[1].disk_stats.write_iops) / 2.0;
            avg_result.disk_stats.total_iops = (run_results[0].disk_stats.total_iops + run_results[1].disk_stats.total_iops) / 2.0;
            avg_result.disk_stats.avg_io_latency_ms = (run_results[0].disk_stats.avg_io_latency_ms + run_results[1].disk_stats.avg_io_latency_ms) / 2.0;
            avg_result.disk_stats.disk_utilization = (run_results[0].disk_stats.disk_utilization + run_results[1].disk_stats.disk_utilization) / 2.0;
            avg_result.disk_stats.disk_bottleneck = run_results[0].disk_stats.disk_bottleneck || run_results[1].disk_stats.disk_bottleneck;
            avg_result.disk_stats.disk_device = run_results[0].disk_stats.disk_device;
            avg_result.disk_stats.available_space_gb = (run_results[0].disk_stats.available_space_gb + run_results[1].disk_stats.available_space_gb) / 2.0;
            avg_result.disk_stats.total_space_gb = (run_results[0].disk_stats.total_space_gb + run_results[1].disk_stats.total_space_gb) / 2.0;
            avg_result.disk_stats.space_utilization = (run_results[0].disk_stats.space_utilization + run_results[1].disk_stats.space_utilization) / 2.0;
        } else {
            // 3次或以上运行，按QPS排序后去掉最大最小值
            vector<pair<double, int>> qps_with_index;
            for (int i = 0; i < num_runs_per_thread_setting; ++i) {
                qps_with_index.push_back({run_results[i].qps, i});
            }
            
            // 按QPS排序（升序）
            sort(qps_with_index.begin(), qps_with_index.end());
            
            // 计算有效数据数量（去掉最大最小值）
            int valid_count = num_runs_per_thread_setting - 2;
            if (valid_count <= 0) valid_count = 1; // 至少保留1个数据
            
            // 选择有效数据的索引
            vector<int> valid_indices;
            if (valid_count == 1) {
                // 如果只有1个有效数据，选择中间值
                valid_indices.push_back(qps_with_index[num_runs_per_thread_setting / 2].second);
            } else {
                // 去掉最小值和最大值，保留中间的数据
                for (int i = 1; i <= valid_count; ++i) {
                    valid_indices.push_back(qps_with_index[i].second);
                }
            }
            
            // 计算平均值
            double sum_total_time = 0.0, sum_qps = 0.0, sum_avg_lat = 0.0, sum_p50 = 0.0, sum_p99 = 0.0;
            double sum_max_lat = 0.0, sum_min_lat = 0.0;
            double sum_cpu_energy = 0.0, sum_memory_energy = 0.0, sum_total_energy = 0.0;
            double sum_avg_power = 0.0, sum_peak_power = 0.0, sum_power_per_query = 0.0, sum_energy_efficiency = 0.0;
            double sum_peak_memory = 0.0;
            
            // 磁盘统计求和
            double sum_read_bytes = 0.0, sum_write_bytes = 0.0, sum_read_ops = 0.0, sum_write_ops = 0.0;
            double sum_read_speed = 0.0, sum_write_speed = 0.0, sum_read_iops = 0.0, sum_write_iops = 0.0;
            double sum_total_iops = 0.0, sum_io_latency = 0.0, sum_disk_util = 0.0;
            double sum_available_space = 0.0, sum_total_space = 0.0, sum_space_util = 0.0;
            bool any_disk_bottleneck = false;
            
            for (int idx : valid_indices) {
                sum_total_time += run_results[idx].total_wall_time_s;
                sum_qps += run_results[idx].qps;
                sum_avg_lat += run_results[idx].avg_latency_ms;
                sum_p50 += run_results[idx].p50_latency_ms;
                sum_p99 += run_results[idx].p99_latency_ms;
                sum_max_lat += run_results[idx].max_latency_ms;
                sum_min_lat += run_results[idx].min_latency_ms;
                sum_cpu_energy += run_results[idx].power_stats.cpu_energy_joules;
                sum_memory_energy += run_results[idx].power_stats.memory_energy_joules;
                sum_total_energy += run_results[idx].power_stats.total_energy_joules;
                sum_avg_power += run_results[idx].power_stats.avg_power_watts;
                sum_peak_power += run_results[idx].power_stats.peak_power_watts;
                sum_power_per_query += run_results[idx].power_stats.power_per_query_mj;
                sum_energy_efficiency += run_results[idx].power_stats.energy_efficiency;
                sum_peak_memory += run_results[idx].peak_memory_mb;
                
                // 磁盘统计
                sum_read_bytes += run_results[idx].disk_stats.read_bytes;
                sum_write_bytes += run_results[idx].disk_stats.write_bytes;
                sum_read_ops += run_results[idx].disk_stats.read_ops;
                sum_write_ops += run_results[idx].disk_stats.write_ops;
                sum_read_speed += run_results[idx].disk_stats.read_speed_mbps;
                sum_write_speed += run_results[idx].disk_stats.write_speed_mbps;
                sum_read_iops += run_results[idx].disk_stats.read_iops;
                sum_write_iops += run_results[idx].disk_stats.write_iops;
                sum_total_iops += run_results[idx].disk_stats.total_iops;
                sum_io_latency += run_results[idx].disk_stats.avg_io_latency_ms;
                sum_disk_util += run_results[idx].disk_stats.disk_utilization;
                sum_available_space += run_results[idx].disk_stats.available_space_gb;
                sum_total_space += run_results[idx].disk_stats.total_space_gb;
                sum_space_util += run_results[idx].disk_stats.space_utilization;
                
                if (run_results[idx].disk_stats.disk_bottleneck) {
                    any_disk_bottleneck = true;
                }
            }

            int count = valid_indices.size();
            avg_result.total_wall_time_s = sum_total_time / count;
            avg_result.qps = sum_qps / count;
            avg_result.avg_latency_ms = sum_avg_lat / count;
            avg_result.p50_latency_ms = sum_p50 / count;
            avg_result.p99_latency_ms = sum_p99 / count;
            avg_result.max_latency_ms = sum_max_lat / count;
            avg_result.min_latency_ms = sum_min_lat / count;
            avg_result.peak_memory_mb = sum_peak_memory / count;
            avg_result.power_stats.cpu_energy_joules = sum_cpu_energy / count;
            avg_result.power_stats.memory_energy_joules = sum_memory_energy / count;
            avg_result.power_stats.total_energy_joules = sum_total_energy / count;
            avg_result.power_stats.avg_power_watts = sum_avg_power / count;
            avg_result.power_stats.peak_power_watts = sum_peak_power / count;
            avg_result.power_stats.power_per_query_mj = sum_power_per_query / count;
            avg_result.power_stats.energy_efficiency = sum_energy_efficiency / count;
            avg_result.power_stats.rapl_available = run_results[0].power_stats.rapl_available;
            
            // 磁盘统计平均
            avg_result.disk_stats.read_bytes = sum_read_bytes / count;
            avg_result.disk_stats.write_bytes = sum_write_bytes / count;
            avg_result.disk_stats.read_ops = sum_read_ops / count;
            avg_result.disk_stats.write_ops = sum_write_ops / count;
            avg_result.disk_stats.read_speed_mbps = sum_read_speed / count;
            avg_result.disk_stats.write_speed_mbps = sum_write_speed / count;
            avg_result.disk_stats.read_iops = sum_read_iops / count;
            avg_result.disk_stats.write_iops = sum_write_iops / count;
            avg_result.disk_stats.total_iops = sum_total_iops / count;
            avg_result.disk_stats.avg_io_latency_ms = sum_io_latency / count;
            avg_result.disk_stats.disk_utilization = sum_disk_util / count;
            avg_result.disk_stats.disk_bottleneck = any_disk_bottleneck;
            avg_result.disk_stats.disk_device = run_results[0].disk_stats.disk_device;
            avg_result.disk_stats.available_space_gb = sum_available_space / count;
            avg_result.disk_stats.total_space_gb = sum_total_space / count;
            avg_result.disk_stats.space_utilization = sum_space_util / count;
        }

        final_results.push_back({threads, avg_result});
    }

    // === 最后，打印格式化的总结果表格 ===
    cout << "\n\n===== Faiss Benchmark Summary with Disk Monitoring (Average of " << (num_runs_per_thread_setting <= 2 ? num_runs_per_thread_setting : num_runs_per_thread_setting - 2) << " runs, excluding min/max) =====" << endl;
    cout << "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
    cout << "| " << left << setw(8) << "Threads" 
         << " | " << setw(10) << "Time(s)" 
         << " | " << setw(12) << "QPS" 
         << " | " << setw(10) << "Avg Power(W)" 
         << " | " << setw(12) << "CPU Energy(J)" 
         << " | " << setw(12) << "Mem Energy(J)" 
         << " | " << setw(15) << "Power/Query(mJ)" 
         << " | " << setw(12) << "Efficiency" 
         << " | " << setw(12) << "Avg Lat(ms)" 
         << " | " << setw(10) << "P50 Lat(ms)" 
         << " | " << setw(10) << "P99 Lat(ms)" 
         << " | " << setw(10) << "Max Lat(ms)" 
         << " | " << setw(10) << "Min Lat(ms)" 
         << " | " << setw(12) << "Peak Mem(MB)"
         << " | " << setw(12) << "Disk Read(MB/s)"
         << " | " << setw(12) << "Disk Write(MB/s)"
         << " | " << setw(10) << "Total IOPS"
         << " | " << setw(12) << "IO Latency(ms)"
         << " | " << setw(12) << "Disk Util(%)"
         << " | " << setw(8) << "Bottleneck" << " |" << endl;
    cout << "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;

    // 创建文件输出流
    ofstream output_file("benchmark_results_with_disk.txt");
    
    // 输出表头到文件
    output_file << "Threads,Time(s),QPS,Avg Power(W),CPU Energy(J),Mem Energy(J),Power/Query(mJ),Efficiency,Avg Lat(ms),P50 Lat(ms),P99 Lat(ms),Max Lat(ms),Min Lat(ms),Peak Mem(MB),Disk Read(MB/s),Disk Write(MB/s),Total IOPS,IO Latency(ms),Disk Util(%),Bottleneck" << endl;

    for (const auto& result_pair : final_results) {
        // 输出到终端
        cout << "| " << left << setw(8) << result_pair.first 
             << " | " << fixed << setprecision(4) << setw(10) << result_pair.second.total_wall_time_s
             << " | " << fixed << setprecision(2) << setw(12) << result_pair.second.qps
             << " | " << fixed << setprecision(2) << setw(10) << result_pair.second.power_stats.avg_power_watts
             << " | " << fixed << setprecision(4) << setw(12) << result_pair.second.power_stats.cpu_energy_joules
             << " | " << fixed << setprecision(4) << setw(12) << result_pair.second.power_stats.memory_energy_joules
             << " | " << fixed << setprecision(4) << setw(15) << result_pair.second.power_stats.power_per_query_mj
             << " | " << fixed << setprecision(2) << setw(12) << result_pair.second.power_stats.energy_efficiency
             << " | " << fixed << setprecision(4) << setw(12) << result_pair.second.avg_latency_ms
             << " | " << fixed << setprecision(4) << setw(10) << result_pair.second.p50_latency_ms
             << " | " << fixed << setprecision(4) << setw(10) << result_pair.second.p99_latency_ms
             << " | " << fixed << setprecision(4) << setw(10) << result_pair.second.max_latency_ms
             << " | " << fixed << setprecision(4) << setw(10) << result_pair.second.min_latency_ms
             << " | " << fixed << setprecision(2) << setw(12) << result_pair.second.peak_memory_mb
             << " | " << fixed << setprecision(2) << setw(12) << result_pair.second.disk_stats.read_speed_mbps
             << " | " << fixed << setprecision(2) << setw(12) << result_pair.second.disk_stats.write_speed_mbps
             << " | " << fixed << setprecision(0) << setw(10) << result_pair.second.disk_stats.total_iops
             << " | " << fixed << setprecision(4) << setw(12) << result_pair.second.disk_stats.avg_io_latency_ms
             << " | " << fixed << setprecision(2) << setw(12) << result_pair.second.disk_stats.disk_utilization
             << " | " << setw(8) << (result_pair.second.disk_stats.disk_bottleneck ? "YES" : "NO") << " |" << endl;
        
        // 输出到文件（CSV格式）
        output_file << result_pair.first << ","
                   << fixed << setprecision(4) << result_pair.second.total_wall_time_s << ","
                   << fixed << setprecision(2) << result_pair.second.qps << ","
                   << fixed << setprecision(2) << result_pair.second.power_stats.avg_power_watts << ","
                   << fixed << setprecision(4) << result_pair.second.power_stats.cpu_energy_joules << ","
                   << fixed << setprecision(4) << result_pair.second.power_stats.memory_energy_joules << ","
                   << fixed << setprecision(4) << result_pair.second.power_stats.power_per_query_mj << ","
                   << fixed << setprecision(2) << result_pair.second.power_stats.energy_efficiency << ","
                   << fixed << setprecision(4) << result_pair.second.avg_latency_ms << ","
                   << fixed << setprecision(4) << result_pair.second.p50_latency_ms << ","
                   << fixed << setprecision(4) << result_pair.second.p99_latency_ms << ","
                   << fixed << setprecision(4) << result_pair.second.max_latency_ms << ","
                   << fixed << setprecision(4) << result_pair.second.min_latency_ms << ","
                   << fixed << setprecision(2) << result_pair.second.peak_memory_mb << ","
                   << fixed << setprecision(2) << result_pair.second.disk_stats.read_speed_mbps << ","
                   << fixed << setprecision(2) << result_pair.second.disk_stats.write_speed_mbps << ","
                   << fixed << setprecision(0) << result_pair.second.disk_stats.total_iops << ","
                   << fixed << setprecision(4) << result_pair.second.disk_stats.avg_io_latency_ms << ","
                   << fixed << setprecision(2) << result_pair.second.disk_stats.disk_utilization << ","
                   << (result_pair.second.disk_stats.disk_bottleneck ? "YES" : "NO") << endl;
    }
    cout << "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
    
    // 关闭文件
    output_file.close();
    cout << "\n结果已保存到 benchmark_results_with_disk.txt 文件中" << endl;

    // ==============================================================================
    // 磁盘瓶颈分析和优化建议
    // ==============================================================================
    cout << "\n" << string(60, '=') << endl;
    cout << "Phase 7: 磁盘瓶颈分析和优化建议" << endl;
    cout << string(60, '=') << endl;

    // 统计磁盘瓶颈情况
    int bottleneck_count = 0;
    double avg_disk_utilization = 0.0;
    double max_disk_utilization = 0.0;
    double avg_io_latency = 0.0;
    double max_io_latency = 0.0;
    double avg_read_speed = 0.0;
    double avg_write_speed = 0.0;
    double avg_total_iops = 0.0;

    for (const auto& result_pair : final_results) {
        if (result_pair.second.disk_stats.disk_bottleneck) {
            bottleneck_count++;
        }
        avg_disk_utilization += result_pair.second.disk_stats.disk_utilization;
        max_disk_utilization = max(max_disk_utilization, result_pair.second.disk_stats.disk_utilization);
        avg_io_latency += result_pair.second.disk_stats.avg_io_latency_ms;
        max_io_latency = max(max_io_latency, result_pair.second.disk_stats.avg_io_latency_ms);
        avg_read_speed += result_pair.second.disk_stats.read_speed_mbps;
        avg_write_speed += result_pair.second.disk_stats.write_speed_mbps;
        avg_total_iops += result_pair.second.disk_stats.total_iops;
    }

    int total_configs = final_results.size();
    avg_disk_utilization /= total_configs;
    avg_io_latency /= total_configs;
    avg_read_speed /= total_configs;
    avg_write_speed /= total_configs;
    avg_total_iops /= total_configs;

    cout << "\n===== 磁盘性能分析 =====" << endl;
    cout << "磁盘设备: " << (final_results.empty() ? "未知" : final_results[0].second.disk_stats.disk_device) << endl;
    cout << "可用空间: " << fixed << setprecision(2) << (final_results.empty() ? 0.0 : final_results[0].second.disk_stats.available_space_gb) << " GB" << endl;
    cout << "总空间: " << fixed << setprecision(2) << (final_results.empty() ? 0.0 : final_results[0].second.disk_stats.total_space_gb) << " GB" << endl;
    cout << "空间利用率: " << fixed << setprecision(2) << (final_results.empty() ? 0.0 : final_results[0].second.disk_stats.space_utilization) << "%" << endl;
    
    cout << "\n===== 磁盘I/O性能统计 =====" << endl;
    cout << "平均磁盘利用率: " << fixed << setprecision(2) << avg_disk_utilization << "%" << endl;
    cout << "最大磁盘利用率: " << fixed << setprecision(2) << max_disk_utilization << "%" << endl;
    cout << "平均IO延迟: " << fixed << setprecision(4) << avg_io_latency << " ms" << endl;
    cout << "最大IO延迟: " << fixed << setprecision(4) << max_io_latency << " ms" << endl;
    cout << "平均读取速度: " << fixed << setprecision(2) << avg_read_speed << " MB/s" << endl;
    cout << "平均写入速度: " << fixed << setprecision(2) << avg_write_speed << " MB/s" << endl;
    cout << "平均总IOPS: " << fixed << setprecision(0) << avg_total_iops << endl;
    
    cout << "\n===== 磁盘瓶颈检测结果 =====" << endl;
    cout << "检测到磁盘瓶颈的配置数量: " << bottleneck_count << "/" << total_configs << endl;
    cout << "磁盘瓶颈比例: " << fixed << setprecision(1) << (double(bottleneck_count) / total_configs * 100.0) << "%" << endl;

    // 找出最优配置
    auto max_efficiency = max_element(final_results.begin(), final_results.end(),
        [](const auto& a, const auto& b) {
            return a.second.power_stats.energy_efficiency < b.second.power_stats.energy_efficiency;
        });
    
    auto min_power_per_query = min_element(final_results.begin(), final_results.end(),
        [](const auto& a, const auto& b) {
            return a.second.power_stats.power_per_query_mj < b.second.power_stats.power_per_query_mj;
        });
    
    auto min_total_energy = min_element(final_results.begin(), final_results.end(),
        [](const auto& a, const auto& b) {
            return a.second.power_stats.total_energy_joules < b.second.power_stats.total_energy_joules;
        });

    // 找出磁盘性能最好的配置
    auto best_disk_performance = min_element(final_results.begin(), final_results.end(),
        [](const auto& a, const auto& b) {
            return a.second.disk_stats.avg_io_latency_ms < b.second.disk_stats.avg_io_latency_ms;
        });

    cout << "\n===== 性能优化建议 =====" << endl;
    cout << "最高能效配置: " << max_efficiency->first << " 线程 (能效: " 
         << fixed << setprecision(2) << max_efficiency->second.power_stats.energy_efficiency << " QPS/W)" << endl;
    cout << "最低每查询功耗配置: " << min_power_per_query->first << " 线程 (每查询: " 
         << fixed << setprecision(4) << min_power_per_query->second.power_stats.power_per_query_mj << " mJ)" << endl;
    cout << "最低总能耗配置: " << min_total_energy->first << " 线程 (总能耗: " 
         << fixed << setprecision(4) << min_total_energy->second.power_stats.total_energy_joules << " J)" << endl;
    cout << "最佳磁盘性能配置: " << best_disk_performance->first << " 线程 (IO延迟: " 
         << fixed << setprecision(4) << best_disk_performance->second.disk_stats.avg_io_latency_ms << " ms)" << endl;

    // 磁盘瓶颈优化建议
    cout << "\n===== 磁盘瓶颈优化建议 =====" << endl;
    if (bottleneck_count > total_configs * 0.5) {
        cout << "⚠️  警告: 超过50%的配置存在磁盘瓶颈，建议采取以下措施:" << endl;
        cout << "   1. 升级到SSD或NVMe存储设备" << endl;
        cout << "   2. 增加磁盘缓存大小" << endl;
        cout << "   3. 使用RAID配置提高I/O性能" << endl;
        cout << "   4. 考虑使用内存映射文件减少磁盘I/O" << endl;
    } else if (bottleneck_count > 0) {
        cout << "ℹ️  部分配置存在磁盘瓶颈，建议:" << endl;
        cout << "   1. 优化线程数配置以减少磁盘竞争" << endl;
        cout << "   2. 考虑使用更快的存储设备" << endl;
        cout << "   3. 调整系统I/O调度器设置" << endl;
    } else {
        cout << "✅ 未检测到明显的磁盘瓶颈，系统I/O性能良好" << endl;
    }

    if (avg_disk_utilization > 80.0) {
        cout << "⚠️  磁盘利用率过高 (" << fixed << setprecision(1) << avg_disk_utilization << "%)，建议:" << endl;
        cout << "   1. 减少并发I/O操作" << endl;
        cout << "   2. 增加磁盘缓存" << endl;
        cout << "   3. 考虑使用更快的存储设备" << endl;
    }

    if (avg_io_latency > 10.0) {
        cout << "⚠️  IO延迟较高 (" << fixed << setprecision(2) << avg_io_latency << " ms)，建议:" << endl;
        cout << "   1. 检查磁盘健康状态" << endl;
        cout << "   2. 优化I/O调度策略" << endl;
        cout << "   3. 考虑使用SSD存储" << endl;
    }

    if (final_results[0].second.disk_stats.space_utilization > 90.0) {
        cout << "⚠️  磁盘空间利用率过高 (" << fixed << setprecision(1) << final_results[0].second.disk_stats.space_utilization << "%)，建议:" << endl;
        cout << "   1. 清理不必要的文件" << endl;
        cout << "   2. 增加磁盘容量" << endl;
        cout << "   3. 使用数据压缩" << endl;
    }

    cout << "\n===== 系统监控信息 =====" << endl;
    if (final_results[0].second.power_stats.rapl_available) {
        cout << "功耗监控: 基于Intel RAPL接口，数据准确" << endl;
    } else {
        cout << "功耗监控: 基于估算方法，数据可能不够准确" << endl;
    }
    
    cout << "磁盘监控: 基于/proc/diskstats，监控设备: " << final_results[0].second.disk_stats.disk_device << endl;

    // ==============================================================================
    // 10. 报告峰值内存
    // ==============================================================================
    cout << "\n" << string(60, '=') << endl;
    string sys = "Linux"; // Assuming Linux, adjust if needed
    if (sys == "Linux" || sys == "Darwin") {
        rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        long peak_memory_bytes = usage.ru_maxrss;
        if (sys == "Linux") {
            peak_memory_bytes *= 1024;
        }
        double peak_memory_mb = peak_memory_bytes / (1024.0 * 1024.0);
        cout << "整个程序运行期间的峰值内存占用: " << fixed << setprecision(2) << peak_memory_mb << " MB" << endl;
    }
    cout << string(60, '=') << endl;

    delete index_final;
    if (!skip_index_building) {
        delete coarse_quantizer;
    }

    return 0;
}
