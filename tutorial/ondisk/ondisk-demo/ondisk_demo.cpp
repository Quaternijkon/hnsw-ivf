#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/invlists/OnDiskInvertedLists.h>
#include <faiss/invlists/InvertedLists.h>

#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <cstdio> // For remove()
#include <memory> // For std::unique_ptr
#include <fstream> // For memory monitoring
#include <sstream> // For memory monitoring
#include <iomanip> // For formatted output
#include <chrono> // For timing
#include <algorithm> // For std::find, std::min

// --- 内存监控功能 ---

// 获取当前实际内存使用量（通过/proc/self/status）
long getCurrentMemoryMB() {
    std::ifstream status_file("/proc/self/status");
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line);
            std::string key, value, unit;
            iss >> key >> value >> unit;
            return std::stol(value) / 1024; // 转换为MB
        }
    }
    return 0;
}

// 监控峰值内存的类
class PeakMemoryMonitor {
private:
    long start_memory_mb;
    long peak_memory_mb;
    bool monitoring;
    
public:
    PeakMemoryMonitor() : start_memory_mb(0), peak_memory_mb(0), monitoring(false) {}
    
    void start() {
        start_memory_mb = getCurrentMemoryMB();
        peak_memory_mb = start_memory_mb;
        monitoring = true;
    }
    
    void update() {
        if (monitoring) {
            long current_memory = getCurrentMemoryMB();
            peak_memory_mb = std::max(peak_memory_mb, current_memory);
        }
    }
    
    long getPeakMemoryMB() const {
        return peak_memory_mb;
    }
    
    long getMemoryIncrease() const {
        return peak_memory_mb - start_memory_mb;
    }
    
    long getCurrentMemoryMB() const {
        return ::getCurrentMemoryMB();
    }
    
    void stop() {
        monitoring = false;
    }
    
    void print(const std::string& phase_name) const {
        std::cout << "  [" << phase_name << "] "
                  << "当前内存: " << std::fixed << std::setprecision(2) 
                  << getCurrentMemoryMB() << "MB, "
                  << "峰值内存: " << peak_memory_mb << "MB, "
                  << "增长: " << getMemoryIncrease() << "MB" << std::endl;
    }
};

// --- 数据文件路径配置 ---
// 支持多个可能的路径
const std::string DATA_DIR1 = "../../cpp/data";
const std::string DATA_DIR2 = "../cpp/data";
const std::string DATA_DIR3 = "../../../tutorial/cpp/data";
const std::string DATA_DIR4 = "../data";

std::string find_data_file(const std::string& filename) {
    std::vector<std::string> dirs = {DATA_DIR1, DATA_DIR2, DATA_DIR3, DATA_DIR4};
    for (const auto& dir : dirs) {
        std::string full_path = dir + "/" + filename;
        std::ifstream test(full_path);
        if (test.good()) {
            test.close();
            return full_path;
        }
    }
    throw std::runtime_error("Cannot find " + filename + " in any of the expected directories:\n" +
                             "  " + DATA_DIR1 + "\n  " + DATA_DIR2 + "\n  " + DATA_DIR3 + "\n  " + DATA_DIR4);
}

// 数据文件路径将在 main 函数中动态查找

// --- 帮助函数：读取 .fbin 格式文件 ---
// .fbin 格式：文件头包含 [nvecs: int32, dim: int32]，然后是数据 [float32[nvecs*dim]]
std::pair<std::vector<float>, std::pair<size_t, size_t>> read_fbin(
    const std::string& filename, 
    size_t start_idx = 0, 
    size_t chunk_size = 0) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename + 
                                 "\nPlease ensure SIFT dataset files exist in one of:\n" +
                                 "  " + DATA_DIR1 + "\n  " + DATA_DIR2 + "\n  " + DATA_DIR3);
    }

    int32_t nvecs_raw, dim_raw;
    f.read(reinterpret_cast<char*>(&nvecs_raw), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&dim_raw), sizeof(int32_t));
    size_t nvecs = static_cast<size_t>(nvecs_raw);
    size_t dim = static_cast<size_t>(dim_raw);

    size_t num_vectors_in_chunk = nvecs;
    if (chunk_size > 0) {
        size_t end_idx = std::min(start_idx + chunk_size, nvecs);
        num_vectors_in_chunk = end_idx - start_idx;
        size_t offset = 8 + start_idx * dim * sizeof(float); // 8 = 2 * sizeof(int32_t)
        f.seekg(offset, std::ios::beg);
    }

    std::vector<float> data(num_vectors_in_chunk * dim);
    f.read(reinterpret_cast<char*>(data.data()), num_vectors_in_chunk * dim * sizeof(float));

    return {data, {nvecs, dim}};
}

// --- 帮助函数：读取 .ivecs 格式的ground truth文件 ---
// .ivecs 格式：每个查询一行，格式为 [dim: int32, id1: int32, id2: int32, ...]
std::vector<std::vector<int32_t>> read_ivecs(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        return {}; // 返回空vector，表示文件不存在
    }
    size_t file_size = f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<int32_t> a(file_size / sizeof(int32_t));
    f.read(reinterpret_cast<char*>(a.data()), file_size);

    int32_t d = a[0];
    std::vector<std::vector<int32_t>> result(a.size() / (d + 1));
    for (size_t i = 0, j = 0; i < result.size(); ++i) {
        result[i].resize(d);
        ++j; // skip dim
        std::copy(a.begin() + j, a.begin() + j + d, result[i].begin());
        j += d;
    }
    return result;
}

// --- 计算召回率 ---
double calculateRecall(const std::vector<faiss::idx_t>& search_results, 
                      const std::vector<std::vector<int32_t>>& groundtruth, 
                      size_t nq, size_t k) {
    if (groundtruth.empty()) return 0.0;
    
    size_t total_found = 0;
    for (size_t i = 0; i < nq; ++i) {
        size_t found_count = 0;
        size_t check_size = std::min(k, static_cast<size_t>(groundtruth[i].size()));
        
        for (size_t j = 0; j < k; ++j) {
            faiss::idx_t neighbor = search_results[i * k + j];
            if (std::find(groundtruth[i].begin(), groundtruth[i].begin() + check_size, neighbor) != 
                groundtruth[i].begin() + check_size) {
                ++found_count;
            }
        }
        total_found += found_count;
    }
    
    return static_cast<double>(total_found) / (nq * k);
}

int main() {
    // --- 0. 参数定义 ---
    // 先从训练文件中获取维度信息
    try {
        // 查找数据文件（在 main 函数中调用，因为 find_data_file 需要抛出异常）
        std::string learn_file = find_data_file("sift_learn.fbin");
        std::string base_file = find_data_file("sift_base.fbin");
        std::string query_file = find_data_file("sift_query.fbin");
        
        std::cout << "Using data files:" << std::endl;
        std::cout << "  Learn: " << learn_file << std::endl;
        std::cout << "  Base: " << base_file << std::endl;
        std::cout << "  Query: " << query_file << std::endl;
        std::cout << std::endl;
        
        auto [_, metadata] = read_fbin(learn_file, 0, 1);
        size_t d = metadata.second;  // 向量维度（SIFT 是 128）
        size_t nt_total = metadata.first;  // 训练向量总数
        
        std::cout << "Dataset info:" << std::endl;
        std::cout << "  Dimension: " << d << std::endl;
        std::cout << "  Training vectors: " << nt_total << std::endl;
        
        // 获取基础数据集信息
        auto [__, base_metadata] = read_fbin(base_file, 0, 1);
        size_t nb_total = base_metadata.first;
        std::cout << "  Base vectors: " << nb_total << std::endl;
        
        // 获取查询数据集信息
        auto [___, query_metadata] = read_fbin(query_file, 0, 1);
        size_t nq_total = query_metadata.first;
        std::cout << "  Query vectors: " << nq_total << std::endl;
        
        // 使用所有可用数据（或根据内存情况调整）
        size_t nt = nt_total;   // 使用所有训练向量
        size_t nb = nb_total;   // 使用所有基础向量
        size_t nq = std::min(nq_total, static_cast<size_t>(1000));  // 限制查询数量用于演示
        
        int nlist = 100;  // 倒排列表（聚类中心）数量
        int k = 100;     // 返回 top-k 结果
        
        std::cout << "\nIndex parameters:" << std::endl;
        std::cout << "  nlist: " << nlist << std::endl;
        std::cout << "  Index type: IVF-Flat (no quantization)" << std::endl;
        std::cout << "  k (top-k): " << k << std::endl;
        std::cout << std::endl;

        std::string main_index_file = "ondisk_demo_flat.faiss";
        std::string ivf_data_file = "ondisk_demo_flat.ivf";

        // --- 清理可能存在的旧文件 ---
        remove(main_index_file.c_str());
        remove(ivf_data_file.c_str());

        // --- 全局内存监控 ---
        PeakMemoryMonitor global_memory_monitor;
        global_memory_monitor.start();
        std::cout << "===== 程序启动 =====" << std::endl;
        global_memory_monitor.print("启动");

    std::cout << "\n===== Phase 1: Build Index =====" << std::endl;

    { // --- 构建作用域 ---
      // --- 1. 初始化阶段 ---
        PeakMemoryMonitor init_memory_monitor;
        init_memory_monitor.start();
        
        std::cout << "\n--- 1.1 初始化阶段 ---" << std::endl;
        std::cout << "Loading training data from " << learn_file << "..." << std::endl;
        auto [train_data_vec, train_meta] = read_fbin(learn_file);
        size_t train_dim = train_meta.second;
        size_t train_nvecs = train_meta.first;
        
        if (train_dim != d) {
            std::cerr << "Error: Dimension mismatch! Expected " << d 
                      << " but got " << train_dim << std::endl;
            return 1;
        }
        
        if (train_nvecs < nt) {
            std::cerr << "Warning: Only " << train_nvecs 
                      << " training vectors available, but " << nt << " requested." << std::endl;
            nt = train_nvecs;
        }
        
        float* train_vectors = train_data_vec.data();
        init_memory_monitor.update();
        init_memory_monitor.print("加载训练数据后");
        global_memory_monitor.update();

        std::unique_ptr<faiss::IndexFlatL2> quantizer(new faiss::IndexFlatL2(d));
        init_memory_monitor.update();
        init_memory_monitor.print("创建量化器后");
        global_memory_monitor.update();
        
        std::unique_ptr<faiss::IndexIVFFlat> index(new faiss::IndexIVFFlat(
            quantizer.get(), d, nlist, faiss::METRIC_L2
        ));
        init_memory_monitor.update();
        init_memory_monitor.print("创建索引后");
        global_memory_monitor.update();
        
        std::cout << "初始化阶段峰值内存: " << std::fixed << std::setprecision(2) 
                  << init_memory_monitor.getPeakMemoryMB() << "MB" << std::endl;

      // --- 1.2 训练阶段 ---
        PeakMemoryMonitor train_memory_monitor;
        train_memory_monitor.start();
        
        std::cout << "\n--- 1.2 训练阶段 ---" << std::endl;
        std::cout << "Training index on " << nt << " vectors..." << std::endl;
        train_memory_monitor.update();
        train_memory_monitor.print("训练开始");
        global_memory_monitor.update();
        
        index->train(nt, train_vectors);
        train_memory_monitor.update();
        train_memory_monitor.print("训练完成");
        global_memory_monitor.update();
        
        std::cout << "训练阶段峰值内存: " << std::fixed << std::setprecision(2) 
                  << train_memory_monitor.getPeakMemoryMB() << "MB" << std::endl;
        
        // 释放训练数据
        train_data_vec.clear();
        train_data_vec.shrink_to_fit();
        train_memory_monitor.update();
        train_memory_monitor.print("释放训练数据后");
        global_memory_monitor.update();

      // --- 1.3 构建阶段（添加数据到 ArrayInvertedLists）---
      // 根据官方示例，先使用 ArrayInvertedLists 添加数据，
      // 然后使用 merge_from_1 合并到 OnDiskInvertedLists 以创建紧凑存储
        PeakMemoryMonitor build_memory_monitor;
        build_memory_monitor.start();
        
        std::cout << "\n--- 1.3 构建阶段（添加数据到内存索引）---" << std::endl;
        std::cout << "Loading base data from " << base_file << "..." << std::endl;
        std::cout << "Adding " << nb << " vectors in batches..." << std::endl;
        size_t batch_size = 10000;  // 每次加载 10000 个向量以减少内存峰值
        std::vector<faiss::idx_t> batch_ids(batch_size);
        
        build_memory_monitor.update();
        build_memory_monitor.print("开始添加数据");
        global_memory_monitor.update();
        
        size_t num_batches = (nb + batch_size - 1) / batch_size;
        for (size_t i = 0; i < num_batches; ++i) {
            size_t start_idx = i * batch_size;
            size_t current_batch_size = std::min(batch_size, nb - start_idx);
            
            // 读取一批数据
            auto [batch_data, _] = read_fbin(base_file, start_idx, current_batch_size);
            
            // 准备 ID
            for(size_t j = 0; j < current_batch_size; ++j) {
                batch_ids[j] = static_cast<faiss::idx_t>(start_idx + j);
            }
            
            // 添加到索引
            index->add_with_ids(current_batch_size, batch_data.data(), batch_ids.data());
            build_memory_monitor.update();
            global_memory_monitor.update();
            
            // 释放批次数据
            batch_data.clear();
            batch_data.shrink_to_fit();
            
            if ((i + 1) % 10 == 0 || i == num_batches - 1) {
                 std::cout << "  Added batch " << (i + 1) << "/" << num_batches 
                           << " (vectors " << start_idx << "-" << (start_idx + current_batch_size - 1) << ")" << std::endl;
                 build_memory_monitor.print("批次 " + std::to_string(i));
            }
        }
        build_memory_monitor.update();
        build_memory_monitor.print("添加数据完成");
        global_memory_monitor.update();
        
        std::cout << "Add complete. Total vectors: " << index->ntotal << std::endl;
        std::cout << "构建阶段峰值内存: " << std::fixed << std::setprecision(2) 
                  << build_memory_monitor.getPeakMemoryMB() << "MB" << std::endl;

      // --- 1.4 合并阶段（ArrayInvertedLists -> OnDiskInvertedLists）---
      // 这是官方推荐的做法：先构建 ArrayInvertedLists，然后合并到 OnDiskInvertedLists
      // merge_from_1 会创建紧凑的存储，避免段错误
        PeakMemoryMonitor merge_memory_monitor;
        merge_memory_monitor.start();
        
        std::cout << "\n--- 1.4 合并阶段（转换到磁盘索引）---" << std::endl;
        std::cout << "Creating OnDiskInvertedLists and merging data..." << std::endl;
        
        faiss::IndexIVF* index_ivf = dynamic_cast<faiss::IndexIVF*>(index.get());
        if (!index_ivf) {
            std::cerr << "Error: Index is not an IVF index" << std::endl;
            return 1;
        }
        
        // 保存当前的 invlists（ArrayInvertedLists）
        faiss::InvertedLists* array_invlists = index_ivf->invlists;
        merge_memory_monitor.update();
        merge_memory_monitor.print("保存 ArrayInvertedLists");
        global_memory_monitor.update();
        
        // 创建新的 OnDiskInvertedLists（空的）
        faiss::OnDiskInvertedLists* ondisk_invlists = 
            new faiss::OnDiskInvertedLists(index_ivf->nlist, index_ivf->code_size, ivf_data_file.c_str());
        merge_memory_monitor.update();
        merge_memory_monitor.print("创建 OnDiskInvertedLists");
        global_memory_monitor.update();
        
        // 使用 merge_from_1 将 ArrayInvertedLists 合并到 OnDiskInvertedLists
        // 这会创建紧凑的存储，确保所有列表都是连续的
        size_t ntotal = ondisk_invlists->merge_from_1(array_invlists, true);
        merge_memory_monitor.update();
        merge_memory_monitor.print("合并完成");
        global_memory_monitor.update();
        
        // 替换索引的 invlists
        index_ivf->invlists = ondisk_invlists;
        index_ivf->own_invlists = true;
        index_ivf->ntotal = ntotal;
        
        // 删除旧的 ArrayInvertedLists（不再需要）
        delete array_invlists;
        merge_memory_monitor.update();
        merge_memory_monitor.print("删除 ArrayInvertedLists 后");
        global_memory_monitor.update();
        
        std::cout << "Merge complete. Total vectors: " << ntotal << std::endl;
        std::cout << "合并阶段峰值内存: " << std::fixed << std::setprecision(2) 
                  << merge_memory_monitor.getPeakMemoryMB() << "MB" << std::endl;
        
      // --- 1.5 写入索引元数据 ---
        PeakMemoryMonitor write_memory_monitor;
        write_memory_monitor.start();
        
        std::cout << "\n--- 1.5 写入索引元数据 ---" << std::endl;
        std::cout << "Writing index metadata -> " << main_index_file << std::endl;
        write_memory_monitor.update();
        global_memory_monitor.update();
        
        faiss::write_index(index.get(), main_index_file.c_str());
        write_memory_monitor.update();
        write_memory_monitor.print("写入完成");
        global_memory_monitor.update();

        std::cout << "Explicitly destroying index..." << std::endl;
        index.reset();
        quantizer.reset();
        write_memory_monitor.update();
        write_memory_monitor.print("销毁索引后");
        global_memory_monitor.update();
        
        std::cout << "Index destroyed." << std::endl;
        std::cout << "写入阶段峰值内存: " << std::fixed << std::setprecision(2) 
                  << write_memory_monitor.getPeakMemoryMB() << "MB" << std::endl;
        
        std::cout << "\n===== Phase 1 总结 =====" << std::endl;
        std::cout << "构建阶段总峰值内存: " << std::fixed << std::setprecision(2) 
                  << global_memory_monitor.getPeakMemoryMB() << "MB" << std::endl;

    } // --- 构建作用域结束 ---
    
    std::cout << "\n===== Phase 2: Search with mmap =====" << std::endl;

    // 在搜索作用域外定义recall和QPS变量，以便在最后输出
    double final_qps = 0.0;
    double final_recall = 0.0;

    { // --- 搜索作用域 (模拟一个新进程) ---
      
      // --- 2.1 加载索引阶段 ---
        PeakMemoryMonitor load_memory_monitor;
        load_memory_monitor.start();
        
        std::cout << "\n--- 2.1 加载索引阶段 ---" << std::endl;
        // 注意：对于 OnDiskInvertedLists，我们不使用 IO_FLAG_MMAP，
        // 因为它包含 IO_FLAG_SKIP_IVF_DATA，会阻止 do_mmap() 被调用。
        // 我们使用 IO_FLAG_ONDISK_SAME_DIR 和 IO_FLAG_READ_ONLY 来确保
        // OnDiskInvertedLists 的文件被正确映射。
        std::cout << "Reading index " << main_index_file << "..." << std::endl;
        load_memory_monitor.update();
        load_memory_monitor.print("加载索引前");
        global_memory_monitor.update();
        
        std::unique_ptr<faiss::Index> index_to_search(
            faiss::read_index(main_index_file.c_str(), 
                              faiss::IO_FLAG_ONDISK_SAME_DIR | faiss::IO_FLAG_READ_ONLY)
        );
        load_memory_monitor.update();
        load_memory_monitor.print("加载索引后");
        global_memory_monitor.update();

        std::cout << "Index loaded. Total vectors: " << index_to_search->ntotal << std::endl;
        std::cout << "加载索引阶段峰值内存: " << std::fixed << std::setprecision(2) 
                  << load_memory_monitor.getPeakMemoryMB() << "MB" << std::endl;

        faiss::IndexIVF* index_ivf_search = dynamic_cast<faiss::IndexIVF*>(index_to_search.get());
        if (!index_ivf_search) {
            std::cerr << "Error: Loaded index is not an IVF index" << std::endl;
            return 1;
        }

        faiss::OnDiskInvertedLists* loaded_invlists = 
            dynamic_cast<faiss::OnDiskInvertedLists*>(index_ivf_search->invlists);
        
        if (!loaded_invlists) {
            std::cerr << "CRITICAL ERROR: Index was NOT loaded with OnDiskInvertedLists!" << std::endl;
            return 1;
        }
        
        std::cout << "SUCCESS: Index confirmed to be using OnDiskInvertedLists." << std::endl;
        std::cout << "  Invlists file: " << loaded_invlists->filename << std::endl;
        std::cout << "  Total size: " << loaded_invlists->totsize << " bytes" << std::endl;
        std::cout << "  Ptr: " << (void*)loaded_invlists->ptr << std::endl;
        std::cout << "  Read only: " << loaded_invlists->read_only << std::endl;
        
        // 验证 ptr 是否有效（应该通过 do_mmap() 被设置）
        if (loaded_invlists->ptr == nullptr) {
            std::cerr << "ERROR: ptr is nullptr! do_mmap() may have failed." << std::endl;
            return 1;
        }

      // --- 2.2 搜索阶段 ---
        PeakMemoryMonitor search_memory_monitor;
        search_memory_monitor.start();
        
        std::cout << "\n--- 2.2 搜索阶段 ---" << std::endl;
        index_ivf_search->nprobe = 10;
        
        std::cout << "Loading query data from " << query_file << "..." << std::endl;
        auto [query_data_vec, query_meta] = read_fbin(query_file);
        size_t query_dim = query_meta.second;
        size_t query_nvecs = query_meta.first;
        
        if (query_dim != d) {
            std::cerr << "Error: Query dimension mismatch! Expected " << d 
                      << " but got " << query_dim << std::endl;
            return 1;
        }
        
        if (query_nvecs < nq) {
            std::cerr << "Warning: Only " << query_nvecs 
                      << " query vectors available, but " << nq << " requested." << std::endl;
            nq = query_nvecs;
        }
        
        float* query_vectors = query_data_vec.data();
        search_memory_monitor.update();
        search_memory_monitor.print("加载查询数据后");
        global_memory_monitor.update();
        
        std::vector<faiss::idx_t> I(nq * k);
        std::vector<float> D(nq * k);
        search_memory_monitor.update();
        search_memory_monitor.print("分配结果缓冲区后");
        global_memory_monitor.update();

        std::cout << "Searching for " << nq << " query vectors..." << std::endl;
        // 记录搜索开始时间
        auto search_start = std::chrono::high_resolution_clock::now();
        index_ivf_search->search(nq, query_vectors, k, D.data(), I.data());
        // 记录搜索结束时间
        auto search_end = std::chrono::high_resolution_clock::now();
        search_memory_monitor.update();
        search_memory_monitor.print("搜索完成");
        global_memory_monitor.update();
        
        // 计算搜索时间（在内存监控停止前）
        double search_time_s = std::chrono::duration<double>(search_end - search_start).count();
        
        std::cout << "搜索阶段峰值内存: " << std::fixed << std::setprecision(2) 
                  << search_memory_monitor.getPeakMemoryMB() << "MB" << std::endl;
        
        // 停止内存监控，后续计算recall和QPS不计入搜索阶段内存
        search_memory_monitor.stop();

      // --- 2.3 计算recall和QPS（在内存监控停止后）---
        // 计算QPS
        if (search_time_s > 0) {
            final_qps = nq / search_time_s;
        }
        
        // 计算recall（如果ground truth文件存在）
        try {
            std::string groundtruth_file = find_data_file("sift_groundtruth.ivecs");
            std::ifstream gt_file_check(groundtruth_file);
            if (gt_file_check.good()) {
                gt_file_check.close();
                std::vector<std::vector<int32_t>> groundtruth = read_ivecs(groundtruth_file);
                if (!groundtruth.empty()) {
                    final_recall = calculateRecall(I, groundtruth, nq, k);
                }
            }
        } catch (const std::exception& e) {
            // groundtruth文件不存在，跳过recall计算
            // std::cerr << "Warning: Cannot find groundtruth file, skipping recall calculation." << std::endl;
        }
        
        std::cout << "\n===== Phase 2 总结 =====" << std::endl;
        std::cout << "搜索阶段总峰值内存: " << std::fixed << std::setprecision(2) 
                  << search_memory_monitor.getPeakMemoryMB() << "MB" << std::endl;

    } // --- 搜索作用域结束 ---

    // --- 最终清理和总结 ---
    std::cout << "\n===== Phase 3: Cleanup =====" << std::endl;
    std::cout << "Cleaning up files..." << std::endl;
    remove(main_index_file.c_str());
    remove(ivf_data_file.c_str());
    
    global_memory_monitor.update();
    std::cout << "\n===== 内存使用总结 =====" << std::endl;
    std::cout << "程序总峰值内存: " << std::fixed << std::setprecision(2) 
              << global_memory_monitor.getPeakMemoryMB() << "MB" << std::endl;
    std::cout << "程序总内存增长: " << std::fixed << std::setprecision(2) 
              << global_memory_monitor.getMemoryIncrease() << "MB" << std::endl;
    std::cout << "当前内存: " << std::fixed << std::setprecision(2) 
              << global_memory_monitor.getCurrentMemoryMB() << "MB" << std::endl;
    
    // --- 输出recall和QPS信息（在内存监控之外）---
    std::cout << "\n===== 搜索性能指标 =====" << std::endl;
    std::cout << "QPS (每秒查询数): " << std::fixed << std::setprecision(2) << final_qps << std::endl;
    if (final_recall > 0.0) {
        std::cout << "Recall@k: " << std::fixed << std::setprecision(4) << final_recall << std::endl;
    } else {
        std::cout << "Recall@k: N/A (groundtruth file not found)" << std::endl;
    }
    
    std::cout << "\nDemo complete." << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
