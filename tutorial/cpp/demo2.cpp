#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <faiss/impl/FaissAssert.h>
#include <omp.h>

using namespace std;

namespace {

const string DATA_DIR = "./sift";
const string LEARN_FILE = DATA_DIR + "/learn.fbin";
const string BASE_FILE = DATA_DIR + "/base.fbin";
const string QUERY_FILE = DATA_DIR + "/query.fbin";
const string GROUNDTRUTH_FILE = DATA_DIR + "/groundtruth.ivecs";

pair<vector<float>, pair<size_t, size_t>> read_fbin(
        const string& filename,
        size_t start_idx = 0,
        size_t chunk_size = 0) {
    ifstream f(filename, ios::binary);
    if (!f.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }

    int32_t nvecs_raw = 0;
    int32_t dim_raw = 0;
    f.read(reinterpret_cast<char*>(&nvecs_raw), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&dim_raw), sizeof(int32_t));
    size_t nvecs = static_cast<size_t>(nvecs_raw);
    size_t dim = static_cast<size_t>(dim_raw);

    size_t num_vectors_in_chunk = nvecs;
    if (chunk_size > 0) {
        size_t end_idx = min(start_idx + chunk_size, nvecs);
        num_vectors_in_chunk = end_idx - start_idx;
        size_t offset = 8 + start_idx * dim * sizeof(float);
        f.seekg(offset, ios::beg);
    }

    vector<float> data(num_vectors_in_chunk * dim);
    f.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));

    return {data, {nvecs, dim}};
}

vector<vector<int32_t>> read_ivecs(const string& filename) {
    ifstream f(filename, ios::binary | ios::ate);
    if (!f.is_open()) {
        return {};
    }

    size_t file_size = static_cast<size_t>(f.tellg());
    f.seekg(0, ios::beg);

    vector<int32_t> buffer(file_size / sizeof(int32_t));
    f.read(reinterpret_cast<char*>(buffer.data()), file_size);

    if (buffer.empty()) {
        return {};
    }

    int32_t d = buffer[0];
    if (d <= 0) {
        return {};
    }

    vector<vector<int32_t>> result(buffer.size() / (d + 1));
    for (size_t i = 0, j = 0; i < result.size(); ++i) {
        result[i].resize(d);
        ++j; // skip dim field
        copy(buffer.begin() + j, buffer.begin() + j + d, result[i].begin());
        j += d;
    }
    return result;
}

struct BenchmarkResult {
    string label;
    double seconds;
    double qps;
    double recall;
};

bool compute_recall(
        const vector<faiss::idx_t>& neighbors,
        size_t nq,
        size_t k,
        const vector<vector<int32_t>>& groundtruth,
        double& out_recall) {
    if (groundtruth.empty()) {
        return false;
    }

    size_t eval_queries = min(nq, groundtruth.size());
    if (eval_queries == 0) {
        return false;
    }

    size_t gt_k = groundtruth.front().size();
    size_t compare_k = min(k, gt_k);
    if (compare_k == 0) {
        return false;
    }

    size_t total_found = 0;
    for (size_t qi = 0; qi < eval_queries; ++qi) {
        const faiss::idx_t* row = neighbors.data() + qi * k;
        for (size_t j = 0; j < k; ++j) {
            faiss::idx_t candidate = row[j];
            const auto& gt_row = groundtruth[qi];
            if (find(gt_row.begin(), gt_row.begin() + compare_k, candidate) !=
                    gt_row.begin() + compare_k) {
                ++total_found;
            }
        }
    }

    double denom = static_cast<double>(eval_queries) * compare_k;
    if (denom == 0.0) {
        return false;
    }

    out_recall = static_cast<double>(total_found) / denom;
    return true;
}

string build_index_path(size_t dim, size_t nlist, size_t M, size_t efc) {
    string base_name = BASE_FILE.substr(BASE_FILE.find_last_of('/') + 1);
    base_name = base_name.substr(0, base_name.find_last_of('.'));
    regex non_alnum("[^a-zA-Z0-9_]");
    string clean = regex_replace(base_name, non_alnum, "_");
    return DATA_DIR + "/" + clean + "_d" + to_string(dim) + "_nlist" +
            to_string(nlist) + "_HNSWM" + to_string(M) + "_efc" +
            to_string(efc) + "_IVFFlat.index";
}

void maybe_build_index(
        const string& index_path,
        size_t d,
        size_t nlist,
        size_t nt,
        size_t nb,
        size_t chunk_size,
        size_t M,
        size_t efconstruction,
        size_t efsearch) {
    ifstream fin(index_path);
    if (fin.good()) {
        return;
    }

    auto* coarse_quantizer = new faiss::IndexHNSWFlat(d, M);
    auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(coarse_quantizer);
    hnsw->hnsw.efConstruction = static_cast<int>(efconstruction);
    hnsw->hnsw.efSearch = static_cast<int>(efsearch);

    auto* train_index = new faiss::IndexIVFFlat(
            coarse_quantizer, d, nlist, faiss::METRIC_L2);
    train_index->verbose = false;

    auto training_pair = read_fbin(LEARN_FILE);
    vector<float>& xt_data = training_pair.first;
    train_index->train(nt, xt_data.data());
    delete train_index;

    auto* index_shell = new faiss::IndexIVFFlat(
            coarse_quantizer, d, nlist, faiss::METRIC_L2);
    faiss::write_index(index_shell, index_path.c_str());
    delete index_shell;

    int IO_FLAG_READ_WRITE = 0;
    faiss::Index* index_ondisk = faiss::read_index(index_path.c_str(), IO_FLAG_READ_WRITE);

    for (size_t i = 0; i < nb; i += chunk_size) {
        auto xb_chunk = read_fbin(BASE_FILE, i, chunk_size);
        size_t current = min(chunk_size, nb - i);
        index_ondisk->add(current, xb_chunk.first.data());
    }

    faiss::write_index(index_ondisk, index_path.c_str());
    delete index_ondisk;
    delete coarse_quantizer;
}

BenchmarkResult run_search_mode(
        faiss::IndexIVF* index_ivf,
        const float* queries,
        size_t nq,
        size_t k,
        bool use_hnsw,
        int hnsw_ef_search,
        const vector<vector<int32_t>>& groundtruth) {
    vector<float> distances(nq * k);
    vector<faiss::idx_t> labels(nq * k);

    auto start = chrono::high_resolution_clock::now();
    if (use_hnsw) {
        index_ivf->search_with_hnsw(
                nq,
                queries,
                k,
                distances.data(),
                labels.data(),
                nullptr,
                hnsw_ef_search);
    } else {
        index_ivf->search(nq, queries, k, distances.data(), labels.data());
    }
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> duration = end - start;
    double seconds = duration.count();
    double qps = seconds > 0 ? static_cast<double>(nq) / seconds : 0.0;

    double recall = -1.0;
    if (!compute_recall(labels, nq, k, groundtruth, recall)) {
        recall = -1.0;
    }

    return {use_hnsw ? "search_with_hnsw" : "search", seconds, qps, recall};
}

void print_result(const BenchmarkResult& result) {
    cout << left << setw(22) << result.label << " | "
         << "time: " << fixed << setprecision(3) << result.seconds << " s"
         << ", QPS: " << setw(10) << result.qps;
    if (result.recall >= 0.0) {
        cout << ", Recall@k: " << setprecision(4) << result.recall;
    } else {
        cout << ", Recall@k: N/A";
    }
    cout << endl;
}

} // namespace

int main() {
    auto train_meta = read_fbin(LEARN_FILE, 0, 1);
    size_t nt = train_meta.second.first;
    size_t d = train_meta.second.second;

    auto base_meta = read_fbin(BASE_FILE, 0, 1);
    size_t nb = base_meta.second.first;
    size_t d_base = base_meta.second.second;

    auto query_meta = read_fbin(QUERY_FILE, 0, 1);
    size_t nq = query_meta.second.first;
    size_t d_query = query_meta.second.second;

    if (d != d_base || d != d_query) {
        throw runtime_error("Dataset dimensions are inconsistent");
    }

    size_t cell_size = 64;
    size_t nlist = nb / cell_size;
    size_t nprobe = 64;
    size_t chunk_size = 100000;
    size_t k = 10;
    size_t M = 32;
    size_t efconstruction = 16;
    size_t efsearch = 40;
    int hnsw_ef_override = static_cast<int>(efsearch * 1);

    string index_path = build_index_path(d, nlist, M, efconstruction);
    maybe_build_index(
            index_path,
            d,
            nlist,
            nt,
            nb,
            chunk_size,
            M,
            efconstruction,
            efsearch);

    int io_flag_mmap = faiss::IO_FLAG_MMAP;
    faiss::Index* generic_index = faiss::read_index(index_path.c_str(), io_flag_mmap);
    auto* index_ivf = dynamic_cast<faiss::IndexIVF*>(generic_index);
    FAISS_THROW_IF_NOT_MSG(index_ivf, "Loaded index is not an IVF index");

    index_ivf->nprobe = nprobe;
    index_ivf->parallel_mode = 0;
    omp_set_num_threads(40);

    auto* quantizer_hnsw = dynamic_cast<faiss::IndexHNSW*>(index_ivf->quantizer);
    FAISS_THROW_IF_NOT_MSG(
            quantizer_hnsw, "Coarse quantizer is not an HNSW index");
    quantizer_hnsw->hnsw.efSearch = static_cast<int>(efsearch);

    auto query_data_pair = read_fbin(QUERY_FILE);
    vector<float>& xq_data = query_data_pair.first;
    auto groundtruth = read_ivecs(GROUNDTRUTH_FILE);

    cout << "Benchmark parameters:\n"
         << "  dimension: " << d << "\n"
         << "  database vectors: " << nb << "\n"
         << "  queries: " << nq << "\n"
         << "  nlist: " << nlist << ", nprobe: " << nprobe << "\n"
         << "  HNSW M: " << M << ", efConstruction: " << efconstruction
         << ", efSearch: " << efsearch << endl;
    if (!groundtruth.empty()) {
        cout << "  groundtruth vectors loaded: " << groundtruth.size() << endl;
    } else {
        cout << "  groundtruth not found or empty; recall will be omitted" << endl;
    }

    auto baseline = run_search_mode(
            index_ivf,
            xq_data.data(),
            nq,
            k,
            false,
            0,
            groundtruth);
    auto hnsw_accel = run_search_mode(
            index_ivf,
            xq_data.data(),
            nq,
            k,
            true,
            hnsw_ef_override,
            groundtruth);

    cout << "\nResults:\n";
    print_result(baseline);
    print_result(hnsw_accel);

    if (baseline.seconds > 0.0 && hnsw_accel.seconds > 0.0) {
        double speedup = baseline.seconds / hnsw_accel.seconds;
        cout << "\nSpeedup (baseline / hnsw): " << fixed << setprecision(3)
             << speedup << "x" << endl;
    }

    delete generic_index;
    return 0;
}
