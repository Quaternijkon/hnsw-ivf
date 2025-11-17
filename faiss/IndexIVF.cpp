/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexIVF.h>

#include <omp.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <chrono>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <limits>
#include <unordered_set>

#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>
#include <faiss/utils/distances.h>

#include <faiss/IndexFlat.h>
#include <faiss/Clustering.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/invlists/DirectMap.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/CodePacker.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>

struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    Timer() : t0(std::chrono::high_resolution_clock::now()) {}
    double elapsed_us() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
};

namespace faiss {

using ScopedIds = InvertedLists::ScopedIds;
using ScopedCodes = InvertedLists::ScopedCodes;

/*****************************************
 * Level1Quantizer implementation
 ******************************************/

Level1Quantizer::Level1Quantizer(Index* quantizer, size_t nlist)
        : quantizer(quantizer), nlist(nlist) {
    // here we set a low # iterations because this is typically used
    // for large clusterings (nb this is not used for the MultiIndex,
    // for which quantizer_trains_alone = true)
    cp.niter = 10;
}

Level1Quantizer::Level1Quantizer() = default;

Level1Quantizer::~Level1Quantizer() {
    if (own_fields) {
        delete quantizer;
    }
}

void Level1Quantizer::train_q1(
        size_t n,
        const float* x,
        bool verbose,
        MetricType metric_type) {
    size_t d = quantizer->d;
    if (quantizer->is_trained && (quantizer->ntotal == nlist)) {
        if (verbose)
            printf("IVF quantizer does not need training.\n");
    } else if (quantizer_trains_alone == 1) {
        if (verbose)
            printf("IVF quantizer trains alone...\n");
        quantizer->verbose = verbose;
        quantizer->train(n, x);
        FAISS_THROW_IF_NOT_MSG(
                quantizer->ntotal == nlist,
                "nlist not consistent with quantizer size");
    } else if (quantizer_trains_alone == 0) {
        if (verbose)
            printf("Training level-1 quantizer on %zd vectors in %zdD\n", n, d);

        Clustering clus(d, nlist, cp);
        quantizer->reset();
        if (clustering_index) {
            clus.train(n, x, *clustering_index);
            quantizer->add(nlist, clus.centroids.data());
        } else {
            clus.train(n, x, *quantizer);
        }
        quantizer->is_trained = true;
    } else if (quantizer_trains_alone == 2) {
        if (verbose) {
            printf("Training L2 quantizer on %zd vectors in %zdD%s\n",
                   n,
                   d,
                   clustering_index ? "(user provided index)" : "");
        }
        // also accept spherical centroids because in that case
        // L2 and IP are equivalent
        FAISS_THROW_IF_NOT(
                metric_type == METRIC_L2 ||
                (metric_type == METRIC_INNER_PRODUCT && cp.spherical));

        Clustering clus(d, nlist, cp);
        if (!clustering_index) {
            IndexFlatL2 assigner(d);
            clus.train(n, x, assigner);
        } else {
            clus.train(n, x, *clustering_index);
        }
        if (verbose) {
            printf("Adding centroids to quantizer\n");
        }
        if (!quantizer->is_trained) {
            if (verbose) {
                printf("But training it first on centroids table...\n");
            }
            quantizer->train(nlist, clus.centroids.data());
        }
        quantizer->add(nlist, clus.centroids.data());
    }
}

size_t Level1Quantizer::coarse_code_size() const {
    size_t nl = nlist - 1;
    size_t nbyte = 0;
    while (nl > 0) {
        nbyte++;
        nl >>= 8;
    }
    return nbyte;
}

void Level1Quantizer::encode_listno(idx_t list_no, uint8_t* code) const {
    // little endian
    size_t nl = nlist - 1;
    while (nl > 0) {
        *code++ = list_no & 0xff;
        list_no >>= 8;
        nl >>= 8;
    }
}

idx_t Level1Quantizer::decode_listno(const uint8_t* code) const {
    size_t nl = nlist - 1;
    int64_t list_no = 0;
    int nbit = 0;
    while (nl > 0) {
        list_no |= int64_t(*code++) << nbit;
        nbit += 8;
        nl >>= 8;
    }
    FAISS_THROW_IF_NOT(list_no >= 0 && list_no < nlist);
    return list_no;
}

/*****************************************
 * IndexIVF implementation
 ******************************************/

IndexIVF::IndexIVF(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t code_size,
        MetricType metric)
        : Index(d, metric),
          IndexIVFInterface(quantizer, nlist),
          invlists(new ArrayInvertedLists(nlist, code_size)),
          own_invlists(true),
          code_size(code_size) {
    FAISS_THROW_IF_NOT(d == quantizer->d);
    is_trained = quantizer->is_trained && (quantizer->ntotal == nlist);
    // Spherical by default if the metric is inner_product
    if (metric_type == METRIC_INNER_PRODUCT) {
        cp.spherical = true;
    }
}

IndexIVF::IndexIVF() = default;

void IndexIVF::add(idx_t n, const float* x) {
    add_with_ids(n, x, nullptr);
}

void IndexIVF::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    // 确保 quantizer 和 nlist 同步
    // 如果不同步，quantizer->assign 可能返回无效的索引
    if (quantizer->ntotal != nlist) {
        FAISS_THROW_IF_NOT_MSG(
                quantizer->ntotal == nlist,
                "quantizer and nlist must be synchronized before add_with_ids");
    }
   
    std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n]);
    quantizer->assign(n, x, coarse_idx.get());
    add_core(n, x, xids, coarse_idx.get(), nullptr, true);  // CHANGE: Pass true for auto_maintain
}

void IndexIVF::add_sa_codes(idx_t n, const uint8_t* codes, const idx_t* xids) {
    size_t coarse_size = coarse_code_size();
    DirectMapAdd dm_adder(direct_map, n, xids);

    for (idx_t i = 0; i < n; i++) {
        const uint8_t* code = codes + (code_size + coarse_size) * i;
        idx_t list_no = decode_listno(code);
        idx_t id = xids ? xids[i] : ntotal + i;
        size_t ofs = invlists->add_entry(list_no, id, code + coarse_size);
        dm_adder.add(i, list_no, ofs);
    }
    ntotal += n;
}

void IndexIVF::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context,
        bool auto_maintain) {
    // do some blocking to avoid excessive allocs
    idx_t bs = 65536;
    if (n > bs) {
        for (idx_t i0 = 0; i0 < n; i0 += bs) {
            idx_t i1 = std::min(n, i0 + bs);
            if (verbose) {
                printf("   IndexIVF::add_with_ids %" PRId64 ":%" PRId64 "\n",
                       i0,
                       i1);
            }
            add_core(
                    i1 - i0,
                    x + i0 * d,
                    xids ? xids + i0 : nullptr,
                    coarse_idx + i0,
                    inverted_list_context,
                    auto_maintain);
        }
        return;
    }
    FAISS_THROW_IF_NOT(coarse_idx);
    FAISS_THROW_IF_NOT(is_trained);
    direct_map.check_can_add(xids);

    size_t nadd = 0, nminus1 = 0;

    for (size_t i = 0; i < n; i++) {
        if (coarse_idx[i] < 0)
            nminus1++;
    }

    std::unique_ptr<uint8_t[]> flat_codes(new uint8_t[n * code_size]);
    encode_vectors(n, x, coarse_idx, flat_codes.get());

    DirectMapAdd dm_adder(direct_map, n, xids);

#pragma omp parallel reduction(+ : nadd)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            idx_t list_no = coarse_idx[i];
            // 添加范围检查，确保 list_no 在有效范围内
            // 这可以防止 quantizer->assign 返回超出 nlist 范围的索引
            if (list_no >= 0 && (size_t)list_no < nlist && list_no % nt == rank) {
                idx_t id = xids ? xids[i] : ntotal + i;
                size_t ofs = invlists->add_entry(
                        list_no,
                        id,
                        flat_codes.get() + i * code_size,
                        inverted_list_context);

                dm_adder.add(i, list_no, ofs);

                nadd++;
            } else if (rank == 0 && list_no == -1) {
                dm_adder.add(i, -1, 0);
            } else if (rank == 0 && list_no >= 0 && (size_t)list_no >= nlist) {
                // 如果 list_no 超出范围，记录警告并跳过
                if (verbose) {
                    printf("Warning: list_no=%zd >= nlist=%zd, skipping vector %zd\n", 
                           (size_t)list_no, nlist, i);
                }
                dm_adder.add(i, -1, 0);
            }
        }
    }

    if (verbose) {
        printf("    added %zd / %" PRId64 " vectors (%zd -1s)\n",
               nadd,
               n,
               nminus1);
    }

    ntotal += n;
    
    // 收集被插入的聚类，用于后续维护
    if (auto_maintain) {
        std::unordered_set<size_t> affected_clusters;
        for (size_t i = 0; i < n; i++) {
            idx_t list_no = coarse_idx[i];
            if (list_no >= 0 && (size_t)list_no < nlist) {
                affected_clusters.insert(list_no);
            }
        }
        
        // 自动维护被插入的聚类
        if (!affected_clusters.empty()) {
            maintain_affected_clusters(affected_clusters, true);
        }
    }
}

void IndexIVF::make_direct_map(bool b) {
    if (b) {
        direct_map.set_type(DirectMap::Array, invlists, ntotal);
    } else {
        direct_map.set_type(DirectMap::NoMap, invlists, ntotal);
    }
}

void IndexIVF::set_direct_map_type(DirectMap::Type type) {
    direct_map.set_type(type, invlists, ntotal);
}

/** It is a sad fact of software that a conceptually simple function like this
 * becomes very complex when you factor in several ways of parallelizing +
 * interrupt/error handling + collecting stats + min/max collection. The
 * codepath that is used 95% of time is the one for parallel_mode = 0 */
void IndexIVF::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);
    const IVFSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
            std::min(nlist, params ? params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    // search function for a subset of queries
    auto sub_search_func = [this, k, nprobe, params](
                                   idx_t n,
                                   const float* x,
                                   float* distances,
                                   idx_t* labels,
                                   IndexIVFStats* ivf_stats) {
        std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

        double t0 = getmillisecs();
        quantizer->search(
                n,
                x,
                nprobe,
                coarse_dis.get(),
                idx.get(),
                params ? params->quantizer_params : nullptr);

        double t1 = getmillisecs();
        invlists->prefetch_lists(idx.get(), n * nprobe);

        search_preassigned(
                n,
                x,
                k,
                idx.get(),
                coarse_dis.get(),
                distances,
                labels,
                false,
                params,
                ivf_stats);
        double t2 = getmillisecs();
        ivf_stats->quantization_time += t1 - t0;
        ivf_stats->search_time += t2 - t0;
    };

    if ((parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT) == 0) {
        int nt = std::min(omp_get_max_threads(), int(n));
        std::vector<IndexIVFStats> stats(nt);
        std::mutex exception_mutex;
        std::string exception_string;

#pragma omp parallel for if (nt > 1)
        for (idx_t slice = 0; slice < nt; slice++) {
            IndexIVFStats local_stats;
            idx_t i0 = n * slice / nt;
            idx_t i1 = n * (slice + 1) / nt;
            if (i1 > i0) {
                try {
                    sub_search_func(
                            i1 - i0,
                            x + i0 * d,
                            distances + i0 * k,
                            labels + i0 * k,
                            &stats[slice]);
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    exception_string = e.what();
                }
            }
        }

        if (!exception_string.empty()) {
            FAISS_THROW_MSG(exception_string.c_str());
        }

        // collect stats
        for (idx_t slice = 0; slice < nt; slice++) {
            indexIVF_stats.add(stats[slice]);
        }
    } else {
        // handle parallelization at level below (or don't run in parallel at
        // all)
        sub_search_func(n, x, distances, labels, &indexIVF_stats);
    }
}

void IndexIVF::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* keys,
        const float* coarse_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* ivf_stats) const {
    FAISS_THROW_IF_NOT(k > 0);

    idx_t nprobe = params ? params->nprobe : this->nprobe;
    nprobe = std::min((idx_t)nlist, nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    const idx_t unlimited_list_size = std::numeric_limits<idx_t>::max();
    idx_t max_codes = params ? params->max_codes : this->max_codes;
    IDSelector* sel = params ? params->sel : nullptr;
    const IDSelectorRange* selr = dynamic_cast<const IDSelectorRange*>(sel);
    if (selr) {
        if (selr->assume_sorted) {
            sel = nullptr; // use special IDSelectorRange processing
        } else {
            selr = nullptr; // use generic processing
        }
    }

    FAISS_THROW_IF_NOT_MSG(
            !(sel && store_pairs),
            "selector and store_pairs cannot be combined");

    FAISS_THROW_IF_NOT_MSG(
            !invlists->use_iterator || (max_codes == 0 && store_pairs == false),
            "iterable inverted lists don't support max_codes and store_pairs");

    size_t nlistv = 0, ndis = 0, nheap = 0;

    using HeapForIP = CMin<float, idx_t>;
    using HeapForL2 = CMax<float, idx_t>;

    bool interrupt = false;
    std::mutex exception_mutex;
    std::string exception_string;

    int pmode = this->parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;
    bool do_heap_init = !(this->parallel_mode & PARALLEL_MODE_NO_HEAP_INIT);

    FAISS_THROW_IF_NOT_MSG(
            max_codes == 0 || pmode == 0 || pmode == 3,
            "max_codes supported only for parallel_mode = 0 or 3");

    if (max_codes == 0) {
        max_codes = unlimited_list_size;
    }

    [[maybe_unused]] bool do_parallel = omp_get_max_threads() >= 2 &&
            (pmode == 0           ? false
                     : pmode == 3 ? n > 1
                     : pmode == 1 ? nprobe > 1
                                  : nprobe * n > 1);

    void* inverted_list_context =
            params ? params->inverted_list_context : nullptr;

#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis, nheap)
    {
        std::unique_ptr<InvertedListScanner> scanner(
                get_InvertedListScanner(store_pairs, sel));

        /*****************************************************
         * Depending on parallel_mode, there are two possible ways
         * to organize the search. Here we define local functions
         * that are in common between the two
         ******************************************************/

        // initialize + reorder a result heap

        auto init_result = [&](float* simi, idx_t* idxi) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_heapify<HeapForIP>(k, simi, idxi);
            } else {
                heap_heapify<HeapForL2>(k, simi, idxi);
            }
        };

        auto add_local_results = [&](const float* local_dis,
                                     const idx_t* local_idx,
                                     float* simi,
                                     idx_t* idxi) {
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
            } else {
                heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
            }
        };

        auto reorder_result = [&](float* simi, idx_t* idxi) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_reorder<HeapForIP>(k, simi, idxi);
            } else {
                heap_reorder<HeapForL2>(k, simi, idxi);
            }
        };

        // single list scan using the current scanner (with query
        // set porperly) and storing results in simi and idxi
        auto scan_one_list = [&](idx_t key,
                                 float coarse_dis_i,
                                 float* simi,
                                 idx_t* idxi,
                                 idx_t list_size_max) {
            if (key < 0) {
                // not enough centroids for multiprobe
                return (size_t)0;
            }
            FAISS_THROW_IF_NOT_FMT(
                    key < (idx_t)nlist,
                    "Invalid key=%" PRId64 " nlist=%zd\n",
                    key,
                    nlist);

            // don't waste time on empty lists
            if (invlists->is_empty(key, inverted_list_context)) {
                return (size_t)0;
            }

            scanner->set_list(key, coarse_dis_i);

            nlistv++;

            try {
                if (invlists->use_iterator) {
                    size_t list_size = 0;

                    std::unique_ptr<InvertedListsIterator> it(
                            invlists->get_iterator(key, inverted_list_context));

                    nheap += scanner->iterate_codes(
                            it.get(), simi, idxi, k, list_size);

                    return list_size;
                } else {
                    size_t list_size = invlists->list_size(key);
                    if (list_size > list_size_max) {
                        list_size = list_size_max;
                    }

                    InvertedLists::ScopedCodes scodes(invlists, key);
                    const uint8_t* codes = scodes.get();

                    std::unique_ptr<InvertedLists::ScopedIds> sids;
                    const idx_t* ids = nullptr;

                    if (!store_pairs) {
                        sids = std::make_unique<InvertedLists::ScopedIds>(
                                invlists, key);
                        ids = sids->get();
                    }

                    if (selr) { // IDSelectorRange
                        // restrict search to a section of the inverted list
                        size_t jmin, jmax;
                        selr->find_sorted_ids_bounds(
                                list_size, ids, &jmin, &jmax);
                        list_size = jmax - jmin;
                        if (list_size == 0) {
                            return (size_t)0;
                        }
                        codes += jmin * code_size;
                        ids += jmin;
                    }

                    nheap += scanner->scan_codes(
                            list_size, codes, ids, simi, idxi, k);

                    return list_size;
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                exception_string =
                        demangle_cpp_symbol(typeid(e).name()) + "  " + e.what();
                interrupt = true;
                return size_t(0);
            }
        };

        /****************************************************
         * Actual loops, depending on parallel_mode
         ****************************************************/

        if (pmode == 0 || pmode == 3) {
#pragma omp for
            for (idx_t i = 0; i < n; i++) {
                if (interrupt) {
                    continue;
                }

                // loop over queries
                scanner->set_query(x + i * d);
                float* simi = distances + i * k;
                idx_t* idxi = labels + i * k;

                init_result(simi, idxi);

                idx_t nscan = 0;

                // loop over probes
                for (size_t ik = 0; ik < nprobe; ik++) {
                    nscan += scan_one_list(
                            keys[i * nprobe + ik],
                            coarse_dis[i * nprobe + ik],
                            simi,
                            idxi,
                            max_codes - nscan);
                    if (nscan >= max_codes) {
                        break;
                    }
                }

                ndis += nscan;
                reorder_result(simi, idxi);

                if (InterruptCallback::is_interrupted()) {
                    interrupt = true;
                }

            } // parallel for
        } else if (pmode == 1) {
            std::vector<idx_t> local_idx(k);
            std::vector<float> local_dis(k);

            for (size_t i = 0; i < n; i++) {
                scanner->set_query(x + i * d);
                init_result(local_dis.data(), local_idx.data());

#pragma omp for schedule(dynamic)
                for (idx_t ik = 0; ik < nprobe; ik++) {
                    ndis += scan_one_list(
                            keys[i * nprobe + ik],
                            coarse_dis[i * nprobe + ik],
                            local_dis.data(),
                            local_idx.data(),
                            unlimited_list_size);

                    // can't do the test on max_codes
                }
                // merge thread-local results

                float* simi = distances + i * k;
                idx_t* idxi = labels + i * k;
#pragma omp single
                init_result(simi, idxi);

#pragma omp barrier
#pragma omp critical
                {
                    add_local_results(
                            local_dis.data(), local_idx.data(), simi, idxi);
                }
#pragma omp barrier
#pragma omp single
                reorder_result(simi, idxi);
            }
        } else if (pmode == 2) {
            std::vector<idx_t> local_idx(k);
            std::vector<float> local_dis(k);

#pragma omp single
            for (int64_t i = 0; i < n; i++) {
                init_result(distances + i * k, labels + i * k);
            }

#pragma omp for schedule(dynamic)
            for (int64_t ij = 0; ij < n * nprobe; ij++) {
                size_t i = ij / nprobe;

                scanner->set_query(x + i * d);
                init_result(local_dis.data(), local_idx.data());
                ndis += scan_one_list(
                        keys[ij],
                        coarse_dis[ij],
                        local_dis.data(),
                        local_idx.data(),
                        unlimited_list_size);
#pragma omp critical
                {
                    add_local_results(
                            local_dis.data(),
                            local_idx.data(),
                            distances + i * k,
                            labels + i * k);
                }
            }
#pragma omp single
            for (int64_t i = 0; i < n; i++) {
                reorder_result(distances + i * k, labels + i * k);
            }
        } else {
            FAISS_THROW_FMT("parallel_mode %d not supported\n", pmode);
        }
    } // parallel section

    if (interrupt) {
        if (!exception_string.empty()) {
            FAISS_THROW_FMT(
                    "search interrupted with: %s", exception_string.c_str());
        } else {
            FAISS_THROW_MSG("computation interrupted");
        }
    }

    if (ivf_stats == nullptr) {
        ivf_stats = &indexIVF_stats;
    }
    ivf_stats->nq += n;
    ivf_stats->nlist += nlistv;
    ivf_stats->ndis += ndis;
    ivf_stats->nheap_updates += nheap;
}
 


void IndexIVF::search_stats(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in,
        QueryLatencyStats* per_query_stats) const {
    FAISS_THROW_IF_NOT(k > 0);
    // MODIFICATION #1: (推荐) 在开始时清零统计数组
    if (per_query_stats) {
        memset(per_query_stats, 0, sizeof(QueryLatencyStats) * n);
    }
    const IVFSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
            std::min(nlist, params ? params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    // search function for a subset of queries
    auto sub_search_func = [this, k, nprobe, params](
                                   idx_t n,
                                   const float* x,
                                   float* distances,
                                   idx_t* labels,
                                   IndexIVFStats* ivf_stats,
                                   QueryLatencyStats* per_query_stats_subset) {
        std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

        // MODIFICATION #4: 添加粗量化阶段的计时逻辑
        if ((parallel_mode == 0) && per_query_stats_subset) {
            Timer quantizer_timer;
            quantizer->search(
                    n, x, nprobe, coarse_dis.get(), idx.get(),
                    params ? params->quantizer_params : nullptr);
            double total_quant_us = quantizer_timer.elapsed_us();
            
            double amortized_quant_us = (n > 0) ? (total_quant_us / n) : 0.0;
            for (idx_t i = 0; i < n; ++i) {
                per_query_stats_subset[i].quantization_us = amortized_quant_us;
            }
        } else {
            // 原始的、无计时的调用
            quantizer->search(
                    n, x, nprobe, coarse_dis.get(), idx.get(),
                    params ? params->quantizer_params : nullptr);
        }

        // double t0 = getmillisecs();
        // quantizer->search(
        //         n,
        //         x,
        //         nprobe,
        //         coarse_dis.get(),
        //         idx.get(),
        //         params ? params->quantizer_params : nullptr);

        // double t1 = getmillisecs();
        invlists->prefetch_lists(idx.get(), n * nprobe);

        search_preassigned_stats(
                n,
                x,
                k,
                idx.get(),
                coarse_dis.get(),
                distances,
                labels,
                false,
                params,
                ivf_stats,
                per_query_stats_subset);
        // double t2 = getmillisecs();
        // ivf_stats->quantization_time += t1 - t0;
        // ivf_stats->search_time += t2 - t0;
    };

    if ((parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT) == 0) {
        int nt = std::min(omp_get_max_threads(), int(n));
        std::vector<IndexIVFStats> stats(nt);
        std::mutex exception_mutex;
        std::string exception_string;

#pragma omp parallel for if (nt > 1)
        for (idx_t slice = 0; slice < nt; slice++) {
            IndexIVFStats local_stats;
            idx_t i0 = n * slice / nt;
            idx_t i1 = n * (slice + 1) / nt;
            if (i1 > i0) {
                try {
                    // MODIFICATION #5: 调用 sub_search_func 时传入             per_query_stats 指针
                    // 注意指针需要偏移
                    sub_search_func(
                            i1 - i0,
                            x + i0 * d,
                            distances + i0 * k,
                            labels + i0 * k,
                            &stats[slice],
                            per_query_stats ? (per_query_stats + i0) : nullptr);
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    exception_string = e.what();
                }
            }
        }

        if (!exception_string.empty()) {
            FAISS_THROW_MSG(exception_string.c_str());
        }

        // collect stats
        for (idx_t slice = 0; slice < nt; slice++) {
            indexIVF_stats.add(stats[slice]);
        }
    } else {
        // MODIFICATION #5 (else 分支): 调用 sub_search_func 时传入 per_query_stats 指针
        // handle parallelization at level below (or don't run in parallel at
        // all)
        sub_search_func(n, x, distances, labels, &indexIVF_stats, per_query_stats);
    }

    // MODIFICATION #6: 计算最终的总耗时
    if ((parallel_mode == 0) && per_query_stats) {
        for (idx_t i = 0; i < n; ++i) {
            per_query_stats[i].total_us = per_query_stats[i].quantization_us +
                                          per_query_stats[i].list_scan_us;
        }
    }
}


void IndexIVF::search_preassigned_stats(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* keys,
        const float* coarse_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* ivf_stats,
        QueryLatencyStats* per_query_stats) const {
    FAISS_THROW_IF_NOT(k > 0);

    idx_t nprobe = params ? params->nprobe : this->nprobe;
    nprobe = std::min((idx_t)nlist, nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    const idx_t unlimited_list_size = std::numeric_limits<idx_t>::max();
    idx_t max_codes = params ? params->max_codes : this->max_codes;
    IDSelector* sel = params ? params->sel : nullptr;
    const IDSelectorRange* selr = dynamic_cast<const IDSelectorRange*>(sel);
    if (selr) {
        if (selr->assume_sorted) {
            sel = nullptr; // use special IDSelectorRange processing
        } else {
            selr = nullptr; // use generic processing
        }
    }

    FAISS_THROW_IF_NOT_MSG(
            !(sel && store_pairs),
            "selector and store_pairs cannot be combined");

    FAISS_THROW_IF_NOT_MSG(
            !invlists->use_iterator || (max_codes == 0 && store_pairs == false),
            "iterable inverted lists don't support max_codes and store_pairs");

    size_t nlistv = 0, ndis = 0, nheap = 0;

    using HeapForIP = CMin<float, idx_t>;
    using HeapForL2 = CMax<float, idx_t>;

    bool interrupt = false;
    std::mutex exception_mutex;
    std::string exception_string;

    int pmode = this->parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;
    bool do_heap_init = !(this->parallel_mode & PARALLEL_MODE_NO_HEAP_INIT);

    FAISS_THROW_IF_NOT_MSG(
            max_codes == 0 || pmode == 0 || pmode == 3,
            "max_codes supported only for parallel_mode = 0 or 3");

    if (max_codes == 0) {
        max_codes = unlimited_list_size;
    }

    [[maybe_unused]] bool do_parallel = omp_get_max_threads() >= 2 &&
            (pmode == 0           ? false
                     : pmode == 3 ? n > 1
                     : pmode == 1 ? nprobe > 1
                                  : nprobe * n > 1);

    void* inverted_list_context =
            params ? params->inverted_list_context : nullptr;

#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis, nheap)
    {
        std::unique_ptr<InvertedListScanner> scanner(
                get_InvertedListScanner(store_pairs, sel));

        /*****************************************************
         * Depending on parallel_mode, there are two possible ways
         * to organize the search. Here we define local functions
         * that are in common between the two
         ******************************************************/

        // initialize + reorder a result heap

        auto init_result = [&](float* simi, idx_t* idxi) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_heapify<HeapForIP>(k, simi, idxi);
            } else {
                heap_heapify<HeapForL2>(k, simi, idxi);
            }
        };

        auto add_local_results = [&](const float* local_dis,
                                     const idx_t* local_idx,
                                     float* simi,
                                     idx_t* idxi) {
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
            } else {
                heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
            }
        };

        auto reorder_result = [&](float* simi, idx_t* idxi) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_reorder<HeapForIP>(k, simi, idxi);
            } else {
                heap_reorder<HeapForL2>(k, simi, idxi);
            }
        };

        // single list scan using the current scanner (with query
        // set porperly) and storing results in simi and idxi
        auto scan_one_list = [&](idx_t key,
                                 float coarse_dis_i,
                                 float* simi,
                                 idx_t* idxi,
                                 idx_t list_size_max) {
            if (key < 0) {
                // not enough centroids for multiprobe
                return (size_t)0;
            }
            FAISS_THROW_IF_NOT_FMT(
                    key < (idx_t)nlist,
                    "Invalid key=%" PRId64 " nlist=%zd\n",
                    key,
                    nlist);

            // don't waste time on empty lists
            if (invlists->is_empty(key, inverted_list_context)) {
                return (size_t)0;
            }

            scanner->set_list(key, coarse_dis_i);

            nlistv++;

            try {
                if (invlists->use_iterator) {
                    size_t list_size = 0;

                    std::unique_ptr<InvertedListsIterator> it(
                            invlists->get_iterator(key, inverted_list_context));

                    nheap += scanner->iterate_codes(
                            it.get(), simi, idxi, k, list_size);

                    return list_size;
                } else {
                    size_t list_size = invlists->list_size(key);
                    if (list_size > list_size_max) {
                        list_size = list_size_max;
                    }

                    InvertedLists::ScopedCodes scodes(invlists, key);
                    const uint8_t* codes = scodes.get();

                    std::unique_ptr<InvertedLists::ScopedIds> sids;
                    const idx_t* ids = nullptr;

                    if (!store_pairs) {
                        sids = std::make_unique<InvertedLists::ScopedIds>(
                                invlists, key);
                        ids = sids->get();
                    }

                    if (selr) { // IDSelectorRange
                        // restrict search to a section of the inverted list
                        size_t jmin, jmax;
                        selr->find_sorted_ids_bounds(
                                list_size, ids, &jmin, &jmax);
                        list_size = jmax - jmin;
                        if (list_size == 0) {
                            return (size_t)0;
                        }
                        codes += jmin * code_size;
                        ids += jmin;
                    }

                    nheap += scanner->scan_codes(
                            list_size, codes, ids, simi, idxi, k);

                    return list_size;
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                exception_string =
                        demangle_cpp_symbol(typeid(e).name()) + "  " + e.what();
                interrupt = true;
                return size_t(0);
            }
        };

        /****************************************************
         * Actual loops, depending on parallel_mode
         ****************************************************/

        if (pmode == 0 || pmode == 3) {
#pragma omp for
            for (idx_t i = 0; i < n; i++) {
                // START OF MODIFICATION #1: 声明计时器
                // 只有在需要统计时才创建计时器对象，避免不必要的开销
                std::unique_ptr<Timer> list_scan_timer = nullptr;
                if (pmode == 0 && per_query_stats != nullptr) {
                    list_scan_timer = std::make_unique<Timer>();
                }
                // END OF MODIFICATION #1
                if (interrupt) {
                    continue;
                }

                // loop over queries
                scanner->set_query(x + i * d);
                float* simi = distances + i * k;
                idx_t* idxi = labels + i * k;

                init_result(simi, idxi);

                idx_t nscan = 0;

                // loop over probes
                for (size_t ik = 0; ik < nprobe; ik++) {
                    nscan += scan_one_list(
                            keys[i * nprobe + ik],
                            coarse_dis[i * nprobe + ik],
                            simi,
                            idxi,
                            max_codes - nscan);
                    if (nscan >= max_codes) {
                        break;
                    }
                }

                ndis += nscan;
                reorder_result(simi, idxi);

                if (InterruptCallback::is_interrupted()) {
                    interrupt = true;
                }

                // START OF MODIFICATION #2: 记录耗时
                // 同样，只有在 pmode=0 且需要统计时才执行
                if (list_scan_timer) { // 如果计时器被创建了
                    per_query_stats[i].list_scan_us = list_scan_timer->elapsed_us();
                }
                // END OF MODIFICATION #2

            } // parallel for
        } else if (pmode == 1) {
            std::vector<idx_t> local_idx(k);
            std::vector<float> local_dis(k);

            for (size_t i = 0; i < n; i++) {
                scanner->set_query(x + i * d);
                init_result(local_dis.data(), local_idx.data());

#pragma omp for schedule(dynamic)
                for (idx_t ik = 0; ik < nprobe; ik++) {
                    ndis += scan_one_list(
                            keys[i * nprobe + ik],
                            coarse_dis[i * nprobe + ik],
                            local_dis.data(),
                            local_idx.data(),
                            unlimited_list_size);

                    // can't do the test on max_codes
                }
                // merge thread-local results

                float* simi = distances + i * k;
                idx_t* idxi = labels + i * k;
#pragma omp single
                init_result(simi, idxi);

#pragma omp barrier
#pragma omp critical
                {
                    add_local_results(
                            local_dis.data(), local_idx.data(), simi, idxi);
                }
#pragma omp barrier
#pragma omp single
                reorder_result(simi, idxi);
            }
        } else if (pmode == 2) {
            std::vector<idx_t> local_idx(k);
            std::vector<float> local_dis(k);

#pragma omp single
            for (int64_t i = 0; i < n; i++) {
                init_result(distances + i * k, labels + i * k);
            }

#pragma omp for schedule(dynamic)
            for (int64_t ij = 0; ij < n * nprobe; ij++) {
                size_t i = ij / nprobe;

                scanner->set_query(x + i * d);
                init_result(local_dis.data(), local_idx.data());
                ndis += scan_one_list(
                        keys[ij],
                        coarse_dis[ij],
                        local_dis.data(),
                        local_idx.data(),
                        unlimited_list_size);
#pragma omp critical
                {
                    add_local_results(
                            local_dis.data(),
                            local_idx.data(),
                            distances + i * k,
                            labels + i * k);
                }
            }
#pragma omp single
            for (int64_t i = 0; i < n; i++) {
                reorder_result(distances + i * k, labels + i * k);
            }
        } else {
            FAISS_THROW_FMT("parallel_mode %d not supported\n", pmode);
        }
    } // parallel section

    if (interrupt) {
        if (!exception_string.empty()) {
            FAISS_THROW_FMT(
                    "search interrupted with: %s", exception_string.c_str());
        } else {
            FAISS_THROW_MSG("computation interrupted");
        }
    }

    if (ivf_stats == nullptr) {
        ivf_stats = &indexIVF_stats;
    }
    ivf_stats->nq += n;
    ivf_stats->nlist += nlistv;
    ivf_stats->ndis += ndis;
    ivf_stats->nheap_updates += nheap;
}

void IndexIVF::range_search(
        idx_t nx,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params_in) const {
    const IVFSearchParameters* params = nullptr;
    const SearchParameters* quantizer_params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
        quantizer_params = params->quantizer_params;
    }
    const size_t nprobe =
            std::min(nlist, params ? params->nprobe : this->nprobe);
    std::unique_ptr<idx_t[]> keys(new idx_t[nx * nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[nx * nprobe]);

    double t0 = getmillisecs();
    quantizer->search(
            nx, x, nprobe, coarse_dis.get(), keys.get(), quantizer_params);
    indexIVF_stats.quantization_time += getmillisecs() - t0;

    t0 = getmillisecs();
    invlists->prefetch_lists(keys.get(), nx * nprobe);

    range_search_preassigned(
            nx,
            x,
            radius,
            keys.get(),
            coarse_dis.get(),
            result,
            false,
            params,
            &indexIVF_stats);

    indexIVF_stats.search_time += getmillisecs() - t0;
}

void IndexIVF::range_search_preassigned(
        idx_t nx,
        const float* x,
        float radius,
        const idx_t* keys,
        const float* coarse_dis,
        RangeSearchResult* result,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    idx_t nprobe = params ? params->nprobe : this->nprobe;
    nprobe = std::min((idx_t)nlist, nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    idx_t max_codes = params ? params->max_codes : this->max_codes;
    IDSelector* sel = params ? params->sel : nullptr;

    FAISS_THROW_IF_NOT_MSG(
            !invlists->use_iterator || (max_codes == 0 && store_pairs == false),
            "iterable inverted lists don't support max_codes and store_pairs");

    size_t nlistv = 0, ndis = 0;

    bool interrupt = false;
    std::mutex exception_mutex;
    std::string exception_string;

    std::vector<RangeSearchPartialResult*> all_pres(omp_get_max_threads());

    int pmode = this->parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;
    // don't start parallel section if single query
    [[maybe_unused]] bool do_parallel = omp_get_max_threads() >= 2 &&
            (pmode == 3           ? false
                     : pmode == 0 ? nx > 1
                     : pmode == 1 ? nprobe > 1
                                  : nprobe * nx > 1);

    void* inverted_list_context =
            params ? params->inverted_list_context : nullptr;

#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis)
    {
        RangeSearchPartialResult pres(result);
        std::unique_ptr<InvertedListScanner> scanner(
                get_InvertedListScanner(store_pairs, sel));
        FAISS_THROW_IF_NOT(scanner.get());
        all_pres[omp_get_thread_num()] = &pres;

        // prepare the list scanning function

        auto scan_list_func = [&](size_t i, size_t ik, RangeQueryResult& qres) {
            idx_t key = keys[i * nprobe + ik]; /* select the list  */
            if (key < 0)
                return;
            FAISS_THROW_IF_NOT_FMT(
                    key < (idx_t)nlist,
                    "Invalid key=%" PRId64 " at ik=%zd nlist=%zd\n",
                    key,
                    ik,
                    nlist);

            if (invlists->is_empty(key, inverted_list_context)) {
                return;
            }

            try {
                size_t list_size = 0;
                scanner->set_list(key, coarse_dis[i * nprobe + ik]);
                if (invlists->use_iterator) {
                    std::unique_ptr<InvertedListsIterator> it(
                            invlists->get_iterator(key, inverted_list_context));

                    scanner->iterate_codes_range(
                            it.get(), radius, qres, list_size);
                } else {
                    InvertedLists::ScopedCodes scodes(invlists, key);
                    InvertedLists::ScopedIds ids(invlists, key);
                    list_size = invlists->list_size(key);

                    scanner->scan_codes_range(
                            list_size, scodes.get(), ids.get(), radius, qres);
                }
                nlistv++;
                ndis += list_size;
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                exception_string =
                        demangle_cpp_symbol(typeid(e).name()) + "  " + e.what();
                interrupt = true;
            }
        };

        if (parallel_mode == 0) {
#pragma omp for
            for (idx_t i = 0; i < nx; i++) {
                scanner->set_query(x + i * d);

                RangeQueryResult& qres = pres.new_result(i);

                for (size_t ik = 0; ik < nprobe; ik++) {
                    scan_list_func(i, ik, qres);
                }
            }

        } else if (parallel_mode == 1) {
            for (size_t i = 0; i < nx; i++) {
                scanner->set_query(x + i * d);

                RangeQueryResult& qres = pres.new_result(i);

#pragma omp for schedule(dynamic)
                for (int64_t ik = 0; ik < nprobe; ik++) {
                    scan_list_func(i, ik, qres);
                }
            }
        } else if (parallel_mode == 2) {
            RangeQueryResult* qres = nullptr;

#pragma omp for schedule(dynamic)
            for (idx_t iik = 0; iik < nx * (idx_t)nprobe; iik++) {
                idx_t i = iik / (idx_t)nprobe;
                idx_t ik = iik % (idx_t)nprobe;
                if (qres == nullptr || qres->qno != i) {
                    qres = &pres.new_result(i);
                    scanner->set_query(x + i * d);
                }
                scan_list_func(i, ik, *qres);
            }
        } else {
            FAISS_THROW_FMT("parallel_mode %d not supported\n", parallel_mode);
        }
        if (parallel_mode == 0) {
            pres.finalize();
        } else {
#pragma omp barrier
#pragma omp single
            RangeSearchPartialResult::merge(all_pres, false);
#pragma omp barrier
        }
    }

    if (interrupt) {
        if (!exception_string.empty()) {
            FAISS_THROW_FMT(
                    "search interrupted with: %s", exception_string.c_str());
        } else {
            FAISS_THROW_MSG("computation interrupted");
        }
    }

    if (stats == nullptr) {
        stats = &indexIVF_stats;
    }
    stats->nq += nx;
    stats->nlist += nlistv;
    stats->ndis += ndis;
}

InvertedListScanner* IndexIVF::get_InvertedListScanner(
        bool /*store_pairs*/,
        const IDSelector* /* sel */) const {
    FAISS_THROW_MSG("get_InvertedListScanner not implemented");
}

void IndexIVF::reconstruct(idx_t key, float* recons) const {
    idx_t lo = direct_map.get(key);
    reconstruct_from_offset(lo_listno(lo), lo_offset(lo), recons);
}

void IndexIVF::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
    FAISS_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));

    for (idx_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = invlists->list_size(list_no);
        ScopedIds idlist(invlists, list_no);

        for (idx_t offset = 0; offset < list_size; offset++) {
            idx_t id = idlist[offset];
            if (!(id >= i0 && id < i0 + ni)) {
                continue;
            }

            float* reconstructed = recons + (id - i0) * d;
            reconstruct_from_offset(list_no, offset, reconstructed);
        }
    }
}

bool IndexIVF::check_ids_sorted() const {
    size_t nflip = 0;

    for (size_t i = 0; i < nlist; i++) {
        size_t list_size = invlists->list_size(i);
        InvertedLists::ScopedIds ids(invlists, i);
        for (size_t j = 0; j + 1 < list_size; j++) {
            if (ids[j + 1] < ids[j]) {
                nflip++;
            }
        }
    }
    return nflip == 0;
}

/* standalone codec interface */
size_t IndexIVF::sa_code_size() const {
    size_t coarse_size = coarse_code_size();
    return code_size + coarse_size;
}

void IndexIVF::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    FAISS_THROW_IF_NOT(is_trained);
    std::unique_ptr<int64_t[]> idx(new int64_t[n]);
    quantizer->assign(n, x, idx.get());
    encode_vectors(n, x, idx.get(), bytes, true);
}

void IndexIVF::search_and_reconstruct(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        float* recons,
        const SearchParameters* params_in) const {
    const IVFSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
            std::min(nlist, params ? params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

    quantizer->search(n, x, nprobe, coarse_dis.get(), idx.get());

    invlists->prefetch_lists(idx.get(), n * nprobe);

    // search_preassigned() with `store_pairs` enabled to obtain the list_no
    // and offset into `codes` for reconstruction
    search_preassigned(
            n,
            x,
            k,
            idx.get(),
            coarse_dis.get(),
            distances,
            labels,
            true /* store_pairs */,
            params);
#pragma omp parallel for if (n * k > 1000)
    for (idx_t ij = 0; ij < n * k; ij++) {
        idx_t key = labels[ij];
        float* reconstructed = recons + ij * d;
        if (key < 0) {
            // Fill with NaNs
            memset(reconstructed, -1, sizeof(*reconstructed) * d);
        } else {
            int list_no = lo_listno(key);
            int offset = lo_offset(key);

            // Update label to the actual id
            labels[ij] = invlists->get_single_id(list_no, offset);

            reconstruct_from_offset(list_no, offset, reconstructed);
        }
    }
}

void IndexIVF::search_and_return_codes(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        uint8_t* codes,
        bool include_listno,
        const SearchParameters* params_in) const {
    const IVFSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
            std::min(nlist, params ? params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

    quantizer->search(n, x, nprobe, coarse_dis.get(), idx.get());

    invlists->prefetch_lists(idx.get(), n * nprobe);

    // search_preassigned() with `store_pairs` enabled to obtain the list_no
    // and offset into `codes` for reconstruction
    search_preassigned(
            n,
            x,
            k,
            idx.get(),
            coarse_dis.get(),
            distances,
            labels,
            true /* store_pairs */,
            params);

    size_t code_size_1 = code_size;
    if (include_listno) {
        code_size_1 += coarse_code_size();
    }

#pragma omp parallel for if (n * k > 1000)
    for (idx_t ij = 0; ij < n * k; ij++) {
        idx_t key = labels[ij];
        uint8_t* code1 = codes + ij * code_size_1;

        if (key < 0) {
            // Fill with 0xff
            memset(code1, -1, code_size_1);
        } else {
            int list_no = lo_listno(key);
            int offset = lo_offset(key);
            const uint8_t* cc = invlists->get_single_code(list_no, offset);

            labels[ij] = invlists->get_single_id(list_no, offset);

            if (include_listno) {
                encode_listno(list_no, code1);
                code1 += code_size_1 - code_size;
            }
            memcpy(code1, cc, code_size);
        }
    }
}

void IndexIVF::reconstruct_from_offset(
        int64_t /*list_no*/,
        int64_t /*offset*/,
        float* /*recons*/) const {
    FAISS_THROW_MSG("reconstruct_from_offset not implemented");
}

void IndexIVF::reset() {
    direct_map.clear();
    invlists->reset();
    ntotal = 0;
}

size_t IndexIVF::remove_ids(const IDSelector& sel) {
    // 收集被删除向量所在的聚类（在删除前）
    std::unordered_set<size_t> affected_clusters;
    
    if (direct_map.type == DirectMap::Hashtable) {
        // 对于Hashtable类型，我们可以从direct_map中获取聚类信息
        const IDSelectorArray* sela = dynamic_cast<const IDSelectorArray*>(&sel);
        if (sela) {
            for (idx_t i = 0; i < sela->n; i++) {
                idx_t id = sela->ids[i];
                auto res = direct_map.hashtable.find(id);
                if (res != direct_map.hashtable.end()) {
                    size_t list_no = lo_listno(res->second);
                    affected_clusters.insert(list_no);
                }
            }
        } else {
            // 对于其他类型的selector，需要扫描所有聚类
            // 这是一个fallback方案，可能效率较低
            for (size_t list_no = 0; list_no < nlist; list_no++) {
                size_t list_size = invlists->list_size(list_no);
                if (list_size > 0) {
                    InvertedLists::ScopedIds ids(invlists, list_no);
                    for (size_t offset = 0; offset < list_size; offset++) {
                        if (sel.is_member(ids[offset])) {
                            affected_clusters.insert(list_no);
                            break;  // 找到至少一个被删除的向量即可
                        }
                    }
                }
            }
        }
    } else if (direct_map.type == DirectMap::Array) {
        // 对于Array类型，需要扫描所有聚类
        for (size_t list_no = 0; list_no < nlist; list_no++) {
            size_t list_size = invlists->list_size(list_no);
            if (list_size > 0) {
                InvertedLists::ScopedIds ids(invlists, list_no);
                for (size_t offset = 0; offset < list_size; offset++) {
                    idx_t id = ids[offset];
                    if (id >= 0 && id < direct_map.array.size()) {
                        idx_t lo = direct_map.array[id];
                        if (lo >= 0 && sel.is_member(id)) {
                            affected_clusters.insert(list_no);
                            break;  // 找到至少一个被删除的向量即可
                        }
                    }
                }
            }
        }
    } else {
        // 对于NoMap类型，需要扫描所有聚类
        for (size_t list_no = 0; list_no < nlist; list_no++) {
            size_t list_size = invlists->list_size(list_no);
            if (list_size > 0) {
                InvertedLists::ScopedIds ids(invlists, list_no);
                for (size_t offset = 0; offset < list_size; offset++) {
                    if (sel.is_member(ids[offset])) {
                        affected_clusters.insert(list_no);
                        break;  // 找到至少一个被删除的向量即可
                    }
                }
            }
        }
    }
    
    // 执行删除操作
    size_t nremove = direct_map.remove_ids(sel, invlists);
    ntotal -= nremove;
    
    // 自动维护被删除向量所在的聚类
    if (!affected_clusters.empty() && nremove > 0) {
        maintain_affected_clusters(affected_clusters, true);
    }
    
    return nremove;
}

void IndexIVF::update_vectors(int n, const idx_t* new_ids, const float* x) {
    if (direct_map.type == DirectMap::Hashtable) {
        // 对于 Hashtable 类型，update_vectors 的实现是：删除旧向量，然后添加新向量
        // 但是，删除操作可能会触发聚类维护，导致 nlist 改变
        // 为了安全起见，我们需要：
        // 1. 先删除向量（这会触发维护）
        // 2. 确保 quantizer 和 nlist 同步
        // 3. 然后添加新向量
        
        IDSelectorArray sel(n, new_ids);
        
        // 记录删除前的 nlist，以便检测变化
        size_t nlist_before = nlist;
        size_t quantizer_ntotal_before = quantizer->ntotal;
        
        // 执行删除操作（这会触发 maintain_affected_clusters）
        size_t nremove = remove_ids(sel);
        FAISS_THROW_IF_NOT_MSG(
                nremove == n, "did not find all entries to remove");
        
        // 检查 nlist 和 quantizer 是否同步
        // 如果不同步，说明删除操作触发了聚类维护（分裂或合并）
        if (quantizer->ntotal != nlist) {
            if (verbose) {
                printf("Warning: After remove_ids, quantizer->ntotal (%zd) != nlist (%zd)\n",
                       quantizer->ntotal, nlist);
                printf("  nlist changed: %zd -> %zd\n", nlist_before, nlist);
                printf("  quantizer->ntotal changed: %zd -> %zd\n", quantizer_ntotal_before, quantizer->ntotal);
            }
            
            // 如果 nlist 增加了（分裂），quantizer 应该已经更新了
            // 但如果不同步，我们需要修复
            if (nlist > quantizer->ntotal) {
                // nlist 增加了但 quantizer 没有同步，添加零向量质心
                std::vector<float> new_centroids((nlist - quantizer->ntotal) * d, 0.0f);
                quantizer->add(nlist - quantizer->ntotal, new_centroids.data());
            } else if (nlist < quantizer->ntotal) {
                // nlist 减少了（不应该发生），重置 quantizer
                std::vector<float> centroids(nlist * d);
                for (size_t i = 0; i < nlist && i < quantizer->ntotal; i++) {
                    quantizer->reconstruct(i, centroids.data() + i * d);
                }
                quantizer->reset();
                quantizer->add(nlist, centroids.data());
            }
        }
        
        // 验证同步
        FAISS_THROW_IF_NOT_MSG(
                quantizer->ntotal == nlist,
                "quantizer and nlist must be synchronized before add_with_ids");
        
        // 现在可以安全地添加新向量
        add_with_ids(n, x, new_ids);
        return;
    }

    FAISS_THROW_IF_NOT(direct_map.type == DirectMap::Array);
    // here it is more tricky because we don't want to introduce holes
    // in continuous range of ids

    FAISS_THROW_IF_NOT(is_trained);
    std::vector<idx_t> assign(n);
    quantizer->assign(n, x, assign.data());

    std::vector<uint8_t> flat_codes(n * code_size);
    encode_vectors(n, x, assign.data(), flat_codes.data());

    direct_map.update_codes(
            invlists, n, new_ids, assign.data(), flat_codes.data());
}

void IndexIVF::train(idx_t n, const float* x) {
    if (verbose) {
        printf("Training level-1 quantizer\n");
    }

    train_q1(n, x, verbose, metric_type);

    if (verbose) {
        printf("Training IVF residual\n");
    }

    // optional subsampling
    idx_t max_nt = train_encoder_num_vectors();
    if (max_nt <= 0) {
        max_nt = (size_t)1 << 35;
    }

    TransformedVectors tv(
            x, fvecs_maybe_subsample(d, (size_t*)&n, max_nt, x, verbose));

    if (by_residual) {
        std::vector<idx_t> assign(n);
        quantizer->assign(n, tv.x, assign.data());

        std::vector<float> residuals(n * d);
        quantizer->compute_residual_n(n, tv.x, residuals.data(), assign.data());

        train_encoder(n, residuals.data(), assign.data());
    } else {
        train_encoder(n, tv.x, nullptr);
    }

    is_trained = true;
}

idx_t IndexIVF::train_encoder_num_vectors() const {
    return 0;
}

void IndexIVF::train_encoder(
        idx_t /*n*/,
        const float* /*x*/,
        const idx_t* assign) {
    // does nothing by default
    if (verbose) {
        printf("IndexIVF: no residual training\n");
    }
}

bool check_compatible_for_merge_expensive_check = true;

void IndexIVF::check_compatible_for_merge(const Index& otherIndex) const {
    // minimal sanity checks
    const IndexIVF* other = dynamic_cast<const IndexIVF*>(&otherIndex);
    FAISS_THROW_IF_NOT(other);
    FAISS_THROW_IF_NOT(other->d == d);
    FAISS_THROW_IF_NOT(other->nlist == nlist);
    FAISS_THROW_IF_NOT(quantizer->ntotal == other->quantizer->ntotal);
    FAISS_THROW_IF_NOT(other->code_size == code_size);
    FAISS_THROW_IF_NOT_MSG(
            typeid(*this) == typeid(*other),
            "can only merge indexes of the same type");
    FAISS_THROW_IF_NOT_MSG(
            this->direct_map.no() && other->direct_map.no(),
            "merge direct_map not implemented");

    if (check_compatible_for_merge_expensive_check) {
        std::vector<float> v(d), v2(d);
        for (size_t i = 0; i < nlist; i++) {
            quantizer->reconstruct(i, v.data());
            other->quantizer->reconstruct(i, v2.data());
            FAISS_THROW_IF_NOT_MSG(
                    v == v2, "coarse quantizers should be the same");
        }
    }
}

void IndexIVF::merge_from(Index& otherIndex, idx_t add_id) {
    check_compatible_for_merge(otherIndex);
    IndexIVF* other = static_cast<IndexIVF*>(&otherIndex);
    invlists->merge_from(other->invlists, add_id);

    ntotal += other->ntotal;
    other->ntotal = 0;
}

CodePacker* IndexIVF::get_CodePacker() const {
    return new CodePackerFlat(code_size);
}

void IndexIVF::replace_invlists(InvertedLists* il, bool own) {
    if (own_invlists) {
        delete invlists;
        invlists = nullptr;
    }
    // FAISS_THROW_IF_NOT (ntotal == 0);
    if (il) {
        FAISS_THROW_IF_NOT(il->nlist == nlist);
        FAISS_THROW_IF_NOT(
                il->code_size == code_size ||
                il->code_size == InvertedLists::INVALID_CODE_SIZE);
    }
    invlists = il;
    own_invlists = own;
}

void IndexIVF::copy_subset_to(
        IndexIVF& other,
        InvertedLists::subset_type_t subset_type,
        idx_t a1,
        idx_t a2) const {
    other.ntotal +=
            invlists->copy_subset_to(*other.invlists, subset_type, a1, a2);
}

IndexIVF::~IndexIVF() {
    if (own_invlists) {
        delete invlists;
    }
}

/*************************************************************************
 * IndexIVFStats
 *************************************************************************/

void IndexIVFStats::reset() {
    memset((void*)this, 0, sizeof(*this));
}

void IndexIVFStats::add(const IndexIVFStats& other) {
    nq += other.nq;
    nlist += other.nlist;
    ndis += other.ndis;
    nheap_updates += other.nheap_updates;
    quantization_time += other.quantization_time;
    search_time += other.search_time;
}

IndexIVFStats indexIVF_stats;

/*************************************************************************
 * InvertedListScanner
 *************************************************************************/

size_t InvertedListScanner::scan_codes(
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        float* simi,
        idx_t* idxi,
        size_t k) const {
    size_t nup = 0;

    if (!keep_max) {
        for (size_t j = 0; j < list_size; j++) {
            float dis = distance_to_code(codes);
            if (dis < simi[0]) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                maxheap_replace_top(k, simi, idxi, dis, id);
                nup++;
            }
            codes += code_size;
        }
    } else {
        for (size_t j = 0; j < list_size; j++) {
            float dis = distance_to_code(codes);
            if (dis > simi[0]) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                minheap_replace_top(k, simi, idxi, dis, id);
                nup++;
            }
            codes += code_size;
        }
    }
    return nup;
}

size_t InvertedListScanner::iterate_codes(
        InvertedListsIterator* it,
        float* simi,
        idx_t* idxi,
        size_t k,
        size_t& list_size) const {
    size_t nup = 0;
    list_size = 0;

    if (!keep_max) {
        for (; it->is_available(); it->next()) {
            auto id_and_codes = it->get_id_and_codes();
            float dis = distance_to_code(id_and_codes.second);
            if (dis < simi[0]) {
                maxheap_replace_top(k, simi, idxi, dis, id_and_codes.first);
                nup++;
            }
            list_size++;
        }
    } else {
        for (; it->is_available(); it->next()) {
            auto id_and_codes = it->get_id_and_codes();
            float dis = distance_to_code(id_and_codes.second);
            if (dis > simi[0]) {
                minheap_replace_top(k, simi, idxi, dis, id_and_codes.first);
                nup++;
            }
            list_size++;
        }
    }
    return nup;
}

void InvertedListScanner::scan_codes_range(
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        float radius,
        RangeQueryResult& res) const {
    for (size_t j = 0; j < list_size; j++) {
        float dis = distance_to_code(codes);
        bool keep = !keep_max
                ? dis < radius
                : dis > radius; // TODO templatize to remove this test
        if (keep) {
            int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
            res.add(dis, id);
        }
        codes += code_size;
    }
}

void InvertedListScanner::iterate_codes_range(
        InvertedListsIterator* it,
        float radius,
        RangeQueryResult& res,
        size_t& list_size) const {
    list_size = 0;
    for (; it->is_available(); it->next()) {
        auto id_and_codes = it->get_id_and_codes();
        float dis = distance_to_code(id_and_codes.second);
        bool keep = !keep_max
                ? dis < radius
                : dis > radius; // TODO templatize to remove this test
        if (keep) {
            res.add(dis, id_and_codes.first);
        }
        list_size++;
    }
}

/*************************************************************************
 * Dynamic Cluster Maintenance Implementation
 *************************************************************************/

size_t IndexIVF::recompute_centroids(bool update_quantizer) {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(invlists != nullptr);
    
    if (verbose) {
        printf("IndexIVF::recompute_centroids: recomputing centroids for %zd clusters\n", nlist);
    }
    
    // 先保存旧的 quantizer 质心（用于空聚类）
    std::vector<float> old_centroids;
    if (update_quantizer && quantizer->ntotal > 0) {
        old_centroids.resize(quantizer->ntotal * d);
        for (size_t i = 0; i < quantizer->ntotal; i++) {
            quantizer->reconstruct(i, old_centroids.data() + i * d);
        }
    }
    
    std::vector<float> new_centroids(nlist * d, 0.0f);
    size_t n_recomputed = 0;
    
    // 遍历所有倒排列表，重新计算质心
    for (size_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = invlists->list_size(list_no);
        if (list_size == 0) {
            // 空列表：使用旧的质心（如果存在），否则使用零向量
            if (list_no < old_centroids.size() / d) {
                memcpy(new_centroids.data() + list_no * d,
                       old_centroids.data() + list_no * d,
                       d * sizeof(float));
            }
            continue;
        }
        
        // 获取该列表中的所有向量
        std::vector<float> list_vectors(list_size * d);
        InvertedLists::ScopedCodes codes(invlists, list_no);
        
        for (size_t offset = 0; offset < list_size; offset++) {
            float* vec = list_vectors.data() + offset * d;
            reconstruct_from_offset(list_no, offset, vec);
            
            // 累加向量
            for (size_t j = 0; j < d; j++) {
                new_centroids[list_no * d + j] += vec[j];
            }
        }
        
        n_recomputed++;
        
        // 计算平均值（质心）
        float inv_size = 1.0f / list_size;
        for (size_t j = 0; j < d; j++) {
            new_centroids[list_no * d + j] *= inv_size;
        }
    }
    
    // 更新quantizer
    if (update_quantizer) {
        // 重置quantizer并添加新质心
        quantizer->reset();
        quantizer->add(nlist, new_centroids.data());
        
        if (verbose) {
            printf("IndexIVF::recompute_centroids: updated %zd centroids in quantizer\n", n_recomputed);
        }
    }
    
    return n_recomputed;
}

size_t IndexIVF::split_large_clusters(
        size_t size_threshold,
        int split_factor,
        size_t min_split_size) {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(invlists != nullptr);
    FAISS_THROW_IF_NOT(split_factor >= 2);
    FAISS_THROW_IF_NOT(direct_map.type == DirectMap::Hashtable || direct_map.type == DirectMap::NoMap);
    
    if (verbose) {
        printf("IndexIVF::split_large_clusters: checking clusters with threshold=%zd, split_factor=%d\n",
               size_threshold, split_factor);
    }
    
    std::vector<size_t> clusters_to_split;
    
    // 找出需要分裂的聚类
    for (size_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = invlists->list_size(list_no);
        if (list_size > size_threshold && list_size >= min_split_size) {
            clusters_to_split.push_back(list_no);
        }
    }
    
    if (clusters_to_split.empty()) {
        if (verbose) {
            printf("IndexIVF::split_large_clusters: no clusters need splitting\n");
        }
        return 0;
    }
    
    if (verbose) {
        printf("IndexIVF::split_large_clusters: found %zd clusters to split\n", clusters_to_split.size());
    }
    
    size_t n_split = 0;
    std::vector<std::vector<float>> new_centroids_to_add;  // 每个元素包含 split_factor - 1 个新质心
    std::vector<idx_t> all_ids_to_reassign;
    std::vector<float> all_vectors_to_reassign;
    std::unordered_set<idx_t> ids_to_remove_set;  // 用于快速查找需要移除的ID
    
    // 第一阶段：收集所有需要分裂的聚类的信息
    for (size_t list_no : clusters_to_split) {
        size_t list_size = invlists->list_size(list_no);
        
        if (verbose) {
            printf("IndexIVF::split_large_clusters: analyzing cluster %zd (size=%zd)\n", list_no, list_size);
        }
        
        // 1. 获取该聚类的所有向量和ID
        std::vector<float> cluster_vectors(list_size * d);
        std::vector<idx_t> cluster_ids(list_size);
        
        InvertedLists::ScopedCodes codes(invlists, list_no);
        InvertedLists::ScopedIds ids(invlists, list_no);
        
        for (size_t offset = 0; offset < list_size; offset++) {
            cluster_ids[offset] = ids[offset];
            reconstruct_from_offset(list_no, offset, cluster_vectors.data() + offset * d);
        }
        
        // 2. 使用k-means将聚类分裂成split_factor个子聚类
        Clustering clus(d, split_factor, cp);
        clus.verbose = false;
        clus.niter = 10;  // 减少迭代次数以提高速度
        
        IndexFlatL2 assigner(d);
        clus.train(list_size, cluster_vectors.data(), assigner);
        
        // 3. 保存新质心：只保存 split_factor - 1 个新质心（第一个子聚类的质心保留在原位置，不替换）
        // 注意：为了简化实现，我们不替换原来的质心，只添加新质心
        // 这样 quantizer->ntotal 会增加 n_split * (split_factor - 1)，与 nlist 的增加一致
        std::vector<float> new_centroids((split_factor - 1) * d);
        for (int i = 1; i < split_factor; i++) {
            memcpy(new_centroids.data() + (i - 1) * d,
                   clus.centroids.data() + i * d,
                   d * sizeof(float));
        }
        new_centroids_to_add.push_back(new_centroids);
        
        // 4. 收集需要重新分配的向量和ID
        all_ids_to_reassign.insert(
            all_ids_to_reassign.end(), cluster_ids.begin(), cluster_ids.end());
        all_vectors_to_reassign.insert(
            all_vectors_to_reassign.end(),
            cluster_vectors.begin(),
            cluster_vectors.end());
        
        // 添加到移除集合
        for (idx_t id : cluster_ids) {
            ids_to_remove_set.insert(id);
        }
        
        n_split++;
    }
    
    // 第二阶段：计算新的nlist大小并扩展invlists
    size_t new_nlist = nlist + n_split * (split_factor - 1);
    
    if (new_nlist > nlist) {
        // 创建新的ArrayInvertedLists
        ArrayInvertedLists* new_invlists = new ArrayInvertedLists(new_nlist, code_size);
        
        // 迁移现有数据（排除需要重新分配的向量）
        for (size_t i = 0; i < nlist; i++) {
            size_t old_size = invlists->list_size(i);
            if (old_size > 0) {
                InvertedLists::ScopedCodes old_codes(invlists, i);
                InvertedLists::ScopedIds old_ids(invlists, i);
                
                // 过滤掉需要移除的ID
                std::vector<idx_t> filtered_ids;
                std::vector<uint8_t> filtered_codes;
                
                for (size_t j = 0; j < old_size; j++) {
                    if (ids_to_remove_set.find(old_ids[j]) == ids_to_remove_set.end()) {
                        filtered_ids.push_back(old_ids[j]);
                        filtered_codes.insert(
                            filtered_codes.end(),
                            old_codes.get() + j * code_size,
                            old_codes.get() + (j + 1) * code_size);
                    }
                }
                
                if (!filtered_ids.empty()) {
                    new_invlists->add_entries(i, filtered_ids.size(), filtered_ids.data(), filtered_codes.data());
                }
            }
        }
        
        // 替换invlists（需要在调用前更新nlist，因为replace_invlists会检查il->nlist == nlist）
        nlist = new_nlist;
        replace_invlists(new_invlists, true);
    } else {
        // 如果不需要扩展，直接从旧列表中移除需要重新分配的向量
        // 注意：这里直接调用 direct_map.remove_ids 而不是 remove_ids，
        // 以避免触发 maintain_affected_clusters 造成递归调用
        if (!all_ids_to_reassign.empty()) {
            IDSelectorArray sel(all_ids_to_reassign.size(), all_ids_to_reassign.data());
            size_t nremove = direct_map.remove_ids(sel, invlists);
            ntotal -= nremove;
        }
    }
    
    // 第三阶段：添加新质心到quantizer
    // 注意：我们不替换原来的质心，只添加新质心
    // 这样 quantizer->ntotal 会增加 n_split * (split_factor - 1)，与 nlist 的增加一致
    for (const auto& centroids : new_centroids_to_add) {
        quantizer->add(split_factor - 1, centroids.data());
    }
    
    // 验证 quantizer->ntotal == nlist
    if (quantizer->ntotal != nlist) {
        FAISS_THROW_IF_NOT_MSG(
            quantizer->ntotal == nlist,
            "quantizer->ntotal must equal nlist after splitting");
    }
    
    // 第四阶段：重新添加向量（会自动分配到新的聚类）
    if (!all_ids_to_reassign.empty()) {
        add_with_ids(
            all_ids_to_reassign.size(),
            all_vectors_to_reassign.data(),
            all_ids_to_reassign.data());
    }
    
    if (verbose) {
        printf("IndexIVF::split_large_clusters: split %zd clusters, new nlist=%zd\n", n_split, nlist);
    }
    
    return n_split;
}

size_t IndexIVF::merge_small_clusters(
        size_t size_threshold,
        size_t min_merge_size) {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(invlists != nullptr);
    FAISS_THROW_IF_NOT(direct_map.type == DirectMap::Hashtable || direct_map.type == DirectMap::NoMap);
    
    if (verbose) {
        printf("IndexIVF::merge_small_clusters: checking clusters with threshold=%zd\n", size_threshold);
    }
    
    std::vector<size_t> small_clusters;
    
    // 找出需要合并的小聚类
    for (size_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = invlists->list_size(list_no);
        if (list_size < size_threshold && list_size >= min_merge_size) {
            small_clusters.push_back(list_no);
        }
    }
    
    if (small_clusters.empty()) {
        if (verbose) {
            printf("IndexIVF::merge_small_clusters: no clusters need merging\n");
        }
        return 0;
    }
    
    if (verbose) {
        printf("IndexIVF::merge_small_clusters: found %zd clusters to merge\n", small_clusters.size());
    }
    
    size_t n_merged = 0;
    
    // 获取所有质心用于查找最近邻
    // 注意：需要确保 quantizer 和 nlist 同步
    if (quantizer->ntotal < nlist) {
        FAISS_THROW_MSG("quantizer->ntotal < nlist in merge_small_clusters");
    }
    std::vector<float> centroids(nlist * d);
    for (size_t i = 0; i < nlist && i < quantizer->ntotal; i++) {
        quantizer->reconstruct(i, centroids.data() + i * d);
    }
    // 如果 nlist > quantizer->ntotal，剩余的位置使用零向量
    if (nlist > quantizer->ntotal) {
        memset(centroids.data() + quantizer->ntotal * d, 0, (nlist - quantizer->ntotal) * d * sizeof(float));
    }
    
    // 对每个小聚类，找到最近的聚类并合并
    for (size_t small_list_no : small_clusters) {
        size_t list_size = invlists->list_size(small_list_no);
        if (list_size == 0) {
            continue;  // 跳过空列表
        }
        
        // 找到最近的聚类（排除自己）
        float* small_centroid = centroids.data() + small_list_no * d;
        float min_dist = std::numeric_limits<float>::max();
        size_t nearest_list_no = nlist;  // 无效值
        
        for (size_t j = 0; j < nlist; j++) {
            if (j == small_list_no || invlists->list_size(j) == 0) {
                continue;
            }
            
            float* other_centroid = centroids.data() + j * d;
            float dist = metric_type == METRIC_L2
                ? fvec_L2sqr(small_centroid, other_centroid, d)
                : fvec_inner_product(small_centroid, other_centroid, d);
            
            if (dist < min_dist) {
                min_dist = dist;
                nearest_list_no = j;
            }
        }
        
        if (nearest_list_no >= nlist) {
            continue;  // 没有找到合适的合并目标
        }
        
        if (verbose) {
            printf("IndexIVF::merge_small_clusters: merging cluster %zd (size=%zd) into %zd\n",
                   small_list_no, list_size, nearest_list_no);
        }
        
        // 获取小聚类中的所有向量和ID
        std::vector<float> cluster_vectors(list_size * d);
        std::vector<idx_t> cluster_ids(list_size);
        
        InvertedLists::ScopedCodes codes(invlists, small_list_no);
        InvertedLists::ScopedIds ids(invlists, small_list_no);
        
        for (size_t offset = 0; offset < list_size; offset++) {
            cluster_ids[offset] = ids[offset];
            reconstruct_from_offset(small_list_no, offset, cluster_vectors.data() + offset * d);
        }
        
        // 从小聚类中移除向量
        // 注意：这里直接调用 direct_map.remove_ids 而不是 remove_ids，
        // 以避免触发 maintain_affected_clusters 造成递归调用
        IDSelectorArray sel(list_size, cluster_ids.data());
        size_t nremove = direct_map.remove_ids(sel, invlists);
        ntotal -= nremove;
        
        // 将向量重新添加到最近的聚类
        // 注意：这里需要强制分配到nearest_list_no，禁用auto_maintain以避免递归调用
        std::vector<idx_t> forced_assign(list_size, nearest_list_no);
        add_core(list_size, cluster_vectors.data(), cluster_ids.data(), forced_assign.data(), nullptr, false);
        
        n_merged++;
    }
    
    if (verbose) {
        printf("IndexIVF::merge_small_clusters: merged %zd clusters\n", n_merged);
    }
    
    return n_merged;
}

IndexIVF::ClusterMaintenanceStats IndexIVF::maintain_clusters(
        size_t split_threshold,
        size_t merge_threshold,
        int split_factor,
        bool update_quantizer) {
    ClusterMaintenanceStats stats;
    
    if (verbose) {
        printf("IndexIVF::maintain_clusters: starting maintenance (split_threshold=%zd, merge_threshold=%zd)\n",
               split_threshold, merge_threshold);
    }
    
    // 1. 重新计算质心
    stats.centroids_recomputed = recompute_centroids(update_quantizer);
    
    // 2. 分裂大聚类
    stats.clusters_split = split_large_clusters(split_threshold, split_factor);
    
    // 3. 合并小聚类
    stats.clusters_merged = merge_small_clusters(merge_threshold);
    
    // 4. 记录最终的nlist
    stats.new_nlist = nlist;
    
    if (verbose) {
        printf("IndexIVF::maintain_clusters: completed\n");
        printf("  - Centroids recomputed: %zd\n", stats.centroids_recomputed);
        printf("  - Clusters split: %zd\n", stats.clusters_split);
        printf("  - Clusters merged: %zd\n", stats.clusters_merged);
        printf("  - New nlist: %zd\n", stats.new_nlist);
    }
    
    return stats;
}

/*************************************************************************
 * Per-Cluster Maintenance Implementation
 *************************************************************************/

bool IndexIVF::recompute_cluster_centroid(size_t list_no, bool update_quantizer) {
    if (quantizer->ntotal != nlist) {
        if (verbose) {
            printf("Warning: recompute_cluster_centroid: skipping due to mismatch (%zd != %zd)\n",
                   quantizer->ntotal, nlist);
        }
        return false;
    }
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(invlists != nullptr);
    FAISS_THROW_IF_NOT(list_no < nlist);
    
    size_t list_size = invlists->list_size(list_no);
    if (list_size == 0) {
        return false;  // 空列表，无需重新计算
    }
    
    if (verbose) {
        printf("IndexIVF::recompute_cluster_centroid: recomputing centroid for cluster %zd (size=%zd)\n", 
               list_no, list_size);
    }
    
    // 计算新质心
    std::vector<float> new_centroid(d, 0.0f);
    
    InvertedLists::ScopedCodes codes(invlists, list_no);
    std::vector<float> vec(d);
    for (size_t offset = 0; offset < list_size; offset++) {
        reconstruct_from_offset(list_no, offset, vec.data());
        
        // 累加向量
        for (size_t j = 0; j < d; j++) {
            new_centroid[j] += vec[j];
        }
    }
    
    // 计算平均值（质心）
    float inv_size = 1.0f / list_size;
    for (size_t j = 0; j < d; j++) {
        new_centroid[j] *= inv_size;
    }
    
    // 更新quantizer
    if (update_quantizer) {
        // 检查 list_no 是否在 quantizer 的有效范围内
        if (list_no >= quantizer->ntotal) {
            if (verbose) {
                printf("Warning: recompute_cluster_centroid: list_no %zd >= quantizer->ntotal %zd, skipping update\n",
                       list_no, quantizer->ntotal);
            }
            return false;
        }
        
        // 更新quantizer中对应位置的质心
        // 注意：quantizer可能是IndexFlat，我们需要直接更新
        IndexFlat* flat_quantizer = dynamic_cast<IndexFlat*>(quantizer);
        if (flat_quantizer) {
            float* quantizer_data = flat_quantizer->get_xb();
            memcpy(quantizer_data + list_no * d, new_centroid.data(), d * sizeof(float));
        } else {
            // 对于其他类型的quantizer，可能需要重新训练或使用其他方法
            // 这里我们尝试重建：先删除旧质心，再添加新质心
            // 但更简单的方法是直接替换（如果quantizer支持）
            FAISS_THROW_MSG("quantizer type not supported for per-cluster centroid update");
        }
        
        if (verbose) {
            printf("IndexIVF::recompute_cluster_centroid: updated centroid in quantizer for cluster %zd\n", list_no);
        }
    }
    
    return true;
}

IndexIVF::ClusterMaintenanceStats IndexIVF::maintain_cluster(
        size_t list_no,
        size_t split_threshold,
        size_t merge_threshold,
        int split_factor,
        bool update_quantizer) {
    if (quantizer->ntotal != nlist) {
        if (verbose) {
            printf("Warning: maintain_cluster: skipping cluster %zd due to mismatch (%zd != %zd)\n",
                    list_no, quantizer->ntotal, nlist);
        }
        return ClusterMaintenanceStats{}; // Skip
    }
    ClusterMaintenanceStats stats;
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(invlists != nullptr);
    FAISS_THROW_IF_NOT(list_no < nlist);
    
    // 确保 quantizer 和 nlist 同步
    // 如果不同步，可能会导致访问越界
    if (quantizer->ntotal != nlist) {
        if (verbose) {
            printf("Warning: maintain_cluster: quantizer->ntotal (%zd) != nlist (%zd), skipping maintenance\n",
                   quantizer->ntotal, nlist);
        }
        return stats;  // 跳过维护，避免崩溃
    }
    
    size_t list_size = invlists->list_size(list_no);
    
    if (verbose) {
        printf("IndexIVF::maintain_cluster: maintaining cluster %zd (size=%zd, split_threshold=%zd, merge_threshold=%zd)\n",
               list_no, list_size, split_threshold, merge_threshold);
    }
    
    // 如果聚类为空，无需维护
    if (list_size == 0) {
        return stats;
    }
    
    // 1. 检查是否需要分裂
    if (list_size > split_threshold && list_size >= 100) {  // min_split_size = 100
        if (verbose) {
            printf("IndexIVF::maintain_cluster: splitting cluster %zd (size=%zd > threshold=%zd)\n",
                   list_no, list_size, split_threshold);
        }
        
        // 获取该聚类的所有向量和ID
        std::vector<float> cluster_vectors(list_size * d);
        std::vector<idx_t> cluster_ids(list_size);
        
        InvertedLists::ScopedCodes codes(invlists, list_no);
        InvertedLists::ScopedIds ids(invlists, list_no);
        
        for (size_t offset = 0; offset < list_size; offset++) {
            cluster_ids[offset] = ids[offset];
            reconstruct_from_offset(list_no, offset, cluster_vectors.data() + offset * d);
        }
        
        // 使用k-means将聚类分裂成split_factor个子聚类
        Clustering clus(d, split_factor, cp);
        clus.verbose = false;
        clus.niter = 10;
        
        IndexFlatL2 assigner(d);
        clus.train(list_size, cluster_vectors.data(), assigner);
        
        // 保存新质心（split_factor - 1个新质心）
        std::vector<float> new_centroids((split_factor - 1) * d);
        for (int i = 1; i < split_factor; i++) {
            memcpy(new_centroids.data() + (i - 1) * d,
                   clus.centroids.data() + i * d,
                   d * sizeof(float));
        }
        
        // 扩展invlists
        size_t new_nlist = nlist + (split_factor - 1);
        ArrayInvertedLists* new_invlists = new ArrayInvertedLists(new_nlist, code_size);
        
        // 迁移现有数据（排除需要重新分配的向量）
        std::unordered_set<idx_t> ids_to_remove_set(cluster_ids.begin(), cluster_ids.end());
        for (size_t i = 0; i < nlist; i++) {
            size_t old_size = invlists->list_size(i);
            if (old_size > 0) {
                InvertedLists::ScopedCodes old_codes(invlists, i);
                InvertedLists::ScopedIds old_ids(invlists, i);
                
                std::vector<idx_t> filtered_ids;
                std::vector<uint8_t> filtered_codes;
                
                for (size_t j = 0; j < old_size; j++) {
                    if (ids_to_remove_set.find(old_ids[j]) == ids_to_remove_set.end()) {
                        filtered_ids.push_back(old_ids[j]);
                        filtered_codes.insert(
                            filtered_codes.end(),
                            old_codes.get() + j * code_size,
                            old_codes.get() + (j + 1) * code_size);
                    }
                }
                
                if (!filtered_ids.empty()) {
                    new_invlists->add_entries(i, filtered_ids.size(), filtered_ids.data(), filtered_codes.data());
                }
            }
        }
        
        // 替换invlists
        nlist = new_nlist;
        replace_invlists(new_invlists, true);
        
        // 添加新质心到quantizer
        quantizer->add(split_factor - 1, new_centroids.data());
        
        // 验证 quantizer 和 nlist 同步
        FAISS_THROW_IF_NOT_MSG(
                quantizer->ntotal == nlist,
                "quantizer and nlist must be synchronized after splitting");
        
        // 更新原聚类的质心（使用第一个子聚类的质心）
        // 检查 list_no 是否在 quantizer 的有效范围内
        if (list_no < quantizer->ntotal) {
            IndexFlat* flat_quantizer = dynamic_cast<IndexFlat*>(quantizer);
            if (flat_quantizer) {
                float* quantizer_data = flat_quantizer->get_xb();
                memcpy(quantizer_data + list_no * d, clus.centroids.data(), d * sizeof(float));
            }
        } else if (verbose) {
            printf("Warning: maintain_cluster: list_no %zd >= quantizer->ntotal %zd, skipping centroid update\n",
                   list_no, quantizer->ntotal);
        }
        
        // 使用k-means分配结果，将向量分配到对应的子聚类
        // 训练完成后，assigner中已经包含了最终的质心（clus.centroids）
        // 理论上可以直接使用，但为了确保使用最新的clus.centroids并保持代码清晰，
        // 我们重置并重新添加
        assigner.reset();
        std::vector<float> distances(list_size);
        std::vector<idx_t> sub_assign(list_size);
        assigner.add(split_factor, clus.centroids.data());
        assigner.search(list_size, cluster_vectors.data(), 1, distances.data(), sub_assign.data());
        
        // 调整分配：将分配结果映射到新的聚类索引
        // 第一个子聚类使用原list_no，其他使用新创建的聚类
        // 新聚类的索引是：list_no + 1, list_no + 2, ..., list_no + (split_factor - 1)
        std::vector<idx_t> final_assign(list_size);
        size_t old_nlist = nlist - (split_factor - 1);  // 分裂前的nlist
        for (size_t i = 0; i < list_size; i++) {
            if (sub_assign[i] == 0) {
                final_assign[i] = list_no;  // 第一个子聚类使用原位置
            } else {
                // 其他子聚类使用新位置
                // 新聚类的索引从 old_nlist 开始
                final_assign[i] = old_nlist + (sub_assign[i] - 1);
            }
        }
        
        // 直接使用add_core添加向量
        // 注意：禁用auto_maintain以避免递归调用maintain_affected_clusters
        add_core(list_size, cluster_vectors.data(), cluster_ids.data(), final_assign.data(), nullptr, false);
        
        stats.clusters_split = 1;
        stats.new_nlist = nlist;
        
        if (verbose) {
            printf("IndexIVF::maintain_cluster: split cluster %zd into %d sub-clusters\n", list_no, split_factor);
        }
        
        return stats;  // 分裂后不需要再检查合并
    }
    
    // 2. 检查是否需要合并
    if (list_size < merge_threshold && list_size >= 1) {  // min_merge_size = 1
        if (verbose) {
            printf("IndexIVF::maintain_cluster: merging cluster %zd (size=%zd < threshold=%zd)\n",
                   list_no, list_size, merge_threshold);
        }
        
        // 获取所有质心用于查找最近邻
        // 注意：需要确保 quantizer 和 nlist 同步
        if (quantizer->ntotal < nlist) {
            FAISS_THROW_MSG("quantizer->ntotal < nlist in maintain_cluster merge operation");
        }
        std::vector<float> centroids(nlist * d);
        for (size_t i = 0; i < nlist && i < quantizer->ntotal; i++) {
            quantizer->reconstruct(i, centroids.data() + i * d);
        }
        // 如果 nlist > quantizer->ntotal，剩余的位置使用零向量
        if (nlist > quantizer->ntotal) {
            memset(centroids.data() + quantizer->ntotal * d, 0, (nlist - quantizer->ntotal) * d * sizeof(float));
        }
        
        // 找到最近的聚类（排除自己）
        float* small_centroid = centroids.data() + list_no * d;
        float min_dist = std::numeric_limits<float>::max();
        size_t nearest_list_no = nlist;
        
        for (size_t j = 0; j < nlist; j++) {
            if (j == list_no || invlists->list_size(j) == 0) {
                continue;
            }
            
            float* other_centroid = centroids.data() + j * d;
            float dist = metric_type == METRIC_L2
                ? fvec_L2sqr(small_centroid, other_centroid, d)
                : fvec_inner_product(small_centroid, other_centroid, d);
            
            if (dist < min_dist) {
                min_dist = dist;
                nearest_list_no = j;
            }
        }
        
        if (nearest_list_no < nlist) {
            // 再次检查 list_no 和 nearest_list_no 的有效性
            if (list_no >= nlist || nearest_list_no >= nlist) {
                if (verbose) {
                    printf("Warning: maintain_cluster: invalid list_no (%zd) or nearest_list_no (%zd) >= nlist (%zd), skipping merge\n",
                           list_no, nearest_list_no, nlist);
                }
                // 如果无效，只重新计算质心
                recompute_cluster_centroid(list_no, update_quantizer);
                stats.centroids_recomputed = 1;
                return stats;
            }
            
            // 再次检查 list_size，可能在检查后发生了变化
            size_t current_list_size = invlists->list_size(list_no);
            if (current_list_size == 0 || current_list_size != list_size) {
                if (verbose) {
                    printf("Warning: maintain_cluster: list_size changed from %zd to %zd, skipping merge\n",
                           list_size, current_list_size);
                }
                // 如果大小变化了，只重新计算质心
                recompute_cluster_centroid(list_no, update_quantizer);
                stats.centroids_recomputed = 1;
                return stats;
            }
            
            // 获取小聚类中的所有向量和ID
            std::vector<float> cluster_vectors(list_size * d);
            std::vector<idx_t> cluster_ids(list_size);
            
            InvertedLists::ScopedCodes codes(invlists, list_no);
            InvertedLists::ScopedIds ids(invlists, list_no);
            
            // 安全地获取向量和ID
            // 注意：list_size 已经在前面检查过了，所以这里应该是安全的
            for (size_t offset = 0; offset < list_size; offset++) {
                cluster_ids[offset] = ids[offset];
                reconstruct_from_offset(list_no, offset, cluster_vectors.data() + offset * d);
            }
            
            // 从小聚类中移除向量
            // 注意：这里直接调用 direct_map.remove_ids 而不是 remove_ids，
            // 以避免触发 maintain_affected_clusters 造成递归调用
            IDSelectorArray sel(list_size, cluster_ids.data());
            size_t nremove = direct_map.remove_ids(sel, invlists);
            ntotal -= nremove;
            
            // 将向量重新添加到最近的聚类
            // 确保 nearest_list_no 仍然有效
            if (nearest_list_no < nlist) {
                std::vector<idx_t> forced_assign(list_size, nearest_list_no);
                add_core(list_size, cluster_vectors.data(), cluster_ids.data(), forced_assign.data(), nullptr, false);
                
                // 重新计算被合并到的聚类的质心
                recompute_cluster_centroid(nearest_list_no, update_quantizer);
                
                stats.clusters_merged = 1;
                
                if (verbose) {
                    printf("IndexIVF::maintain_cluster: merged cluster %zd into cluster %zd\n", list_no, nearest_list_no);
                }
            } else {
                if (verbose) {
                    printf("Warning: maintain_cluster: nearest_list_no %zd >= nlist %zd after remove, skipping add\n",
                           nearest_list_no, nlist);
                }
            }
            
            return stats;  // 合并后不需要再重新计算原聚类
        }
    }
    
    // 3. 如果聚类大小适中，只重新计算质心
    recompute_cluster_centroid(list_no, update_quantizer);
    stats.centroids_recomputed = 1;
    
    if (verbose) {
        printf("IndexIVF::maintain_cluster: recomputed centroid for cluster %zd\n", list_no);
    }
    
    return stats;
}

void IndexIVF::maintain_affected_clusters(
    const std::unordered_set<size_t>& affected_clusters,
    bool update_quantizer) {
if (affected_clusters.empty()) {
    return;
}

// CHANGE: Add sync check to prevent errors
if (quantizer->ntotal != nlist) {
    if (verbose) {
        printf("Warning: maintain_affected_clusters: skipping due to quantizer/ntotal mismatch (%zd != %zd)\n",
               quantizer->ntotal, nlist);
    }
    return;
}

// 暂时禁用自动维护以避免递归问题
// TODO: 重新设计维护逻辑以支持增量维护
// CHANGE: Remove the disabling printf and return to enable maintenance
// if (verbose) {
//     printf("IndexIVF::maintain_affected_clusters: auto-maintenance temporarily disabled\n");
// }
// return;

// 计算平均聚类大小，用于确定阈值
size_t total_vectors = ntotal;
size_t avg_cluster_size = (nlist > 0 && total_vectors > 0) ? (total_vectors / nlist) : 0;

// 设置阈值
size_t split_threshold = avg_cluster_size * 3;
if (split_threshold < 100) split_threshold = 100; // 最小阈值

size_t merge_threshold = avg_cluster_size / 3;
if (merge_threshold < 10) merge_threshold = 10; // 最小阈值

if (verbose) {
    printf("IndexIVF::maintain_affected_clusters: maintaining %zd clusters (split_threshold=%zd, merge_threshold=%zd)\n",
           affected_clusters.size(), split_threshold, merge_threshold);
}

// 对每个受影响的聚类进行维护
// 注意：在维护过程中，nlist 可能会改变（分裂会增加nlist）
// 所以我们需要在每次迭代时重新检查 list_no 的有效性
std::vector<size_t> clusters_to_maintain(affected_clusters.begin(), affected_clusters.end());
for (size_t list_no : clusters_to_maintain) {
    // 在每次迭代时检查 list_no 是否仍然有效
    // 如果 nlist 在维护过程中增加了，某些 list_no 可能仍然有效
    if (list_no < nlist) {
        // 在调用 maintain_cluster 之前，再次检查 quantizer 和 nlist 是否同步
        // 如果不同步，跳过这个聚类
        if (quantizer->ntotal == nlist) {
            maintain_cluster(list_no, split_threshold, merge_threshold, 2, update_quantizer);
        } else {
            if (verbose) {
                printf("Warning: maintain_affected_clusters: skipping cluster %zd due to quantizer/nlist mismatch (%zd != %zd)\n",
                       list_no, quantizer->ntotal, nlist);
            }
        }
    }
}
}

} // namespace faiss
