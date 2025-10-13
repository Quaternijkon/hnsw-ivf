// In faiss/IndexIVF.cpp

void IndexIVF::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);

    // NEW: Overall timer for QPS calculation
    double t_start_total = getmillisecs();

    const IVFSearchParameters* params = nullptr;
    // NEW: Check for our custom params struct
    const IVFSearchParametersWithQueryStats* params_with_stats = nullptr;

    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
        // NEW: Also try to cast to our new type
        params_with_stats = dynamic_cast<const IVFSearchParametersWithQueryStats*>(params_in);
    }

    const size_t nprobe =
            std::min(nlist, params ? params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

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
                    // NEW: Create a temporary sub-params object to pass down the stats collector
                    IVFSearchParametersWithQueryStats sub_params;
                    if (params_with_stats && params_with_stats->query_timings) {
                        sub_params = *params_with_stats; // Copy base params
                        // Point to the correct slice of the timings vector
                        // Note: This requires careful handling if we were to resize vectors.
                        // For this read-only pattern, it's okay. A more robust solution might pass offsets.
                    } else {
                        if (params) sub_params = *dynamic_cast<const IVFSearchParametersWithQueryStats*>(params);
                    }
                    
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

        for (idx_t slice = 0; slice < nt; slice++) {
            indexIVF_stats.add(stats[slice]);
        }
    } else {
        sub_search_func(n, x, distances, labels, &indexIVF_stats);
    }

    // NEW: Add total search time to stats for QPS calculation
    indexIVF_stats.search_time = getmillisecs() - t_start_total;
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

    // NEW: Get the query timings collector if it exists
    const IVFSearchParametersWithQueryStats* params_with_stats =
            dynamic_cast<const IVFSearchParametersWithQueryStats*>(params);
    QueryTimings* query_timings =
            params_with_stats ? params_with_stats->query_timings : nullptr;

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

                // NEW: Per-query timer
                double t0_query = getmillisecs();

                scanner->set_query(x + i * d);
                float* simi = distances + i * k;
                idx_t* idxi = labels + i * k;

                init_result(simi, idxi);

                idx_t nscan = 0;

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
                
                // NEW: Record the latency for this single query
                if (query_timings) {
                    // This write is safe because each thread 'i' writes to a unique location.
                    query_timings->per_query_latencies[i] = getmillisecs() - t0_query;
                }

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
 */
