/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_INDEX_IVF_HNSW_H
#define FAISS_INDEX_IVF_HNSW_H

#include <stdint.h>

#include <faiss/IndexIVF.h>
#include <faiss/IndexHNSW.h>

namespace faiss {

/** Inverted file with HNSW quantizer and stored vectors.
 * 
 * This index combines IVFFlat with HNSW quantizer for faster coarse quantization.
 * The inverted file uses HNSW to pre-select the vectors to be searched,
 * and the code array contains the raw float entries (like IVFFlat).
 */
struct IndexIVFHNSW : IndexIVF {
    /// HNSW parameters
    size_t M = 32;              ///< number of neighbors in HNSW graph
    size_t efConstruction = 40; ///< construction time search depth
    size_t efSearch = 16;       ///< search time search depth

    /// Index file management
    std::string index_file_path;     ///< path to index file on disk
    size_t add_chunk_size = 100000;  ///< chunk size for adding vectors
    bool auto_save = false;          ///< automatically save after train/add
    bool use_mmap = false;           ///< use mmap when loading from disk

    /** Constructor
     * 
     * @param d          dimensionality of vectors
     * @param nlist      number of inverted lists (clusters)
     * @param M          HNSW M parameter
     * @param metric     distance metric (L2 or inner product)
     */
    IndexIVFHNSW(
            size_t d,
            size_t nlist,
            size_t M = 32,
            MetricType metric = METRIC_L2);

    /** Constructor with external quantizer
     * 
     * @param quantizer  external HNSW quantizer (will be owned by this object)
     * @param d          dimensionality of vectors
     * @param nlist      number of inverted lists
     * @param metric     distance metric
     */
    IndexIVFHNSW(
            Index* quantizer,
            size_t d,
            size_t nlist,
            MetricType metric = METRIC_L2);

    /// Set HNSW parameters
    void set_hnsw_parameters(size_t M, size_t efConstruction, size_t efSearch);

    /// Get the HNSW quantizer
    IndexHNSW* get_hnsw_quantizer();
    const IndexHNSW* get_hnsw_quantizer() const;

    /** Set index file path and enable auto save/load
     * @param path       path to index file
     * @param auto_save  automatically save after train/add
     */
    void set_index_file(const std::string& path, bool auto_save = true);

    /** Load index from disk
     * If index file exists, load it; otherwise return false
     * @param use_mmap   use memory mapping for large indices
     * @return true if loaded successfully
     */
    bool load_from_disk(bool use_mmap = true);

    /** Save index to disk
     * Save to the path specified by set_index_file()
     * @return true if saved successfully
     */
    bool save_to_disk();

    /** Static method to load index from file
     * @param filename   path to index file
     * @param use_mmap   use memory mapping
     * @return pointer to loaded index (caller owns it)
     */
    static IndexIVFHNSW* load(const std::string& filename, bool use_mmap = true);

    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel) const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    /// Train the index (trains HNSW quantizer and IVF structure)
    void train(idx_t n, const float* x) override;

    IndexIVFHNSW();

    ~IndexIVFHNSW() override;
};

} // namespace faiss

#endif

