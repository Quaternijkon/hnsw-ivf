/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <omp.h>
#include <algorithm>
#include <cstddef>
#include <map>
#include <random>
#include <set>

#include <gtest/gtest.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/FaissAssert.h>

namespace {

// stores all ivf lists, used to verify the context
// object is passed to the iterator
class TestContext {
   public:
    TestContext() {}

    void save_code(size_t list_no, const uint8_t* code, size_t code_size) {
        list_nos.emplace(id, list_no);
        codes.emplace(id, std::vector<uint8_t>(code_size));
        for (size_t i = 0; i < code_size; i++) {
            codes[id][i] = code[i];
        }
        id++;
    }

    // id to codes map
    std::unordered_map<faiss::idx_t, std::vector<uint8_t>> codes;
    // id to list_no map
    std::unordered_map<faiss::idx_t, size_t> list_nos;
    faiss::idx_t id = 0;
    std::set<size_t> lists_probed;
};

// the iterator that iterates over the codes stored in context object
class TestInvertedListIterator : public faiss::InvertedListsIterator {
   public:
    TestInvertedListIterator(size_t list_no, TestContext* context)
            : list_no{list_no}, context{context} {
        it = context->codes.cbegin();
        seek_next();
    }
    ~TestInvertedListIterator() override {}

    // move the cursor to the first valid entry
    void seek_next() {
        while (it != context->codes.cend() &&
               context->list_nos[it->first] != list_no) {
            it++;
        }
    }

    virtual bool is_available() const override {
        return it != context->codes.cend();
    }

    virtual void next() override {
        it++;
        seek_next();
    }

    virtual std::pair<faiss::idx_t, const uint8_t*> get_id_and_codes()
            override {
        if (it == context->codes.cend()) {
            FAISS_THROW_MSG("invalid state");
        }
        return std::make_pair(it->first, it->second.data());
    }

   private:
    size_t list_no;
    TestContext* context;
    decltype(context->codes.cbegin()) it;
};

class TestInvertedLists : public faiss::InvertedLists {
   public:
    TestInvertedLists(size_t nlist, size_t code_size)
            : faiss::InvertedLists(nlist, code_size) {
        use_iterator = true;
    }

    ~TestInvertedLists() override {}
    size_t list_size(size_t /*list_no*/) const override {
        FAISS_THROW_MSG("unexpected call");
    }

    faiss::InvertedListsIterator* get_iterator(size_t list_no, void* context)
            const override {
        auto testContext = (TestContext*)context;
        testContext->lists_probed.insert(list_no);
        return new TestInvertedListIterator(list_no, testContext);
    }

    const uint8_t* get_codes(size_t /* list_no */) const override {
        FAISS_THROW_MSG("unexpected call");
    }

    const faiss::idx_t* get_ids(size_t /* list_no */) const override {
        FAISS_THROW_MSG("unexpected call");
    }

    // store the codes in context object
    size_t add_entry(
            size_t list_no,
            faiss::idx_t /*theid*/,
            const uint8_t* code,
            void* context) override {
        auto testContext = (TestContext*)context;
        testContext->save_code(list_no, code, code_size);
        return 0;
    }

    size_t add_entries(
            size_t /*list_no*/,
            size_t /*n_entry*/,
            const faiss::idx_t* /*ids*/,
            const uint8_t* /*code*/) override {
        FAISS_THROW_MSG("unexpected call");
    }

    void update_entries(
            size_t /*list_no*/,
            size_t /*offset*/,
            size_t /*n_entry*/,
            const faiss::idx_t* /*ids*/,
            const uint8_t* /*code*/) override {
        FAISS_THROW_MSG("unexpected call");
    }

    void resize(size_t /*list_no*/, size_t /*new_size*/) override {
        FAISS_THROW_MSG("unexpected call");
    }
};
} // namespace

TEST(IVF, list_context) {
    // this test verifies that the context object is passed
    // to the InvertedListsIterator and InvertedLists::add_entry.
    // the test InvertedLists and InvertedListsIterator reads/writes
    // to the test context object.
    // the test verifies the context object is modified as expected.

    constexpr int d = 32;      // dimension
    constexpr int nb = 100000; // database size
    constexpr int nlist = 100;

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    // disable parallism, or we need to make Context object
    // thread-safe
    omp_set_num_threads(1);

    faiss::IndexFlatL2 quantizer(d); // the other index
    faiss::IndexIVFFlat index(&quantizer, d, nlist);
    TestInvertedLists inverted_lists(nlist, index.code_size);
    index.replace_invlists(&inverted_lists);
    {
        // training
        constexpr size_t nt = 1500; // nb of training vectors
        std::vector<float> trainvecs(nt * d);
        for (size_t i = 0; i < nt * d; i++) {
            trainvecs[i] = distrib(rng);
        }
        index.verbose = true;
        index.train(nt, trainvecs.data());
    }
    TestContext context;
    std::vector<float> query_vector;
    constexpr faiss::idx_t query_vector_id = 100;
    {
        // populating the database
        std::vector<float> database(nb * d);
        for (size_t i = 0; i < nb * d; i++) {
            database[i] = distrib(rng);
            // populate the query vector
            if (i >= query_vector_id * d && i < query_vector_id * d + d) {
                query_vector.push_back(database[i]);
            }
        }
        std::vector<faiss::idx_t> coarse_idx(nb);
        index.quantizer->assign(nb, database.data(), coarse_idx.data());
        // pass dummy ids, the acutal ids are assigned in TextContext object
        std::vector<faiss::idx_t> xids(nb, 42);
        index.add_core(
                nb, database.data(), xids.data(), coarse_idx.data(), &context);

        // check the context object get updated
        EXPECT_EQ(nb, context.id) << "should have added all ids";
        EXPECT_EQ(nb, context.codes.size())
                << "should have correct number of codes";
        EXPECT_EQ(nb, context.list_nos.size())
                << "should have correct number of list numbers";
    }
    {
        constexpr size_t num_vecs = 5; // number of vectors
        std::vector<float> vecs(num_vecs * d);
        for (size_t i = 0; i < num_vecs * d; i++) {
            vecs[i] = distrib(rng);
        }
        const size_t codeSize = index.sa_code_size();
        std::vector<uint8_t> encodedData(num_vecs * codeSize);
        index.sa_encode(num_vecs, vecs.data(), encodedData.data());
        std::vector<float> decodedVecs(num_vecs * d);
        index.sa_decode(num_vecs, encodedData.data(), decodedVecs.data());
        EXPECT_EQ(vecs, decodedVecs)
                << "decoded vectors should be the same as the original vectors that were encoded";
    }
    {
        constexpr faiss::idx_t k = 100;
        constexpr size_t nprobe = 10;
        std::vector<float> distances(k);
        std::vector<faiss::idx_t> labels(k);
        faiss::SearchParametersIVF params;
        params.inverted_list_context = &context;
        params.nprobe = nprobe;
        index.search(
                1,
                query_vector.data(),
                k,
                distances.data(),
                labels.data(),
                &params);
        EXPECT_EQ(nprobe, context.lists_probed.size())
                << "should probe nprobe lists";

        // check the result contains the query vector, the probablity of
        // this fail should be low
        auto query_vector_listno = context.list_nos[query_vector_id];
        auto& lists_probed = context.lists_probed;
        EXPECT_TRUE(
                std::find(
                        lists_probed.cbegin(),
                        lists_probed.cend(),
                        query_vector_listno) != lists_probed.cend())
                << "should probe the list of the query vector";
        EXPECT_TRUE(
                std::find(labels.cbegin(), labels.cend(), query_vector_id) !=
                labels.cend())
                << "should return the query vector";
    }
}

// Test for search_with_hnsw functionality
#include <faiss/IndexHNSW.h>

TEST(IVF, search_with_hnsw) {
    // Test search_with_hnsw with HNSW quantizer
    constexpr int d = 32;       // dimension
    constexpr int nb = 10000;   // database size
    constexpr int nlist = 100;  // number of clusters
    constexpr int M = 16;       // HNSW parameter
    constexpr int nq = 10;      // number of queries
    constexpr int k = 10;       // number of nearest neighbors
    constexpr int nprobe = 8;   // number of probes

    std::mt19937 rng(12345);
    std::uniform_real_distribution<> distrib;

    // Create HNSW quantizer
    faiss::IndexHNSWFlat hnsw_quantizer(d, M);
    hnsw_quantizer.hnsw.efConstruction = 40;
    hnsw_quantizer.hnsw.efSearch = 32;

    // Create IVF index with HNSW quantizer
    faiss::IndexIVFFlat index(&hnsw_quantizer, d, nlist);
    index.own_fields = false;  // Don't own quantizer
    
    // Generate training data
    constexpr size_t nt = 2000;
    std::vector<float> trainvecs(nt * d);
    for (size_t i = 0; i < nt * d; i++) {
        trainvecs[i] = distrib(rng);
    }
    
    // Train index
    index.train(nt, trainvecs.data());
    EXPECT_TRUE(index.is_trained);

    // Generate database
    std::vector<float> database(nb * d);
    for (size_t i = 0; i < nb * d; i++) {
        database[i] = distrib(rng);
    }
    
    // Add vectors to index
    index.add(nb, database.data());
    EXPECT_EQ(index.ntotal, nb);

    // Generate queries
    std::vector<float> queries(nq * d);
    for (size_t i = 0; i < nq * d; i++) {
        queries[i] = distrib(rng);
    }
    
    // Set nprobe
    index.nprobe = nprobe;

    // Search with search_with_hnsw
    std::vector<float> distances_hnsw(nq * k);
    std::vector<faiss::idx_t> labels_hnsw(nq * k);
    index.search_with_hnsw(
            nq, queries.data(), k, distances_hnsw.data(), labels_hnsw.data());

    // Search with regular search for comparison
    std::vector<float> distances_regular(nq * k);
    std::vector<faiss::idx_t> labels_regular(nq * k);
    index.search(
            nq, queries.data(), k, distances_regular.data(), labels_regular.data());

    // Compare results - they should be identical since we're using the same
    // underlying search_preassigned function
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            EXPECT_EQ(labels_hnsw[i * k + j], labels_regular[i * k + j])
                    << "Labels mismatch at query " << i << " neighbor " << j;
            EXPECT_NEAR(
                    distances_hnsw[i * k + j],
                    distances_regular[i * k + j],
                    1e-5)
                    << "Distances mismatch at query " << i << " neighbor " << j;
        }
    }
}

TEST(IVF, search_with_hnsw_fallback) {
    // Test that search_with_hnsw falls back to regular search when quantizer
    // is not HNSW
    constexpr int d = 32;
    constexpr int nb = 1000;
    constexpr int nlist = 10;
    constexpr int nq = 5;
    constexpr int k = 5;

    std::mt19937 rng(12345);
    std::uniform_real_distribution<> distrib;

    // Create flat quantizer (not HNSW)
    faiss::IndexFlatL2 flat_quantizer(d);
    faiss::IndexIVFFlat index(&flat_quantizer, d, nlist);
    index.own_fields = false;

    // Generate and train
    std::vector<float> trainvecs(500 * d);
    for (size_t i = 0; i < 500 * d; i++) {
        trainvecs[i] = distrib(rng);
    }
    index.train(500, trainvecs.data());

    // Generate and add database
    std::vector<float> database(nb * d);
    for (size_t i = 0; i < nb * d; i++) {
        database[i] = distrib(rng);
    }
    index.add(nb, database.data());

    // Generate queries
    std::vector<float> queries(nq * d);
    for (size_t i = 0; i < nq * d; i++) {
        queries[i] = distrib(rng);
    }

    // Both search methods should return the same results
    std::vector<float> distances_hnsw(nq * k);
    std::vector<faiss::idx_t> labels_hnsw(nq * k);
    index.search_with_hnsw(
            nq, queries.data(), k, distances_hnsw.data(), labels_hnsw.data());

    std::vector<float> distances_regular(nq * k);
    std::vector<faiss::idx_t> labels_regular(nq * k);
    index.search(
            nq, queries.data(), k, distances_regular.data(), labels_regular.data());

    for (int i = 0; i < nq * k; i++) {
        EXPECT_EQ(labels_hnsw[i], labels_regular[i]);
        EXPECT_NEAR(distances_hnsw[i], distances_regular[i], 1e-5);
    }
}
