// Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// - Neither the name of Salesforce.com nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <unordered_set>
#include <distributions/vector.hpp>
#include <loom/common.hpp>
#include <loom/protobuf.hpp>
#include <loom/product_model.hpp>
#include <loom/product_mixture.hpp>

namespace loom
{

struct CrossCat : noncopyable
{
    typedef FastProductMixture ProductMixture;
    struct Kind
    {
        ProductModel model;
        ProductMixture mixture;
        std::unordered_set<size_t> featureids;
    };

    ValueSchema schema;
    std::vector<ProductValue> tares;
    ValueSplitter splitter;
    protobuf::HyperPrior hyper_prior;
    Clustering::Shared topology;
    distributions::Packed_<Kind> kinds;
    std::vector<uint32_t> featureid_to_kindid;

    void model_load (const char * filename);
    void model_dump (const char * filename) const;

    void tares_load (const char * filename, rng_t & rng);

    void mixture_init_unobserved (
            size_t empty_group_count,
            rng_t & rng);
    void mixture_load (
            const char * dirname,
            size_t empty_group_count,
            rng_t & rng);
    void mixture_dump (
            const char * dirname,
            const std::vector<std::vector<uint32_t>> & sorted_to_globals) const;

    std::vector<std::vector<uint32_t>> get_sorted_groupids () const;

    void update_splitter ();
    void update_tares (
            std::vector<ProductValue> & temp_values,
            rng_t & rng);

    void value_split (
            const ProductValue & full_value,
            std::vector<ProductValue> & partial_values) const;

    void diff_split (
            const ProductValue::Diff & full_diff,
            std::vector<ProductValue::Diff> & partial_diffs,
            std::vector<ProductValue> temp) const;

    void value_join (
            ProductValue & full_value,
            const std::vector<ProductValue> & partial_values) const;

    void diff_join (
            ProductValue::Diff & full_diff,
            const std::vector<ProductValue::Diff> & partial_diffss) const;

    void normalize_small (
            std::vector<ProductValue::Diff> & partial_diffss) const;

    float score_data (rng_t & rng) const;

    void validate () const;

private:

    std::string get_mixture_filename (
            const char * dirname,
            size_t kindid) const;
};

inline void CrossCat::value_split (
        const ProductValue & full_value,
        std::vector<ProductValue> & partial_values) const
{
    splitter.split(full_value, partial_values);
}

inline void CrossCat::diff_split (
        const ProductValue::Diff & full_diff,
        std::vector<ProductValue::Diff> & partial_diffs,
        std::vector<ProductValue> temp) const
{
    const size_t kind_count = kinds.size();
    partial_diffs.resize(kind_count);
    for (auto & diff : partial_diffs) {
        diff.Clear();
        * diff.mutable_tares() = full_diff.tares();
    }
    splitter.split(full_diff.pos(), temp, [&partial_diffs](size_t i){
        return partial_diffs[i].mutable_pos();
    });
    splitter.split(full_diff.neg(), temp, [&partial_diffs](size_t i){
        return partial_diffs[i].mutable_neg();
    });
}

inline void CrossCat::value_join (
        ProductValue & full_value,
        const std::vector<ProductValue> & partial_values) const
{
    splitter.join(full_value, partial_values);
}

inline void CrossCat::diff_join (
        ProductValue::Diff & full_diff,
        const std::vector<ProductValue::Diff> & partial_diffs) const
{
    if (LOOM_UNLIKELY(partial_diffs.empty())) {
        schema.clear(full_diff);
    } else {
        * full_diff.mutable_tares() = partial_diffs.front().tares();
        splitter.join(* full_diff.mutable_pos(), [&partial_diffs](size_t i){
            return & partial_diffs[i].pos();
        });
        splitter.join(* full_diff.mutable_neg(), [&partial_diffs](size_t i){
            return & partial_diffs[i].neg();
        });
    }
}

inline void CrossCat::normalize_small (
        std::vector<ProductValue::Diff> & partial_diffs) const
{
    if (LOOM_DEBUG_LEVEL >= 1) {
        LOOM_ASSERT_EQ(partial_diffs.size(), kinds.size());
    }
#ifdef LOOM_NORMALZE_SMALL_DURING_INFERENCE
    // TODO allow normalize_small optimization during inference
    auto diff = partial_diffs.begin();
    for (auto & kind : kinds) {
        kind.model.schema.normalize_small(*diff++);
    }
#endif // LOOM_NORMALZE_SMALL_DURING_INFERENCE
}

inline void CrossCat::validate () const
{
    if (LOOM_DEBUG_LEVEL >= 1) {
        LOOM_ASSERT_LT(0, schema.total_size());
        ValueSchema expected_schema;
        for (const auto & kind : kinds) {
            kind.model.validate();
            kind.mixture.validate(kind.model);
            expected_schema += kind.model.schema;
        }
        LOOM_ASSERT_EQ(schema, expected_schema);
        for (auto & tare : tares) {
            schema.validate(tare);
        }
    }
    if (LOOM_DEBUG_LEVEL >= 2) {
        splitter.validate(schema, featureid_to_kindid, kinds.size());
        for (size_t f = 0; f < featureid_to_kindid.size(); ++f) {
            size_t k = featureid_to_kindid[f];
            const auto & featureids = kinds[k].featureids;
            LOOM_ASSERT(
                featureids.find(f) != featureids.end(),
                "kind.featureids is missing " << f);
        }
        for (size_t k = 0; k < kinds.size(); ++k) {
            for (size_t f : kinds[k].featureids) {
                LOOM_ASSERT_EQ(featureid_to_kindid[f], k);
            }
        }
        for (size_t k = 0; k < kinds.size(); ++k) {
            LOOM_ASSERT_EQ(kinds[k].model.tares.size(), tares.size());
        }
    }
    if (LOOM_DEBUG_LEVEL >= 3) {
        std::vector<size_t> row_counts;
        for (const auto & kind : kinds) {
            row_counts.push_back(kind.mixture.count_rows());
        }
        for (size_t k = 1; k < kinds.size(); ++k) {
            LOOM_ASSERT_EQ(row_counts[k], row_counts[0]);
            LOOM_ASSERT_EQ(
                kinds[k].mixture.maintaining_cache,
                kinds[0].mixture.maintaining_cache);
        }
        std::vector<ProductValue> partial_tares;
        for (size_t id = 0; id < tares.size(); ++id) {
            splitter.split(tares[id], partial_tares);
            for (size_t k = 0; k < kinds.size(); ++k) {
                LOOM_ASSERT_EQ(partial_tares[k], kinds[k].model.tares[id]);
            }
        }
    }
}

inline void CrossCat::update_splitter ()
{
    splitter.init(schema, featureid_to_kindid, kinds.size());
}

} // namespace loom
