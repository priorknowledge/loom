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

#include <loom/product_model.hpp>

namespace loom
{

template<bool cached> struct ProductMixture_;
typedef ProductMixture_<false> SmallProductMixture;
typedef ProductMixture_<true> FastProductMixture;

template<bool cached>
struct ProductMixture_
{
    typedef protobuf::ProductValue Value;
    struct Feature
    {
        template<class T>
        struct Container
        {
            typedef IndexedVector<typename T::template Mixture<cached>::t> t;
        };
    };
    typedef ForEachFeatureType<Feature> Features;

    struct TareCache
    {
        VectorFloat scores;
        distributions::Packed_<uint32_t> counts;
    };

    typename Clustering::Mixture<cached>::t clustering;
    Features features;
    std::vector<TareCache> tare_caches;
    distributions::MixtureIdTracker id_tracker;
    bool maintaining_cache;

    void init_unobserved (
            const ProductModel & model,
            const std::vector<int> & counts,
            rng_t & rng);

    void load_step_1_of_3 (
            const ProductModel & model,
            const char * filename,
            size_t empty_group_count);

    void load_step_2_of_3 (
            const ProductModel & model,
            size_t featureid,
            size_t empty_group_count,
            rng_t & rng);

    void load_step_3_of_3 (
            const ProductModel & model,
            rng_t & rng);

    void dump (
            const char * filename,
            const std::vector<uint32_t> & sorted_to_global) const;

    void add_value (
            const ProductModel & model,
            size_t groupid,
            const Value & value,
            rng_t & rng);

    void remove_value (
            const ProductModel & model,
            size_t groupid,
            const Value & value,
            rng_t & rng);

    void add_diff (
            const ProductModel & model,
            size_t groupid,
            const Value::Diff & diff,
            rng_t & rng);

    void remove_diff (
            const ProductModel & model,
            size_t groupid,
            const Value::Diff & diff,
            rng_t & rng);

    void add_diff_step_1_of_2 (
            const ProductModel & model,
            size_t groupid,
            const Value::Diff & diff,
            rng_t & rng);

    void add_diff_step_2_of_2 (
            const ProductModel & model,
            rng_t & rng);

    void remove_unobserved_value (
            const ProductModel & model,
            size_t groupid);

    void score_value (
            const ProductModel & model,
            const Value & value,
            VectorFloat & scores,
            rng_t & rng) const;

    void score_diff (
            const ProductModel & model,
            const Value::Diff & diff,
            VectorFloat & scores,
            rng_t & rng) const;

    float score_feature (
            const ProductModel & model,
            size_t featureid,
            rng_t & rng) const;

    float score_data (
            const ProductModel & model,
            rng_t & rng) const;

    size_t sample_value (
            const ProductModel & model,
            const VectorFloat & probs,
            Value & value,
            rng_t & rng) const;

    template<class OtherMixture>
    void move_feature_to (
            size_t featureid,
            ProductModel & source_model, OtherMixture & source_mixture,
            ProductModel & destin_model, OtherMixture & destin_mixture);

    template<bool other_cached>
    void validate_subset (const ProductMixture_<other_cached> & other) const;

    void init_feature_cache (
            const ProductModel & model,
            size_t featureid,
            rng_t & rng);

    void init_tare_cache (
            const ProductModel & model,
            rng_t & rng);

    void validate (const ProductModel & model) const;

    size_t count_rows () const
    {
        const auto & counts = clustering.counts();
        return std::accumulate(counts.begin(), counts.end(), 0);
    }

private:

    void _add_tare_cache (const ProductModel & model, rng_t & rng);
    void _remove_tare_cache (size_t groupid);
    void _update_tare_cache (
            const ProductModel & model,
            size_t groupid,
            rng_t & rng);

    struct validate_fun;
    struct clear_fun;
    struct load_group_fun;
    struct init_groups_fun;
    struct init_feature_cache_fun;
    struct init_unobserved_fun;
    struct sort_groups_fun;
    struct dump_group_fun;
    struct add_group_fun;
    struct add_value_fun;
    struct remove_group_fun;
    struct remove_value_fun;
    struct add_diff_fun;
    struct score_value_fun;
    struct score_value_group_fun;
    struct score_feature_fun;
    struct score_data_fun;
    struct sample_fun;

    template<class OtherMixture>
    struct move_feature_to_fun;

    template<bool other_cached>
    struct validate_subset_fun;
};

template<bool cached>
struct ProductMixture_<cached>::validate_fun
{
    const size_t group_count;
    const ProductModel::Features & shareds;
    const Features & mixtures;
    const bool maintaining_cache;

    template<class T>
    void operator() (T * t)
    {
        LOOM_ASSERT_EQ(shareds[t].size(), mixtures[t].size());
        for (size_t i = 0, size = shareds[t].size(); i < size; ++i) {
            const auto & shared = shareds[t][i];
            const auto & mixture = mixtures[t][i];
            LOOM_ASSERT_EQ(mixture.groups().size(), group_count);
            if (maintaining_cache) {
                mixture.validate(shared);
            } else {
                for (const auto & group : mixture.groups()) {
                    group.validate(shared);
                }
            }
        }
    }
};

template<bool cached>
inline void ProductMixture_<cached>::validate (
        const ProductModel & model) const
{
    if (LOOM_DEBUG_LEVEL >= 1) {
        model.schema.validate(features);
    }
    if (LOOM_DEBUG_LEVEL >= 2) {
        const size_t group_count = clustering.counts().size();
        validate_fun fun = {
            group_count,
            model.features,
            features,
            maintaining_cache};
        for_each_feature_type(fun);
        if (maintaining_cache) {
            LOOM_ASSERT_EQ(tare_caches.size(), model.tares.size());
            for (auto & tare_cache : tare_caches) {
                if (cached) {
                    LOOM_ASSERT_EQ(tare_cache.scores.size(), group_count);
                    LOOM_ASSERT_EQ(tare_cache.counts.size(), 0);
                } else {
                    LOOM_ASSERT_EQ(tare_cache.scores.size(), 0);
                    LOOM_ASSERT_EQ(tare_cache.counts.size(), group_count);
                }
            }
        }
        LOOM_ASSERT_EQ(id_tracker.packed_size(), group_count);
    }
}

} // namespace loom
