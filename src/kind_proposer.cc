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

#include <unordered_set>
#include <distributions/random.hpp>
#include <distributions/vector_math.hpp>
#include <distributions/trivial_hash.hpp>
#include <loom/kind_proposer.hpp>

#define LOOM_ASSERT_CLOSE(x, y) \
    LOOM_ASSERT_LT(fabs((x) - (y)) / ((x) + (y) + 1e-20), 1e-4)

namespace loom
{

void KindProposer::model_load (
        const CrossCat & cross_cat,
        ProductModel & model)
{
    model.clear();
    for (const auto & kind : cross_cat.kinds) {
        model.extend(kind.model);
    }
    LOOM_ASSERT_EQ(model.schema, cross_cat.schema);
}

void KindProposer::model_load (const CrossCat & cross_cat)
{
    if (const size_t kind_count = kinds.size()) {
        auto & model = kinds[0].model;
        model_load(cross_cat, model);
        for (size_t i = 1; i < kind_count; ++i) {
            kinds[i].model = model;
        }
    }
}

void KindProposer::mixture_init_unobserved (
        const CrossCat & cross_cat,
        rng_t & rng)
{
    const size_t kind_count = cross_cat.kinds.size();
    LOOM_ASSERT_LT(0, kind_count);
    kinds.resize(kind_count);
    model_load(cross_cat);
    for (size_t i = 0; i < kind_count; ++i) {
        const auto & counts = cross_cat.kinds[i].mixture.clustering.counts();
        kinds[i].mixture.init_unobserved(kinds[i].model, counts, rng);
    }
}

//----------------------------------------------------------------------------
// Block Pitman-Yor Sampler
//
// This sampler follows the math in
// $DISTRIBUTIONS_PATH/src/clustering.hpp
// distributions::Clustering<int>::PitmanYor::sample_assignments(...)

class KindProposer::BlockPitmanYorSampler
{
public:

    BlockPitmanYorSampler (
            const distributions::Clustering<int>::PitmanYor & topology,
            const std::vector<VectorFloat> & likelihoods,
            std::vector<uint32_t> & assignments);

    void run (size_t iterations, rng_t & rng);

    typedef std::unordered_set<uint32_t, distributions::TrivialHash<uint32_t>>
        IdSet;

private:

    void validate () const;

    float get_likelihood_empty () const;
    std::vector<uint32_t> get_counts_from_assignments () const;
    IdSet get_empty_kinds_from_counts () const;
    VectorFloat get_prior_from_counts () const;
    void add_empty_kind (size_t kindid);
    void remove_empty_kind (size_t kindid);

    static float compute_posterior (
            const VectorFloat & prior_in,
            const VectorFloat & likelihood_in,
            VectorFloat & posterior_out);

    const float alpha_;
    const float d_;
    const size_t feature_count_;
    const size_t kind_count_;
    const std::vector<VectorFloat> & likelihoods_;
    std::vector<uint32_t> & assignments_;
    std::vector<uint32_t> counts_;
    IdSet empty_kinds_;
    size_t empty_kind_count_;
    VectorFloat prior_;
    VectorFloat posterior_;
};

KindProposer::BlockPitmanYorSampler::BlockPitmanYorSampler (
        const distributions::Clustering<int>::PitmanYor & topology,
        const std::vector<VectorFloat> & likelihoods,
        std::vector<uint32_t> & assignments) :
    alpha_(topology.alpha),
    d_(topology.d),
    feature_count_(likelihoods.size()),
    kind_count_(likelihoods[0].size()),
    likelihoods_(likelihoods),
    assignments_(assignments),
    counts_(get_counts_from_assignments()),
    empty_kinds_(get_empty_kinds_from_counts()),
    empty_kind_count_(empty_kinds_.size()),
    prior_(get_prior_from_counts()),
    posterior_(kind_count_)
{
    LOOM_ASSERT_LT(0, alpha_);
    LOOM_ASSERT_LE(0, d_);
    LOOM_ASSERT_LT(d_, 1);

    LOOM_ASSERT_LT(0, likelihoods.size());
    LOOM_ASSERT_EQ(likelihoods.size(), assignments.size());
    for (const auto & likelihood : likelihoods) {
        LOOM_ASSERT_EQ(likelihood.size(), kind_count_);
    }
}

inline std::vector<uint32_t>
    KindProposer::BlockPitmanYorSampler::get_counts_from_assignments () const
{
    std::vector<uint32_t> counts(kind_count_, 0);
    for (size_t f = 0; f < feature_count_; ++f) {
        size_t k = assignments_[f];
        LOOM_ASSERT1(k < kind_count_, "bad kind id: " << k);
        ++counts[k];
    }
    return counts;
}

inline KindProposer::BlockPitmanYorSampler::IdSet
    KindProposer::BlockPitmanYorSampler::get_empty_kinds_from_counts () const
{
    IdSet empty_kinds;
    for (size_t k = 0; k < kind_count_; ++k) {
        if (counts_[k] == 0) {
            empty_kinds.insert(k);
        }
    }
    return empty_kinds;
}

inline VectorFloat
    KindProposer::BlockPitmanYorSampler::get_prior_from_counts () const
{
    VectorFloat prior(kind_count_);
    const float likelihood_empty = get_likelihood_empty();
    for (size_t k = 0; k < kind_count_; ++k) {
        if (auto count = counts_[k]) {
            prior[k] = count - d_;
        } else {
            prior[k] = likelihood_empty;
        }
    }
    return prior;
}

inline void KindProposer::BlockPitmanYorSampler::validate () const
{
    std::vector<uint32_t> expected_counts = get_counts_from_assignments();
    for (size_t k = 0; k < kind_count_; ++k) {
        LOOM_ASSERT_EQ(counts_[k], expected_counts[k]);
    }

    LOOM_ASSERT_EQ(empty_kind_count_, empty_kinds_.size());
    for (size_t k = 0; k < kind_count_; ++k) {
        bool in_empty_kinds = (empty_kinds_.find(k) != empty_kinds_.end());
        bool has_zero_count = (counts_[k] == 0);
        LOOM_ASSERT_EQ(in_empty_kinds, has_zero_count);
    }

    VectorFloat expected_prior = get_prior_from_counts();
    for (size_t k = 0; k < kind_count_; ++k) {
        LOOM_ASSERT_CLOSE(prior_[k], expected_prior[k]);
    }
}

inline float KindProposer::BlockPitmanYorSampler::get_likelihood_empty () const
{
    if (empty_kind_count_) {
        float nonempty_kind_count = kind_count_ - empty_kind_count_;
        return (alpha_ + d_ * nonempty_kind_count) / empty_kind_count_;
    } else {
        return 0.f;
    }
}

inline void KindProposer::BlockPitmanYorSampler::add_empty_kind (size_t kindid)
{
    empty_kinds_.insert(kindid);
    ++empty_kind_count_;
    const float likelihood_empty = get_likelihood_empty();
    for (auto k : empty_kinds_) {
        prior_[k] = likelihood_empty;
    }
}

inline void KindProposer::BlockPitmanYorSampler::remove_empty_kind (size_t kindid)
{
    empty_kinds_.erase(kindid);
    --empty_kind_count_;
    const float likelihood_empty = get_likelihood_empty();
    for (auto k : empty_kinds_) {
        prior_[k] = likelihood_empty;
    }
}

inline float KindProposer::BlockPitmanYorSampler::compute_posterior (
        const VectorFloat & prior_in,
        const VectorFloat & likelihood_in,
        VectorFloat & posterior_out)
{
    const size_t size = prior_in.size();
    const float * __restrict__ prior =
        DIST_ASSUME_ALIGNED(prior_in.data());
    const float * __restrict__ likelihood =
        DIST_ASSUME_ALIGNED(likelihood_in.data());
    float * __restrict__ posterior =
        DIST_ASSUME_ALIGNED(posterior_out.data());

    float total = 0;
    for (size_t i = 0; i < size; ++i) {
        total += posterior[i] = prior[i] * likelihood[i];
    }
    return total;
}

using distributions::sample_from_likelihoods;

void KindProposer::BlockPitmanYorSampler::run (
        size_t iterations,
        rng_t & rng)
{
    LOOM_ASSERT_LT(0, iterations);

    for (size_t i = 0; i < iterations; ++i) {
        for (size_t f = 0; f < feature_count_; ++f) {
            size_t k = assignments_[f];

            if (--counts_[k] == 0) {
                add_empty_kind(k);
            } else {
                prior_[k] = counts_[k] - d_;
            }

            const VectorFloat & likelihood = likelihoods_[f];
            float total = compute_posterior(prior_, likelihood, posterior_);
            k = sample_from_likelihoods(rng, posterior_, total);
            assignments_[f] = k;

            if (counts_[k]++ == 0) {
                remove_empty_kind(k);
            }
            prior_[k] = counts_[k] - d_;

            if (LOOM_DEBUG_LEVEL >= 3) {
                validate();
            }
        }
    }
}

KindProposer::Timers KindProposer::infer_assignments (
        const ProductValue & tare,
        const CrossCat & cross_cat,
        std::vector<uint32_t> & featureid_to_kindid,
        size_t iterations,
        bool parallel,
        rng_t & rng)
{
    LOOM_ASSERT_LT(0, iterations);

    ProductModel model;
    model_load(cross_cat, model);
    const auto seed = rng();
    const size_t feature_count = featureid_to_kindid.size();
    const size_t kind_count = kinds.size();
    std::vector<VectorFloat> likelihoods(feature_count);
    for (auto & likelihood : likelihoods) {
        likelihood.resize(kind_count);
    }

    Timers timers = {0, 0, 0};

    if (tare.observed().sparsity() != ProductValue::Observed::NONE) {
        TimedScope timer(timers.tare);

        model.add_value(tare, rng);

        #pragma omp parallel for if(parallel) schedule(dynamic, 1)
        for (size_t k = 0; k < kind_count; ++k) {
            kinds[k].mixture.add_tare(model, tare, rng);
        }
    }
    {
        TimedScope timer(timers.score);

        #pragma omp parallel for if(parallel) schedule(dynamic, 1)
        for (size_t f = 0; f < feature_count; ++f) {
            rng_t rng(seed + f);
            VectorFloat & scores = likelihoods[f];
            for (size_t k = 0; k < kind_count; ++k) {
                const auto & mixture = kinds[k].mixture;
                scores[k] = mixture.score_feature(model, f, rng);
            }
            distributions::scores_to_likelihoods(scores);
        }
    }
    {
        TimedScope timer(timers.sample);

        BlockPitmanYorSampler sampler(
                cross_cat.topology,
                likelihoods,
                featureid_to_kindid);

        sampler.run(iterations, rng);
    }

    return timers;
}

} // namespace loom
