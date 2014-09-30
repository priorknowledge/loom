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

#include <loom/scorer.hpp>

namespace loom
{

RestrictionScorerKind::RestrictionScorerKind (
        const CrossCat::Kind & kind,
        const ProductValue::Diff & conditional,
        rng_t & rng) :
    kind_(kind),
    prior_(),
    likelihoods_(kind.model.schema.total_size()),
    cache_()
{
    kind.mixture.score_diff(kind.model, conditional, prior_, rng);
}

void RestrictionScorerKind::set_value (
        const ProductValue & value,
        rng_t & rng)
{
    cache_.clear();

    std::vector<VectorFloat *> feature_scores;
    feature_scores.reserve(kind_.model.schema.observed_count(value.observed()));
    kind_.model.schema.for_each(value.observed(), [&](size_t i){
        feature_scores.push_back(&likelihoods_[i]);
        kind_.mixture.score_value_features(
            kind_.model,
            value,
            feature_scores,
            rng);
    });
}

float RestrictionScorerKind::_compute_score (
        const ProductValue::Observed & restriction) const
{
    // never freed
    static thread_local VectorFloat * scores = nullptr;
    construct_if_null(scores);

    const size_t size = prior_.size();
    * scores = prior_;
    kind_.model.schema.for_each(restriction, [&](size_t i){
        const auto & likelihood = likelihoods_[i];
        if (LOOM_DEBUG_LEVEL >= 1) {
            LOOM_ASSERT_EQ(likelihood.size(), size);
        }
        distributions::vector_add(size, scores->data(), likelihood.data());
    });
    return distributions::log_sum_exp(*scores);
}

RestrictionScorer::RestrictionScorer (
        const CrossCat & cross_cat,
        const ProductValue::Diff & conditional,
        rng_t & rng) :
    cross_cat_(cross_cat),
    kinds_(cross_cat.kinds.size(), nullptr)
{
    const size_t kind_count = cross_cat_.kinds.size();
    std::vector<ProductValue::Diff> partial_diffs(kind_count);
    cross_cat_.splitter.split(conditional, partial_diffs);
    for (size_t k = 0; k < kind_count; ++k) {
        kinds_[k] = new RestrictionScorerKind(
            cross_cat_.kinds[k],
            partial_diffs[k],
            rng);
    }
}

RestrictionScorer::~RestrictionScorer ()
{
    for (auto kind : kinds_) {
        delete kind;
    }
}

void RestrictionScorer::set_value (const ProductValue & value, rng_t & rng)
{
    std::vector<ProductValue> partial_values;

    const size_t kind_count = cross_cat_.kinds.size();
    cross_cat_.splitter.split(value, partial_values);
    for (size_t k = 0; k < kind_count; ++k) {
        kinds_[k]->set_value(partial_values[k], rng);
    }
}

float RestrictionScorer::get_score (
        const ProductValue::Observed & restriction) const
{
    // never freed
    static thread_local std::vector<ProductValue::Observed> *
    partial_observeds = nullptr;
    construct_if_null(partial_observeds);

    const size_t kind_count = cross_cat_.kinds.size();
    cross_cat_.splitter.split(restriction, *partial_observeds);
    float score = 0;
    for (size_t k = 0; k < kind_count; ++k) {
        score += kinds_[k]->get_score((*partial_observeds)[k]);
    }
    return score;
}

} // namespace loom
