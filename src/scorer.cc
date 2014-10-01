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
    restriction_to_hash_(),
    pos_to_hash_(),
    hash_to_score_()
{
    kind.mixture.score_diff(kind.model, conditional, prior_, rng);
}

void RestrictionScorerKind::add_restriction (
        const ProductValue::Observed & restriction)
{
    // never freed
    static thread_local Map::value_type * to_insert = nullptr;
    construct_if_null(to_insert);

    restriction.SerializeToString(
        const_cast<std::string *>(& to_insert->first));
    auto inserted = restriction_to_hash_.insert(*to_insert);
    uint32_t & hash = inserted.first->second;
    if (LOOM_UNLIKELY(inserted.second)) {
        hash = hash_to_score_.size();
        hash_to_score_.push_back(NAN);
    }
    pos_to_hash_.push_back(hash);

    if (LOOM_DEBUG_LEVEL >= 1) {
        LOOM_ASSERT_EQ(restriction_to_hash_.size(), hash_to_score_.size());
    }
}

void RestrictionScorerKind::set_value (
        const ProductValue & value,
        rng_t & rng)
{
    std::vector<VectorFloat *> feature_scores;
    feature_scores.reserve(kind_.model.schema.observed_count(value.observed()));
    kind_.model.schema.for_each(value.observed(), [&](size_t i){
        if (LOOM_DEBUG_LEVEL >= 1) {
            LOOM_ASSERT_LT(i, likelihoods_.size());
        }
        feature_scores.push_back(&likelihoods_[i]);
    });
    kind_.mixture.score_value_features(
        kind_.model,
        value,
        feature_scores,
        rng);

    for (const auto & pair : restriction_to_hash_) {
        hash_to_score_[pair.second] = _compute_score(pair.first);
    }
}

float RestrictionScorerKind::_compute_score (const std::string & message) const
{
    // never freed
    static thread_local ProductValue::Observed * restriction = nullptr;
    static thread_local VectorFloat * scores = nullptr;
    construct_if_null(restriction);
    construct_if_null(scores);

    restriction->ParseFromString(message);
    if (LOOM_DEBUG_LEVEL >= 1) {
        kind_.model.schema.validate(*restriction);
    }
    *scores = prior_;
    kind_.model.schema.for_each(*restriction, [&](size_t i){
        if (LOOM_DEBUG_LEVEL >= 1) {
            LOOM_ASSERT_LT(i, likelihoods_.size());
            LOOM_ASSERT_EQ(likelihoods_[i].size(), scores->size());
        }
        distributions::vector_add(
                scores->size(),
                scores->data(),
                likelihoods_[i].data());
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

void RestrictionScorer::add_restriction (
        const ProductValue::Observed & restriction)
{
    // never freed
    static thread_local std::vector<ProductValue::Observed> *
    partial_restrictions = nullptr;
    construct_if_null(partial_restrictions);

    const size_t kind_count = cross_cat_.kinds.size();
    cross_cat_.splitter.split(restriction, *partial_restrictions);
    for (size_t k = 0; k < kind_count; ++k) {
        kinds_[k]->add_restriction((*partial_restrictions)[k]);
    }
}

void RestrictionScorer::set_value (const ProductValue & value, rng_t & rng)
{
    // never freed
    static thread_local std::vector<ProductValue> * partial_values = nullptr;
    construct_if_null(partial_values);

    const size_t kind_count = cross_cat_.kinds.size();
    cross_cat_.splitter.split(value, *partial_values);
    for (size_t k = 0; k < kind_count; ++k) {
        kinds_[k]->set_value((*partial_values)[k], rng);
    }
}

} // namespace loom
