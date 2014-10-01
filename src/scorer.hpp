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

#include <unordered_map>
#include <loom/cross_cat.hpp>

namespace loom
{

class RestrictionScorerKind
{
    typedef std::unordered_map<std::string, uint32_t> Map;

    const CrossCat::Kind & kind_;
    VectorFloat prior_;
    std::vector<VectorFloat> likelihoods_;
    Map restriction_to_hash_;
    std::vector<uint32_t> pos_to_hash_;
    std::vector<float> hash_to_score_;

public:

    RestrictionScorerKind (
            const CrossCat::Kind & kind,
            const ProductValue::Diff & conditional,
            rng_t & rng);

    void add_restriction (const ProductValue::Observed & restriction);
    void set_value (const ProductValue & value, rng_t & rng);

    float get_score (size_t i) const
    {
        if (LOOM_DEBUG_LEVEL >= 1) {
            LOOM_ASSERT_LT(i, pos_to_hash_.size());
            LOOM_ASSERT_LT(pos_to_hash_[i], hash_to_score_.size());
        }
        auto hash = pos_to_hash_[i];
        return hash_to_score_[hash];
    }

private:

    float _compute_score (const std::string & message) const;
};

class RestrictionScorer : noncopyable
{
    const CrossCat & cross_cat_;
    std::vector<RestrictionScorerKind *> kinds_;

public:

    RestrictionScorer (
            const CrossCat & cross_cat,
            const ProductValue::Diff & conditional,
            rng_t & rng);

    ~RestrictionScorer ();

    void add_restriction (const ProductValue::Observed & restriction);
    void set_value (const ProductValue & value, rng_t & rng);

    float get_score (size_t i) const
    {
        float score = 0;
        for (const auto * kind : kinds_) {
            score += kind->get_score(i);
        }
        return score;
    }
};

} // namespace loom
