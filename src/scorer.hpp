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
    typedef std::unordered_map<std::string, float> Cache;

    const CrossCat::Kind & kind_;
    VectorFloat prior_;
    std::vector<VectorFloat> likelihoods_;
    mutable Cache cache_;

public:

    RestrictionScorerKind (
            const CrossCat::Kind & kind,
            const ProductValue::Diff & conditional,
            rng_t & rng);

    void set_value (const ProductValue & value, rng_t & rng);

    float get_score (
            const ProductValue::Observed & restriction) const
    {
        auto pair = _cache_find(restriction);
        float & score = pair.first->second;
        bool inserted = pair.second;
        if (LOOM_UNLIKELY(inserted)) {
            score = _compute_score(restriction);
        }
        return score;
    }

private:

    std::pair<Cache::iterator, bool> _cache_find (
            const ProductValue::Observed & restriction) const
    {
        // never freed
        static thread_local Cache::value_type * pair = nullptr;
        construct_if_null(pair);

        restriction.SerializeToString(const_cast<std::string *>(& pair->first));
        return cache_.insert(*pair);
    }

    float _compute_score (
            const ProductValue::Observed & restriction) const;
};

class RestrictionScorer
{
    const CrossCat & cross_cat_;
    std::vector<RestrictionScorerKind *> kinds_;

public:

    RestrictionScorer (
            const CrossCat & cross_cat,
            const ProductValue::Diff & conditional,
            rng_t & rng);

    ~RestrictionScorer ();

    void set_value (const ProductValue & value, rng_t & rng);

    float get_score (
            const ProductValue::Observed & restriction) const;
};

} // namespace loom
