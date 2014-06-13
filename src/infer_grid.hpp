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

#include <vector>
#include <distributions/random.hpp>
#include <distributions/io/protobuf.hpp>
#include <loom/common.hpp>
#include <loom/models.hpp>

namespace loom
{

//----------------------------------------------------------------------------
// Generic Sampler
//
// This conforms to the Visitor interface implicit in
// hyper_prior.hpp for_each_gridpoint(const _ & grid, Visitor &)

template<class Mixture>
class InferShared
{
public:

    typedef typename Mixture::Shared Shared;

    InferShared (
            Shared & shared,
            const Mixture & mixture,
            rng_t & rng) :
        shared_(shared),
        mixture_(mixture),
        rng_(rng)
    {
    }

    const Shared & shared () const
    {
        return shared_;
    }

    Shared & add ()
    {
        hypotheses_.push_back(shared_);
        return hypotheses_.back();
    }

    void done ()
    {
        const size_t size = hypotheses_.size();
        if (size == 1) {

            shared_ = hypotheses_[0];

        } else if (size > 1) {

            scores_.resize(size);
            mixture_.score_data_grid(hypotheses_, scores_, rng_);
            size_t i = sample_from_scores_overwrite(rng_, scores_);
            shared_ = hypotheses_[i];
        }
        hypotheses_.clear();
        scores_.clear();
    }

private:

    Shared & shared_;
    const Mixture & mixture_;
    std::vector<Shared> hypotheses_;
    VectorFloat scores_;
    rng_t & rng_;
};

//----------------------------------------------------------------------------
// Clustering

template<class GridPrior>
Clustering::Shared sample_clustering_posterior (
        const GridPrior & grid_prior,
        const std::vector<int> & counts,
        rng_t & rng)
{
    const size_t grid_size = grid_prior.size();
    LOOM_ASSERT_LT(0, grid_size);

    Clustering::Shared shared;
    if (grid_size == 1) {
        shared.protobuf_load(grid_prior.Get(0));
    } else {
        VectorFloat scores(grid_size);
        for (size_t i = 0; i < grid_size; ++i) {
            shared.protobuf_load(grid_prior.Get(i));
            scores[i] = shared.score_counts(counts);
        }
        size_t i = distributions::sample_from_scores_overwrite(rng, scores);
        shared.protobuf_load(grid_prior.Get(i));
    }
    return shared;
}

template<class GridPrior>
Clustering::Shared sample_clustering_prior (
        const GridPrior & grid_prior,
        rng_t & rng)
{
    const size_t grid_size = grid_prior.size();
    LOOM_ASSERT_LT(0, grid_size);

    size_t i = distributions::sample_int(rng, 0, grid_size - 1);
    Clustering::Shared shared;
    shared.protobuf_load(grid_prior.Get(i));
    return shared;
}

} // namespace loom
