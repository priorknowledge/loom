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

#include <utility>
#include <typeinfo>
#include <loom/common.hpp>
#include <loom/cross_cat.hpp>
#include <loom/timer.hpp>

namespace loom
{

struct KindProposer
{
    struct Kind
    {
        ProductModel model;
        ProductModel::SmallMixture mixture;
    };

    std::vector<Kind> kinds;

    void clear () { kinds.clear(); }

    void model_load (const CrossCat & cross_cat);

    void mixture_init_unobserved (
            const CrossCat & cross_cat,
            rng_t & rng);

    std::pair<usec_t, usec_t> infer_assignments (
            const CrossCat & cross_cat,
            std::vector<uint32_t> & featureid_to_kindid,
            size_t iterations,
            bool parallel,
            rng_t & rng) const;

    void validate (const CrossCat & cross_cat) const;

private:

    static void model_load (
            const CrossCat & cross_cat,
            ProductModel & model);

    class BlockPitmanYorSampler;
};

inline void KindProposer::validate (const CrossCat & cross_cat) const
{
    if (LOOM_DEBUG_LEVEL >= 1) {
        LOOM_ASSERT_EQ(kinds.size(), cross_cat.kinds.size());
        for (const auto & kind : kinds) {
            LOOM_ASSERT_EQ(kind.model.schema, cross_cat.schema);
            kind.mixture.validate(kind.model);
        }
        for (size_t i = 0; i < kinds.size(); ++i) {
            size_t proposer_group_count =
                kinds[i].mixture.clustering.counts().size();
            size_t cross_cat_group_count =
                cross_cat.kinds[i].mixture.clustering.counts().size();
            LOOM_ASSERT_EQ(proposer_group_count, cross_cat_group_count);
        }
    }
}

} // namespace loom
