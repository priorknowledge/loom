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

#include <loom/cat_kernel.hpp>
#include <loom/hyper_kernel.hpp>
#include <loom/kind_proposer.hpp>
#include <loom/infer_grid.hpp>

namespace loom
{

void generate_rows (
        const protobuf::Config::Generate & config,
        CrossCat & cross_cat,
        const char * rows_out,
        rng_t & rng)
{
    const size_t kind_count = cross_cat.kinds.size();
    const size_t row_count = config.row_count();
    const float density = config.density();
    LOOM_ASSERT_LE(0.0, density);
    LOOM_ASSERT_LE(density, 1.0);
    VectorFloat scores;
    std::vector<ProductModel::Value> partial_values(kind_count);
    protobuf::Row row;
    protobuf::OutFile rows(rows_out);

    for (auto & kind : cross_cat.kinds) {
        kind.model.realize(rng);
    }

    for (size_t id = 0; id < row_count; ++id) {

        for (size_t k = 0; k < kind_count; ++k) {
            auto & kind = cross_cat.kinds[k];
            ProductModel & model = kind.model;
            auto & mixture = kind.mixture;
            ProductModel::Value & value = partial_values[k];

            scores.resize(mixture.clustering.counts().size());
            mixture.clustering.score_value(model.clustering, scores);
            distributions::scores_to_probs(scores);
            const VectorFloat & probs = scores;

            auto & observed = * value.mutable_observed();
            observed.Clear();
            observed.set_sparsity(ProductModel::Value::Observed::DENSE);
            const size_t feature_count = kind.featureids.size();
            for (size_t f = 0; f < feature_count; ++f) {
                observed.add_dense(
                    distributions::sample_bernoulli(rng, density));
            }
            size_t groupid = mixture.sample_value(model, probs, value, rng);

            model.add_value(value, rng);
            mixture.add_value(model, groupid, value, rng);
        }

        row.set_id(id);
        cross_cat.value_join(* row.mutable_data(), partial_values);
        rows.write_stream(row);
    }
}

} // namespace loom
