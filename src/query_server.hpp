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

#include <loom/cross_cat.hpp>

namespace loom
{

class QueryServer
{
public:

    typedef protobuf::Query::Request Request;
    typedef protobuf::Query::Response Response;

    QueryServer (const CrossCat & cross_cat) :
        cross_cat_(cross_cat),
        temp_diff_(),
        partial_diffs_(),
        result_factors_(),
        temp_values_(),
        scores_(),
        timer_()
    {
    }

    void score_row (
            rng_t & rng,
            const Request & request,
            Response & response);

    void sample_row (
            rng_t & rng,
            const Request & request,
            Response & response);

private:

    const CrossCat & cross_cat_;
    ProductValue::Diff temp_diff_;
    std::vector<ProductValue::Diff> partial_diffs_;
    std::vector<std::vector<ProductValue::Diff>> result_factors_;
    std::vector<ProductValue> temp_values_;
    VectorFloat scores_;
    Timer timer_;
};

inline void QueryServer::score_row (
        rng_t & rng,
        const Request & request,
        Response & response)
{
    Timer::Scope timer(timer_);

    response.Clear();
    response.set_id(request.id());
    if (not cross_cat_.schema.is_valid(request.score().data())) {
        response.add_error("invalid query data");
        return;
    }

    cross_cat_.diff_split(
        request.score().data(),
        partial_diffs_,
        temp_values_);

    const size_t kind_count = cross_cat_.kinds.size();
    float score = 0.f;
    for (size_t i = 0; i < kind_count; ++i) {
        const ProductValue::Diff & diff = partial_diffs_[i];
        auto & kind = cross_cat_.kinds[i];
        const ProductModel & model = kind.model;
        auto & mixture = kind.mixture;

        if (diff.tares_size()) {
            mixture.score_diff(model, diff, scores_, rng);
        } else {
            mixture.score_value(model, diff.pos(), scores_, rng);
        }
        score += distributions::log_sum_exp(scores_);
    }
    response.mutable_score()->set_score(score);
}

inline void QueryServer::sample_row (
        rng_t & rng,
        const Request & request,
        Response & response)
{
    Timer::Scope timer(timer_);

    response.Clear();
    response.set_id(request.id());
    if (not cross_cat_.schema.is_valid(request.sample().data())) {
        response.add_error("invalid request.sample.data");
        return;
    }
    for (auto id : request.sample().data().tares()) {
        if (id >= cross_cat_.tares.size()) {
            response.add_error("invalid request.sample.data.tares");
        }
    }
    if (not cross_cat_.schema.is_valid(request.sample().to_sample())) {
        response.add_error("invalid request.sample.to_sample");
        return;
    }

    const size_t sample_count = request.sample().sample_count();
    if (sample_count == 0) {
        return;
    }

    cross_cat_.schema.clear(temp_diff_);
    * temp_diff_.mutable_pos()->mutable_observed() =
        request.sample().to_sample();
    cross_cat_.schema.fill_data_with_zeros(* temp_diff_.mutable_pos());
    result_factors_.resize(sample_count);
    cross_cat_.diff_split(temp_diff_, result_factors_.front(), temp_values_);
    std::fill(
        result_factors_.begin() + 1,
        result_factors_.end(),
        result_factors_.front());

    cross_cat_.diff_split(
        request.sample().data(),
        partial_diffs_,
        temp_values_);
    const size_t kind_count = cross_cat_.kinds.size();
    for (size_t i = 0; i < kind_count; ++i) {
        const auto & to_sample = result_factors_.front()[i].pos().observed();
        if (cross_cat_.schema.observed_count(to_sample)) {
            const ProductValue::Diff & diff = partial_diffs_[i];
            auto & kind = cross_cat_.kinds[i];
            const ProductModel & model = kind.model;
            auto & mixture = kind.mixture;

            if (diff.tares_size()) {
                mixture.score_diff(model, diff, scores_, rng);
            } else {
                mixture.score_value(model, diff.pos(), scores_, rng);
            }
            distributions::scores_to_probs(scores_);
            const VectorFloat & probs = scores_;

            for (auto & result_values : result_factors_) {
                ProductValue & value = * result_values[i].mutable_pos();
                mixture.sample_value(model, probs, value, rng);
            }
        }
    }

    for (const auto & result_values : result_factors_) {
        auto & sample = * response.mutable_sample()->add_samples();
        cross_cat_.diff_join(sample, result_values);
    }
}

} // namespace loom
