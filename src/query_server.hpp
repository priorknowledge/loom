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
        partial_values_(),
        result_factors_(),
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
    std::vector<ProductValue> partial_values_;
    std::vector<std::vector<ProductValue>> result_factors_;
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
        response.set_error("invalid query data");
        return;
    }

    cross_cat_.value_split(request.score().data(), partial_values_);

    const size_t kind_count = cross_cat_.kinds.size();
    for (size_t i = 0; i < kind_count; ++i) {
        const ProductValue & value = partial_values_[i];
        auto & kind = cross_cat_.kinds[i];
        const ProductModel & model = kind.model;
        auto & mixture = kind.mixture;

        mixture.score_value(model, value, scores_, rng);
    }
    response.mutable_score()->set_score(distributions::log_sum_exp(scores_));
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
        response.set_error("invalid request.sample.data");
        return;
    }
    if (not cross_cat_.schema.is_valid(request.sample().to_sample())) {
        response.set_error("invalid request.sample.to_sample");
        return;
    }
    const size_t sample_count = request.sample().sample_count();
    if (sample_count == 0) {
        return;
    }

    cross_cat_.value_split(request.sample().data(), partial_values_);
    result_factors_.resize(sample_count);
    cross_cat_.observed_split(
        request.sample().to_sample(),
        result_factors_.front());
    std::fill(
        result_factors_.begin() + 1,
        result_factors_.end(),
        result_factors_.front());

    const size_t kind_count = cross_cat_.kinds.size();
    for (size_t i = 0; i < kind_count; ++i) {
        const auto & to_sample = result_factors_.front()[i].observed();
        if (cross_cat_.schema.observed_count(to_sample)) {
            const ProductValue & value = partial_values_[i];
            auto & kind = cross_cat_.kinds[i];
            const ProductModel & model = kind.model;
            auto & mixture = kind.mixture;

            mixture.score_value(model, value, scores_, rng);
            distributions::scores_to_probs(scores_);
            const VectorFloat & probs = scores_;

            for (auto & result_values : result_factors_) {
                mixture.sample_value(model, probs, result_values[i], rng);
            }
        }
    }

    for (const auto & result_values : result_factors_) {
        auto & sample = * response.mutable_sample()->add_samples();
        cross_cat_.value_join(sample, result_values);
    }
}

} // namespace loom
