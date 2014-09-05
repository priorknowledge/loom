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

#include <loom/query_server.hpp>

namespace loom
{

void QueryServer::serve (
        rng_t & rng,
        const char * requests_in,
        const char * responses_out)
{
    protobuf::InFile query_stream(requests_in);
    protobuf::OutFile response_stream(responses_out);
    protobuf::Query::Request request;
    protobuf::Query::Response response;

    while (query_stream.try_read_stream(request)) {
        response.Clear();
        response.set_id(request.id());
        if (request.has_sample()) {
            sample_rows(rng, request, response);
        }
        if (request.has_score()) {
            score_row(rng, request, response);
        }
        if (request.has_entropy()) {
            estimate_entropy(rng, request, response);
        }
        response_stream.write_stream(response);
        response_stream.flush();
    }
}

void QueryServer::score_row (
        rng_t & rng,
        const Request & request,
        Response & response)
{
    Timer::Scope timer(timer_);

    if (not schema().is_valid(request.score().data())) {
        response.add_error("invalid request.score.data");
        return;
    }
    for (auto id : request.sample().data().tares()) {
        if (id >= tares().size()) {
            response.add_error("invalid request.score.data.tares");
            return;
        }
    }

    VectorFloat latent_scores(cross_cats_.size(), 0.f);
    const size_t latent_count = cross_cats_.size();
    for (size_t l = 0; l < latent_count; ++l) {
        const auto & cross_cat = * cross_cats_[l];
        float & score = latent_scores[l];

        cross_cat.splitter.split(
            request.score().data(),
            partial_diffs_,
            temp_values_);

        const size_t kind_count = cross_cat.kinds.size();
        for (size_t k = 0; k < kind_count; ++k) {
            const ProductValue::Diff & diff = partial_diffs_[k];
            auto & kind = cross_cat.kinds[k];
            const ProductModel & model = kind.model;
            auto & mixture = kind.mixture;

            if (diff.tares_size()) {
                mixture.score_diff(model, diff, scores_, rng);
            } else {
                mixture.score_value(model, diff.pos(), scores_, rng);
            }
            score += distributions::log_sum_exp(scores_);
        }
    }
    float score = distributions::log_sum_exp(latent_scores)
                - distributions::fast_log(latent_count);
    response.mutable_score()->set_score(score);
}

void QueryServer::sample_rows (
        rng_t & rng,
        const Request & request,
        Response & response)
{
    Timer::Scope timer(timer_);

    if (not schema().is_valid(request.sample().data())) {
        response.add_error("invalid request.sample.data");
        return;
    }
    for (auto id : request.sample().data().tares()) {
        if (id >= tares().size()) {
            response.add_error("invalid request.sample.data.tares");
            return;
        }
    }
    if (not schema().is_valid(request.sample().to_sample())) {
        response.add_error("invalid request.sample.to_sample");
        return;
    }

    const size_t latent_count = cross_cats_.size();
    std::vector<std::vector<VectorFloat>> latent_kind_scores(latent_count);
    VectorFloat latent_scores(latent_count, 0.f);
    {
        std::vector<ProductValue::Diff> conditional_diffs;
        for (size_t l = 0; l < latent_count; ++l) {
            const auto & cross_cat = * cross_cats_[l];
            auto & kind_scores = latent_kind_scores[l];
            cross_cat.splitter.split(
                request.sample().data(),
                conditional_diffs,
                temp_values_);

            const size_t kind_count = cross_cat.kinds.size();
            kind_scores.resize(kind_count);
            for (size_t k = 0; k < kind_count; ++k) {
                const ProductValue::Diff & diff = conditional_diffs[k];
                auto & kind = cross_cat.kinds[k];
                const ProductModel & model = kind.model;
                auto & mixture = kind.mixture;
                auto & scores = kind_scores[k];

                if (diff.tares_size()) {
                    mixture.score_diff(model, diff, scores, rng);
                } else {
                    mixture.score_value(model, diff.pos(), scores, rng);
                }

                latent_scores[l] += distributions::log_sum_exp(scores);
                distributions::scores_to_probs(scores);
            }
        }

        distributions::scores_to_probs(latent_scores);
    }

    const size_t sample_count = request.sample().sample_count();
    std::vector<size_t> latent_counts(latent_count, 0);
    for (size_t s = 0; s < sample_count; ++s) {
        size_t l = distributions::sample_discrete(
            rng,
            latent_scores.size(),
            latent_scores.data());
        ++latent_counts[l];
    }

    ProductValue::Diff blank;
    schema().clear(blank);
    * blank.mutable_pos()->mutable_observed() = request.sample().to_sample();
    schema().fill_data_with_zeros(* blank.mutable_pos());

    std::vector<ProductValue::Diff> result_diffs;
    for (size_t l = 0; l < latent_count; ++l) {
        const auto & cross_cat = * cross_cats_[l];
        auto & kind_scores = latent_kind_scores[l];

        for (size_t s = 0; s < latent_counts[l]; ++s) {
            cross_cat.splitter.split(blank, result_diffs, temp_values_);

            const size_t kind_count = cross_cat.kinds.size();
            for (size_t k = 0; k < kind_count; ++k) {
                if (cross_cat.schema.observed_count(
                    result_diffs[k].pos().observed()))
                {
                    auto & kind = cross_cat.kinds[k];
                    const ProductModel & model = kind.model;
                    auto & mixture = kind.mixture;
                    auto & probs = kind_scores[k];

                    ProductValue & value = * result_diffs[k].mutable_pos();
                    mixture.sample_value(model, probs, value, rng);
                }
            }

            auto & sample = * response.mutable_sample()->add_samples();
            cross_cat.splitter.join(sample, result_diffs);
        }
    }
}

void QueryServer::estimate_entropy (
        rng_t & rng,
        const Request & request,
        Response & response)
{
    Timer::Scope timer(timer_);

    if (not schema().is_valid(request.entropy().conditional())) {
        response.add_error("invalid request.entropy.conditional");
        return;
    }
    for (auto id : request.entropy().conditional().tares()) {
        if (id >= tares().size()) {
            response.add_error("invalid request.entropy.conditional.tares");
            return;
        }
    }
    for (const auto & feature_set : request.entropy().feature_sets()) {
        if (not schema().is_valid(feature_set)) {
            response.add_error("invalid request.entropy.feature_sets");
            return;
        }
    }
    if (request.entropy().sample_count() <= 1) {
        response.add_error("invalid request.entropy.sample_count");
        return;
    }

    TODO("sample rows");
    TODO("for each feature_set: entropy = mean(score(s) for s in samples)");
    rng();  // pacify gcc
}

} // namespace loom
