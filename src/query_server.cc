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
        Timer::Scope timer(timer_);
        response.Clear();
        response.set_id(request.id());
        Errors & errors = * response.mutable_error();
        if (request.has_sample() and validate(request.sample(), errors)) {
            call(rng, request.sample(), * response.mutable_sample());
        }
        if (request.has_score() and validate(request.score(), errors)) {
            call(rng, request.score(), * response.mutable_score());
        }
        if (request.has_entropy() and validate(request.entropy(), errors)) {
            call(rng, request.entropy(), * response.mutable_entropy());
        }
        response_stream.write_stream(response);
        response_stream.flush();
    }
}

bool QueryServer::validate (
        const Query::Sample::Request & request,
        Errors & errors) const
{
    if (not schema().is_valid(request.data())) {
        * errors.Add() = "invalid request.sample.data";
        return false;
    }
    for (auto id : request.data().tares()) {
        if (id >= tares().size()) {
            * errors.Add() = "invalid request.sample.data.tares";
            return false;
        }
    }
    if (not schema().is_valid(request.to_sample())) {
        * errors.Add() = "invalid request.sample.to_sample";
        return false;
    }

    return true;
}

void QueryServer::call (
        rng_t & rng,
        const Query::Sample::Request & request,
        Query::Sample::Response & response)
{
    const size_t latent_count = cross_cats_.size();
    std::vector<std::vector<VectorFloat>> latent_kind_scores(latent_count);
    VectorFloat latent_scores(latent_count, 0.f);
    {
        std::vector<ProductValue::Diff> conditional_diffs;
        for (size_t l = 0; l < latent_count; ++l) {
            const auto & cross_cat = * cross_cats_[l];
            auto & kind_scores = latent_kind_scores[l];
            cross_cat.splitter.split(
                request.data(),
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

    const size_t sample_count = request.sample_count();
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
    * blank.mutable_pos()->mutable_observed() = request.to_sample();
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

            auto & sample = * response.add_samples();
            cross_cat.splitter.join(sample, result_diffs);
        }
    }
}

bool QueryServer::validate (
        const Query::Score::Request & request,
        Errors & errors) const
{
    if (not schema().is_valid(request.data())) {
        * errors.Add() = "invalid request.score.data";
        return false;
    }
    for (auto id : request.data().tares()) {
        if (id >= tares().size()) {
            * errors.Add() = "invalid request.score.data.tares";
            return false;
        }
    }

    return true;
}

void QueryServer::call (
        rng_t & rng,
        const Query::Score::Request & request,
        Query::Score::Response & response)
{
    VectorFloat latent_scores(cross_cats_.size(), 0.f);
    const size_t latent_count = cross_cats_.size();
    for (size_t l = 0; l < latent_count; ++l) {
        const auto & cross_cat = * cross_cats_[l];
        float & score = latent_scores[l];

        cross_cat.splitter.split(
            request.data(),
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
    response.set_score(score);
}

bool QueryServer::validate (
        const Query::Entropy::Request & request,
        Errors & errors) const
{
    if (not schema().is_valid(request.conditional())) {
        * errors.Add() = "invalid request.entropy.conditional";
        return false;
    }
    for (auto id : request.conditional().tares()) {
        if (id >= tares().size()) {
            * errors.Add() = "invalid request.entropy.conditional.tares";
            return false;
        }
    }
    for (const auto & feature_set : request.feature_sets()) {
        if (not schema().is_valid(feature_set)) {
            * errors.Add() = "invalid request.entropy.feature_sets";
            return false;
        }
    }
    if (request.sample_count() <= 1) {
        * errors.Add() = "invalid request.entropy.sample_count";
        return false;
    }

    return true;
}

struct QueryServer::Restrictor
{
    enum { undefined = ~uint32_t(0) };

    Restrictor (
            const ValueSchema & schema,
            const ProductValue::Observed & full_observed) :
        schema_(schema),
        end_(),
        packed_(schema.total_size(), undefined)
    {
        end_.booleans_size = schema.booleans_size;
        end_.counts_size = end_.booleans_size + schema.counts_size;
        end_.reals_size = end_.counts_size + schema.reals_size;

        ValueSchema pos;
        schema.for_each(full_observed, [&](size_t absolute){
            if (absolute < end_.booleans_size) {
                packed_[absolute] = pos.booleans_size++;
            } else if (absolute < end_.counts_size) {
                packed_[absolute] = pos.counts_size++;
            } else if (absolute < end_.reals_size) {
                packed_[absolute] = pos.reals_size++;
            }
        });
    }

    void operator() (
        const ProductValue & full_value,
        ProductValue & partial_value) const
    {
        schema_.for_each(partial_value.observed(), [&](size_t absolute){
            size_t packed = packed_[absolute];
            LOOM_ASSERT1(packed != undefined, "undefined pos: " << absolute);
            if (absolute < end_.booleans_size) {
                partial_value.add_booleans(full_value.booleans(packed));
            } else if (absolute < end_.counts_size) {
                partial_value.add_counts(full_value.counts(packed));
            } else if (absolute < end_.reals_size) {
                partial_value.add_reals(full_value.reals(packed));
            }
        });
    }

private:

    ValueSchema schema_;
    ValueSchema end_;
    std::vector<uint32_t> packed_;
};

void QueryServer::call (
        rng_t & rng,
        const Query::Entropy::Request & request,
        Query::Entropy::Response & response)
{
    Query::Sample::Request sample_request;
    Query::Sample::Response sample_response;
    * sample_request.mutable_data() = request.conditional();
    sample_request.set_sample_count(request.sample_count());
    auto & to_sample = * sample_request.mutable_to_sample();
    schema().clear(to_sample);
    schema().normalize_dense(to_sample);
    for (const auto & feature_set : request.feature_sets()) {
        schema().for_each(feature_set, [&](int i){
            to_sample.set_dense(i, true);
        });
    }
    call(rng, sample_request, sample_response);

    Query::Score::Request score_request;
    Query::Score::Response score_response;
    ProductValue & partial_value =
        * score_request.mutable_data()->mutable_pos();
    VectorFloat scores;
    const Restrictor restrict(schema(), to_sample);
    for (const auto & feature_set : request.feature_sets()) {
        scores.clear();
        * partial_value.mutable_observed() = feature_set;
        for (const auto & full_sample : sample_response.samples()) {
            partial_value.clear_booleans();
            partial_value.clear_counts();
            partial_value.clear_reals();
            restrict(full_sample.pos(), partial_value);
            call(rng, score_request, score_response);
            scores.push_back(score_response.score());
        }
        float mean = 0;
        for (float score : scores) {
            mean += score;
        }
        mean /= scores.size();
        float variance = 0;
        for (float score : scores) {
            variance += (score - mean) * (score - mean);
        }
        variance /= scores.size() - 1;
        response.add_means(mean);
        response.add_variances(variance);
    }
}

} // namespace loom
