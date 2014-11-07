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
#include <loom/compressed_vector.hpp>
#include <loom/scorer.hpp>
#include <loom/cat_kernel.hpp>

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
        if (request.has_score_derivative() and
            validate(request.score_derivative(), errors))
        {
            call(
                rng,
                request.score_derivative(),
                * response.mutable_score_derivative());
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
        Query::Sample::Response & response) const
{
    const size_t latent_count = cross_cats_.size();
    std::vector<std::vector<VectorFloat>> latent_kind_scores(latent_count);
    VectorFloat latent_scores(latent_count, 0.f);
    {
        std::vector<ProductValue::Diff> conditional_diffs;
        for (size_t l = 0; l < latent_count; ++l) {
            const auto & cross_cat = * cross_cats_[l];
            auto & kind_scores = latent_kind_scores[l];
            cross_cat.splitter.split(request.data(), conditional_diffs);

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
            cross_cat.splitter.split(blank, result_diffs);

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
        Query::Score::Response & response) const
{
    // not freed
    static thread_local std::vector<ProductValue::Diff> *
    partial_diffs = nullptr;
    static thread_local VectorFloat * scores = nullptr;
    construct_if_null(partial_diffs);
    construct_if_null(scores);

    const auto NONE = ProductValue::Observed::NONE;
    VectorFloat latent_scores(cross_cats_.size(), 0.f);
    const size_t latent_count = cross_cats_.size();
    for (size_t l = 0; l < latent_count; ++l) {
        const auto & cross_cat = * cross_cats_[l];
        float & score = latent_scores[l];

        cross_cat.splitter.split(request.data(), *partial_diffs);

        const size_t kind_count = cross_cat.kinds.size();
        for (size_t k = 0; k < kind_count; ++k) {
            ProductValue::Diff & diff = (*partial_diffs)[k];
            cross_cat.splitter.schema(k).normalize_small(diff);
            auto & kind = cross_cat.kinds[k];
            const ProductModel & model = kind.model;
            auto & mixture = kind.mixture;

            if (diff.tares_size()) {
                mixture.score_diff(model, diff, *scores, rng);
                score += distributions::log_sum_exp(*scores);
            } else if (diff.pos().observed().sparsity() != NONE) {
                mixture.score_value(model, diff.pos(), *scores, rng);
                score += distributions::log_sum_exp(*scores);
            }
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
    for (const auto & feature_set : request.row_sets()) {
        if (not schema().is_valid(feature_set)) {
            * errors.Add() = "invalid request.entropy.row_sets";
            return false;
        }
    }
    for (const auto & feature_set : request.col_sets()) {
        if (not schema().is_valid(feature_set)) {
            * errors.Add() = "invalid request.entropy.col_sets";
            return false;
        }
    }
    if (request.sample_count() <= 1) {
        * errors.Add() = "invalid request.entropy.sample_count";
        return false;
    }

    return true;
}

namespace
{
class Accum
{
    typedef distributions::NormalInverseChiSq::Shared Shared;
    typedef distributions::NormalInverseChiSq::Group Group;

    Group group_;

public:

    Accum ()
    {
        static rng_t rng;
        static Shared shared;
        group_.init(shared, rng);
    }

    void add (float x)
    {
        static rng_t rng;
        static Shared shared;
        group_.add_value(shared, x, rng);
    }

    float mean () const
    {
        return group_.mean;
    }

    float variance () const
    {
        return group_.count_times_variance / (group_.count - 1);
    }
};
} // anonymous namespace

void QueryServer::call (
        rng_t & rng,
        const Query::Entropy::Request & request,
        Query::Entropy::Response & response) const
{
    Query::Sample::Request sample_request;
    Query::Sample::Response sample_response;
    Errors errors;

    * sample_request.mutable_data() = request.conditional();
    sample_request.set_sample_count(request.sample_count());
    auto & to_sample = * sample_request.mutable_to_sample();
    schema().clear(to_sample);
    schema().normalize_dense(to_sample);
    for (const auto & feature_set : request.row_sets()) {
        schema().for_each(feature_set, [&](int i){
            to_sample.set_dense(i, true);
        });
    }
    for (const auto & feature_set : request.col_sets()) {
        schema().for_each(feature_set, [&](int i){
            to_sample.set_dense(i, true);
        });
    }
    LOOM_ASSERT1(validate(sample_request, errors), errors);
    call(rng, sample_request, sample_response);

    Query::Score::Request score_request;
    Query::Score::Response score_response;
    * score_request.mutable_data() = request.conditional();
    LOOM_ASSERT1(validate(score_request, errors), errors);
    call(rng, score_request, score_response);
    const float base_score = score_response.score();

    const size_t row_count = request.row_sets_size();
    const size_t col_count = request.col_sets_size();
    const size_t cell_count = row_count * col_count;
    const size_t latent_count = cross_cats_.size();

    std::vector<RestrictionScorer *> scorers(latent_count, nullptr);
    for (size_t l = 0; l < latent_count; ++l) {
        scorers[l] = new RestrictionScorer(
            *cross_cats_[l],
            request.conditional(),
            rng);
    }
    const float score_shift =
        distributions::fast_log(latent_count) + base_score;

    CompressedVector<ProductValue::Observed> tasks;
    ProductValue::Observed union_set;
    for (size_t i = 0; i < row_count; ++i) {
        ProductValue::Observed row_set = request.row_sets(i);
        schema().normalize_dense(row_set);
        for (size_t j = 0; j < col_count; ++j) {
            union_set = row_set;
            schema().for_each(request.col_sets(j), [&](size_t f){
                union_set.set_dense(f, true);
            });
            schema().normalize_small(union_set);
            tasks.push_back(union_set);
        }
    }
    tasks.init_index();

    const size_t task_count = tasks.unique_count();
    for (size_t t = 0; t < task_count; ++t) {
        tasks.unique_value(t, union_set);
        for (size_t l = 0; l < latent_count; ++l) {
            scorers[l]->add_restriction(union_set);
        }
    }

    std::vector<Accum> accums(task_count);
    #pragma omp parallel if(config_.query().parallel())
    {
        VectorFloat scores(latent_count);
        for (const auto & sample : sample_response.samples()) {

            #pragma omp barrier
            #pragma omp for
            for (size_t l = 0; l < latent_count; ++l) {
                scorers[l]->set_value(sample.pos(), rng);
            }

            #pragma omp barrier
            #pragma omp for
            for (size_t t = 0; t < task_count; ++t) {
                for (size_t l = 0; l < latent_count; ++l) {
                    scores[l] = scorers[l]->get_score(t);
                }
                float score = score_shift - distributions::log_sum_exp(scores);

                // FIXME this should be atomic
                accums[t].add(score);
            }
        }
    }

    for (auto scorer : scorers) {
        delete scorer;
    }
    for (size_t i = 0; i < cell_count; ++i) {
        const Accum & accum = accums[tasks.unique_id(i)];
        response.add_means(accum.mean());
        response.add_variances(accum.variance() / request.sample_count());
    }
}

bool QueryServer::validate (
        const Query::ScoreDerivative::Request & request,
        Errors & errors) const
{
    if (not schema().is_valid(request.update_data())) {
        * errors.Add() = "invalid request.score_derivative.update_data";
        return false;
    }
    for (auto id : request.update_data().tares()) {
        if (id >= tares().size()) {
            * errors.Add() = "invalid request.score_derivative.update_data";
            return false;
        }
    }
    for (const ProductValue::Diff & score_data : request.score_data()) {
        if (not schema().is_valid(score_data)) {
            * errors.Add() = "invalid request.score_derivative.score_data";
            return false;
        }
        for (auto id : score_data.tares()) {
            if (id >= tares().size()) {
                * errors.Add() = "invalid request.score_derivative.score_data";
                return false;
            }
        }
    }

    return true;
}

class QueryServer::Scorer
{
    const QueryServer & server_;
    Query::Score::Request request_;
    Query::Score::Response response_;
    rng_t & rng_;

public:

    Scorer (const QueryServer * server, rng_t & rng) :
        server_(* server),
        rng_(rng)
    {
    }

    float operator() (const ProductValue::Diff & diff)
    {
        * request_.mutable_data() = diff;
        server_.call(rng_, request_, response_);
        return response_.score();
    }

    float operator() (const protobuf::Row & row)
    {
        return operator()(row.diff());
    }
};

class QueryServer::AssignmentHypothesis
{
    CatKernel cat_kernel_;
    const protobuf::Row & row_;
    rng_t & rng_;
    protobuf::Assignment assignment_;

public:

    AssignmentHypothesis (
            const protobuf::Config & config,
            const CrossCat * cross_cat,
            const protobuf::Row & row,
            rng_t & rng) :
        cat_kernel_(config.kernels().cat(), * const_cast<CrossCat*>(cross_cat)),
        row_(row),
        rng_(rng),
        assignment_()
    {
        cat_kernel_.add_row(rng_, row_, assignment_);
    }

    ~AssignmentHypothesis ()
    {
        cat_kernel_.remove_row(rng_, row_, assignment_);
    }
};

// not threadsafe
void QueryServer::call (
        rng_t & rng,
        const Query::ScoreDerivative::Request & request,
        Query::ScoreDerivative::Response & response) const
{
    const size_t row_count =
        cross_cats_[0]->kinds[0].mixture.clustering.sample_size();

    protobuf::Row row;
    Scorer get_score(this, rng);

    protobuf::Row update_row;
    update_row.set_id(0);
    * update_row.mutable_diff() = request.update_data();

    typedef std::pair<int, float> ScoreDiff;
    std::vector<ScoreDiff> score_diffs;

    if (request.score_data_size() == 0) {
        protobuf::InFile all_rows(rows_in_);
        while (all_rows.try_read_stream(row)) {
            score_diffs.push_back(ScoreDiff(row.id(), -get_score(row)));
        }
    } else {
        for (size_t i = 0; i < request.score_data_size(); ++i) {
            float score = get_score(request.score_data(i));
            score_diffs.push_back(ScoreDiff(i, -score));
        }
    }

    std::vector<AssignmentHypothesis *> hypotheses;
    for (const auto * cross_cat : cross_cats_) {
        hypotheses.push_back(
            new AssignmentHypothesis(config_, cross_cat, update_row, rng));
    }

    if (request.score_data_size() == 0) {
        protobuf::InFile all_rows(rows_in_);
        for (size_t i = 0; all_rows.try_read_stream(row); ++i) {
            score_diffs[i].second += get_score(row);
            score_diffs[i].second *= row_count;
        }
    } else {
        for (size_t i = 0; i < request.score_data_size(); ++i) {
            score_diffs[i].second += get_score(request.score_data(i));
            score_diffs[i].second *= row_count;
        }
    }

    for (auto hypothesis : hypotheses) {
        delete hypothesis;
    }

    std::sort(score_diffs.begin(), score_diffs.end(),
            [](const ScoreDiff & a, const ScoreDiff & b) {
                return a.second > b.second;
            });
    if (score_diffs.size() > request.row_limit()) {
        score_diffs.resize(request.row_limit());
    }

    for (const auto & score_diff : score_diffs) {
        response.add_ids(score_diff.first);
        response.add_score_diffs(score_diff.second);
    }
}

} // namespace loom
