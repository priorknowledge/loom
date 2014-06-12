#pragma once

#include <loom/cross_cat.hpp>

namespace loom
{

class QueryServer
{
public:

    typedef protobuf::Query::Request Request;
    typedef protobuf::Query::Response Response;
    typedef CrossCat::Value Value;

    QueryServer (const CrossCat & cross_cat) :
        cross_cat_(cross_cat),
        value_join_(cross_cat),
        to_sample_(),
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
    CrossCat::ValueJoiner value_join_;
    Value to_sample_;
    std::vector<Value> partial_values_;
    std::vector<std::vector<Value>> result_factors_;
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
        const Value & value = partial_values_[i];
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
        response.set_error("invalid request data");
        return;
    }
    if (request.sample().data().observed_size() != request.sample().to_sample_size()) {
        response.set_error("observed size != to_sample size");
        return;
    }
    const size_t sample_count = request.sample().sample_count();
    if (sample_count == 0) {
        return;
    }

    cross_cat_.value_split(request.sample().data(), partial_values_);
    * to_sample_.mutable_observed() = request.sample().to_sample();
    result_factors_.resize(sample_count);
    cross_cat_.value_split_observed(to_sample_, result_factors_.front());
    std::fill(
        result_factors_.begin() + 1,
        result_factors_.end(),
        result_factors_.front());

    const size_t kind_count = cross_cat_.kinds.size();
    for (size_t i = 0; i < kind_count; ++i) {
        if (cross_cat_.schema.observed_count(result_factors_.front()[i])) {
            const Value & value = partial_values_[i];
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
        value_join_(sample, result_values);
    }
}

} // namespace loom
