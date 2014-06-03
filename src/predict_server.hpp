#pragma once

#include <loom/cross_cat.hpp>

namespace loom
{

class PredictServer
{
public:

    typedef protobuf::PreQL::Predict::Query Query;
    typedef protobuf::PreQL::Predict::Result Result;
    typedef CrossCat::Value Value;

    PredictServer (const CrossCat & cross_cat) :
        cross_cat_(cross_cat),
        value_join_(cross_cat),
        partial_values_(),
        scores_(),
        timer_()
    {
    }

    void predict_row (
            rng_t & rng,
            const Query & query,
            Result & result);

private:

    const CrossCat & cross_cat_;
    CrossCat::ValueJoiner value_join_;
    std::vector<Value> partial_values_;
    VectorFloat scores_;
    Timer timer_;
};

inline void PredictServer::predict_row (
        rng_t & rng,
        const Query & query,
        Result & result)
{
    Timer::Scope timer(timer_);

    result.Clear();
    result.set_id(query.id());
    if (not cross_cat_.schema.is_valid(query.data())) {
        result.set_error("invalid query data");
        return;
    }
    if (query.data().observed_size() != query.to_predict_size()) {
        result.set_error("observed size != to_predict size");
        return;
    }
    const size_t sample_count = query.sample_count();
    if (sample_count == 0) {
        return;
    }

    cross_cat_.value_split(query.data(), partial_values_);
    std::vector<std::vector<Value>> result_factors(1);
    {
        Value sample;
        * sample.mutable_observed() = query.to_predict();
        cross_cat_.value_split_observed(sample, result_factors[0]);
        result_factors.resize(sample_count, result_factors[0]);
    }

    const size_t kind_count = cross_cat_.kinds.size();
    for (size_t i = 0; i < kind_count; ++i) {
        const Value & value = partial_values_[i];
        auto & kind = cross_cat_.kinds[i];
        const ProductModel & model = kind.model;
        auto & mixture = kind.mixture;

        mixture.score_value(model, value, scores_, rng);
        distributions::scores_to_probs(scores_);
        const VectorFloat & probs = scores_;

        for (auto & result_values : result_factors) {
            mixture.sample_value(model, probs, result_values[i], rng);
        }
    }

    for (const auto & result_values : result_factors) {
        value_join_(* result.add_samples(), result_values);
    }
}

} // namespace loom
