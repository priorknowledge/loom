#pragma once

#include <thread>
#include <vector>
#include <unordered_map>
#include <loom/common.hpp>
#include <loom/cross_cat.hpp>
#include <loom/assignments.hpp>
#include <loom/timer.hpp>
#include <loom/logger.hpp>

namespace loom
{

class Loom : noncopyable
{
public:

    typedef ProductModel::Value Value;

    Loom (
            rng_t & rng,
            const protobuf::Config & config,
            const char * model_in,
            const char * groups_in = nullptr,
            const char * assign_in = nullptr);

    void dump (
            const char * model_out = nullptr,
            const char * groups_out = nullptr,
            const char * assign_out = nullptr) const;

    void infer_single_pass (
            rng_t & rng,
            const char * rows_in,
            const char * assign_out = nullptr);

    void infer_multi_pass (
            rng_t & rng,
            const char * rows_in);

    void posterior_enum (
            rng_t & rng,
            const char * rows_in,
            const char * samples_out);

    void predict (
            rng_t & rng,
            const char * queries_in,
            const char * results_out);

private:

    void log_metrics (Logger::Message & message);

    void dump_posterior_enum (
            protobuf::PosteriorEnum::Sample & message,
            rng_t & rng);

    const protobuf::Config & config_;
    CrossCat cross_cat_;
    Assignments assignments_;
};

} // namespace loom
