#pragma once

#include <thread>
#include <vector>
#include <unordered_map>
#include "common.hpp"
#include "cross_cat.hpp"
#include "algorithm8.hpp"
#include "assignments.hpp"
#include "message_queue.hpp"
#include "timer.hpp"
#include "logger.hpp"

namespace loom
{

class Loom : noncopyable
{
    class Algorithm8Kernel;

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

    void log_iter_metrics (
            size_t iter,
            Algorithm8Kernel * kernel = nullptr);

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

    void validate_cross_cat () const;
    void validate_algorithm8 () const;
    void validate () const
    {
        validate_cross_cat();
        validate_algorithm8();
    }

    size_t count_untracked_rows () const;

private:

    void resize_kinds ();

    void dump_posterior_enum (
            protobuf::PosteriorEnum::Sample & message,
            rng_t & rng);

    void prepare_algorithm8 (rng_t & rng);

    size_t run_algorithm8 (
            bool init_cache,
            rng_t & rng);

    void cleanup_algorithm8 (rng_t & rng);

    void resize_algorithm8 (rng_t & rng);

    void run_hyper_inference (rng_t & rng);

    void add_featureless_kind (rng_t & rng);

    void remove_featureless_kind (size_t kindid);

    void init_featureless_kinds (
            size_t featureless_kind_count,
            rng_t & rng);

    void move_feature_to_kind (
            size_t featureid,
            size_t new_kindid,
            bool init_cache,
            rng_t & rng);

    void add_row_noassign (
            rng_t & rng,
            const protobuf::SparseRow & row);

    void add_row (
            rng_t & rng,
            const protobuf::SparseRow & row,
            protobuf::Assignment & assignment_out);

    bool try_add_row (
            rng_t & rng,
            const protobuf::SparseRow & row);

    bool try_add_row_algorithm8 (
            rng_t & rng,
            const protobuf::SparseRow & row);

    void remove_row (
            rng_t & rng,
            const protobuf::SparseRow & row);

    void remove_row_algorithm8 (
            rng_t & rng,
            const protobuf::SparseRow & row);

    void predict_row (
            rng_t & rng,
            const protobuf::PreQL::Predict::Query & query,
            protobuf::PreQL::Predict::Result & result);

    struct Algorithm8Task
    {
        std::vector<Value> partial_values;
        Value full_value;
        bool next_action_is_add;
    };

    void algorithm8_work (
            const size_t kindid,
            rng_t::result_type seed);

    void algorithm8_work_add (
            size_t kindid,
            const Value & partial_value,
            const Value & full_value,
            rng_t & rng);

    void algorithm8_work_remove (
            size_t kindid,
            const Value & partial_value,
            rng_t & rng);

    void infer_hypers (rng_t & rng);

    const protobuf::Config & config_;
    CrossCat cross_cat_;
    Algorithm8 algorithm8_;
    Assignments assignments_;
    CrossCat::ValueJoiner value_join_;
    Value unobserved_;
    std::vector<Value> partial_values_;
    VectorFloat scores_;
    ParallelQueue<Algorithm8Task> algorithm8_queues_;
    std::vector<std::thread> algorithm8_workers_;
    std::unordered_map<std::string, Timer> timers_;
};

inline void Loom::validate_cross_cat () const
{
    cross_cat_.validate();
    assignments_.validate();
    const size_t kind_count = cross_cat_.kinds.size();
    LOOM_ASSERT_EQ(assignments_.kind_count(), kind_count);
    LOOM_ASSERT_EQ(partial_values_.size(), kind_count);
}

inline void Loom::validate_algorithm8 () const
{
    auto config = config_.kernels().kind();
    algorithm8_.validate(cross_cat_);
    if (config.row_queue_size() and not algorithm8_.kinds.empty()) {
        LOOM_ASSERT_EQ(algorithm8_workers_.size(), algorithm8_queues_.size());
        LOOM_ASSERT_LE(cross_cat_.kinds.size(), algorithm8_queues_.size());
        algorithm8_queues_.assert_ready();
    }
}

} // namespace loom
