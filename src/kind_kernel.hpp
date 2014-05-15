#pragma once

#include <thread>
#include <loom/cross_cat.hpp>
#include <loom/algorithm8.hpp>
#include <loom/assignments.hpp>
#include <loom/message_queue.hpp>
#include <loom/timer.hpp>
#include <loom/logger.hpp>

namespace loom
{

class KindKernel : noncopyable
{
public:

    typedef ProductModel::Value Value;

    KindKernel (
            const protobuf::Config::Kernels & config,
            CrossCat & cross_cat,
            Assignments & assignments,
            rng_t::result_type seed);

    ~KindKernel ();

    bool try_add_row (const protobuf::SparseRow & row);
    void remove_row (const protobuf::SparseRow & row);
    bool try_run ();
    void update_hypers () { algorithm8_.model_update(cross_cat_); }
    void validate () const;
    void log_metrics (Logger::Message & message);

private:

    void add_featureless_kind ();
    void remove_featureless_kind (size_t kindid);
    void init_featureless_kinds (size_t featureless_kind_count);

    void move_feature_to_kind (
            size_t featureid,
            size_t new_kindid);

    void resize_worker_pool ();

    struct Task
    {
        std::vector<Value> partial_values;
        Value full_value;
        bool next_action_is_add;
    };

    void process_tasks (
            const size_t kindid,
            rng_t::result_type seed);

    void process_add_task (
            size_t kindid,
            const Value & partial_value,
            const Value & full_value,
            VectorFloat & scores,
            rng_t & rng);

    void process_remove_task (
            size_t kindid,
            const Value & partial_value,
            rng_t & rng);

    const size_t empty_group_count_;
    const size_t empty_kind_count_;
    const size_t iterations_;
    const bool score_parallel_;
    const bool init_cache_;

    CrossCat & cross_cat_;
    Assignments & assignments_;
    Algorithm8 algorithm8_;
    ParallelQueue<Task> queues_;
    std::vector<std::thread> workers_;
    std::vector<Value> partial_values_;
    Value unobserved_;
    VectorFloat scores_;
    rng_t rng_;

    size_t total_count_;
    size_t change_count_;
    size_t birth_count_;
    size_t death_count_;
    usec_t score_time_;
    usec_t sample_time_;
    Timer timer_;
};

inline void KindKernel::validate () const
{
    cross_cat_.validate();
    algorithm8_.validate(cross_cat_);
    assignments_.validate();
    const size_t kind_count = cross_cat_.kinds.size();
    LOOM_ASSERT_EQ(assignments_.kind_count(), kind_count);
    if (queues_.capacity()) {
        LOOM_ASSERT_EQ(workers_.size(), queues_.size());
        LOOM_ASSERT_LE(kind_count, queues_.size());
        queues_.assert_ready();
    }
}

inline void KindKernel::log_metrics (Logger::Message & message)
{
    auto & status = * message.mutable_kernel_status()->mutable_kind();
    status.set_total_count(total_count_);
    status.set_change_count(change_count_);
    status.set_birth_count(birth_count_);
    status.set_death_count(death_count_);
    status.set_score_time(score_time_);
    status.set_sample_time(sample_time_);
    status.set_total_time(timer_.total());
    timer_.clear();
}

//----------------------------------------------------------------------------
// low-level operations

inline bool KindKernel::try_add_row (const protobuf::SparseRow & row)
{
    Timer::Scope timer(timer_);
    bool already_added = not assignments_.rowids().try_push(row.id());
    if (LOOM_UNLIKELY(already_added)) {
        return false;
    }

    LOOM_ASSERT_EQ(cross_cat_.kinds.size(), algorithm8_.kinds.size());
    const size_t kind_count = cross_cat_.kinds.size();

    if (workers_.empty()) {

        const Value & full_value = row.data();
        cross_cat_.value_split(full_value, partial_values_);
        for (size_t i = 0; i < kind_count; ++i) {
            process_add_task(i, partial_values_[i], full_value, scores_, rng_);
        }

    } else {

        auto * envelope = queues_.producer_alloc();
        Task & task = envelope->message;
        task.next_action_is_add = true;
        task.full_value = row.data();
        cross_cat_.value_split(task.full_value, task.partial_values);
        queues_.producer_send(envelope, kind_count);
    }

    return true;
}

inline void KindKernel::process_add_task (
        size_t kindid,
        const Value & partial_value,
        const Value & full_value,
        VectorFloat & scores,
        rng_t & rng)
{
    auto & kind = cross_cat_.kinds[kindid];
    const ProductModel & partial_model = kind.model;
    const ProductModel & full_model = algorithm8_.model;
    auto & partial_mixture = kind.mixture;
    auto & full_mixture = algorithm8_.kinds[kindid].mixture;

    partial_mixture.score_value(partial_model, partial_value, scores, rng);
    size_t groupid = sample_from_scores_overwrite(rng, scores);
    partial_mixture.add_value(partial_model, groupid, partial_value, rng);
    full_mixture.add_value(full_model, groupid, full_value, rng);
    size_t global_groupid =
        partial_mixture.id_tracker.packed_to_global(groupid);
    assignments_.groupids(kindid).push(global_groupid);
}

inline void KindKernel::remove_row (const protobuf::SparseRow & row)
{
    Timer::Scope timer(timer_);
    const auto rowid = assignments_.rowids().pop();
    if (LOOM_DEBUG_LEVEL >= 1) {
        LOOM_ASSERT_EQ(rowid, row.id());
    }

    LOOM_ASSERT_EQ(cross_cat_.kinds.size(), algorithm8_.kinds.size());
    const size_t kind_count = cross_cat_.kinds.size();

    if (workers_.empty()) {

        cross_cat_.value_split(row.data(), partial_values_);
        LOOM_ASSERT_EQ(partial_values_.size(), kind_count); // DEBUG
        for (size_t i = 0; i < kind_count; ++i) {
            process_remove_task(i, partial_values_[i], rng_);
        }

    } else {

        auto * envelope = queues_.producer_alloc();
        Task & task = envelope->message;
        task.next_action_is_add = false;
        cross_cat_.value_split(row.data(), task.partial_values);
        queues_.producer_send(envelope, kind_count);
    }
}

inline void KindKernel::process_remove_task (
        size_t kindid,
        const Value & partial_value,
        rng_t & rng)
{
    auto & kind = cross_cat_.kinds[kindid];
    const ProductModel & partial_model = kind.model;
    auto & partial_mixture = kind.mixture;
    const ProductModel & full_model = algorithm8_.model;
    auto & full_mixture = algorithm8_.kinds[kindid].mixture;

    auto global_groupid = assignments_.groupids(kindid).pop();
    auto groupid = partial_mixture.id_tracker.global_to_packed(global_groupid);
    partial_mixture.remove_value(partial_model, groupid, partial_value, rng);
    full_mixture.remove_value(full_model, groupid, unobserved_, rng);
}

} // namespace loom
