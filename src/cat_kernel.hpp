#pragma once

#include <thread>
#include <loom/cross_cat.hpp>
#include <loom/assignments.hpp>
#include <loom/shared_queue.hpp>
#include <loom/timer.hpp>
#include <loom/logger.hpp>

namespace loom
{

using ::distributions::sample_from_scores_overwrite;

class CatKernel : noncopyable
{
public:

    typedef ProductModel::Value Value;

    CatKernel (
            const protobuf::Config::Kernels::Cat & config,
            CrossCat & cross_cat);

    ~CatKernel ();

    void add_row_noassign (
            rng_t & rng,
            const protobuf::SparseRow & row);

    void add_row (
            rng_t & rng,
            const protobuf::SparseRow & row,
            protobuf::Assignment & assignment_out);

    bool try_add_row (
            rng_t & rng,
            const protobuf::SparseRow & row,
            Assignments & assignments);

    void remove_row (
            rng_t & rng,
            const protobuf::SparseRow & row,
            Assignments & assignments);

    void wait (Assignments & assignments, rng_t & rng);

    void log_metrics (Logger::Message & message);

private:

    typedef Assignments::Queue<Assignments::Value> Groupids;

    struct Task
    {
        enum Action { add, remove, exit };
        Action action;
        std::vector<Value> partial_values;
    };

    void process_tasks (
            const size_t kindid,
            size_t consumer_position,
            rng_t::result_type seed,
            Assignments * assignments);

    void process_add_task (
            CrossCat::Kind & kind,
            const Value & partial_value,
            VectorFloat & scores,
            Groupids & groupids,
            rng_t & rng);

    void process_remove_task (
            CrossCat::Kind & kind,
            const Value & partial_value,
            Groupids & groupids,
            rng_t & rng);

    CrossCat & cross_cat_;
    pipeline::SharedQueue<Task> task_queue_;
    std::vector<std::thread> workers_;
    std::vector<Value> partial_values_;
    VectorFloat scores_;
    Timer timer_;
};

inline void CatKernel::log_metrics (Logger::Message & message)
{
    auto & status = * message.mutable_kernel_status()->mutable_cat();
    status.set_total_time(timer_.total());
    timer_.clear();
}

inline void CatKernel::add_row_noassign (
        rng_t & rng,
        const protobuf::SparseRow & row)
{
    Timer::Scope timer(timer_);
    cross_cat_.value_split(row.data(), partial_values_);

    const size_t kind_count = cross_cat_.kinds.size();
    for (size_t i = 0; i < kind_count; ++i) {
        const Value & partial_value = partial_values_[i];
        auto & kind = cross_cat_.kinds[i];
        const ProductModel & model = kind.model;
        auto & mixture = kind.mixture;

        mixture.score_value(model, partial_value, scores_, rng);
        size_t groupid = sample_from_scores_overwrite(rng, scores_);
        mixture.add_value(model, groupid, partial_value, rng);
    }
}

inline void CatKernel::add_row (
        rng_t & rng,
        const protobuf::SparseRow & row,
        protobuf::Assignment & assignment_out)
{
    Timer::Scope timer(timer_);
    cross_cat_.value_split(row.data(), partial_values_);
    assignment_out.set_rowid(row.id());
    assignment_out.clear_groupids();

    const size_t kind_count = cross_cat_.kinds.size();
    for (size_t i = 0; i < kind_count; ++i) {
        const Value & partial_value = partial_values_[i];
        auto & kind = cross_cat_.kinds[i];
        const ProductModel & model = kind.model;
        auto & mixture = kind.mixture;

        mixture.score_value(model, partial_value, scores_, rng);
        size_t groupid = sample_from_scores_overwrite(rng, scores_);
        mixture.add_value(model, groupid, partial_value, rng);
        assignment_out.add_groupids(groupid);
    }
}

inline bool CatKernel::try_add_row (
        rng_t & rng,
        const protobuf::SparseRow & row,
        Assignments & assignments)
{
    Timer::Scope timer(timer_);
    bool already_added = not assignments.rowids().try_push(row.id());
    if (LOOM_UNLIKELY(already_added)) {
        return false;
    }

    if (workers_.empty()) {

        cross_cat_.value_split(row.data(), partial_values_);
        const size_t kind_count = cross_cat_.kinds.size();
        for (size_t i = 0; i < kind_count; ++i) {
            process_add_task(
                cross_cat_.kinds[i],
                partial_values_[i],
                scores_,
                assignments.groupids(i),
                rng);
        }

    } else {

        task_queue_.produce([&](Task & task){
            task.action = Task::add;
            cross_cat_.value_split(row.data(), task.partial_values);
        });
    }

    return true;
}

inline void CatKernel::process_add_task (
        CrossCat::Kind & kind,
        const Value & partial_value,
        VectorFloat & scores,
        Groupids & groupids,
        rng_t & rng)
{
    const ProductModel & model = kind.model;
    auto & mixture = kind.mixture;

    mixture.score_value(model, partial_value, scores, rng);
    size_t groupid = sample_from_scores_overwrite(rng, scores);
    mixture.add_value(model, groupid, partial_value, rng);
    size_t global_groupid = mixture.id_tracker.packed_to_global(groupid);
    groupids.push(global_groupid);
}

inline void CatKernel::remove_row (
        rng_t & rng,
        const protobuf::SparseRow & row,
        Assignments & assignments)
{
    Timer::Scope timer(timer_);
    const auto rowid = assignments.rowids().pop();
    if (LOOM_DEBUG_LEVEL >= 1) {
        LOOM_ASSERT_EQ(rowid, row.id());
    }

    if (workers_.empty()) {

        cross_cat_.value_split(row.data(), partial_values_);
        const size_t kind_count = cross_cat_.kinds.size();
        for (size_t i = 0; i < kind_count; ++i) {
            process_remove_task(
                cross_cat_.kinds[i],
                partial_values_[i],
                assignments.groupids(i),
                rng);
        }

    } else {

        task_queue_.produce([&](Task & task){
            task.action = Task::remove;
            cross_cat_.value_split(row.data(), task.partial_values);
        });
    }
}

inline void CatKernel::process_remove_task (
        CrossCat::Kind & kind,
        const Value & partial_value,
        Groupids & groupids,
        rng_t & rng)
{
    const ProductModel & model = kind.model;
    auto & mixture = kind.mixture;

    auto global_groupid = groupids.pop();
    auto groupid = mixture.id_tracker.global_to_packed(global_groupid);
    mixture.remove_value(model, groupid, partial_value, rng);
}

} // namespace loom
