#pragma once

#include <thread>
#include <loom/cross_cat.hpp>
#include <loom/assignments.hpp>
#include <loom/timer.hpp>
#include <loom/logger.hpp>

namespace loom
{

using ::distributions::sample_from_scores_overwrite;

class CatKernel : noncopyable
{
public:

    typedef ProductModel::Value Value;
    typedef Assignments::Queue<Assignments::Value> Groupids;

    CatKernel (
            const protobuf::Config::Kernels::Cat & config,
            CrossCat & cross_cat) :
        cross_cat_(cross_cat),
        partial_values_(),
        scores_(),
        timer_()
    {
        LOOM_ASSERT_LT(0, config.empty_group_count());
    }

    void add_row_noassign (
            rng_t & rng,
            const protobuf::Row & row);

    void add_row (
            rng_t & rng,
            const protobuf::Row & row,
            protobuf::Row & assignment_out);

    void add_row (
            rng_t & rng,
            const protobuf::Row & row,
            Assignments & assignments);

    void process_add_task (
            CrossCat::Kind & kind,
            const Value & partial_value,
            VectorFloat & scores,
            Groupids & groupids,
            rng_t & rng);

    void remove_row (
            rng_t & rng,
            const protobuf::Row & row,
            Assignments & assignments);

    void process_remove_task (
            CrossCat::Kind & kind,
            const Value & partial_value,
            Groupids & groupids,
            rng_t & rng);

    void log_metrics (Logger::Message & message);

private:

    CrossCat & cross_cat_;
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
        const protobuf::Row & row)
{
    Timer::Scope timer(timer_);
    cross_cat_.value_split(row.data(), partial_values_);

    const size_t kind_count = cross_cat_.kinds.size();
    for (size_t i = 0; i < kind_count; ++i) {
        const Value & partial_value = partial_values_[i];
        auto & kind = cross_cat_.kinds[i];
        ProductModel & model = kind.model;
        auto & mixture = kind.mixture;

        model.add_value(partial_value, rng);
        mixture.score_value(model, partial_value, scores_, rng);
        size_t groupid = sample_from_scores_overwrite(rng, scores_);
        mixture.add_value(model, groupid, partial_value, rng);
    }
}

inline void CatKernel::add_row (
        rng_t & rng,
        const protobuf::Row & row,
        protobuf::Row & assignment_out)
{
    Timer::Scope timer(timer_);
    cross_cat_.value_split(row.data(), partial_values_);
    assignment_out.set_id(row.id());
    auto & groupids_out = * assignment_out.mutable_data()->mutable_counts();
    groupids_out.Clear();

    const size_t kind_count = cross_cat_.kinds.size();
    for (size_t i = 0; i < kind_count; ++i) {
        const Value & partial_value = partial_values_[i];
        auto & kind = cross_cat_.kinds[i];
        ProductModel & model = kind.model;
        auto & mixture = kind.mixture;

        model.add_value(partial_value, rng);
        mixture.score_value(model, partial_value, scores_, rng);
        size_t groupid = sample_from_scores_overwrite(rng, scores_);
        mixture.add_value(model, groupid, partial_value, rng);
        groupids_out.Add(groupid);
    }
}

inline void CatKernel::add_row (
        rng_t & rng,
        const protobuf::Row & row,
        Assignments & assignments)
{
    Timer::Scope timer(timer_);
    bool ok = assignments.rowids().try_push(row.id());
    LOOM_ASSERT1(ok, "duplicate row: " << row.id());

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
}

inline void CatKernel::process_add_task (
        CrossCat::Kind & kind,
        const Value & partial_value,
        VectorFloat & scores,
        Groupids & groupids,
        rng_t & rng)
{
    ProductModel & model = kind.model;
    auto & mixture = kind.mixture;

    model.add_value(partial_value, rng);
    mixture.score_value(model, partial_value, scores, rng);
    size_t groupid = sample_from_scores_overwrite(rng, scores);
    mixture.add_value(model, groupid, partial_value, rng);
    size_t global_groupid = mixture.id_tracker.packed_to_global(groupid);
    groupids.push(global_groupid);
}

inline void CatKernel::remove_row (
        rng_t & rng,
        const protobuf::Row & row,
        Assignments & assignments)
{
    Timer::Scope timer(timer_);
    const auto rowid = assignments.rowids().pop();
    if (LOOM_DEBUG_LEVEL >= 1) {
        LOOM_ASSERT_EQ(rowid, row.id());
    }

    cross_cat_.value_split(row.data(), partial_values_);
    const size_t kind_count = cross_cat_.kinds.size();
    for (size_t i = 0; i < kind_count; ++i) {
        process_remove_task(
            cross_cat_.kinds[i],
            partial_values_[i],
            assignments.groupids(i),
            rng);
    }
}

inline void CatKernel::process_remove_task (
        CrossCat::Kind & kind,
        const Value & partial_value,
        Groupids & groupids,
        rng_t & rng)
{
    ProductModel & model = kind.model;
    auto & mixture = kind.mixture;

    auto global_groupid = groupids.pop();
    auto groupid = mixture.id_tracker.global_to_packed(global_groupid);
    mixture.remove_value(model, groupid, partial_value, rng);
    model.remove_value(partial_value, rng);
}

} // namespace loom
