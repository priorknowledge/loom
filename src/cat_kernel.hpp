#pragma once

#include <loom/cross_cat.hpp>
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
            CrossCat & cross_cat) :
        cross_cat_(cross_cat),
        partial_values_(cross_cat.kinds.size())
    {
        LOOM_ASSERT_LT(0, config.empty_group_count());
    }

    void resize () { partial_values_.resize(cross_cat_.kinds.size()); }

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

    void validate ();

    void log_metrics (Logger::Message & message);

private:

    CrossCat & cross_cat_;
    std::vector<Value> partial_values_;
    VectorFloat scores_;
    Timer timer_;
};

inline void CatKernel::validate ()
{
    const size_t kind_count = cross_cat_.kinds.size();
    LOOM_ASSERT_EQ(partial_values_.size(), kind_count);
}

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
        const Value & value = partial_values_[i];
        auto & kind = cross_cat_.kinds[i];
        const ProductModel & model = kind.model;
        auto & mixture = kind.mixture;

        mixture.score_value(model, value, scores_, rng);
        size_t groupid = sample_from_scores_overwrite(rng, scores_);
        mixture.add_value(model, groupid, value, rng);
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
        const Value & value = partial_values_[i];
        auto & kind = cross_cat_.kinds[i];
        const ProductModel & model = kind.model;
        auto & mixture = kind.mixture;

        mixture.score_value(model, value, scores_, rng);
        size_t groupid = sample_from_scores_overwrite(rng, scores_);
        mixture.add_value(model, groupid, value, rng);
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

    cross_cat_.value_split(row.data(), partial_values_);

    const auto seed = rng();
    const size_t kind_count = cross_cat_.kinds.size();
    //#pragma omp parallel
    {
        rng_t rng;
        //#pragma omp for schedule(static)
        for (size_t i = 0; i < kind_count; ++i) {
            rng.seed(seed + i);
            const Value & value = partial_values_[i];
            auto & kind = cross_cat_.kinds[i];
            const ProductModel & model = kind.model;
            auto & mixture = kind.mixture;

            mixture.score_value(model, value, scores_, rng);
            size_t groupid = sample_from_scores_overwrite(rng, scores_);
            mixture.add_value(model, groupid, value, rng);
            size_t global_groupid =
                mixture.id_tracker.packed_to_global(groupid);
            assignments.groupids(i).push(global_groupid);
        }
    }

    return true;
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

    cross_cat_.value_split(row.data(), partial_values_);

    const auto seed = rng();
    const size_t kind_count = cross_cat_.kinds.size();
    //#pragma omp parallel
    {
        rng_t rng;
        //#pragma omp for schedule(static)
        for (size_t i = 0; i < kind_count; ++i) {
            rng.seed(seed + i);
            const Value & value = partial_values_[i];
            auto & kind = cross_cat_.kinds[i];
            const ProductModel & model = kind.model;
            auto & mixture = kind.mixture;

            auto global_groupid = assignments.groupids(i).pop();
            auto groupid = mixture.id_tracker.global_to_packed(global_groupid);
            mixture.remove_value(model, groupid, value, rng);
        }
    }
}

} // namespace loom
