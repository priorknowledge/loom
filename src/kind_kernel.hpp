#pragma once

#include <thread>
#include <loom/cross_cat.hpp>
#include <loom/assignments.hpp>
#include <loom/kind_proposer.hpp>
#include <loom/pipeline.hpp>
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

    void add_row (const protobuf::SparseRow & row);
    void remove_row (const protobuf::SparseRow & row);
    bool try_run ();
    void update_hypers () { kind_proposer_.model_update(cross_cat_); }
    void validate () const;
    void log_metrics (Logger::Message & message);

    size_t add_to_cross_cat (
            size_t kindid,
            const Value & partial_value,
            VectorFloat & scores,
            rng_t & rng);

    void add_to_kind_proposer (
            size_t kindid,
            size_t groupid,
            const Value & full_value,
            rng_t & rng);

    size_t remove_from_cross_cat (
            size_t kindid,
            const Value & partial_value,
            rng_t & rng);

    void remove_from_kind_proposer (
            size_t kindid,
            size_t groupid,
            rng_t & rng);

private:

    void add_featureless_kind ();
    void remove_featureless_kind (size_t kindid);
    void init_featureless_kinds (size_t featureless_kind_count);

    void move_feature_to_kind (
            size_t featureid,
            size_t new_kindid);

    const size_t empty_group_count_;
    const size_t empty_kind_count_;
    const size_t iterations_;
    const bool score_parallel_;
    const bool init_cache_;

    CrossCat & cross_cat_;
    Assignments & assignments_;
    KindProposer kind_proposer_;
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
    kind_proposer_.validate(cross_cat_);
    assignments_.validate();
    const size_t kind_count = cross_cat_.kinds.size();
    LOOM_ASSERT_EQ(assignments_.kind_count(), kind_count);
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

inline void KindKernel::add_row (const protobuf::SparseRow & row)
{
    Timer::Scope timer(timer_);
    bool ok =  assignments_.rowids().try_push(row.id());
    LOOM_ASSERT1(ok, "duplicate row: " << row.id());

    LOOM_ASSERT_EQ(cross_cat_.kinds.size(), kind_proposer_.kinds.size());
    const size_t kind_count = cross_cat_.kinds.size();

    const Value & full_value = row.data();
    cross_cat_.value_split(full_value, partial_values_);
    for (size_t i = 0; i < kind_count; ++i) {
        auto groupid = add_to_cross_cat(i, partial_values_[i], scores_, rng_);
        add_to_kind_proposer(i, groupid, full_value, rng_);
    }
}

inline size_t KindKernel::add_to_cross_cat (
        size_t kindid,
        const Value & value,
        VectorFloat & scores,
        rng_t & rng)
{
    LOOM_ASSERT3(kindid < cross_cat_.kinds.size(), "bad kindid: " << kindid);
    auto & kind = cross_cat_.kinds[kindid];
    const ProductModel & model = kind.model;
    auto & mixture = kind.mixture;

    mixture.score_value(model, value, scores, rng);
    size_t groupid = sample_from_scores_overwrite(rng, scores);
    mixture.add_value(model, groupid, value, rng);
    size_t global_groupid = mixture.id_tracker.packed_to_global(groupid);
    assignments_.groupids(kindid).push(global_groupid);
    return groupid;
}

inline void KindKernel::add_to_kind_proposer (
        size_t kindid,
        size_t groupid,
        const Value & value,
        rng_t & rng)
{
    LOOM_ASSERT3(kindid < cross_cat_.kinds.size(), "bad kindid: " << kindid);
    const ProductModel & model = kind_proposer_.model;
    auto & mixture = kind_proposer_.kinds[kindid].mixture;

    mixture.add_value(model, groupid, value, rng);
}

inline void KindKernel::remove_row (const protobuf::SparseRow & row)
{
    Timer::Scope timer(timer_);
    const auto rowid = assignments_.rowids().pop();
    if (LOOM_DEBUG_LEVEL >= 1) {
        LOOM_ASSERT_EQ(rowid, row.id());
    }

    LOOM_ASSERT_EQ(cross_cat_.kinds.size(), kind_proposer_.kinds.size());
    const size_t kind_count = cross_cat_.kinds.size();

    cross_cat_.value_split(row.data(), partial_values_);
    for (size_t i = 0; i < kind_count; ++i) {
        auto groupid = remove_from_cross_cat(i, partial_values_[i], rng_);
        remove_from_kind_proposer(i, groupid, rng_);
    }
}

inline size_t KindKernel::remove_from_cross_cat (
        size_t kindid,
        const Value & value,
        rng_t & rng)
{
    LOOM_ASSERT3(kindid < cross_cat_.kinds.size(), "bad kindid: " << kindid);
    auto & kind = cross_cat_.kinds[kindid];
    const ProductModel & model = kind.model;
    auto & mixture = kind.mixture;

    auto global_groupid = assignments_.groupids(kindid).pop();
    auto groupid = mixture.id_tracker.global_to_packed(global_groupid);
    mixture.remove_value(model, groupid, value, rng);
    return groupid;
}

inline void KindKernel::remove_from_kind_proposer (
        size_t kindid,
        size_t groupid,
        rng_t & rng)
{
    LOOM_ASSERT3(kindid < cross_cat_.kinds.size(), "bad kindid: " << kindid);
    const ProductModel & model = kind_proposer_.model;
    auto & mixture = kind_proposer_.kinds[kindid].mixture;

    mixture.remove_value(model, groupid, unobserved_, rng);
}

} // namespace loom
