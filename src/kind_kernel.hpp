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

    KindKernel (
            const protobuf::Config::Kernels & config,
            CrossCat & cross_cat,
            Assignments & assignments,
            rng_t::result_type seed);

    ~KindKernel ();

    void add_row (const protobuf::Row & row);
    void remove_row (const protobuf::Row & row);
    bool try_run ();
    void init_cache ();
    void validate () const;
    void log_metrics (Logger::Message & message);

    size_t add_to_cross_cat (
            size_t kindid,
            const ProductValue::Diff & partial_diff,
            VectorFloat & scores,
            rng_t & rng);

    void add_to_kind_proposer (
            size_t kindid,
            size_t groupid,
            const ProductValue::Diff & diff,
            rng_t & rng);

    size_t remove_from_cross_cat (
            size_t kindid,
            const ProductValue::Diff & partial_diff,
            rng_t & rng);

    void remove_from_kind_proposer (
            size_t kindid,
            size_t groupid);

private:

    void add_featureless_kind (bool maintaining_cache);
    void remove_featureless_kind (size_t kindid);
    void init_featureless_kinds (
            size_t featureless_kind_count,
            bool maintaining_cache);
    size_t move_features (
            const std::vector<uint32_t> & old_kindids,
            const std::vector<uint32_t> & new_kindids);

    void move_feature_to_kind (
            size_t featureid,
            size_t new_kindid);

    const size_t empty_group_count_;
    const size_t empty_kind_count_;
    const size_t iterations_;
    const bool score_parallel_;

    CrossCat & cross_cat_;
    Assignments & assignments_;
    KindProposer kind_proposer_;
    std::vector<ProductValue::Diff> partial_diffs_;
    std::vector<ProductValue> temp_values_;
    VectorFloat scores_;
    rng_t rng_;

    size_t total_count_;
    size_t change_count_;
    size_t birth_count_;
    size_t death_count_;
    usec_t tare_time_;
    usec_t score_time_;
    usec_t sample_time_;
    Timer timer_;
};

inline void KindKernel::validate () const
{
    cross_cat_.validate();
    if (not kind_proposer_.kinds.empty()) {
        kind_proposer_.validate(cross_cat_);
    }
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
    status.set_tare_time(tare_time_);
    status.set_score_time(score_time_);
    status.set_sample_time(sample_time_);
    status.set_total_time(timer_.total());
    timer_.clear();
}

//----------------------------------------------------------------------------
// low-level operations

inline void KindKernel::add_row (const protobuf::Row & row)
{
    Timer::Scope timer(timer_);
    bool ok = assignments_.rowids().try_push(row.id());
    LOOM_ASSERT1(ok, "duplicate row: " << row.id());

    LOOM_ASSERT_EQ(cross_cat_.kinds.size(), kind_proposer_.kinds.size());
    const size_t kind_count = cross_cat_.kinds.size();

    cross_cat_.diff_split(row.diff(), partial_diffs_, temp_values_);
    cross_cat_.normalize_small(partial_diffs_);
    for (size_t i = 0; i < kind_count; ++i) {
        auto groupid = add_to_cross_cat(i, partial_diffs_[i], scores_, rng_);
        add_to_kind_proposer(i, groupid, row.diff(), rng_);
    }
}

inline size_t KindKernel::add_to_cross_cat (
        size_t kindid,
        const ProductValue::Diff & partial_diff,
        VectorFloat & scores,
        rng_t & rng)
{
    LOOM_ASSERT3(kindid < cross_cat_.kinds.size(), "bad kindid: " << kindid);
    auto & kind = cross_cat_.kinds[kindid];
    ProductModel & model = kind.model;
    auto & mixture = kind.mixture;

    size_t groupid;
    if (cross_cat_.tares.empty()) {
        auto & value = partial_diff.pos();
        model.add_value(value, rng);
        mixture.score_value(model, value, scores, rng);
        groupid = sample_from_scores_overwrite(rng, scores);
        mixture.add_value(model, groupid, value, rng);
    } else {
        model.add_diff(partial_diff, rng);
        mixture.score_diff(model, partial_diff, scores, rng);
        groupid = sample_from_scores_overwrite(rng, scores);
        mixture.add_diff(model, groupid, partial_diff, rng);
    }
    size_t global_groupid = mixture.id_tracker.packed_to_global(groupid);
    assignments_.groupids(kindid).push(global_groupid);
    return groupid;
}

inline void KindKernel::add_to_kind_proposer (
        size_t kindid,
        size_t groupid,
        const ProductValue::Diff & diff,
        rng_t & rng)
{
    LOOM_ASSERT3(kindid < cross_cat_.kinds.size(), "bad kindid: " << kindid);
    auto & kind = kind_proposer_.kinds[kindid];
    ProductModel & model = kind.model;
    auto & mixture = kind.mixture;

    if (cross_cat_.tares.empty()) {
        auto & value = diff.pos();
        model.add_value(value, rng);
        mixture.add_value(model, groupid, value, rng);
    } else {
        model.add_diff(diff, rng);
#define DEBUG_LAZY_ADD_DIFF
#ifdef DEBUG_LAZY_ADD_DIFF
        mixture.add_diff_step_1_of_2(model, groupid, diff, rng);
#else // DEBUG_LAZY_ADD_DIFF
        mixture.add_diff(model, groupid, diff, rng);
#endif // DEBUG_LAZY_ADD_DIFF
    }
}

inline void KindKernel::remove_row (const protobuf::Row & row)
{
    Timer::Scope timer(timer_);
    const auto rowid = assignments_.rowids().pop();
    if (LOOM_DEBUG_LEVEL >= 1) {
        LOOM_ASSERT_EQ(rowid, row.id());
    }

    LOOM_ASSERT_EQ(cross_cat_.kinds.size(), kind_proposer_.kinds.size());
    const size_t kind_count = cross_cat_.kinds.size();

    cross_cat_.diff_split(row.diff(), partial_diffs_, temp_values_);
    cross_cat_.normalize_small(partial_diffs_);
    for (size_t i = 0; i < kind_count; ++i) {
        auto groupid = remove_from_cross_cat(i, partial_diffs_[i], rng_);
        remove_from_kind_proposer(i, groupid);
    }
}

inline size_t KindKernel::remove_from_cross_cat (
        size_t kindid,
        const ProductValue::Diff & partial_diff,
        rng_t & rng)
{
    LOOM_ASSERT3(kindid < cross_cat_.kinds.size(), "bad kindid: " << kindid);
    auto & kind = cross_cat_.kinds[kindid];
    ProductModel & model = kind.model;
    auto & mixture = kind.mixture;

    auto global_groupid = assignments_.groupids(kindid).pop();
    auto groupid = mixture.id_tracker.global_to_packed(global_groupid);
    if (cross_cat_.tares.empty()) {
        auto & value = partial_diff.pos();
        mixture.remove_value(model, groupid, value, rng);
        model.remove_value(value, rng);
    } else {
        mixture.remove_diff(model, groupid, partial_diff, rng);
        model.remove_diff(partial_diff, rng);
    }
    return groupid;
}

inline void KindKernel::remove_from_kind_proposer (
        size_t kindid,
        size_t groupid)
{
    LOOM_ASSERT3(kindid < cross_cat_.kinds.size(), "bad kindid: " << kindid);
    auto & kind = kind_proposer_.kinds[kindid];
    ProductModel & model = kind.model;
    auto & mixture = kind.mixture;

    mixture.remove_unobserved_value(model, groupid);
}

} // namespace loom
