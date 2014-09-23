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

#include <loom/kind_kernel.hpp>
#include <loom/infer_grid.hpp>

namespace loom
{

KindKernel::KindKernel (
        const protobuf::Config::Kernels & config,
        CrossCat & cross_cat,
        Assignments & assignments,
        rng_t::result_type seed) :
    empty_group_count_(config.cat().empty_group_count()),
    empty_kind_count_(config.kind().empty_kind_count()),
    iterations_(config.kind().iterations()),
    score_parallel_(config.kind().score_parallel()),

    cross_cat_(cross_cat),
    assignments_(assignments),
    kind_proposer_(),
    partial_diffs_(),
    scores_(),
    rng_(seed),

    total_count_(0),
    change_count_(0),
    birth_count_(0),
    death_count_(0),
    tare_time_(0),
    score_time_(0),
    sample_time_(0),
    timer_()
{
    Timer::Scope timer(timer_);
    LOOM_ASSERT_LT(0, iterations_);
    LOOM_ASSERT_LT(0, empty_kind_count_);
    if (LOOM_DEBUG_LEVEL >= 1) {
        auto assigned_row_count = assignments_.row_count();
        auto cross_cat_row_count = cross_cat_.kinds[0].mixture.count_rows();
        LOOM_ASSERT_EQ(assigned_row_count, cross_cat_row_count);
    }

    init_featureless_kinds(empty_kind_count_, true);
    kind_proposer_.mixture_init_unobserved(cross_cat_, rng_);

    validate();
}

KindKernel::~KindKernel ()
{
    kind_proposer_.clear();
    init_featureless_kinds(0, true);

    validate();
}

size_t KindKernel::move_features (
        const std::vector<uint32_t> & old_kindids,
        const std::vector<uint32_t> & new_kindids)
{
    size_t change_count = 0;
    const size_t feature_count = old_kindids.size();
    for (size_t featureid = 0; featureid < feature_count; ++featureid) {
        size_t old_kindid = old_kindids[featureid];
        size_t new_kindid = new_kindids[featureid];
        if (new_kindid != old_kindid) {
            move_feature_to_kind(featureid, new_kindid);
            ++change_count;
        }
    }
    total_count_ = feature_count;
    change_count_ = change_count;

    size_t kind_count = cross_cat_.kinds.size();
    std::vector<size_t> kind_states(kind_count, 0);
    for (auto kindid : old_kindids) {
        kind_states[kindid] = 1;
    }
    for (auto kindid : new_kindids) {
        kind_states[kindid] |= 2;
    }
    size_t state_counts[4] = {0, 0, 0, 0};
    for (auto state : kind_states) {
        state_counts[state] += 1;
    }
    death_count_ = state_counts[1];
    birth_count_ = state_counts[2];

    return change_count;
}

bool KindKernel::try_run ()
{
    Timer::Scope timer(timer_);

    if (LOOM_DEBUG_LEVEL >= 1) {
        auto assigned_row_count = assignments_.row_count();
        auto cross_cat_row_count = cross_cat_.kinds[0].mixture.count_rows();
        auto proposer_row_count = kind_proposer_.kinds[0].mixture.count_rows();
        LOOM_ASSERT_EQ(assigned_row_count, cross_cat_row_count);
        LOOM_ASSERT_EQ(proposer_row_count, cross_cat_row_count);
    }

    validate();

    const auto old_kindids = cross_cat_.featureid_to_kindid;
    auto new_kindids = old_kindids;
    auto times = kind_proposer_.infer_assignments(
            cross_cat_,
            new_kindids,
            iterations_,
            score_parallel_,
            rng_);
    tare_time_ = times.tare;
    score_time_ = times.score;
    sample_time_ = times.sample;

    for (auto & kind : cross_cat_.kinds) {
        kind.mixture.maintaining_cache = false;
    }
    for (auto & kind : kind_proposer_.kinds) {
        kind.mixture.maintaining_cache = false;
    }
    size_t change_count = move_features(old_kindids, new_kindids);
    init_featureless_kinds(empty_kind_count_, false);
    kind_proposer_.mixture_init_unobserved(cross_cat_, rng_);

    validate();

    return change_count > 0;
}

void KindKernel::add_featureless_kind (bool maintaining_cache)
{
    auto & kind = cross_cat_.kinds.packed_add();
    auto & model = kind.model;
    auto & mixture = kind.mixture;
    model.clear();
    mixture.maintaining_cache = maintaining_cache;

    const auto & grid_prior = cross_cat_.hyper_prior.clustering();
    if (grid_prior.size()) {
        model.clustering = sample_clustering_prior(grid_prior, rng_);
    } else {
        model.clustering = cross_cat_.kinds[0].model.clustering;
    }

    const size_t row_count = assignments_.row_count();
    const std::vector<int> assignment_vector =
        model.clustering.sample_assignments(row_count, rng_);
    size_t group_count = 0;
    for (size_t groupid : assignment_vector) {
        group_count = std::max(group_count, 1 + groupid);
    }
    group_count += empty_group_count_;
    std::vector<int> counts(group_count, 0);
    auto & assignments = assignments_.packed_add();
    for (int groupid : assignment_vector) {
        assignments.push(groupid);
        ++counts[groupid];
    }
    mixture.init_unobserved(model, counts, rng_);
}

void KindKernel::remove_featureless_kind (size_t kindid)
{
    LOOM_ASSERT(
        cross_cat_.kinds[kindid].featureids.empty(),
        "cannot remove nonempty kind: " << kindid);

    cross_cat_.kinds.packed_remove(kindid);
    assignments_.packed_remove(kindid);

    // this is simpler than keeping a MixtureIdTracker for kinds
    if (kindid < cross_cat_.kinds.size()) {
        for (auto featureid : cross_cat_.kinds[kindid].featureids) {
            cross_cat_.featureid_to_kindid[featureid] = kindid;
        }
    }
}

void KindKernel::init_featureless_kinds (
        size_t featureless_kind_count,
        bool maintaining_cache)
{
    for (int i = cross_cat_.kinds.size() - 1; i >= 0; --i) {
        if (cross_cat_.kinds[i].featureids.empty()) {
            remove_featureless_kind(i);
        }
    }

    for (size_t i = 0; i < featureless_kind_count; ++i) {
        add_featureless_kind(maintaining_cache);
    }

    cross_cat_.update_splitter();
    cross_cat_.update_tares(temp_values_, rng_);

    cross_cat_.validate();
    assignments_.validate();
}

void KindKernel::move_feature_to_kind (
        size_t featureid,
        size_t new_kindid)
{
    size_t old_kindid = cross_cat_.featureid_to_kindid[featureid];
    LOOM_ASSERT_NE(new_kindid, old_kindid);

    CrossCat::Kind & old_kind = cross_cat_.kinds[old_kindid];
    CrossCat::Kind & new_kind = cross_cat_.kinds[new_kindid];
    KindProposer::Kind & proposed_kind = kind_proposer_.kinds[new_kindid];

    proposed_kind.mixture.move_feature_to(
        featureid,
        old_kind.model, old_kind.mixture,
        new_kind.model, new_kind.mixture);

    old_kind.featureids.erase(featureid);
    new_kind.featureids.insert(featureid);
    cross_cat_.featureid_to_kindid[featureid] = new_kindid;

    // TODO do this less frequently:
    cross_cat_.update_splitter();
    cross_cat_.update_tares(temp_values_, rng_);

    cross_cat_.validate();
    assignments_.validate();
}

void KindKernel::init_cache ()
{
    LOOM_ASSERT1(not kind_proposer_.kinds.empty(), "kind_proposer is empty");

    kind_proposer_.model_load(cross_cat_);

    const size_t kind_count = cross_cat_.kinds.size();
    const size_t feature_count = cross_cat_.featureid_to_kindid.size();

    bool hyper_kernel_has_already_initialized_mixtures =
        cross_cat_.kinds[0].mixture.maintaining_cache;

    for (size_t kindid = 0; kindid < kind_count; ++kindid) {
        cross_cat_.kinds[kindid].mixture.maintaining_cache = true;
        kind_proposer_.kinds[kindid].mixture.maintaining_cache = true;
    }

    if (not hyper_kernel_has_already_initialized_mixtures) {
        const size_t task_count = feature_count + feature_count;
        const auto seed = rng_();

        #pragma omp parallel for if(score_parallel_) schedule(dynamic, 1)
        for (size_t taskid = 0; taskid < task_count; ++taskid) {
            rng_t rng(seed + taskid);
            if (taskid < feature_count) {
                size_t featureid = taskid;
                size_t kindid = cross_cat_.featureid_to_kindid[featureid];
                auto & kind = cross_cat_.kinds[kindid];
                kind.mixture.init_feature_cache(kind.model, featureid, rng);
            } else {
                size_t featureid = taskid - feature_count;
                size_t kindid = cross_cat_.featureid_to_kindid[featureid];
                auto & kind = kind_proposer_.kinds[kindid];
                kind.mixture.init_feature_cache(kind.model, featureid, rng);
            }
        }
    }

    if (not cross_cat_.tares.empty()) {
        const size_t task_count = kind_count + kind_count;
        const auto seed = rng_();

        #pragma omp parallel for if(score_parallel_) schedule(dynamic, 1)
        for (size_t taskid = 0; taskid < task_count; ++taskid) {
            rng_t rng(seed + taskid);
            if (taskid < kind_count) {
                size_t kindid = taskid;
                auto & kind = cross_cat_.kinds[kindid];
                kind.mixture.init_tare_cache(kind.model, rng);
            } else {
                size_t kindid = taskid - kind_count;
                auto & kind = kind_proposer_.kinds[kindid];
                kind.mixture.init_tare_cache(kind.model, rng);
            }
        }
    }

    validate();
}

} // namespace loom
