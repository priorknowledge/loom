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

#include <loom/loom.hpp>
#include <loom/cat_kernel.hpp>
#include <loom/cat_pipeline.hpp>
#include <loom/hyper_kernel.hpp>
#include <loom/kind_kernel.hpp>
#include <loom/kind_pipeline.hpp>
#include <loom/query_server.hpp>
#include <loom/stream_interval.hpp>
#include <loom/differ.hpp>
#include <loom/generate.hpp>

namespace loom
{

//----------------------------------------------------------------------------
// Loom

Loom::Loom (
        rng_t & rng,
        const protobuf::Config & config,
        const char * model_in,
        const char * groups_in,
        const char * assign_in,
        const char * tare_in) :
    config_(config),
    cross_cat_(),
    assignments_(),
    tare_()
{
    cross_cat_.model_load(model_in);
    const size_t kind_count = cross_cat_.kinds.size();
    LOOM_ASSERT(kind_count, "no kinds, loom is empty");
    assignments_.init(kind_count);

    const size_t empty_group_count =
        config_.kernels().cat().empty_group_count();
    LOOM_ASSERT_LT(0, empty_group_count);
    if (groups_in) {
        cross_cat_.mixture_load(groups_in, empty_group_count, rng);
    } else {
        cross_cat_.mixture_init_unobserved(empty_group_count, rng);
    }

    if (assign_in) {
        assignments_.load(assign_in);
        for (const auto & kind : cross_cat_.kinds) {
            LOOM_ASSERT_LE(
                assignments_.row_count(),
                kind.mixture.clustering.sample_size());
        }
        LOOM_ASSERT_EQ(assignments_.kind_count(), cross_cat_.kinds.size());
    }

    if (tare_in) {
        protobuf::InFile(tare_in).read(tare_);
        cross_cat_.schema.normalize_small(* tare_.mutable_observed());
    } else {
        tare_.mutable_observed()->set_sparsity(ProductValue::Observed::NONE);
    }

    cross_cat_.validate();
    cross_cat_.schema.validate(tare_);
    assignments_.validate();
}

//----------------------------------------------------------------------------
// High level operations

void Loom::dump (
        const char * model_out,
        const char * groups_out,
        const char * assign_out) const
{
    if (model_out) {
        cross_cat_.model_dump(model_out);
    }

    if (groups_out or assign_out) {
        std::vector<std::vector<uint32_t>> sorted_to_globals =
            cross_cat_.get_sorted_groupids();

        if (groups_out) {
            cross_cat_.mixture_dump(groups_out, sorted_to_globals);
        }

        if (assign_out) {
            assignments_.dump(assign_out, sorted_to_globals);
        }
    }
}

void Loom::infer_single_pass (
        rng_t & rng,
        const char * rows_in,
        const char * assign_out)
{
    protobuf::InFile rows(rows_in);
    protobuf::Row row;
    Differ differ(cross_cat_.schema, tare_);
    CatKernel cat_kernel(config_.kernels().cat(), cross_cat_);

    if (assign_out) {

        protobuf::OutFile assignments(assign_out);
        protobuf::Assignment assignment;

        while (rows.try_read_stream(row)) {
            differ.fill_in(row);
            cat_kernel.add_row(rng, row, assignment);
            assignments.write_stream(assignment);
        }

    } else {

        while (rows.try_read_stream(row)) {
            differ.fill_in(row);
            cat_kernel.add_row_noassign(rng, row);
        }
    }
}

void Loom::log_metrics (Logger::Message & message)
{
    auto & summary = * message.mutable_summary();
    cross_cat_.topology.protobuf_dump(
        * summary.mutable_model_hypers());

    for (const auto & kind : cross_cat_.kinds) {
        if (not kind.featureids.empty()) {
            auto group_count = kind.mixture.clustering.counts().size()
                             - config_.kernels().cat().empty_group_count();
            summary.add_category_counts(group_count);
            summary.add_feature_counts(kind.featureids.size());
            kind.model.clustering.protobuf_dump(* summary.add_kind_hypers());
        }
    }

    rng_t rng;
    float score = cross_cat_.score_data(rng);
    auto & scores = * message.mutable_scores();
    size_t data_count = assignments_.row_count();
    float kl_divergence = data_count
                        ? (-score - log(data_count)) / data_count
                        : 0;
    scores.set_assigned_object_count(assignments_.row_count());
    scores.set_score(score);
    scores.set_kl_divergence(kl_divergence);
}

void Loom::infer_multi_pass (
        rng_t & rng,
        const char * rows_in,
        const char * checkpoint_in,
        const char * checkpoint_out)
{
    StreamInterval rows(rows_in);
    CombinedSchedule schedule(config_.schedule());
    schedule.annealing.set_extra_passes(
        schedule.accelerating.extra_passes(assignments_.row_count()));

    protobuf::Checkpoint checkpoint;
    if (checkpoint_in) {
        protobuf::InFile(checkpoint_in).read(checkpoint);
        rng.seed(checkpoint.seed());
        rows.load(checkpoint.rows());
        schedule.load(checkpoint.schedule());
        checkpoint.set_tardis_iter(checkpoint.tardis_iter() + 1);
    } else {
        size_t row_count =
            protobuf::InFile::stream_stats(rows_in).message_count;
        checkpoint.set_row_count(row_count);
        if (assignments_.row_count()) {
            rows.init_from_assignments(assignments_);
        }
        checkpoint.set_tardis_iter(0);
        logger([&](Logger::Message & message){
            message.set_iter(checkpoint.tardis_iter());
            log_metrics(message);
        });
    }
    LOOM_ASSERT_LT(assignments_.row_count(), checkpoint.row_count());

    checkpoint.set_finished(false);
    if (config_.kernels().kind().iterations() and schedule.disabling.test()) {
        infer_kind_structure(rows, checkpoint, schedule, rng) ||
        infer_cat_structure(rows, checkpoint, schedule, rng);
    } else {
        infer_cat_structure(rows, checkpoint, schedule, rng);
    }

    if (checkpoint_out) {
        checkpoint.set_seed(rng());
        rows.dump(* checkpoint.mutable_rows());
        schedule.dump(* checkpoint.mutable_schedule());
        protobuf::OutFile(checkpoint_out).write(checkpoint);
    }
}

bool Loom::infer_kind_structure_sequential (
        StreamInterval & rows,
        Checkpoint & checkpoint,
        CombinedSchedule & schedule,
        rng_t & rng)
{
    Differ differ(cross_cat_.schema, tare_);
    KindKernel kind_kernel(config_.kernels(), cross_cat_, assignments_, rng());
    HyperKernel hyper_kernel(config_.kernels().hyper(), cross_cat_);
    protobuf::Row row;

    while (LOOM_LIKELY(assignments_.row_count() != checkpoint.row_count())) {
        if (schedule.annealing.next_action_is_add()) {

            rows.read_unassigned(row);
            differ.fill_in(row);
            kind_kernel.add_row(row);
            schedule.batching.add();

        } else {

            rows.read_assigned(row);
            differ.fill_in(row);
            kind_kernel.remove_row(row);
            schedule.batching.remove();
        }

        if (LOOM_UNLIKELY(schedule.batching.test())) {
            schedule.annealing.set_extra_passes(
                schedule.accelerating.extra_passes(
                    assignments_.row_count()));
            schedule.disabling.run(kind_kernel.try_run());
            hyper_kernel.try_run(rng);
            kind_kernel.update_hypers();
            checkpoint.set_tardis_iter(checkpoint.tardis_iter() + 1);
            logger([&](Logger::Message & message){
                message.set_iter(checkpoint.tardis_iter());
                log_metrics(message);
                kind_kernel.log_metrics(message);
                hyper_kernel.log_metrics(message);
            });
            if (schedule.checkpointing.test()) {
                return false;
            }
            if (not schedule.disabling.test()) {
                return false;
            }
        }
    }

    checkpoint.set_finished(true);
    checkpoint.set_tardis_iter(checkpoint.tardis_iter() + 1);
    logger([&](Logger::Message & message){
        message.set_iter(checkpoint.tardis_iter());
        log_metrics(message);
        kind_kernel.log_metrics(message);
    });
    return true;
}

bool Loom::infer_kind_structure_parallel (
        StreamInterval & rows,
        Checkpoint & checkpoint,
        CombinedSchedule & schedule,
        rng_t & rng)
{
    KindKernel kind_kernel(config_.kernels(), cross_cat_, assignments_, rng());
    HyperKernel hyper_kernel(config_.kernels().hyper(), cross_cat_);
    KindPipeline pipeline(
        config_.kernels().kind(),
        tare_,
        cross_cat_,
        rows,
        assignments_,
        kind_kernel,
        rng);

    size_t row_count = assignments_.row_count();
    while (LOOM_LIKELY(row_count != checkpoint.row_count())) {
        if (schedule.annealing.next_action_is_add()) {

            ++row_count;
            pipeline.add_row();
            schedule.batching.add();

        } else {

            --row_count;
            pipeline.remove_row();
            schedule.batching.remove();
        }

        if (LOOM_UNLIKELY(schedule.batching.test())) {
            pipeline.wait();
            LOOM_ASSERT_EQ(assignments_.row_count(), row_count);
            schedule.annealing.set_extra_passes(
                schedule.accelerating.extra_passes(
                    assignments_.row_count()));
            schedule.disabling.run(pipeline.try_run());
            hyper_kernel.try_run(rng);
            pipeline.update_hypers();
            checkpoint.set_tardis_iter(checkpoint.tardis_iter() + 1);
            logger([&](Logger::Message & message){
                message.set_iter(checkpoint.tardis_iter());
                log_metrics(message);
                pipeline.log_metrics(message);
                hyper_kernel.log_metrics(message);
            });
            if (schedule.checkpointing.test()) {
                return false;
            }
            if (not schedule.disabling.test()) {
                return false;
            }
        }
    }

    pipeline.wait();
    checkpoint.set_finished(true);
    checkpoint.set_tardis_iter(checkpoint.tardis_iter() + 1);
    logger([&](Logger::Message & message){
        message.set_iter(checkpoint.tardis_iter());
        log_metrics(message);
        pipeline.log_metrics(message);
    });
    return true;
}

bool Loom::infer_cat_structure_sequential (
        StreamInterval & rows,
        Checkpoint & checkpoint,
        CombinedSchedule & schedule,
        rng_t & rng)
{
    Differ differ(cross_cat_.schema, tare_);
    CatKernel cat_kernel(config_.kernels().cat(), cross_cat_);
    HyperKernel hyper_kernel(config_.kernels().hyper(), cross_cat_);
    protobuf::Row row;

    while (LOOM_LIKELY(assignments_.row_count() != checkpoint.row_count())) {
        if (schedule.annealing.next_action_is_add()) {

            rows.read_unassigned(row);
            differ.fill_in(row);
            cat_kernel.add_row(rng, row, assignments_);
            schedule.batching.add();

        } else {

            rows.read_assigned(row);
            differ.fill_in(row);
            cat_kernel.remove_row(rng, row, assignments_);
            schedule.batching.remove();
        }

        if (LOOM_UNLIKELY(schedule.batching.test())) {
            schedule.annealing.set_extra_passes(
                schedule.accelerating.extra_passes(
                    assignments_.row_count()));
            hyper_kernel.try_run(rng);
            checkpoint.set_tardis_iter(checkpoint.tardis_iter() + 1);
            logger([&](Logger::Message & message){
                message.set_iter(checkpoint.tardis_iter());
                log_metrics(message);
                cat_kernel.log_metrics(message);
                hyper_kernel.log_metrics(message);
            });
            if (schedule.checkpointing.test()) {
                return false;
            }
        }
    }

    checkpoint.set_finished(true);
    checkpoint.set_tardis_iter(checkpoint.tardis_iter() + 1);
    logger([&](Logger::Message & message){
        message.set_iter(checkpoint.tardis_iter());
        log_metrics(message);
        cat_kernel.log_metrics(message);
    });
    return true;
}

bool Loom::infer_cat_structure_parallel (
        StreamInterval & rows,
        Checkpoint & checkpoint,
        CombinedSchedule & schedule,
        rng_t & rng)
{
    CatKernel cat_kernel(config_.kernels().cat(), cross_cat_);
    HyperKernel hyper_kernel(config_.kernels().hyper(), cross_cat_);
    CatPipeline pipeline(
        config_.kernels().cat(),
        tare_,
        cross_cat_,
        rows,
        assignments_,
        cat_kernel,
        rng);

    size_t row_count = assignments_.row_count();
    while (LOOM_LIKELY(row_count != checkpoint.row_count())) {
        if (schedule.annealing.next_action_is_add()) {

            ++row_count;
            pipeline.add_row();
            schedule.batching.add();

        } else {

            --row_count;
            pipeline.remove_row();
            schedule.batching.remove();
        }

        if (LOOM_UNLIKELY(schedule.batching.test())) {
            pipeline.wait();
            LOOM_ASSERT_EQ(assignments_.row_count(), row_count);
            schedule.annealing.set_extra_passes(
                schedule.accelerating.extra_passes(row_count));
            hyper_kernel.try_run(rng);
            checkpoint.set_tardis_iter(checkpoint.tardis_iter() + 1);
            logger([&](Logger::Message & message){
                message.set_iter(checkpoint.tardis_iter());
                log_metrics(message);
                hyper_kernel.log_metrics(message);
            });
            if (schedule.checkpointing.test()) {
                return false;
            }
        }
    }

    pipeline.wait();
    checkpoint.set_finished(true);
    checkpoint.set_tardis_iter(checkpoint.tardis_iter() + 1);
    logger([&](Logger::Message & message){
        message.set_iter(checkpoint.tardis_iter());
        log_metrics(message);
    });
    return true;
}

void Loom::posterior_enum (
        rng_t & rng,
        const char * rows_in,
        const char * samples_out)
{
    const size_t sample_count = config_.posterior_enum().sample_count();
    const size_t sample_skip = config_.posterior_enum().sample_skip();
    LOOM_ASSERT_LE(1, sample_count);
    LOOM_ASSERT(sample_skip > 0 or sample_count == 1, "zero diversity");

    CatKernel cat_kernel(config_.kernels().cat(), cross_cat_);
    HyperKernel hyper_kernel(config_.kernels().hyper(), cross_cat_);

    auto rows = protobuf_stream_load<protobuf::Row>(rows_in);
    Differ differ(cross_cat_.schema, tare_);
    for (auto & row : rows) {
        differ.fill_in(row);
    }

    LOOM_ASSERT_LT(0, rows.size());
    if (assignments_.rowids().empty()) {
        for (const auto & row : rows) {
            LOOM_ASSERT(row.has_data(), "row.data has not been set");
            cat_kernel.add_row(rng, row, assignments_);
        }
    }

    protobuf::OutFile sample_stream(samples_out);
    protobuf::PosteriorEnum::Sample sample;

    if (config_.kernels().kind().iterations() > 0) {

        KindKernel kind_kernel(
            config_.kernels(),
            cross_cat_,
            assignments_,
            rng());

        for (size_t i = 0; i < sample_count; ++i) {
            for (size_t t = 0; t < sample_skip; ++t) {
                for (const auto & row : rows) {
                    kind_kernel.remove_row(row);
                    kind_kernel.add_row(row);
                }
                kind_kernel.try_run();
                hyper_kernel.try_run(rng);
                kind_kernel.update_hypers();
            }
            dump_posterior_enum(sample, rng);
            sample_stream.write_stream(sample);
        }

    } else {

        for (size_t i = 0; i < sample_count; ++i) {
            for (size_t t = 0; t < sample_skip; ++t) {
                for (const auto & row : rows) {
                    cat_kernel.remove_row(rng, row, assignments_);
                    cat_kernel.add_row(rng, row, assignments_);
                }
                hyper_kernel.try_run(rng);
            }
            dump_posterior_enum(sample, rng);
            sample_stream.write_stream(sample);
        }
    }
}

inline void Loom::dump_posterior_enum (
        protobuf::PosteriorEnum::Sample & message,
        rng_t & rng)
{
    float score = cross_cat_.score_data(rng);
    const size_t row_count = assignments_.row_count();
    const size_t kind_count = assignments_.kind_count();
    const auto & rowids = assignments_.rowids();

    message.Clear();
    for (size_t kindid = 0; kindid < kind_count; ++kindid) {
        const auto & kind = cross_cat_.kinds[kindid];
        if (not kind.featureids.empty()) {
            const auto & groupids = assignments_.groupids(kindid);
            auto & message_kind = * message.add_kinds();
            for (auto featureid : kind.featureids) {
                message_kind.add_featureids(featureid);
            }
            std::unordered_map<size_t, std::vector<size_t>> groupids_map;
            for (size_t i = 0; i < row_count; ++i) {
                groupids_map[groupids[i]].push_back(rowids[i]);
            }
            for (const auto & pair : groupids_map) {
                auto & message_group = * message_kind.add_groups();
                for (const auto & rowid : pair.second) {
                    message_group.add_rowids(rowid);
                }
            }
        }
    }
    message.set_score(score);
}

void Loom::generate (
        rng_t & rng,
        const char * rows_out)
{
    LOOM_ASSERT_EQ(assignments_.row_count(), 0);

    HyperKernel(config_.kernels().hyper(), cross_cat_).try_run(rng);

    generate_rows(config_.generate(), cross_cat_, rows_out, rng);
}


void Loom::query (
        rng_t & rng,
        const char * requests_in,
        const char * responses_out)
{
    protobuf::InFile query_stream(requests_in);
    protobuf::OutFile response_stream(responses_out);
    protobuf::Query::Request request;
    protobuf::Query::Response response;

    QueryServer server(cross_cat_);

    while (query_stream.try_read_stream(request)) {
        if (request.has_sample()) {
            server.sample_row(rng, request, response);
        }
        if (request.has_score()) {
            server.score_row(rng, request, response);
        }
        response_stream.write_stream(response);
        response_stream.flush();
    }
}
} // namespace loom
