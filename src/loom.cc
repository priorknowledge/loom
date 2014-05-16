#include <loom/loom.hpp>
#include <loom/schedules.hpp>
#include <loom/cat_kernel.hpp>
#include <loom/hyper_kernel.hpp>
#include <loom/kind_kernel.hpp>
#include <loom/predict_server.hpp>
#include <loom/stream_interval.hpp>

namespace loom
{

//----------------------------------------------------------------------------
// Loom

Loom::Loom (
        rng_t & rng,
        const protobuf::Config & config,
        const char * model_in,
        const char * groups_in,
        const char * assign_in) :
    config_(config),
    cross_cat_(),
    assignments_()
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
        cross_cat_.mixture_init_empty(empty_group_count, rng);
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

    cross_cat_.validate();
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
    protobuf::SparseRow row;
    CatKernel cat_kernel(config_.kernels().cat(), cross_cat_);

    if (assign_out) {

        protobuf::OutFile assignments(assign_out);
        protobuf::Assignment assignment;

        while (rows.try_read_stream(row)) {
            cat_kernel.add_row(rng, row, assignment);
            assignments.write_stream(assignment);
        }

    } else {

        while (rows.try_read_stream(row)) {
            cat_kernel.add_row_noassign(rng, row);
        }
    }
}

void Loom::log_metrics (Logger::Message & message)
{
    auto & summary = * message.mutable_summary();
    distributions::clustering_dump(
        cross_cat_.feature_clustering,
        * summary.mutable_model_hypers());

    for (const auto & kind : cross_cat_.kinds) {
        if (not kind.featureids.empty()) {
            auto group_count = kind.mixture.clustering.counts().size()
                             - config_.kernels().cat().empty_group_count();
            summary.add_category_counts(group_count);
            summary.add_feature_counts(kind.featureids.size());
            distributions::clustering_dump(
                kind.model.clustering,
                * summary.add_kind_hypers());
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
        const char * rows_in)
{
    const double cat_extra_passes = config_.schedule().cat_passes();
    const double kind_extra_passes = config_.schedule().kind_passes();
    LOOM_ASSERT_LE(0, cat_extra_passes);
    LOOM_ASSERT_LE(0, kind_extra_passes);
    LOOM_ASSERT_LT(0, cat_extra_passes + kind_extra_passes);

    StreamInterval rows(rows_in);
    if (assignments_.row_count()) {
        protobuf::SparseRow row;
        rows.init_and_read_assigned(row, assignments_);
        TODO("remove the row that was just read");
    }

    size_t tardis_iter = 0;
    logger([&](Logger::Message & message){
        message.set_iter(tardis_iter);
        log_metrics(message);
        rows.log_metrics(message);
    });

    bool finished = false;
    if (config_.kernels().kind().iterations() > 0 and kind_extra_passes > 0) {
        finished = try_infer_kind_structure(rows, tardis_iter, rng);
    }
    if (not finished) {
        infer_cat_structure(rows, tardis_iter, rng);
    }
}

bool Loom::try_infer_kind_structure (
        StreamInterval & rows,
        size_t & tardis_iter,
        rng_t & rng)
{
    KindKernel kind_kernel(config_.kernels(), cross_cat_, assignments_, rng());
    HyperKernel hyper_kernel(config_.kernels().hyper(), cross_cat_);

    typedef BatchedAnnealingSchedule Schedule;
    Schedule schedule(
        config_.schedule().cat_passes() + config_.schedule().kind_passes(),
        assignments_.row_count());
    protobuf::SparseRow row;

    const size_t max_reject_iters = config_.schedule().max_reject_iters();
    size_t reject_iters = 0;
    bool adding = true;
    while (LOOM_LIKELY(adding)) {
        switch (schedule.next_action()) {

            case Schedule::add:
                rows.read_unassigned(row);
                adding = kind_kernel.try_add_row(row);
                break;

            case Schedule::remove:
                rows.read_assigned(row);
                kind_kernel.remove_row(row);
                break;

            case Schedule::process_batch:
                if (kind_kernel.try_run()) {
                    reject_iters = 0;
                } else {
                    reject_iters += 1;
                    if (reject_iters > max_reject_iters) {
                        return false;  // exit early
                    }
                }
                if (hyper_kernel.try_run(rng)) {
                    kind_kernel.update_hypers();
                }
                ++tardis_iter;
                logger([&](Logger::Message & message){
                    message.set_iter(tardis_iter);
                    log_metrics(message);
                    rows.log_metrics(message);
                    kind_kernel.log_metrics(message);
                    hyper_kernel.log_metrics(message);
                });
                break;
        }
    }

    ++tardis_iter;
    logger([&](Logger::Message & message){
        message.set_iter(tardis_iter);
        log_metrics(message);
        rows.log_metrics(message);
        kind_kernel.log_metrics(message);
    });

    return true;
}

void Loom::infer_cat_structure (
        StreamInterval & rows,
        size_t & tardis_iter,
        rng_t & rng)
{
    CatKernel cat_kernel(config_.kernels().cat(), cross_cat_);
    HyperKernel hyper_kernel(config_.kernels().hyper(), cross_cat_);

    typedef BatchedAnnealingSchedule Schedule;
    Schedule schedule(
        config_.schedule().cat_passes(),
        assignments_.row_count());
    protobuf::SparseRow row;

    bool adding = true;
    while (LOOM_LIKELY(adding)) {
        switch (schedule.next_action()) {

            case Schedule::add:
                rows.read_unassigned(row);
                adding = cat_kernel.try_add_row(rng, row, assignments_);
                break;

            case Schedule::remove:
                rows.read_assigned(row);
                cat_kernel.remove_row(rng, row, assignments_);
                break;

            case Schedule::process_batch:
                hyper_kernel.try_run(rng);
                ++tardis_iter;
                logger([&](Logger::Message & message){
                    message.set_iter(tardis_iter);
                    log_metrics(message);
                    rows.log_metrics(message);
                    cat_kernel.log_metrics(message);
                    hyper_kernel.log_metrics(message);
                });
                break;
        }
    }

    ++tardis_iter;
    logger([&](Logger::Message & message){
        message.set_iter(tardis_iter);
        log_metrics(message);
        rows.log_metrics(message);
        cat_kernel.log_metrics(message);
    });
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

    const auto rows = protobuf_stream_load<protobuf::SparseRow>(rows_in);
    LOOM_ASSERT_LT(0, rows.size());
    if (assignments_.rowids().empty()) {
        for (const auto & row : rows) {
            bool adding_data = cat_kernel.try_add_row(rng, row, assignments_);
            LOOM_ASSERT(adding_data, "duplicate row: " << row.id());
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
                    kind_kernel.try_add_row(row);
                }
                kind_kernel.try_run();
                if (hyper_kernel.try_run(rng)) {
                    kind_kernel.update_hypers();
                }
            }
            dump_posterior_enum(sample, rng);
            sample_stream.write_stream(sample);
        }

    } else {

        for (size_t i = 0; i < sample_count; ++i) {
            for (size_t t = 0; t < sample_skip; ++t) {
                for (const auto & row : rows) {
                    cat_kernel.remove_row(rng, row, assignments_);
                    cat_kernel.try_add_row(rng, row, assignments_);
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

void Loom::predict (
        rng_t & rng,
        const char * queries_in,
        const char * results_out)
{
    protobuf::InFile query_stream(queries_in);
    protobuf::OutFile result_stream(results_out);
    protobuf::PreQL::Predict::Query query;
    protobuf::PreQL::Predict::Result result;

    PredictServer server(cross_cat_);

    while (query_stream.try_read_stream(query)) {
        server.predict_row(rng, query, result);
        result_stream.write_stream(result);
        result_stream.flush();
    }
}

} // namespace loom
