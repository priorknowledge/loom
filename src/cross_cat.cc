#include <sstream>
#include <iomanip>
#include <distributions/io/protobuf.hpp>
#include "protobuf_stream.hpp"
#include "cross_cat.hpp"
#include "infer_grid.hpp"

namespace loom
{

void CrossCat::model_load (const char * filename)
{
    protobuf::CrossCat message;
    protobuf::InFile(filename).read(message);

    schema.clear();
    featureid_to_kindid.clear();
    kinds.clear();

    const size_t kind_count = message.kinds_size();
    kinds.resize(kind_count);
    for (size_t kindid = 0; kindid < kind_count; ++kindid) {
        auto & kind = kinds[kindid];
        const auto & message_kind = message.kinds(kindid);

        kind.featureids.clear();
        std::vector<size_t> ordered_featureids;
        for (size_t i = 0; i < message_kind.featureids_size(); ++i) {
            size_t featureid = message_kind.featureids(i);
            kind.featureids.insert(featureid);
            ordered_featureids.push_back(featureid);
        }

        kind.model.load(message_kind.product_model(), ordered_featureids);
        schema += kind.model.schema;
    }

    distributions::clustering_load(
        feature_clustering,
        message.feature_clustering());

    for (size_t i = 0; i < message.featureid_to_kindid_size(); ++i) {
        featureid_to_kindid.push_back(message.featureid_to_kindid(i));
    }

    hyper_prior = message.hyper_prior();
}

void CrossCat::model_dump (const char * filename) const
{
    protobuf::CrossCat message;

    for (const auto & kind : kinds) {
        auto & message_kind = * message.add_kinds();

        std::vector<size_t> ordered_featureids(
            kind.featureids.begin(),
            kind.featureids.end());
        std::sort(ordered_featureids.begin(), ordered_featureids.end());

        for (size_t i : ordered_featureids) {
            message_kind.add_featureids(i);
        }

        kind.model.dump(* message_kind.mutable_product_model());
    }

    distributions::clustering_dump(
        feature_clustering,
        * message.mutable_feature_clustering());

    for (size_t kindid : featureid_to_kindid) {
        message.add_featureid_to_kindid(kindid);
    }

    * message.mutable_hyper_prior() = hyper_prior;

    protobuf::OutFile(filename).write(message);
}

std::string CrossCat::get_mixture_filename (
        const char * dirname,
        size_t kindid) const
{
    LOOM_ASSERT_LE(kindid, kinds.size());
    std::ostringstream filename;
    filename << dirname << "/mixture." <<
        std::setfill('0') << std::setw(3) << kindid << ".pbs.gz";
    return filename.str();
}

void CrossCat::mixture_load (
        const char * dirname,
        size_t empty_group_count,
        rng_t & rng)
{
    const size_t kind_count = kinds.size();
    const size_t feature_count = featureid_to_kindid.size();
    const auto seed = rng();

    #pragma omp parallel
    {
        rng_t rng;

        #pragma omp for schedule(dynamic, 1)
        for (size_t kindid = 0; kindid < kind_count; ++kindid) {
            rng.seed(seed + kindid);
            Kind & kind = kinds[kindid];
            std::string filename = get_mixture_filename(dirname, kindid);
            kind.mixture.load_step_1_of_2(
                kind.model,
                filename.c_str(),
                empty_group_count);
        }

        #pragma omp for schedule(dynamic, 1)
        for (size_t featureid = 0; featureid < feature_count; ++featureid) {
            rng.seed(seed + kind_count + featureid);
            size_t kindid = featureid_to_kindid[featureid];
            auto & kind = kinds[kindid];
            kind.mixture.load_step_2_of_2(
                kind.model,
                featureid,
                empty_group_count,
                rng);
        }
    }

    for (size_t kindid = 0; kindid < kind_count; ++kindid) {
        Kind & kind = kinds[kindid];
        kind.mixture.validate(kind.model);
    }
}

void CrossCat::mixture_dump (
        const char * dirname,
        const std::vector<std::vector<uint32_t>> & sorted_to_globals) const
{
    const size_t kind_count = kinds.size();
    LOOM_ASSERT(kind_count, "kind_count == 0, nothing to do");
    for (size_t kindid = 0; kindid < kind_count; ++kindid) {
        const Kind & kind = kinds[kindid];
        const auto & sorted_to_global = sorted_to_globals[kindid];
        std::string filename = get_mixture_filename(dirname, kindid);
        kind.mixture.dump(kind.model, filename.c_str(), sorted_to_global);
    }
}

std::vector<std::vector<uint32_t>> CrossCat::get_sorted_groupids () const
{
    std::vector<std::vector<uint32_t>> sorted_to_globals(kinds.size());
    for (size_t k = 0; k < kinds.size(); ++k) {
        const auto & mixture = kinds[k].mixture;
        const auto & counts = mixture.clustering.counts();
        const auto & id_tracker = mixture.id_tracker;
        const size_t group_count = counts.size();
        std::vector<uint32_t> & sorted_to_global = sorted_to_globals[k];

        for (size_t packed = 0; packed < group_count; ++packed) {
            if (counts[packed]) {
                sorted_to_global.push_back(packed);
            }
        }
        std::sort(
            sorted_to_global.begin(),
            sorted_to_global.end(),
            [&](uint32_t x, uint32_t y) { return counts[x] > counts[y]; });
        for (uint32_t & packed : sorted_to_global) {
            packed = id_tracker.packed_to_global(packed);
        }
    }
    return sorted_to_globals;
}

inline void CrossCat::infer_clustering_hypers (rng_t & rng)
{
    const auto & grid_prior = hyper_prior.outer_prior();
    if (grid_prior.size()) {
        std::vector<int> counts;
        counts.reserve(kinds.size());
        for (const auto & kind : kinds) {
            counts.push_back(kind.featureids.size());
        }
        feature_clustering =
            sample_clustering_posterior(grid_prior, counts, rng);
    }
}

void CrossCat::infer_hypers (rng_t & rng)
{
    const size_t kind_count = kinds.size();
    const size_t feature_count = featureid_to_kindid.size();
    const auto & inner_prior = hyper_prior.inner_prior();
    const size_t task_count = 1 + kind_count + feature_count;
    const auto seed = rng();

    #pragma omp parallel
    {
        rng_t rng;

        #pragma omp for schedule(dynamic, 1)
        for (size_t taskid = 0; taskid < task_count; ++taskid) {
            rng.seed(seed + taskid);
            if (taskid == 0) {

                infer_clustering_hypers(rng);

            } else if (taskid < 1 + kind_count) {

                size_t kindid = taskid - 1;
                auto & kind = kinds[kindid];
                kind.mixture.infer_clustering_hypers(
                    kind.model,
                    inner_prior,
                    rng);

            } else {

                size_t featureid = taskid - 1 - kind_count;
                size_t kindid = featureid_to_kindid[featureid];
                auto & kind = kinds[kindid];
                kind.mixture.infer_feature_hypers(
                    kind.model,
                    inner_prior,
                    featureid,
                    rng);
            }
        }
    }
}

float CrossCat::score_data (rng_t & rng) const
{
    float score = 0;
    std::vector<int> feature_counts;
    for (const auto & kind : kinds) {
        if (size_t feature_count = kind.featureids.size()) {
            feature_counts.push_back(feature_count);
            score += kind.mixture.score_data(kind.model, rng);
        }
    }
    score += feature_clustering.score_counts(feature_counts);
    return score;
}


} // namespace loom
