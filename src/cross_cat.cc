#include <sstream>
#include <iomanip>
#include <distributions/io/protobuf.hpp>
#include "protobuf_stream.hpp"
#include "cross_cat.hpp"

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

void CrossCat::mixture_load (const char * dirname, rng_t & rng)
{
    const size_t kind_count = kinds.size();
    LOOM_ASSERT(kind_count, "kind_count == 0, nothing to do");
    for (size_t kindid = 0; kindid < kind_count; ++kindid) {
        Kind & kind = kinds[kindid];
        std::string filename = get_mixture_filename(dirname, kindid);
        kind.mixture.load(kind.model, filename.c_str(), rng);
    }
}

void CrossCat::mixture_dump (const char * dirname) const
{
    const size_t kind_count = kinds.size();
    LOOM_ASSERT(kind_count, "kind_count == 0, nothing to do");
    for (size_t kindid = 0; kindid < kind_count; ++kindid) {
        const Kind & kind = kinds[kindid];
        std::string filename = get_mixture_filename(dirname, kindid);
        kind.mixture.dump(kind.model, filename.c_str());
    }
}

void CrossCat::infer_hypers (rng_t & rng)
{
    const size_t kind_count = kinds.size();
    const size_t feature_count = featureid_to_kindid.size();
    const auto & inner_prior = hyper_prior.inner_prior();
    const auto seed = rng();

    // TODO run outer clustering hyper inference
    //const auto & outer_prior = hyper_prior.outer_prior();

    #pragma omp parallel
    {
        rng_t rng;

        #pragma omp for schedule(dynamic, 1)
        for (size_t kindid = 0; kindid < kind_count; ++kindid) {
            auto & kind = kinds[kindid];
            auto & model = kind.model;
            auto & mixture = kind.mixture;
            rng.seed(seed + kindid);
            mixture.infer_clustering_hypers(model, inner_prior, rng);
        }

        #pragma omp for schedule(dynamic, 1)
        for (size_t featureid = 0; featureid < feature_count; ++featureid) {
            size_t kindid = featureid_to_kindid[featureid];
            auto & kind = kinds[kindid];
            auto & model = kind.model;
            auto & mixture = kind.mixture; // TODO switch to better mixture type
            rng.seed(seed + featureid);
            mixture.infer_feature_hypers(model, inner_prior, featureid, rng);
        }
    }
}

float CrossCat::score_data (rng_t & rng) const
{
    float score = 0;
    std::vector<int> feature_counts;
    for (const auto & kind : kinds) {
        feature_counts.push_back(kind.featureids.size());
        score += kind.mixture.score_data(kind.model, rng);
    }
    score += feature_clustering.score_counts(feature_counts);
    return score;
}


} // namespace loom
