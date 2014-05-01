#include <sstream>
#include <iomanip>
#include <omp.h>
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

    distributions::clustering_load(clustering, message.clustering());

    for (size_t i = 0; i < message.featureid_to_kindid_size(); ++i) {
        featureid_to_kindid.push_back(message.featureid_to_kindid(i));
    }
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

void CrossCat::mixture_dump (const char * dirname)
{
    const size_t kind_count = kinds.size();
    LOOM_ASSERT(kind_count, "kind_count == 0, nothing to do");
    for (size_t kindid = 0; kindid < kind_count; ++kindid) {
        Kind & kind = kinds[kindid];
        std::string filename = get_mixture_filename(dirname, kindid);
        kind.mixture.dump(kind.model, filename.c_str());
    }
}

void CrossCat::move_feature_to_kind (
        size_t featureid,
        size_t new_kindid)
{
    size_t old_kindid = featureid_to_kindid[featureid];
    LOOM_ASSERT_NE(new_kindid, old_kindid);
    auto & old_kind = kinds[old_kindid];
    auto & new_kind = kinds[new_kindid];

    old_kind.model.move_feature_to(featureid, new_kind.model);
    old_kind.mixture.move_feature_to(featureid, new_kind.mixture);

    old_kind.featureids.erase(featureid);
    new_kind.featureids.insert(featureid);

    featureid_to_kindid[featureid] = new_kindid;
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
        rng_t rng(seed + omp_get_thread_num());

        #pragma omp for schedule(dynamic, 1)
        for (size_t kindid = 0; kindid < kind_count; ++kindid) {
            auto & kind = kinds[kindid];
            auto & model = kind.model;
            auto & mixture = kind.mixture;
            mixture.infer_clustering_hypers(model, inner_prior, rng);
        }

        #pragma omp for schedule(dynamic, 1)
        for (size_t featureid = 0; featureid < feature_count; ++featureid) {
            size_t kindid = featureid_to_kindid[featureid];
            auto & kind = kinds[kindid];
            auto & model = kind.model;
            auto & mixture = kind.mixture; // TODO switch to better mixture type
            mixture.infer_feature_hypers(model, inner_prior, featureid, rng);
        }
    }
}

} // namespace loom
