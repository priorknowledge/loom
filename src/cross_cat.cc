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

void CrossCat::move_feature_to_kind (
        size_t featureid,
        size_t new_kindid,
        rng_t & rng)
{
    size_t old_kindid = featureid_to_kindid[featureid];
    LOOM_ASSERT_NE(new_kindid, old_kindid);

    ProductModel & old_model = kinds[old_kindid].model;
    ProductModel & new_model = kinds[new_kindid].model;
    auto & old_mixture = kinds[old_kindid].mixture;
    auto & new_mixture = kinds[new_kindid].mixture;
    ProductModel::move_shared_to(
        featureid,
        old_model, old_mixture,
        new_model, new_mixture,
        rng);

    kinds[old_kindid].featureids.erase(featureid);
    kinds[new_kindid].featureids.insert(featureid);
    featureid_to_kindid[featureid] = new_kindid;
}

} // namespace loom
