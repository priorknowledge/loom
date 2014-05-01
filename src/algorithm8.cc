#include "algorithm8.hpp"
#include <distributions/random.hpp>
#include <distributions/vector_math.hpp>

namespace loom
{

void Algorithm8::clear ()
{
    TODO("clear");
}

void Algorithm8::model_load (CrossCat & cross_cat)
{
    TODO("load model");
}

void Algorithm8::model_dump (CrossCat & cross_cat)
{
    TODO("dump model");
}

void Algorithm8::mixture_dump (CrossCat & cross_cat)
{
    TODO("dump mixtures");
}

void Algorithm8::mixture_init_empty (rng_t & rng, size_t ephemeral_kind_count)
{
    LOOM_ASSERT(ephemeral_kind_count > 0, "no ephemeral kinds");
    TODO("add ephemeral kinds");
    for (auto & kind : kinds) {
        kind.mixture.init_empty(model, rng);
    }
}

void Algorithm8::infer_assignments (
        std::vector<size_t> & featureid_to_kindid,
        size_t iterations,
        rng_t & rng)
{
    const size_t feature_count = featureid_to_kindid.size();
    const size_t kind_count = kinds.size();
    std::vector<VectorFloat> likelihoods(feature_count);

    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t featureid = 0; featureid < feature_count; ++featureid) {
        VectorFloat & scores = likelihoods[featureid];
        scores.resize(kind_count);
        for (size_t kindid = 0; kindid < feature_count; ++kindid) {
            const auto & mixture = kinds[kindid].mixture;
            scores[kindid] = mixture.score_feature(model, featureid, rng);
        }
        distributions::scores_to_likelihoods(scores);
    }

    TODO("do something with likelihoods");
}

} // namespace loom
