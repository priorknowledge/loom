#include "algorithm8.hpp"
#include <unordered_set>
#include <distributions/random.hpp>
#include <distributions/vector_math.hpp>
#include <distributions/trivial_hash.hpp>

#define LOOM_ASSERT_CLOSE(x, y) LOOM_ASSERT_LT(fabs((x) - (y)), 1e-4)

namespace loom
{

void Algorithm8::clear ()
{
    model.clear();
    kinds.clear();
}

void Algorithm8::model_load (CrossCat &)
{
    TODO("load model");
}

void Algorithm8::mixture_init_empty (rng_t & rng, size_t kind_count)
{
    LOOM_ASSERT_LT(0, kind_count);
    kinds.resize(kind_count);
    for (auto & kind : kinds) {
        kind.mixture.init_empty(model, rng);
    }
}

void Algorithm8::infer_assignments (
        std::vector<uint32_t> & featureid_to_kindid,
        size_t iterations,
        rng_t & rng) const
{
    LOOM_ASSERT_LT(0, iterations);

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

    sample_assignments(
            model.clustering,
            likelihoods,
            featureid_to_kindid,
            iterations,
            rng);
}

inline float compute_posterior (
        const VectorFloat & prior_in,
        const VectorFloat & likelihood_in,
        VectorFloat & posterior_out)
{
    const size_t size = prior_in.size();
    const float * __restrict__ prior =
        DIST_ASSUME_ALIGNED(prior_in.data());
    const float * __restrict__ likelihood =
        DIST_ASSUME_ALIGNED(likelihood_in.data());
    float * __restrict__ posterior =
        DIST_ASSUME_ALIGNED(posterior_out.data());

    float total = 0;
    for (size_t i = 0; i < size; ++i) {
        total += posterior[i] = prior[i] * likelihood[i];
    }
    return total;
}

using distributions::sample_from_likelihoods;

// This sampler follows the math in
//   $DISTRIBUTIONS_PATH/src/clustering.hpp
//   distributions::Clustering<int>::PitmanYor::sample_assignments(...)
//
void Algorithm8::sample_assignments (
        const distributions::Clustering<int>::PitmanYor & clustering,
        const std::vector<VectorFloat> & likelihoods,
        std::vector<uint32_t> & assignments,
        size_t iterations,
        rng_t & rng)
{
    LOOM_ASSERT_LT(0, iterations);
    LOOM_ASSERT_LT(0, likelihoods.size());
    LOOM_ASSERT_EQ(likelihoods.size(), assignments.size());

    const size_t feature_count = likelihoods.size();
    const size_t kind_count = likelihoods[0].size();
    const float alpha = clustering.alpha;
    const float d = clustering.d;

    std::vector<uint32_t> counts(kind_count, 0);
    for (size_t f = 0; f < feature_count; ++f) {
        size_t k = assignments[f];
        ++counts[k];
    }

    std::unordered_set<uint32_t, distributions::TrivialHash<uint32_t>>
        empty_kinds;
    for (size_t k = 0; k < kind_count; ++k) {
        if (counts[k] == 0) {
            empty_kinds.insert(k);
        }
    }

    auto get_likelihood_empty = [&](){
        if (empty_kinds.empty()) {
            return 0.f;
        } else {
            float empty_kind_count = empty_kinds.size();
            float nonempty_kind_count = kind_count - empty_kind_count;
            return (alpha + d * nonempty_kind_count) / empty_kind_count;
        }
    };


    VectorFloat prior(kind_count);
    {
        const float likelihood_empty = get_likelihood_empty();
        for (size_t k = 0; k < kind_count; ++k) {
            if (auto count = counts[k]) {
                prior[k] = count - d;
            } else {
                prior[k] = likelihood_empty;
            }
        }
    }

    VectorFloat posterior(kind_count);
    for (size_t i = 0; i < iterations; ++i) {
    for (size_t f = 0; f < feature_count; ++f) {

        float total = compute_posterior(prior, likelihoods[f], posterior);
        size_t new_k = sample_from_likelihoods(rng, posterior, total);
        size_t old_k = assignments[f];
        if (LOOM_UNLIKELY(new_k != old_k)) {
            assignments[f] = new_k;

            size_t old_empty_kind_count = empty_kinds.size();
            {
                const float likelihood_empty = get_likelihood_empty();
                if (--counts[old_k] == 0) {
                    prior[old_k] = likelihood_empty;
                    empty_kinds.insert(old_k);
                } else {
                    prior[old_k] = counts[old_k] - d;
                }
                if (counts[new_k]++ == 0) {
                    empty_kinds.erase(new_k);
                }
                prior[new_k] = counts[new_k] - d;
            }
            size_t new_empty_kind_count = empty_kinds.size();

            if (new_empty_kind_count != old_empty_kind_count) {
                const float likelihood_empty = get_likelihood_empty();
                for (auto k : empty_kinds) {
                    prior[k] = likelihood_empty;
                }
            }
        }

        if (LOOM_DEBUG_LEVEL >= 3) {
            const float likelihood_empty = get_likelihood_empty();
            for (size_t k = 0; k < kind_count; ++k) {
                if (auto count = counts[k]) {
                    LOOM_ASSERT_CLOSE(prior[k], count - d);
                } else {
                    LOOM_ASSERT_CLOSE(prior[k], likelihood_empty);
                }
            }
        }
    }}
}

} // namespace loom
