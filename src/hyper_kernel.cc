#include <type_traits>
#include "infer_grid.hpp"
#include "hyper_kernel.hpp"

namespace loom
{

template<class GridPrior>
inline void HyperKernel::infer_outer_clustering_hypers (
        const GridPrior & grid_prior,
        rng_t & rng)
{
    if (grid_prior.size()) {
        std::vector<int> counts;
        counts.reserve(cross_cat_.kinds.size());
        for (const auto & kind : cross_cat_.kinds) {
            counts.push_back(kind.featureids.size());
        }
        cross_cat_.feature_clustering =
            sample_clustering_posterior(grid_prior, counts, rng);
    }
}

inline void HyperKernel::infer_inner_clustering_hypers (
        ProductModel & model,
        ProductMixture & mixture,
        const HyperPrior & hyper_prior,
        rng_t & rng)
{
    const auto & grid_prior = hyper_prior.clustering();
    if (grid_prior.size()) {
        // this extraneous copy is needed for init below
        std::vector<int> counts = mixture.clustering.counts();
        model.clustering = sample_clustering_posterior(grid_prior, counts, rng);
        mixture.clustering.init(model.clustering, counts);
    }
}

struct HyperKernel::infer_feature_hypers_fun
{
    const HyperPrior & hyper_prior;
    ProductMixture::Features & mixtures;
    rng_t & rng;

    template<class T>
    void operator() (
            T * t,
            size_t i,
            typename T::Shared & shared)
    {
        // TODO optimize mixture to cache score_data(...)
        auto & mixture = mixtures[t][i];
        typedef typename std::remove_reference<decltype(mixture)>::type Mixture;
        InferShared<Mixture> infer_shared(shared, mixture, rng);
        const auto & grid_prior = protobuf::GridPriors<T>::get(hyper_prior);
        distributions::for_each_gridpoint(grid_prior, infer_shared);
        mixture.init(shared, rng);
    }

    void operator() (
        DPD * t,
        size_t i,
        DPD::Shared & shared)
    {
        // TODO implement DPD inference
        auto & mixture = mixtures[t][i];
        mixture.init(shared, rng);
    }
};

inline void HyperKernel::infer_feature_hypers (
        ProductModel & model,
        ProductMixture & mixture,
        const HyperPrior & hyper_prior,
        size_t featureid,
        rng_t & rng)
{
    infer_feature_hypers_fun fun = {hyper_prior, mixture.features, rng};
    for_one_feature(fun, model.features, featureid);
}

void HyperKernel::run (rng_t & rng, bool parallel)
{
    const size_t kind_count = cross_cat_.kinds.size();
    const size_t feature_count = cross_cat_.featureid_to_kindid.size();
    const auto & outer_prior = cross_cat_.hyper_prior.outer_prior();
    const auto & inner_prior = cross_cat_.hyper_prior.inner_prior();
    const size_t task_count = 1 + kind_count + feature_count;
    const auto seed = rng();

    #pragma omp parallel if (parallel)
    {
        rng_t rng;

        #pragma omp for schedule(dynamic, 1)
        for (size_t taskid = 0; taskid < task_count; ++taskid) {
            rng.seed(seed + taskid);
            if (taskid == 0) {

                infer_outer_clustering_hypers(outer_prior, rng);

            } else if (taskid < 1 + kind_count) {

                size_t kindid = taskid - 1;
                auto & kind = cross_cat_.kinds[kindid];
                infer_inner_clustering_hypers(
                    kind.model,
                    kind.mixture,
                    inner_prior,
                    rng);

            } else {

                size_t featureid = taskid - 1 - kind_count;
                size_t kindid = cross_cat_.featureid_to_kindid[featureid];
                auto & kind = cross_cat_.kinds[kindid];
                infer_feature_hypers(
                    kind.model,
                    kind.mixture,
                    inner_prior,
                    featureid,
                    rng);
            }
        }
    }
}

} // namespace loom

