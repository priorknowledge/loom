#pragma once

#include <loom/cross_cat.hpp>
#include <loom/timer.hpp>

namespace loom
{

//----------------------------------------------------------------------------
// HyperKernel
//
// This kernel infers all hyperparameters in parallel, namely:
// * outer clustering hyperparameters
// * inner clustering hyperparameters for each kind
// * feature hyperparameters for each feature

class HyperKernel
{
public:

    HyperKernel (CrossCat & cross_cat) :
        cross_cat_(cross_cat)
    {
    }

    void run (rng_t & rng, bool parallel);

    Timer & timer () { return timer_; }

private:

    typedef CrossCat::ProductMixture ProductMixture;
    typedef protobuf::ProductModel::HyperPrior HyperPrior;

    template<class GridPrior>
    void infer_outer_clustering_hypers (
            const GridPrior & grid_prior,
            rng_t & rng);

    static void infer_inner_clustering_hypers (
            ProductModel & model,
            ProductMixture & mixture,
            const HyperPrior & hyper_prior,
            rng_t & rng);

    static void infer_feature_hypers (
            ProductModel & model,
            ProductMixture & mixture,
            const HyperPrior & hyper_prior,
            size_t featureid,
            rng_t & rng);

    struct infer_feature_hypers_fun;

private:

    CrossCat & cross_cat_;
    Timer timer_;
};

} // namespace loom
