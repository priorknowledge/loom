#pragma once

#include <loom/cross_cat.hpp>
#include <loom/timer.hpp>
#include <loom/logger.hpp>

namespace loom
{

//----------------------------------------------------------------------------
// HyperKernel
//
// This kernel infers all hyperparameters in parallel, namely:
// * outer clustering hyperparameters
// * inner clustering hyperparameters for each kind
// * feature hyperparameters for each feature

class HyperKernel : noncopyable
{
public:

    HyperKernel (
            const protobuf::Config::Kernels::Hyper & config,
            CrossCat & cross_cat) :
        run_(config.run()),
        parallel_(config.parallel()),
        cross_cat_(cross_cat),
        timer_()
    {
    }

    bool try_run (rng_t & rng)
    {
        if (run_) {
            run(rng);
        }
        return run_;
    }

    void run (rng_t & rng);

    void log_metrics (Logger::Message & message);

private:

    typedef CrossCat::ProductMixture ProductMixture;
    typedef protobuf::HyperPrior HyperPrior;

    template<class GridPrior>
    void infer_topology_hypers (
            const GridPrior & grid_prior,
            rng_t & rng);

    static void infer_clustering_hypers (
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

    const bool run_;
    const bool parallel_;
    CrossCat & cross_cat_;
    Timer timer_;
};

inline void HyperKernel::log_metrics (Logger::Message & message)
{
    auto & status = * message.mutable_kernel_status()->mutable_hyper();
    status.set_total_time(timer_.total());
    timer_.clear();
}

} // namespace loom
