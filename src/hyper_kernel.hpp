// Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// - Neither the name of Salesforce.com nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
