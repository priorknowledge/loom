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

#include <type_traits>
#include <loom/infer_grid.hpp>
#include <loom/hyper_kernel.hpp>
#include <loom/hyper_prior.hpp>

namespace loom
{

using distributions::sample_from_scores_overwrite;
using distributions::fast_log;
using distributions::fast_lgamma;

template<class GridPrior>
inline void HyperKernel::infer_topology_hypers (
        const GridPrior & grid_prior,
        rng_t & rng)
{
    if (grid_prior.size()) {
        std::vector<int> counts;
        counts.reserve(cross_cat_.kinds.size());
        for (const auto & kind : cross_cat_.kinds) {
            counts.push_back(kind.featureids.size());
        }
        cross_cat_.topology =
            sample_clustering_posterior(grid_prior, counts, rng);
    }
}

inline void HyperKernel::infer_clustering_hypers (
        ProductModel & model,
        ProductMixture & mixture,
        const HyperPrior & hyper_prior,
        rng_t & rng)
{
    const auto & grid_prior = hyper_prior.clustering();
    if (grid_prior.size()) {
        const auto & counts = mixture.clustering.counts();
        model.clustering = sample_clustering_posterior(grid_prior, counts, rng);
        mixture.clustering.init(model.clustering);
    }
}

struct HyperKernel::infer_feature_hypers_fun
{
    const HyperPrior & hyper_prior;
    ProductMixture::Features & mixtures;
    rng_t & rng;

    template<class T>
    void operator() (T * t, size_t i, typename T::Shared & shared)
    {
        auto & mixture = mixtures[t][i];
        typedef typename std::remove_reference<decltype(mixture)>::type Mixture;
        InferShared<Mixture> infer_shared(shared, mixture, rng);
        const auto & grid_prior = protobuf::Fields<T>::get(hyper_prior);
        for_each_gridpoint(grid_prior, infer_shared);
        mixture.init(shared, rng);
    }

    void operator() (DPD * t, size_t i, DPD::Shared & shared);
};

void HyperKernel::infer_feature_hypers_fun::operator() (
        DPD * t,
        size_t i,
        DPD::Shared & shared)
{
    auto & mixture = mixtures[t][i];
    typedef typename std::remove_reference<decltype(mixture)>::type Mixture;
    InferShared<Mixture> infer_shared(shared, mixture, rng);
    const auto & grid_prior = protobuf::Fields<DPD>::get(hyper_prior);
    VectorFloat scores;

    // sample aux_counts
    typedef uint32_t count_t;
    std::unordered_map<DPD::Value, count_t> aux_counts;
    for (const auto & group : mixture.groups()) {
        for (const auto & i : group.counts) {
            auto value = i.first;
            auto count = i.second;
            float beta = shared.betas.get(value);
            LOOM_ASSERT_LT(0, beta);
            float log_prior = log(shared.alpha * beta);
            distributions::get_log_stirling1_row(count, scores);
            LOOM_ASSERT_EQ(scores.size(), count + 1);
            for (size_t k = 0; k <= count; ++k) {
                scores[k] += k * log_prior;
            }
            size_t aux_count = sample_from_scores_overwrite(rng, scores);
            LOOM_ASSERT_LT(0, aux_count);
            aux_counts[value] += aux_count;
        }
    }

    // only infer hypers if all values have been observed
    if (LOOM_LIKELY(aux_counts.size() == shared.betas.size())) {

        // grid gibbs gamma | aux_counts
        if (grid_prior.gamma_size()) {
            size_t aux_total = 0;
            for (const auto & i : aux_counts) {
                aux_total += i.second;
            }
            scores.clear();
            scores.reserve(grid_prior.gamma_size());
            for (float gamma : grid_prior.gamma()) {
                float score = aux_counts.size() * fast_log(gamma)
                            + fast_lgamma(gamma)
                            - fast_lgamma(gamma + aux_total);
                scores.push_back(score);
            }
            size_t index = sample_from_scores_overwrite(rng, scores);
            shared.gamma = grid_prior.gamma(index);
        }

        // sample beta0, betas | aux_counts, gamma
        if (grid_prior.alpha_size()) {
            std::vector<DPD::Value> values;
            std::vector<float> betas;
            values.reserve(aux_counts.size() + 1);
            betas.reserve(aux_counts.size() + 1);
            for (const auto & i : aux_counts) {
                values.push_back(i.first);
                betas.push_back(i.second);
            }
            values.push_back(DPD::Model::OTHER());
            betas.push_back(shared.gamma);

            distributions::sample_dirichlet_safe(
                rng,
                betas.size(),
                betas.data(),
                betas.data(),
                DPD::Model::MIN_BETA());

            for (size_t i = 0, size = aux_counts.size(); i < size; ++i) {
                shared.betas.get(values[i]) = betas[i];
            }
            shared.beta0 = betas.back();
        }

        // grid gibbs alpha | beta0, betas, gamma
        if (grid_prior.alpha_size()) {
            mixture.init(shared, rng);
            for (auto alpha : grid_prior.alpha()) {
                infer_shared.add().alpha = alpha;
            }
            infer_shared.done();
        }
    }

    mixture.init(shared, rng);
}

inline void HyperKernel::infer_feature_hypers (
        ProductModel & model,
        ProductMixture & mixture,
        const HyperPrior & hyper_prior,
        size_t featureid,
        rng_t & rng)
{
    infer_feature_hypers_fun fun = {hyper_prior, mixture.features, rng};
    for_one_feature(fun, model.features, featureid);
    mixture.maintaining_cache = true;
}

void HyperKernel::run (rng_t & rng)
{
    Timer::Scope timer(timer_);
    LOOM_ASSERT(run_, "hyper kernel should not be run");

    const size_t kind_count = cross_cat_.kinds.size();
    const size_t feature_count = cross_cat_.featureid_to_kindid.size();
    const size_t task_count = 1 + kind_count + feature_count;
    const auto seed = rng();

    #pragma omp parallel for if(parallel_) schedule(dynamic, 1)
    for (size_t taskid = 0; taskid < task_count; ++taskid) {
        rng_t rng(seed + taskid);
        if (taskid == 0) {

            infer_topology_hypers(cross_cat_.hyper_prior.topology(), rng);

        } else if (taskid < 1 + kind_count) {

            size_t kindid = taskid - 1;
            auto & kind = cross_cat_.kinds[kindid];
            infer_clustering_hypers(
                kind.model,
                kind.mixture,
                cross_cat_.hyper_prior,
                rng);

        } else {

            size_t featureid = taskid - 1 - kind_count;
            size_t kindid = cross_cat_.featureid_to_kindid[featureid];
            auto & kind = cross_cat_.kinds[kindid];
            infer_feature_hypers(
                kind.model,
                kind.mixture,
                cross_cat_.hyper_prior,
                featureid,
                rng);
        }
    }
}

} // namespace loom

