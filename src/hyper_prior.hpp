#pragma once

#include <loom/protobuf.hpp>

namespace loom
{

//----------------------------------------------------------------------------
// Grid Priors

template<class Visitor>
inline void for_each_gridpoint (
        const protobuf::HyperPrior::BetaBernoulli & grid,
        Visitor & visitor)
{
    for (auto alpha : grid.alpha()) {
        visitor.add().alpha = alpha;
    }
    visitor.done();

    for (auto beta : grid.beta()) {
        visitor.add().beta = beta;
    }
    visitor.done();
}

template<class Visitor>
inline void for_each_gridpoint (
        const protobuf::HyperPrior::DirichletDiscrete & grid,
        Visitor & visitor)
{
    int dim = visitor.shared().dim;

    for (int i = 0; i < dim; ++i) {
        for (auto alpha : grid.alpha()) {
            visitor.add().alphas[i] = alpha;
        }
        visitor.done();
    }
}

template<class Visitor>
inline void for_each_gridpoint (
        const protobuf::HyperPrior::DirichletProcessDiscrete & grid,
        Visitor & visitor)
{
    for (auto gamma : grid.gamma()) {
        visitor.add().gamma = gamma;
    }
    visitor.done();

    for (auto alpha : grid.alpha()) {
        visitor.add().alpha = alpha;
    }
    visitor.done();
}

template<class Visitor>
inline void for_each_gridpoint (
        const protobuf::HyperPrior::GammaPoisson & grid,
        Visitor & visitor)
{
    for (auto alpha : grid.alpha()) {
        visitor.add().alpha = alpha;
    }
    visitor.done();

    for (auto inv_beta : grid.inv_beta()) {
        visitor.add().inv_beta = inv_beta;
    }
    visitor.done();
}

template<class Visitor>
inline void for_each_gridpoint (
        const protobuf::HyperPrior::BetaNegativeBinomial & grid,
        Visitor & visitor)
{
    for (auto alpha : grid.alpha()) {
        visitor.add().alpha = alpha;
    }
    visitor.done();

    for (auto beta : grid.beta()) {
        visitor.add().beta = beta;
    }
    visitor.done();

    for (auto r : grid.r()) {
        visitor.add().r = r;
    }
    visitor.done();
}

template<class Visitor>
inline void for_each_gridpoint (
        const protobuf::HyperPrior::NormalInverseChiSq & grid,
        Visitor & visitor)
{
    for (auto mu : grid.mu()) {
        visitor.add().mu = mu;
    }
    visitor.done();

    for (auto kappa : grid.kappa()) {
        visitor.add().kappa = kappa;
    }
    visitor.done();

    for (auto sigmasq : grid.sigmasq()) {
        visitor.add().sigmasq = sigmasq;
    }
    visitor.done();

    for (auto nu : grid.nu()) {
        visitor.add().nu = nu;
    }
    visitor.done();
}

} // namespace distributions
