#pragma once

#include <distributions/mixture.hpp>
#include <distributions/clustering.hpp>
#include <distributions/models/dd.hpp>
#include <distributions/models/dpd.hpp>
#include <distributions/models/nich.hpp>
#include <distributions/models/gp.hpp>

namespace loom
{

template<bool cond, class X, class Y> struct static_if;
template<class X, class Y> struct static_if<true, X, Y> { typedef X t; };
template<class X, class Y> struct static_if<false, X, Y> { typedef Y t; };

template<class Model>
struct BaseModel
{
    template<bool cached>
    struct Mixture
    {
        typedef typename static_if<
            cached,
            typename Model::CachedMixture,
            typename Model::SimpleMixture>::t t;
    };
};


struct Clustering : BaseModel<Clustering>
{
    typedef typename distributions::Clustering<int>::PitmanYor Model;
    typedef Model Shared;
    typedef Model::Mixture CachedMixture;
    typedef distributions::MixtureDriver<Model, int> SimpleMixture;
};

template<int max_dim>
struct DirichletDiscrete : BaseModel<DirichletDiscrete<max_dim>>
{
    typedef distributions::dirichlet_discrete::Value Value;
    typedef distributions::dirichlet_discrete::Shared<max_dim> Shared;
    typedef distributions::dirichlet_discrete::Group<max_dim> Group;
    typedef distributions::dirichlet_discrete::Sampler<max_dim> Sampler;
    typedef distributions::dirichlet_discrete::Mixture<max_dim> CachedMixture;
    typedef distributions::MixtureSlave<Shared> SimpleMixture;
};

struct DirichletProcessDiscrete : BaseModel<DirichletProcessDiscrete>
{
    typedef distributions::dirichlet_process_discrete::Value Value;
    typedef distributions::dirichlet_process_discrete::Shared Shared;
    typedef distributions::dirichlet_process_discrete::Group Group;
    typedef distributions::dirichlet_process_discrete::Sampler Sampler;
    typedef distributions::dirichlet_process_discrete::Mixture CachedMixture;
    typedef distributions::MixtureSlave<Shared> SimpleMixture;
};

struct GammaPoisson : BaseModel<GammaPoisson>
{
    typedef distributions::gamma_poisson::Value Value;
    typedef distributions::gamma_poisson::Shared Shared;
    typedef distributions::gamma_poisson::Group Group;
    typedef distributions::gamma_poisson::Sampler Sampler;
    typedef distributions::gamma_poisson::Mixture CachedMixture;
    typedef distributions::MixtureSlave<Shared> SimpleMixture;
};

struct NormalInverseChiSq : BaseModel<NormalInverseChiSq>
{
    typedef distributions::normal_inverse_chi_sq::Value Value;
    typedef distributions::normal_inverse_chi_sq::Shared Shared;
    typedef distributions::normal_inverse_chi_sq::Group Group;
    typedef distributions::normal_inverse_chi_sq::Sampler Sampler;
    typedef distributions::normal_inverse_chi_sq::Mixture CachedMixture;
    typedef distributions::MixtureSlave<Shared> SimpleMixture;
};

} // namespace loom
