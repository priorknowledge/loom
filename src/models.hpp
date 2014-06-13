#pragma once

#include <type_traits>
#include <distributions/mixture.hpp>
#include <distributions/clustering.hpp>
#include <distributions/models/bb.hpp>
#include <distributions/models/dd.hpp>
#include <distributions/models/dpd.hpp>
#include <distributions/models/gp.hpp>
#include <distributions/models/nich.hpp>

namespace loom
{

//----------------------------------------------------------------------------
// Generics

template<class Model>
struct BaseModel
{
    template<bool cached>
    struct Mixture
    {
        typedef typename std::conditional<
            cached,
            typename Model::FastMixture,
            typename Model::SmallMixture>::type t;
    };

    static Model * null () { return static_cast<Model *>(nullptr); }
};

template<class Wrapper, class Model_>
struct FeatureModel : BaseModel<Wrapper>
{
    typedef Model_ Model;
    typedef typename Model::Value Value;
    typedef typename Model::Shared Shared;
    typedef typename Model::Group Group;
    typedef typename Model::Sampler Sampler;
    typedef typename Model::FastMixture FastMixture;
    typedef typename Model::SmallMixture SmallMixture;

    static Wrapper * null () { return static_cast<Wrapper *>(nullptr); }
};

//----------------------------------------------------------------------------
// Models

struct Clustering : BaseModel<Clustering>
{
    typedef typename distributions::Clustering<int>::PitmanYor Model;
    typedef Model Shared;
    typedef Model::Mixture FastMixture;
    typedef distributions::MixtureDriver<Model, int> SmallMixture;
};

struct BetaBernoulli : FeatureModel<
        BetaBernoulli,
        distributions::BetaBernoulli>
{};

template<int max_dim>
struct DirichletDiscrete : FeatureModel<
        DirichletDiscrete<max_dim>,
        distributions::DirichletDiscrete<max_dim>>
{};

struct DirichletProcessDiscrete : FeatureModel<
        DirichletProcessDiscrete,
        distributions::DirichletProcessDiscrete>
{};

struct GammaPoisson : FeatureModel<
        GammaPoisson,
        distributions::GammaPoisson>
{};

struct NormalInverseChiSq : FeatureModel<
        NormalInverseChiSq,
        distributions::NormalInverseChiSq>
{};

//----------------------------------------------------------------------------
// Feature types

typedef BetaBernoulli BB;
typedef DirichletDiscrete<16> DD16;
typedef DirichletDiscrete<256> DD256;
typedef DirichletProcessDiscrete DPD;
typedef GammaPoisson GP;
typedef NormalInverseChiSq NICH;

template<class Fun>
inline void for_each_feature_type (Fun & fun)
{
    fun(BB::null());
    fun(DD16::null());
    fun(DD256::null());
    fun(DPD::null());
    fun(GP::null());
    fun(NICH::null());
}

template<class Fun>
inline bool for_some_feature_type (Fun & fun)
{
    return fun(BB::null())
        or fun(DD16::null())
        or fun(DD256::null())
        or fun(DPD::null())
        or fun(GP::null())
        or fun(NICH::null());
}

template<class Derived>
class ForEachFeatureType
{
    typedef typename Derived::template Container<BB>::t BBs;
    typedef typename Derived::template Container<DD16>::t DD16s;
    typedef typename Derived::template Container<DD256>::t DD256s;
    typedef typename Derived::template Container<DPD>::t DPDs;
    typedef typename Derived::template Container<GP>::t GPs;
    typedef typename Derived::template Container<NICH>::t NICHs;

public:

    BBs bb;
    DD16s dd16;
    DD256s dd256;
    DPDs dpd;
    GPs gp;
    NICHs nich;

    BBs & operator[] (BB *) { return bb; }
    DD16s & operator[] (DD16 *) { return dd16; }
    DD256s & operator[] (DD256 *) { return dd256; }
    DPDs & operator[] (DPD *) { return dpd; }
    GPs & operator[] (GP *) { return gp; }
    NICHs & operator[] (NICH *) { return nich; }

    const BBs & operator[] (BB *) const { return bb; }
    const DD16s & operator[] (DD16 *) const { return dd16; }
    const DD256s & operator[] (DD256 *) const { return dd256; }
    const DPDs & operator[] (DPD *) const { return dpd; }
    const GPs & operator[] (GP *) const { return gp; }
    const NICHs & operator[] (NICH *) const { return nich; }
};

} // namespace loom
