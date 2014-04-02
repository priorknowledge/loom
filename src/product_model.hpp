#pragma once

#include <vector>
#include <distributions/clustering.hpp>
#include <distributions/models/dd.hpp>
#include <distributions/models/dpd.hpp>
#include <distributions/models/nich.hpp>
#include <distributions/models/gp.hpp>
#include <distributions/protobuf.hpp>
#include "common.hpp"
#include "protobuf.hpp"

namespace loom
{

using distributions::rng_t;
using distributions::VectorFloat;


struct ProductModel
{
    typedef distributions::Clustering<int>::PitmanYor Clustering;

    Clustering clustering;
    std::vector<distributions::DirichletDiscrete<16>> dd;
    std::vector<distributions::DirichletProcessDiscrete> dpd;
    std::vector<distributions::GammaPoisson> gp;
    std::vector<distributions::NormalInverseChiSq> nich;

    void load (const char * filename);
};


struct ProductMixture
{
    typedef protobuf::ProductModel::SparseValue Value;

    const ProductModel & model;
    size_t empty_groupid;
    ProductModel::Clustering::Mixture clustering;
    std::vector<distributions::DirichletDiscrete<16>::Classifier> dd;
    std::vector<distributions::DirichletProcessDiscrete::Classifier> dpd;
    std::vector<distributions::GammaPoisson::Classifier> gp;
    std::vector<distributions::NormalInverseChiSq::Classifier> nich;

    ProductMixture (const ProductModel & m) : model(m) {}

    void init (rng_t & rng);
    void add_group (rng_t & rng);
    void remove_group (size_t groupid);
    void add_value (size_t groupid, const Value & value, rng_t & rng);
    void remove_value (size_t groupid, const Value & value, rng_t & rng);
    void score (const Value & value, VectorFloat & scores, rng_t & rng);

    void load (const char * filename) { TODO("load"); }
    void dump (const char * filename);

private:

    template<class Model>
    void init_factors (
            const std::vector<Model> & models,
            std::vector<typename Model::Classifier> & mixtures,
            rng_t & rng);

    template<class Fun>
    void apply_dense (Fun & fun);

    template<class Fun>
    void apply_sparse (Fun & fun, const Value & value);

    struct score_fun;
    struct add_group_fun;
    struct remove_group_fun;
    struct add_value_fun;
    struct remove_value_fun;
    struct dump_group_fun;
};

template<class Model>
void ProductMixture::init_factors (
        const std::vector<Model> & models,
        std::vector<typename Model::Classifier> & mixtures,
        rng_t & rng)
{
    const size_t count = models.size();
    mixtures.clear();
    mixtures.resize(count);
    for (size_t i = 0; i < count; ++i) {
        const auto & model = models[i];
        auto & mixture = mixtures[i];
        mixture.groups.resize(1);
        model.group_init(mixture.groups[0], rng);
        model.classifier_init(mixture, rng);
    }
}

inline void ProductMixture::init (rng_t & rng)
{
    empty_groupid = 0;

    clustering.counts.resize(1);
    clustering.counts[0] = 0;
    model.clustering.mixture_init(clustering);

    init_factors(model.dd, dd, rng);
    init_factors(model.dpd, dpd, rng);
    init_factors(model.gp, gp, rng);
    init_factors(model.nich, nich, rng);
}

template<class Fun>
inline void ProductMixture::apply_dense (Fun & fun)
{
    //TODO("implement bb");
    for (size_t i = 0; i < dd.size(); ++i) {
        fun(model.dd[i], dd[i]);
    }
    for (size_t i = 0; i < dpd.size(); ++i) {
        fun(model.dpd[i], dpd[i]);
    }
    for (size_t i = 0; i < gp.size(); ++i) {
        fun(model.gp[i], gp[i]);
    }
    for (size_t i = 0; i < nich.size(); ++i) {
        fun(model.nich[i], nich[i]);
    }
}

template<class Fun>
inline void ProductMixture::apply_sparse (Fun & fun, const Value & value)
{
    size_t observed_pos = 0;

    if (value.booleans_size()) {
        TODO("implement bb");
    }

    if (value.counts_size()) {
        size_t data_pos = 0;
        for (size_t i = 0; i < dd.size(); ++i) {
            if (value.observed(observed_pos++)) {
                fun(model.dd[i], dd[i], value.counts(data_pos++));
            }
        }
        for (size_t i = 0; i < dpd.size(); ++i) {
            if (value.observed(observed_pos++)) {
                fun(model.dpd[i], dpd[i], value.counts(data_pos++));
            }
        }
        for (size_t i = 0; i < gp.size(); ++i) {
            if (value.observed(observed_pos++)) {
                fun(model.gp[i], gp[i], value.counts(data_pos++));
            }
        }
    }

    if (value.reals_size()) {
        size_t data_pos = 0;
        for (size_t i = 0; i < nich.size(); ++i) {
            if (value.observed(observed_pos++)) {
                fun(model.nich[i], nich[i], value.reals(data_pos++));
            }
        }
    }
}

struct ProductMixture::add_group_fun
{
    rng_t & rng;

    template<class Model>
    void operator() (
            const Model & model,
            typename Model::Classifier & mixture)
    {
        model.classifier_add_group(mixture, rng);
    }
};

inline void ProductMixture::add_group (rng_t & rng)
{
    model.clustering.mixture_add_group(clustering);
    add_group_fun fun = {rng};
    apply_dense(fun);
}

struct ProductMixture::remove_group_fun
{
    const size_t groupid;

    template<class Model>
    void operator() (
            const Model & model,
            typename Model::Classifier & mixture)
    {
        model.classifier_remove_group(mixture, groupid);
    }
};

inline void ProductMixture::remove_group (size_t groupid)
{
    LOOM_ASSERT2(groupid != empty_groupid, "cannot remove empty group");
    if (empty_groupid == clustering.counts.size() - 1) {
        empty_groupid = groupid;
    }

    model.clustering.mixture_remove_group(clustering, groupid);
    remove_group_fun fun = {groupid};
    apply_dense(fun);
}

struct ProductMixture::add_value_fun
{
    const size_t groupid;
    rng_t & rng;

    template<class Model>
    void operator() (
        const Model & model,
        typename Model::Classifier & mixture,
        const typename Model::Value & value)
    {
        model.classifier_add_value(mixture, groupid, value, rng);
    }
};

inline void ProductMixture::add_value (
        size_t groupid,
        const Value & value,
        rng_t & rng)
{
    if (unlikely(groupid == empty_groupid)) {
        empty_groupid = clustering.counts.size();
        add_group(rng);
    }

    model.clustering.mixture_add_value(clustering, groupid);
    add_value_fun fun = {groupid, rng};
    apply_sparse(fun, value);
}

struct ProductMixture::remove_value_fun
{
    const size_t groupid;
    rng_t & rng;

    template<class Model>
    void operator() (
        const Model & model,
        typename Model::Classifier & mixture,
        const typename Model::Value & value)
    {
        model.classifier_remove_value(mixture, groupid, value, rng);
    }
};

inline void ProductMixture::remove_value (
        size_t groupid,
        const Value & value,
        rng_t & rng)
{
    LOOM_ASSERT2(groupid != empty_groupid, "cannot remove empty group");

    model.clustering.mixture_remove_value(clustering, groupid);
    remove_value_fun fun = {groupid, rng};
    apply_sparse(fun, value);

    if (unlikely(clustering.counts[groupid] == 0)) {
        remove_group(groupid);
    }
}

struct ProductMixture::score_fun
{
    VectorFloat & scores;
    rng_t & rng;

    template<class Model>
    void operator() (
        const Model & model,
        const typename Model::Classifier & mixture,
        const typename Model::Value & value)
    {
        model.classifier_score(mixture, value, scores, rng);
    }
};

inline void ProductMixture::score (
        const Value & value,
        VectorFloat & scores,
        rng_t & rng)
{
    model.clustering.mixture_score(clustering, scores);
    score_fun fun = {scores, rng};
    apply_sparse(fun, value);
}

} // namespace loom
