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

enum { DD_DIM = 32 };

struct ProductModel
{
    typedef protobuf::ProductModel::SparseValue Value;
    typedef distributions::Clustering<int>::PitmanYor Clustering;
    struct Mixture;

    Clustering clustering;
    std::vector<distributions::DirichletDiscrete<DD_DIM>> dd;
    std::vector<distributions::DirichletProcessDiscrete> dpd;
    std::vector<distributions::GammaPoisson> gp;
    std::vector<distributions::NormalInverseChiSq> nich;
    size_t feature_count;

    void load (const protobuf::ProductModel & message);

    void mixture_load (Mixture & mixture, const char * filename) const;
    void mixture_dump (Mixture & mixture, const char * filename) const;
    void mixture_init (Mixture & mixture, rng_t & rng) const;
    void mixture_add_group (Mixture & mixture, rng_t & rng) const;
    void mixture_remove_group (Mixture & mixture, size_t groupid) const;
    void mixture_add_value (
            Mixture & mixture,
            size_t groupid,
            const Value & value,
            rng_t & rng) const;
    void mixture_remove_value (
            Mixture & mixture,
            size_t groupid,
            const Value & value,
            rng_t & rng) const;
    void mixture_score (
            Mixture & mixture,
            const Value & value,
            VectorFloat & scores,
            rng_t & rng) const;

private:

    template<class Model>
    void mixture_init_factors (
            const std::vector<Model> & models,
            std::vector<typename Model::Classifier> & mixtures,
            rng_t & rng) const;

    template<class Fun>
    void apply_dense (Fun & fun, Mixture & mixture) const;

    template<class Fun>
    void apply_sparse (Fun & fun, Mixture & mixture, const Value & value) const;

    struct score_fun;
    struct add_group_fun;
    struct remove_group_fun;
    struct add_value_fun;
    struct remove_value_fun;
    struct dump_group_fun;
};

struct ProductModel::Mixture
{
    size_t empty_groupid;
    ProductModel::Clustering::Mixture clustering;
    std::vector<distributions::DirichletDiscrete<DD_DIM>::Classifier> dd;
    std::vector<distributions::DirichletProcessDiscrete::Classifier> dpd;
    std::vector<distributions::GammaPoisson::Classifier> gp;
    std::vector<distributions::NormalInverseChiSq::Classifier> nich;
};

template<class Model>
void ProductModel::mixture_init_factors (
        const std::vector<Model> & models,
        std::vector<typename Model::Classifier> & mixtures,
        rng_t & rng) const
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

inline void ProductModel::mixture_init (
        Mixture & mixture,
        rng_t & rng) const
{
    mixture.empty_groupid = 0;

    mixture.clustering.counts.resize(1);
    mixture.clustering.counts[0] = 0;
    clustering.mixture_init(mixture.clustering);

    mixture_init_factors(dd, mixture.dd, rng);
    mixture_init_factors(dpd, mixture.dpd, rng);
    mixture_init_factors(gp, mixture.gp, rng);
    mixture_init_factors(nich, mixture.nich, rng);
}

template<class Fun>
inline void ProductModel::apply_dense (
        Fun & fun,
        Mixture & mixture) const
{
    //TODO("implement bb");
    for (size_t i = 0; i < dd.size(); ++i) {
        fun(dd[i], mixture.dd[i]);
    }
    for (size_t i = 0; i < dpd.size(); ++i) {
        fun(dpd[i], mixture.dpd[i]);
    }
    for (size_t i = 0; i < gp.size(); ++i) {
        fun(gp[i], mixture.gp[i]);
    }
    for (size_t i = 0; i < nich.size(); ++i) {
        fun(nich[i], mixture.nich[i]);
    }
}

template<class Fun>
inline void ProductModel::apply_sparse (
        Fun & fun,
        Mixture & mixture,
        const Value & value) const
{
    size_t observed_pos = 0;

    if (value.booleans_size()) {
        TODO("implement bb");
    }

    if (value.counts_size()) {
        size_t data_pos = 0;
        for (size_t i = 0; i < dd.size(); ++i) {
            if (value.observed(observed_pos++)) {
                fun(dd[i], mixture.dd[i], value.counts(data_pos++));
            }
        }
        for (size_t i = 0; i < dpd.size(); ++i) {
            if (value.observed(observed_pos++)) {
                fun(dpd[i], mixture.dpd[i], value.counts(data_pos++));
            }
        }
        for (size_t i = 0; i < gp.size(); ++i) {
            if (value.observed(observed_pos++)) {
                fun(gp[i], mixture.gp[i], value.counts(data_pos++));
            }
        }
    }

    if (value.reals_size()) {
        size_t data_pos = 0;
        for (size_t i = 0; i < nich.size(); ++i) {
            if (value.observed(observed_pos++)) {
                fun(nich[i], mixture.nich[i], value.reals(data_pos++));
            }
        }
    }
}

struct ProductModel::add_group_fun
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

inline void ProductModel::mixture_add_group (
        Mixture & mixture,
        rng_t & rng) const
{
    clustering.mixture_add_group(mixture.clustering);
    add_group_fun fun = {rng};
    apply_dense(fun, mixture);
}

struct ProductModel::remove_group_fun
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

inline void ProductModel::mixture_remove_group (
        Mixture & mixture,
        size_t groupid) const
{
    LOOM_ASSERT2(groupid != mixture.empty_groupid, "cannot remove empty group");
    if (mixture.empty_groupid == mixture.clustering.counts.size() - 1) {
        mixture.empty_groupid = groupid;
    }

    clustering.mixture_remove_group(mixture.clustering, groupid);
    remove_group_fun fun = {groupid};
    apply_dense(fun, mixture);
}

struct ProductModel::add_value_fun
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

inline void ProductModel::mixture_add_value (
        Mixture & mixture,
        size_t groupid,
        const Value & value,
        rng_t & rng) const
{
    if (unlikely(groupid == mixture.empty_groupid)) {
        mixture.empty_groupid = mixture.clustering.counts.size();
        mixture_add_group(mixture, rng);
    }

    clustering.mixture_add_value(mixture.clustering, groupid);
    add_value_fun fun = {groupid, rng};
    apply_sparse(fun, mixture, value);
}

struct ProductModel::remove_value_fun
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

inline void ProductModel::mixture_remove_value (
        Mixture & mixture,
        size_t groupid,
        const Value & value,
        rng_t & rng) const
{
    LOOM_ASSERT2(groupid != mixture.empty_groupid, "cannot remove empty group");

    clustering.mixture_remove_value(mixture.clustering, groupid);
    remove_value_fun fun = {groupid, rng};
    apply_sparse(fun, mixture, value);

    if (unlikely(mixture.clustering.counts[groupid] == 0)) {
        mixture_remove_group(mixture, groupid);
    }
}

struct ProductModel::score_fun
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

inline void ProductModel::mixture_score (
        Mixture & mixture,
        const Value & value,
        VectorFloat & scores,
        rng_t & rng) const
{
    scores.resize(mixture.clustering.counts.size());
    clustering.mixture_score(mixture.clustering, scores);
    score_fun fun = {scores, rng};
    apply_sparse(fun, mixture, value);
}

} // namespace loom
