#pragma once

#include <vector>
#include <distributions/clustering.hpp>
#include <distributions/models/dd.hpp>
#include <distributions/models/dpd.hpp>
#include <distributions/models/nich.hpp>
#include <distributions/models/gp.hpp>
#include <distributions/mixture.hpp>
#include <distributions/io/protobuf.hpp>
#include "common.hpp"
#include "protobuf.hpp"

namespace loom
{

using distributions::rng_t;
using distributions::VectorFloat;

enum { DD_DIM = 256 };

struct ProductModel
{
    typedef protobuf::ProductModel::SparseValue Value;
    typedef distributions::Clustering<int>::PitmanYor Clustering;
    struct Mixture;

    protobuf::SparseValueSchema schema;
    Clustering clustering;
    std::vector<distributions::DirichletDiscrete<DD_DIM>> dd;
    std::vector<distributions::DirichletProcessDiscrete> dpd;
    std::vector<distributions::GammaPoisson> gp;
    std::vector<distributions::NormalInverseChiSq> nich;

    void load (const protobuf::ProductModel & message);
};

struct ProductModel::Mixture
{
    ProductModel::Clustering::Mixture clustering;
    std::vector<distributions::DirichletDiscrete<DD_DIM>::Mixture> dd;
    std::vector<distributions::DirichletProcessDiscrete::Mixture> dpd;
    std::vector<distributions::GammaPoisson::Mixture> gp;
    std::vector<distributions::NormalInverseChiSq::Mixture> nich;
    distributions::MixtureIdTracker id_tracker;

    void init_empty (const ProductModel & model, rng_t & rng);
    void load (const ProductModel & model, const char * filename, rng_t & rng);
    void dump (const ProductModel & model, const char * filename);
    void add_value (
            const ProductModel & model,
            size_t groupid,
            const Value & value,
            rng_t & rng);
    void remove_value (
            const ProductModel & model,
            size_t groupid,
            const Value & value,
            rng_t & rng);
    void score (
            const ProductModel & model,
            const Value & value,
            VectorFloat & scores,
            rng_t & rng);

private:

    template<class Model>
    void init_empty_factors (
            const std::vector<Model> & models,
            std::vector<typename Model::Mixture> & mixtures,
            rng_t & rng);

    template<class Fun>
    void apply_dense (const ProductModel & model, Fun & fun);

    template<class Fun>
    void apply_sparse (
            const ProductModel & model,
            Fun & fun,
            const Value & value);

    struct load_group_fun;
    struct init_fun;
    struct dump_group_fun;
    struct add_group_fun;
    struct add_value_fun;
    struct remove_group_fun;
    struct remove_value_fun;
    struct score_fun;
};

template<class Fun>
inline void ProductModel::Mixture::apply_dense (
        const ProductModel & model,
        Fun & fun)
{
    //TODO("implement bb");
    for (size_t i = 0; i < dd.size(); ++i) {
        fun(i, model.dd[i], dd[i]);
    }
    for (size_t i = 0; i < dpd.size(); ++i) {
        fun(i, model.dpd[i], dpd[i]);
    }
    for (size_t i = 0; i < gp.size(); ++i) {
        fun(i, model.gp[i], gp[i]);
    }
    for (size_t i = 0; i < nich.size(); ++i) {
        fun(i, model.nich[i], nich[i]);
    }
}

template<class Fun>
inline void ProductModel::Mixture::apply_sparse (
        const ProductModel & model,
        Fun & fun,
        const Value & value)
{
    if (LOOM_DEBUG_LEVEL >= 2) {
        model.schema.validate(value);
    }

    size_t absolute_pos = 0;

    if (value.booleans_size()) {
        TODO("implement bb");
    } else {
        absolute_pos += 0;
    }

    if (value.counts_size()) {
        size_t packed_pos = 0;
        for (size_t i = 0; i < dd.size(); ++i) {
            if (value.observed(absolute_pos++)) {
                fun(model.dd[i], dd[i], value.counts(packed_pos++));
            }
        }
        for (size_t i = 0; i < dpd.size(); ++i) {
            if (value.observed(absolute_pos++)) {
                fun(model.dpd[i], dpd[i], value.counts(packed_pos++));
            }
        }
        for (size_t i = 0; i < gp.size(); ++i) {
            if (value.observed(absolute_pos++)) {
                fun(model.gp[i], gp[i], value.counts(packed_pos++));
            }
        }
    } else {
        absolute_pos += dd.size() + dpd.size() + gp.size();
    }

    if (value.reals_size()) {
        size_t packed_pos = 0;
        for (size_t i = 0; i < nich.size(); ++i) {
            if (value.observed(absolute_pos++)) {
                fun(model.nich[i], nich[i], value.reals(packed_pos++));
            }
        }
    }
}

struct ProductModel::Mixture::add_group_fun
{
    rng_t & rng;

    template<class Model>
    void operator() (
            size_t,
            const Model & model,
            typename Model::Mixture & mixture)
    {
        mixture.add_group(model, rng);
    }
};

struct ProductModel::Mixture::add_value_fun
{
    const size_t groupid;
    rng_t & rng;

    template<class Model>
    void operator() (
        const Model & model,
        typename Model::Mixture & mixture,
        const typename Model::Value & value)
    {
        mixture.add_value(model, groupid, value, rng);
    }
};

inline void ProductModel::Mixture::add_value (
        const ProductModel & model,
        size_t groupid,
        const Value & value,
        rng_t & rng)
{
    bool add_group = clustering.add_value(model.clustering, groupid);
    if (LOOM_UNLIKELY(add_group)) {
        add_group_fun fun = {rng};
        apply_dense(model, fun);
        id_tracker.add_group();
    }

    add_value_fun fun = {groupid, rng};
    apply_sparse(model, fun, value);
}

struct ProductModel::Mixture::remove_group_fun
{
    const size_t groupid;

    template<class Model>
    void operator() (
            size_t,
            const Model & model,
            typename Model::Mixture & mixture)
    {
        mixture.remove_group(model, groupid);
    }
};

struct ProductModel::Mixture::remove_value_fun
{
    const size_t groupid;
    rng_t & rng;

    template<class Model>
    void operator() (
        const Model & model,
        typename Model::Mixture & mixture,
        const typename Model::Value & value)
    {
        mixture.remove_value(model, groupid, value, rng);
    }
};

inline void ProductModel::Mixture::remove_value (
        const ProductModel & model,
        size_t groupid,
        const Value & value,
        rng_t & rng)
{
    bool remove_group = clustering.remove_value(model.clustering, groupid);
    if (LOOM_UNLIKELY(remove_group)) {
        remove_group_fun fun = {groupid};
        apply_dense(model, fun);
        id_tracker.remove_group(groupid);
    }

    remove_value_fun fun = {groupid, rng};
    apply_sparse(model, fun, value);
}

struct ProductModel::Mixture::score_fun
{
    VectorFloat & scores;
    rng_t & rng;

    template<class Model>
    void operator() (
        const Model & model,
        const typename Model::Mixture & mixture,
        const typename Model::Value & value)
    {
        mixture.score_value(model, value, scores, rng);
    }
};

inline void ProductModel::Mixture::score (
        const ProductModel & model,
        const Value & value,
        VectorFloat & scores,
        rng_t & rng)
{
    scores.resize(clustering.counts.size());
    clustering.score(model.clustering, scores);
    score_fun fun = {scores, rng};
    apply_sparse(model, fun, value);
}

} // namespace loom
