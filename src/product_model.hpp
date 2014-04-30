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

namespace distributions {
// Kludge because ProductModel::sample_value masks this lookup
// otherwise. Once we refactor ProductModel to fit the same pattern,
// these go away.
using gamma_poisson::sample_value;
using normal_inverse_chi_sq::sample_value;
using dirichlet_discrete::sample_value;
using dirichlet_process_discrete::sample_value;
}

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
    std::vector<distributions::dirichlet_discrete::Shared<DD_DIM>> dd;
    std::vector<distributions::dirichlet_process_discrete::Shared> dpd;
    std::vector<distributions::gamma_poisson::Shared> gp;
    std::vector<distributions::normal_inverse_chi_sq::Shared> nich;

    void load (const protobuf::ProductModel & message);
};

struct ProductModel::Mixture
{
    ProductModel::Clustering::Mixture clustering;
    std::vector<distributions::dirichlet_discrete::Mixture<DD_DIM>> dd;
    std::vector<distributions::dirichlet_process_discrete::Mixture> dpd;
    std::vector<distributions::gamma_poisson::Mixture> gp;
    std::vector<distributions::normal_inverse_chi_sq::Mixture> nich;
    distributions::MixtureIdTracker id_tracker;

    void init_empty (
            const ProductModel & model,
            rng_t & rng,
            size_t empty_group_count = 1);
    void load (
            const ProductModel & model,
            const char * filename,
            rng_t & rng,
            size_t empty_roup_count = 1);
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
    void sample_value (
            const ProductModel & model,
            const VectorFloat & probs,
            Value & value,
            rng_t & rng);

private:

    void _validate (const ProductModel & model);

    template<class Mixture>
    void init_empty_factors (
            size_t empty_group_count,
            const std::vector<typename Mixture::Shared> & shareds,
            std::vector<Mixture> & mixtures,
            rng_t & rng);

    template<class Fun>
    void apply_dense (const ProductModel & model, Fun & fun);

    template<class Fun>
    void apply_sparse (
            const ProductModel & model,
            Fun & fun,
            const Value & value);

    template<class Fun>
    void set_sparse (
            const ProductModel & model,
            Fun & fun,
            Value & value);

    struct validate_fun;
    struct load_group_fun;
    struct init_fun;
    struct dump_group_fun;
    struct add_group_fun;
    struct add_value_fun;
    struct remove_group_fun;
    struct remove_value_fun;
    struct score_fun;
    struct sample_fun;
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

template<class Fun>
inline void ProductModel::Mixture::set_sparse (
        const ProductModel & model,
        Fun & fun,
        Value & value)
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
                value.set_counts(packed_pos++, fun(model.dd[i], dd[i]));
            }
        }
        for (size_t i = 0; i < dpd.size(); ++i) {
            if (value.observed(absolute_pos++)) {
                value.set_counts(packed_pos++, fun(model.dpd[i], dpd[i]));
            }
        }
        for (size_t i = 0; i < gp.size(); ++i) {
            if (value.observed(absolute_pos++)) {
                value.set_counts(packed_pos++, fun(model.gp[i], gp[i]));
            }
        }
    } else {
        absolute_pos += dd.size() + dpd.size() + gp.size();
    }

    if (value.reals_size()) {
        size_t packed_pos = 0;
        for (size_t i = 0; i < nich.size(); ++i) {
            if (value.observed(absolute_pos++)) {
                value.set_reals(packed_pos++, fun(model.nich[i], nich[i]));
            }
        }
    }
}

struct ProductModel::Mixture::validate_fun
{
    const size_t group_count;

    template<class Mixture>
    void operator() (
            size_t,
            const typename Mixture::Shared &,
            const Mixture & mixture)
    {
        LOOM_ASSERT_EQ(mixture.groups().size(), group_count);
    }
};

inline void ProductModel::Mixture::_validate (
        const ProductModel & model)
{
    if (LOOM_DEBUG_LEVEL >= 2) {
        const size_t group_count = clustering.counts().size();
        validate_fun fun = {group_count};
        apply_dense(model, fun);
        LOOM_ASSERT_EQ(id_tracker.packed_size(), group_count);
    }
}

struct ProductModel::Mixture::add_group_fun
{
    rng_t & rng;

    template<class Mixture>
    void operator() (
            size_t,
            const typename Mixture::Shared & model,
            Mixture & mixture)
    {
        mixture.add_group(model, rng);
    }
};

struct ProductModel::Mixture::add_value_fun
{
    const size_t groupid;
    rng_t & rng;

    template<class Mixture>
    void operator() (
        const typename Mixture::Shared & model,
        Mixture & mixture,
        const typename Mixture::Value & value)
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
    add_value_fun fun = {groupid, rng};
    apply_sparse(model, fun, value);

    if (LOOM_UNLIKELY(add_group)) {
        add_group_fun fun = {rng};
        apply_dense(model, fun);
        id_tracker.add_group();
        _validate(model);
    }
}

struct ProductModel::Mixture::remove_group_fun
{
    const size_t groupid;

    template<class Mixture>
    void operator() (
            size_t,
            const typename Mixture::Shared & model,
            Mixture & mixture)
    {
        mixture.remove_group(model, groupid);
    }
};

struct ProductModel::Mixture::remove_value_fun
{
    const size_t groupid;
    rng_t & rng;

    template<class Mixture>
    void operator() (
        const typename Mixture::Shared & model,
        Mixture & mixture,
        const typename Mixture::Value & value)
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
    remove_value_fun fun = {groupid, rng};
    apply_sparse(model, fun, value);

    if (LOOM_UNLIKELY(remove_group)) {
        remove_group_fun fun = {groupid};
        apply_dense(model, fun);
        id_tracker.remove_group(groupid);
        _validate(model);
    }
}

struct ProductModel::Mixture::score_fun
{
    VectorFloat & scores;
    rng_t & rng;

    template<class Mixture>
    void operator() (
        const typename Mixture::Shared & model,
        const Mixture & mixture,
        const typename Mixture::Value & value)
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
    scores.resize(clustering.counts().size());
    clustering.score_value(model.clustering, scores);
    score_fun fun = {scores, rng};
    apply_sparse(model, fun, value);
}

struct ProductModel::Mixture::sample_fun
{
    const size_t groupid;
    rng_t & rng;

    template<class Mixture>
    typename Mixture::Value operator() (
        const typename Mixture::Shared & model,
        const Mixture & mixture)
    {
        return distributions::sample_value(model, mixture.groups(groupid), rng);
    }
};

inline void ProductModel::Mixture::sample_value (
        const ProductModel & model,
        const VectorFloat & probs,
        Value & value,
        rng_t & rng)
{
    size_t groupid = distributions::sample_from_probs(rng, probs);
    sample_fun fun = {groupid, rng};
    set_sparse(model, fun, value);
}

} // namespace loom
