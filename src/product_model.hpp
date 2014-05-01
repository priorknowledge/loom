#pragma once

#include <vector>
#include <distributions/io/protobuf.hpp>
#include "common.hpp"
#include "protobuf.hpp"
#include "models.hpp"
#include "infer_grid.hpp"

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

enum { DD_DIM = 256 };

struct ProductModel
{
    typedef protobuf::ProductModel::SparseValue Value;

    protobuf::SparseValueSchema schema;
    Clustering::Shared clustering;
    std::vector<DirichletDiscrete<DD_DIM>::Shared> dd;
    std::vector<DirichletProcessDiscrete::Shared> dpd;
    std::vector<GammaPoisson::Shared> gp;
    std::vector<NormalInverseChiSq::Shared> nich;

    void load (const protobuf::ProductModel_Shared & message);

    template<bool cached> struct Mixture;
    typedef Mixture<false> SimpleMixture;
    typedef Mixture<true> CachedMixture;
};

template<bool cached>
struct ProductModel::Mixture
{
    typename Clustering::Mixture<cached>::t clustering;
    std::vector<typename DirichletDiscrete<DD_DIM>::Mixture<cached>::t> dd;
    std::vector<typename DirichletProcessDiscrete::Mixture<cached>::t> dpd;
    std::vector<typename GammaPoisson::Mixture<cached>::t> gp;
    std::vector<typename NormalInverseChiSq::Mixture<cached>::t> nich;
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

    void infer_hypers (
            ProductModel & model,
            const protobuf::ProductModel::HyperPrior & hyper_prior,
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
    void apply_dense (ProductModel & model, Fun & fun);

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
    struct infer_hypers_fun;
};

template<bool cached>
template<class Fun>
inline void ProductModel::Mixture<cached>::apply_dense (
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

template<bool cached>
template<class Fun>
inline void ProductModel::Mixture<cached>::apply_dense (
        ProductModel & model,
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

template<bool cached>
template<class Fun>
inline void ProductModel::Mixture<cached>::apply_sparse (
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

template<bool cached>
template<class Fun>
inline void ProductModel::Mixture<cached>::set_sparse (
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

template<bool cached>
struct ProductModel::Mixture<cached>::validate_fun
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

template<bool cached>
inline void ProductModel::Mixture<cached>::_validate (
        const ProductModel & model)
{
    if (LOOM_DEBUG_LEVEL >= 2) {
        const size_t group_count = clustering.counts().size();
        validate_fun fun = {group_count};
        apply_dense(model, fun);
        LOOM_ASSERT_EQ(id_tracker.packed_size(), group_count);
    }
}

template<bool cached>
struct ProductModel::Mixture<cached>::add_group_fun
{
    rng_t & rng;

    template<class Mixture>
    void operator() (
            size_t,
            const typename Mixture::Shared & shared,
            Mixture & mixture)
    {
        mixture.add_group(shared, rng);
    }
};

template<bool cached>
struct ProductModel::Mixture<cached>::add_value_fun
{
    const size_t groupid;
    rng_t & rng;

    template<class Mixture>
    void operator() (
        const typename Mixture::Shared & shared,
        Mixture & mixture,
        const typename Mixture::Value & value)
    {
        mixture.add_value(shared, groupid, value, rng);
    }
};

template<bool cached>
inline void ProductModel::Mixture<cached>::add_value (
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

template<bool cached>
struct ProductModel::Mixture<cached>::remove_group_fun
{
    const size_t groupid;

    template<class Mixture>
    void operator() (
            size_t,
            const typename Mixture::Shared & shared,
            Mixture & mixture)
    {
        mixture.remove_group(shared, groupid);
    }
};

template<bool cached>
struct ProductModel::Mixture<cached>::remove_value_fun
{
    const size_t groupid;
    rng_t & rng;

    template<class Mixture>
    void operator() (
        const typename Mixture::Shared & shared,
        Mixture & mixture,
        const typename Mixture::Value & value)
    {
        mixture.remove_value(shared, groupid, value, rng);
    }
};

template<bool cached>
inline void ProductModel::Mixture<cached>::remove_value (
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

template<bool cached>
struct ProductModel::Mixture<cached>::score_fun
{
    VectorFloat & scores;
    rng_t & rng;

    template<class Mixture>
    void operator() (
        const typename Mixture::Shared & shared,
        const Mixture & mixture,
        const typename Mixture::Value & value)
    {
        mixture.score_value(shared, value, scores, rng);
    }
};

template<bool cached>
inline void ProductModel::Mixture<cached>::score (
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

template<bool cached>
struct ProductModel::Mixture<cached>::sample_fun
{
    const size_t groupid;
    rng_t & rng;

    template<class Mixture>
    typename Mixture::Value operator() (
        const typename Mixture::Shared & shared,
        const Mixture & mixture)
    {
        return distributions::sample_value(
            shared,
            mixture.groups(groupid),
            rng);
    }
};

template<bool cached>
inline void ProductModel::Mixture<cached>::sample_value (
        const ProductModel & model,
        const VectorFloat & probs,
        Value & value,
        rng_t & rng)
{
    size_t groupid = distributions::sample_from_probs(rng, probs);
    sample_fun fun = {groupid, rng};
    set_sparse(model, fun, value);
}


template<bool cached>
template<class MixtureT>
void ProductModel::Mixture<cached>::init_empty_factors (
        size_t empty_group_count,
        const std::vector<typename MixtureT::Shared> & shareds,
        std::vector<MixtureT> & mixtures,
        rng_t & rng)
{
    const size_t shared_count = shareds.size();
    mixtures.clear();
    mixtures.resize(shared_count);
    for (size_t i = 0; i < shared_count; ++i) {
        const auto & shared = shareds[i];
        auto & mixture = mixtures[i];
        mixture.groups().resize(empty_group_count);
        for (auto & group : mixture.groups()) {
            group.init(shared, rng);
        }
        mixture.init(shared, rng);
    }
}

template<bool cached>
void ProductModel::Mixture<cached>::init_empty (
        const ProductModel & model,
        rng_t & rng,
        size_t empty_group_count)
{
    std::vector<int> counts(empty_group_count, 0);
    clustering.init(model.clustering, counts);

    init_empty_factors(empty_group_count, model.dd, dd, rng);
    init_empty_factors(empty_group_count, model.dpd, dpd, rng);
    init_empty_factors(empty_group_count, model.gp, gp, rng);
    init_empty_factors(empty_group_count, model.nich, nich, rng);

    id_tracker.init(empty_group_count);

    _validate(model);
}

template<bool cached>
struct ProductModel::Mixture<cached>::load_group_fun
{
    size_t groupid;
    const protobuf::ProductModel::Group & message;

    template<class Mixture>
    void operator() (
            size_t index,
            const typename Mixture::Shared & shared,
            Mixture & mixture)
    {
        mixture.groups().resize(mixture.groups().size() + 1);
        distributions::group_load(
            shared,
            mixture.groups(groupid),
            protobuf::Groups<typename Mixture::Group>::get(message).Get(index));
    }
};

template<bool cached>
struct ProductModel::Mixture<cached>::init_fun
{
    rng_t & rng;

    template<class Mixture>
    void operator() (
            size_t,
            const typename Mixture::Shared & shared,
            Mixture & mixture)
    {
        mixture.init(shared, rng);
    }
};

template<bool cached>
void ProductModel::Mixture<cached>::load (
        const ProductModel & model,
        const char * filename,
        rng_t & rng,
        size_t empty_group_count)
{
    init_empty(model, rng, empty_group_count);
    protobuf::InFile groups(filename);
    protobuf::ProductModel::Group message;
    for (size_t groupid = 0; groups.try_read_stream(message); ++groupid) {
        clustering.add_value(model.clustering, groupid, message.count());
        load_group_fun fun = {groupid, message};
        apply_dense(model, fun);
    }
    init_fun fun = {rng};
    apply_dense(model, fun);
    id_tracker.init(clustering.counts().size());
    _validate(model);
}

template<bool cached>
struct ProductModel::Mixture<cached>::dump_group_fun
{
    size_t groupid;
    protobuf::ProductModel::Group & message;

    template<class Mixture>
    void operator() (
            size_t,
            const typename Mixture::Shared & shared,
            const Mixture & mixture)
    {
        distributions::group_dump(
            shared,
            mixture.groups(groupid),
            * protobuf::Groups<typename Mixture::Group>::get(message).Add());
    }
};

template<bool cached>
void ProductModel::Mixture<cached>::dump (
        const ProductModel & model,
        const char * filename)
{
    protobuf::OutFile groups_stream(filename);
    protobuf::ProductModel::Group message;
    const size_t group_count = clustering.counts().size();
    for (size_t i = 0; i < group_count; ++i) {
        bool group_is_not_empty = clustering.counts(i);
        if (group_is_not_empty) {
            message.set_count(clustering.counts(i));
            dump_group_fun fun = {i, message};
            apply_dense(model, fun);
            groups_stream.write_stream(message);
            message.Clear();
        }
    }
}

template<bool cached>
struct ProductModel::Mixture<cached>::infer_hypers_fun
{
    const protobuf::ProductModel_HyperPrior & hyper_prior;
    rng_t & rng;

    template<class Mixture>
    void operator() (
            size_t,
            typename Mixture::Shared & shared,
            Mixture & mixture)
    {
        InferShared<Mixture> infer_shared(shared, mixture, rng);
        const auto & grid_prior =
            protobuf::GridPriors<typename Mixture::Shared>::get(hyper_prior);
        distributions::for_each_gridpoint(grid_prior, infer_shared);
    }

    template<class Mixture>
    void operator() (
            size_t,
            DirichletProcessDiscrete::Shared &,
            Mixture &)
    {
        // TODO implement DPD inference
    }
};

template<bool cached>
void ProductModel::Mixture<cached>::infer_hypers (
        ProductModel & model,
        const protobuf::ProductModel_HyperPrior & hyper_prior,
        rng_t & rng)
{
    // TODO infer clustering hypers

    infer_hypers_fun fun = {hyper_prior, rng};
    apply_dense(model, fun);
}

} // namespace loom
