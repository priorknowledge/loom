#pragma once

#include <vector>
#include <distributions/io/protobuf.hpp>
#include "common.hpp"
#include "indexed_vector.hpp"
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
    IndexedVector<DirichletDiscrete<DD_DIM>::Shared> dd;
    IndexedVector<DirichletProcessDiscrete::Shared> dpd;
    IndexedVector<GammaPoisson::Shared> gp;
    IndexedVector<NormalInverseChiSq::Shared> nich;

    void clear ();

    void load (
            const protobuf::ProductModel_Shared & message,
            const std::vector<size_t> & featureids);

    void update_schema ();

    template<class SourceMixture, class DestinMixture>
    static void move_groups_to (
            size_t featureid,
            SourceMixture & source_mixture,
            DestinMixture & destin_mixture);

    template<class Mixture>
    static void move_shared_to (
            size_t featureid,
            ProductModel & source_model, Mixture & source_mixture,
            ProductModel & destin_model, Mixture & destin_mixture,
            rng_t & rng);

    template<bool cached> struct Mixture;
    typedef Mixture<false> SimpleMixture;
    typedef Mixture<true> CachedMixture;

private:

    struct move_groups_to_fun;
    struct move_shared_to_fun;
};

template<bool cached>
struct ProductModel::Mixture
{
    typename Clustering::Mixture<cached>::t clustering;
    IndexedVector<typename DirichletDiscrete<DD_DIM>::Mixture<cached>::t> dd;
    IndexedVector<typename DirichletProcessDiscrete::Mixture<cached>::t> dpd;
    IndexedVector<typename GammaPoisson::Mixture<cached>::t> gp;
    IndexedVector<typename NormalInverseChiSq::Mixture<cached>::t> nich;
    distributions::MixtureIdTracker id_tracker;

    void init_empty (
            const ProductModel & model,
            rng_t & rng,
            size_t empty_group_count = 1);

    void init_featureless (
            const ProductModel & model,
            const std::vector<int> & counts);

    void load (
            const ProductModel & model,
            const char * filename,
            rng_t & rng,
            size_t empty_group_count = 1);

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

    void score_value (
            const ProductModel & model,
            const Value & value,
            VectorFloat & scores,
            rng_t & rng);

    float score_feature (
            const ProductModel & model,
            size_t featureid,
            rng_t & rng) const;

    void sample_value (
            const ProductModel & model,
            const VectorFloat & probs,
            Value & value,
            rng_t & rng);

    void infer_clustering_hypers (
            ProductModel & model,
            const protobuf::ProductModel::HyperPrior & hyper_prior,
            rng_t & rng) const;

    void infer_feature_hypers (
            ProductModel & model,
            const protobuf::ProductModel::HyperPrior & hyper_prior,
            size_t featureid,
            rng_t & rng) const;

private:

    void _validate (const ProductModel & model);

    template<class Mixture>
    void init_empty_factors (
            size_t empty_group_count,
            const IndexedVector<typename Mixture::Shared> & shareds,
            IndexedVector<Mixture> & mixtures,
            rng_t & rng);

    template<class Fun>
    void apply_dense (ProductModel & model, Fun & fun) const;

    template<class Fun>
    void apply_dense (const ProductModel & model, Fun & fun);

    template<class Fun>
    void apply_to_feature (
            ProductModel & model,
            Fun & fun,
            size_t featureid) const;

    template<class Fun>
    void apply_to_feature (
            const ProductModel & model,
            Fun & fun,
            size_t featureid) const;

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
    struct score_value_fun;
    struct score_feature_fun;
    struct sample_fun;
    struct infer_hypers_fun;
};

template<bool cached>
template<class Fun>
inline void ProductModel::Mixture<cached>::apply_dense (
        const ProductModel & model,
        Fun & fun)
{
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
        Fun & fun) const
{
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
inline void ProductModel::Mixture<cached>::apply_to_feature (
        ProductModel & model,
        Fun & fun,
        size_t featureid) const
{
    if (auto maybe_pos = dd.try_find_pos(featureid)) {
        size_t i = maybe_pos.value();
        fun(i, model.dd[i], dd[i]);
    } else if (auto maybe_pos = dpd.try_find_pos(featureid)) {
        size_t i = maybe_pos.value();
        fun(i, model.dpd[i], dpd[i]);
    } else if (auto maybe_pos = gp.try_find_pos(featureid)) {
        size_t i = maybe_pos.value();
        fun(i, model.gp[i], gp[i]);
    } else if (auto maybe_pos = nich.try_find_pos(featureid)) {
        size_t i = maybe_pos.value();
        fun(i, model.nich[i], nich[i]);
    } else {
        LOOM_ERROR("feature not found: " << featureid);
    }
}

template<bool cached>
template<class Fun>
inline void ProductModel::Mixture<cached>::apply_to_feature (
        const ProductModel & model,
        Fun & fun,
        size_t featureid) const
{
    if (auto maybe_pos = dd.try_find_pos(featureid)) {
        size_t i = maybe_pos.value();
        fun(i, model.dd[i], dd[i]);
    } else if (auto maybe_pos = dpd.try_find_pos(featureid)) {
        size_t i = maybe_pos.value();
        fun(i, model.dpd[i], dpd[i]);
    } else if (auto maybe_pos = gp.try_find_pos(featureid)) {
        size_t i = maybe_pos.value();
        fun(i, model.gp[i], gp[i]);
    } else if (auto maybe_pos = nich.try_find_pos(featureid)) {
        size_t i = maybe_pos.value();
        fun(i, model.nich[i], nich[i]);
    } else {
        LOOM_ERROR("feature not found: " << featureid);
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
struct ProductModel::Mixture<cached>::score_value_fun
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
inline void ProductModel::Mixture<cached>::score_value (
        const ProductModel & model,
        const Value & value,
        VectorFloat & scores,
        rng_t & rng)
{
    scores.resize(clustering.counts().size());
    clustering.score_value(model.clustering, scores);
    score_value_fun fun = {scores, rng};
    apply_sparse(model, fun, value);
}

template<bool cached>
struct ProductModel::Mixture<cached>::score_feature_fun
{
    rng_t & rng;
    float score;

    template<class Mixture>
    void operator() (
            size_t,
            const typename Mixture::Shared & shared,
            const Mixture & mixture)
    {
        score = mixture.score_mixture(shared, rng);
    }
};

template<bool cached>
inline float ProductModel::Mixture<cached>::score_feature (
        const ProductModel & model,
        size_t featureid,
        rng_t & rng) const
{
    score_feature_fun fun = {rng, NAN};
    apply_to_feature(model, fun, featureid);
    return fun.score;
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
        const IndexedVector<typename MixtureT::Shared> & shareds,
        IndexedVector<MixtureT> & mixtures,
        rng_t & rng)
{
    mixtures.clear();
    for (size_t i = 0; i < shareds.size(); ++i) {
        const auto & shared = shareds[i];
        auto & mixture = mixtures.insert(shareds.index(i));
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
void ProductModel::Mixture<cached>::init_featureless (
        const ProductModel & model,
        const std::vector<int> & counts)
{
    clustering.init(model.clustering, counts);

    LOOM_ASSERT(
        model.schema.total_size() == 0,
        "cannot init_featureless with features present");

    id_tracker.init(counts.size());

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
            const Mixture &)
    {
        // TODO implement DPD inference
    }
};

template<bool cached>
void ProductModel::Mixture<cached>::infer_feature_hypers (
        ProductModel & model,
        const protobuf::ProductModel_HyperPrior & hyper_prior,
        size_t featureid,
        rng_t & rng) const
{
    infer_hypers_fun fun = {hyper_prior, rng};
    apply_to_feature(model, fun, featureid);
}

template<bool cached>
void ProductModel::Mixture<cached>::infer_clustering_hypers (
        ProductModel &,
        const protobuf::ProductModel_HyperPrior &,
        rng_t &) const
{
    // TODO infer clustering hypers
}

struct ProductModel::move_groups_to_fun
{
    const size_t featureid;

    template<class SourceMixture, class DestinMixture>
    bool operator() (
            IndexedVector<SourceMixture> & sources,
            IndexedVector<DestinMixture> & destins) const
    {
        if (sources.try_find_pos(featureid)) {

            SourceMixture & source = sources.find(featureid);
            DestinMixture & destin = destins.find_or_insert(featureid);
            destin.groups() = std::move(source.groups());

            return true;
        } else {
            return false;
        }
    }
};

template<class SourceMixture, class DestinMixture>
inline void ProductModel::move_groups_to (
        size_t featureid,
        SourceMixture & source,
        DestinMixture & destin)
{
    move_groups_to_fun fun = {featureid};

    bool found =
        fun(source.dd, destin.dd) or
        fun(source.dpd, destin.dpd) or
        fun(source.gp, destin.gp) or
        fun(source.nich, destin.nich);

    LOOM_ASSERT(found, "feature not found: " << featureid);
}

struct ProductModel::move_shared_to_fun
{
    const size_t featureid;
    rng_t & rng;

    template<class Shared, class Mixture>
    bool operator() (
            IndexedVector<Shared> & source_shareds,
            IndexedVector<Mixture> & source_mixtures,
            IndexedVector<Shared> & destin_shareds,
            IndexedVector<Mixture> & destin_mixtures) const
    {
        if (source_shareds.try_find_pos(featureid)) {

            Shared & source_shared = source_shareds.find(featureid);
            Shared & destin_shared = destin_shareds.insert(featureid);
            destin_shared = std::move(source_shared);
            source_shareds.remove(featureid);

            // assume move_groups_to has already been called
            source_mixtures.remove(featureid);
            destin_mixtures.find(featureid).init(destin_shared, rng);

            return true;
        } else {
            return false;
        }
    }
};

template<class MixtureT>
inline void ProductModel::move_shared_to (
        size_t featureid,
        ProductModel & source_model, MixtureT & source_mixture,
        ProductModel & destin_model, MixtureT & destin_mixture,
        rng_t & rng)
{
    move_shared_to_fun fun = {featureid, rng};

    bool found =
        fun(source_model.dd, source_mixture.dd,
            destin_model.dd, destin_mixture.dd) or
        fun(source_model.dpd, source_mixture.dpd,
            destin_model.dpd, destin_mixture.dpd) or
        fun(source_model.gp, source_mixture.gp,
            destin_model.gp, destin_mixture.gp) or
        fun(source_model.nich, source_mixture.nich,
            destin_model.nich, destin_mixture.nich);

    LOOM_ASSERT(found, "feature not found: " << featureid);

    source_model.update_schema();
    destin_model.update_schema();
}

} // namespace loom
