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

//----------------------------------------------------------------------------
// Generics

template<class, bool> struct Reference;
template<class T> struct Reference<T, false> { typedef T & t; };
template<class T> struct Reference<T, true> { typedef const T & t; };

template<class Fun, class X, bool X_const, class Y, bool Y_const>
struct for_each_feature_fun
{
    Fun & fun;
    typename Reference<X, X_const>::t xs;
    typename Reference<Y, Y_const>::t ys;

    template<class T>
    void operator() (T * t)
    {
        auto & x = xs[t];
        auto & y = ys[t];
        for (size_t i = 0, size = x.size(); i < size; ++i) {
            fun(i, x[i], y[i]);
        }
    }
};

template<class Fun, class X, bool X_const, class Y, bool Y_const>
inline void for_each_feature (
        Fun & fun,
        typename Reference<X, X_const>::t xs,
        typename Reference<Y, Y_const>::t ys)
{
    for_each_feature_fun<Fun, X, X_const, Y, Y_const> loop = {fun, xs, ys};
    for_each_feature_type(loop);
}

//----------------------------------------------------------------------------
// Product Model

enum { DD_DIM = 256 };

struct ProductModel
{
    typedef protobuf::ProductModel::SparseValue Value;
    struct Feature_
    {
        template<class T>
        struct Container { typedef IndexedVector<typename T::Shared> t; };
    };
    typedef ForEachFeatureType<Feature_> Features;

    protobuf::SparseValueSchema schema;
    Clustering::Shared clustering;
    Features features;

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

    void extend (const ProductModel & other);

    template<bool cached> struct Mixture;
    typedef Mixture<false> SimpleMixture;
    typedef Mixture<true> CachedMixture;

private:

    template<class SourceMixture, class DestinMixture>
    struct move_groups_to_fun;

    template<class MixtureT>
    struct move_shared_to_fun;

    struct extend_fun;
    struct clear_fun;
};

template<bool cached>
struct ProductModel::Mixture
{
    struct Feature_
    {
        template<class T>
        struct Container
        {
            typedef IndexedVector<typename T::template Mixture<cached>::t> t;
        };
    };
    typedef ForEachFeatureType<Feature_> Features;

    typename Clustering::Mixture<cached>::t clustering;
    Features features;
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

    void validate (const ProductModel & model) const;

private:

    template<class Fun>
    void read_sparse_value (
            const ProductModel & model,
            Fun & fun,
            const Value & value);

    template<class Fun>
    void write_sparse_value (
            const ProductModel & model,
            Fun & fun,
            Value & value);

    struct for_each_feature_fun_1;
    struct for_each_feature_fun_2;
    struct for_each_feature_fun_3;
    struct validate_fun;
    struct load_group_fun;
    struct init_fun;
    struct init_empty_factors_fun;
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
inline void ProductModel::Mixture<cached>::read_sparse_value (
        const ProductModel & model,
        Fun & fun,
        const Value & value)
{
    if (LOOM_DEBUG_LEVEL >= 2) {
        model.schema.validate(value);
    }

    // HACK --------------------------------
    auto & dd = features.dd256;
    auto & dpd = features.dpd;
    auto & gp = features.gp;
    auto & nich = features.nich;
    auto & model_dd = model.features.dd256;
    auto & model_dpd = model.features.dpd;
    auto & model_gp = model.features.gp;
    auto & model_nich = model.features.nich;
    //--------------------------------------

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
                fun(model_dd[i], dd[i], value.counts(packed_pos++));
            }
        }
        for (size_t i = 0; i < dpd.size(); ++i) {
            if (value.observed(absolute_pos++)) {
                fun(model_dpd[i], dpd[i], value.counts(packed_pos++));
            }
        }
        for (size_t i = 0; i < gp.size(); ++i) {
            if (value.observed(absolute_pos++)) {
                fun(model_gp[i], gp[i], value.counts(packed_pos++));
            }
        }
    } else {
        absolute_pos += dd.size() + dpd.size() + gp.size();
    }

    if (value.reals_size()) {
        size_t packed_pos = 0;
        for (size_t i = 0; i < nich.size(); ++i) {
            if (value.observed(absolute_pos++)) {
                fun(model_nich[i], nich[i], value.reals(packed_pos++));
            }
        }
    }
}

template<bool cached>
template<class Fun>
inline void ProductModel::Mixture<cached>::write_sparse_value (
        const ProductModel & model,
        Fun & fun,
        Value & value)
{
    if (LOOM_DEBUG_LEVEL >= 2) {
        model.schema.validate(value);
    }

    // HACK --------------------------------
    auto & dd = features.dd256;
    auto & dpd = features.dpd;
    auto & gp = features.gp;
    auto & nich = features.nich;
    auto & model_dd = model.features.dd256;
    auto & model_dpd = model.features.dpd;
    auto & model_gp = model.features.gp;
    auto & model_nich = model.features.nich;
    //--------------------------------------

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
                value.set_counts(packed_pos++, fun(model_dd[i], dd[i]));
            }
        }
        for (size_t i = 0; i < dpd.size(); ++i) {
            if (value.observed(absolute_pos++)) {
                value.set_counts(packed_pos++, fun(model_dpd[i], dpd[i]));
            }
        }
        for (size_t i = 0; i < gp.size(); ++i) {
            if (value.observed(absolute_pos++)) {
                value.set_counts(packed_pos++, fun(model_gp[i], gp[i]));
            }
        }
    } else {
        absolute_pos += dd.size() + dpd.size() + gp.size();
    }

    if (value.reals_size()) {
        size_t packed_pos = 0;
        for (size_t i = 0; i < nich.size(); ++i) {
            if (value.observed(absolute_pos++)) {
                value.set_reals(packed_pos++, fun(model_nich[i], nich[i]));
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
inline void ProductModel::Mixture<cached>::validate (
        const ProductModel & model) const
{
    if (LOOM_DEBUG_LEVEL >= 2) {
        const size_t group_count = clustering.counts().size();
        validate_fun fun = {group_count};
        for_each_feature
            <validate_fun, ProductModel::Features, true, Features, true>
            (fun, model.features, features);
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
    read_sparse_value(model, fun, value);

    if (LOOM_UNLIKELY(add_group)) {
        add_group_fun fun = {rng};
        for_each_feature
            <add_group_fun, ProductModel::Features, true, Features, false>
            (fun, model.features, features);
        id_tracker.add_group();
        validate(model);
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
    read_sparse_value(model, fun, value);

    if (LOOM_UNLIKELY(remove_group)) {
        remove_group_fun fun = {groupid};
        for_each_feature
            <remove_group_fun, ProductModel::Features, true, Features, false>
            (fun, model.features, features);
        id_tracker.remove_group(groupid);
        validate(model);
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
    read_sparse_value(model, fun, value);
}

template<bool cached>
struct ProductModel::Mixture<cached>::score_feature_fun
{
    const ProductModel::Features & shared_features;
    const Features & mixture_features;
    size_t featureid;
    rng_t & rng;
    float score;

    template<class T>
    bool operator() (T * t)
    {
        const auto & shareds = shared_features[t];
        if (auto maybe_pos = shareds.try_find_pos(featureid)) {
            const auto & mixtures = mixture_features[t];
            size_t i = maybe_pos.value();
            score = mixtures[i].score_mixture(shareds[i], rng);
            return true;
        } else {
            return false;
        }
    }
};

template<bool cached>
inline float ProductModel::Mixture<cached>::score_feature (
        const ProductModel & model,
        size_t featureid,
        rng_t & rng) const
{
    score_feature_fun fun = {model.features, features, featureid, rng, NAN};
    bool found = for_some_feature_type(fun);
    LOOM_ASSERT(found, "feature not found: " << featureid);
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
    write_sparse_value(model, fun, value);
}


template<bool cached>
struct ProductModel::Mixture<cached>::init_empty_factors_fun
{
    size_t empty_group_count;
    const ProductModel::Features & shared_features;
    Features & mixture_features;
    rng_t & rng;

    template<class T>
    void operator() (T * t)
    {
        const auto & shareds = shared_features[t];
        auto & mixtures = mixture_features[t];

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
};

template<bool cached>
void ProductModel::Mixture<cached>::init_empty (
        const ProductModel & model,
        rng_t & rng,
        size_t empty_group_count)
{
    std::vector<int> counts(empty_group_count, 0);
    clustering.init(model.clustering, counts);

    init_empty_factors_fun fun = {
        empty_group_count,
        model.features,
        features,
        rng};
    for_each_feature_type(fun);

    id_tracker.init(empty_group_count);

    validate(model);
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

    validate(model);
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
        for_each_feature
            <load_group_fun, ProductModel::Features, true, Features, false>
            (fun, model.features, features);
    }
    init_fun fun = {rng};
    for_each_feature
        <init_fun, ProductModel::Features, true, Features, false>
        (fun, model.features, features);
    id_tracker.init(clustering.counts().size());
    validate(model);
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
            for_each_feature
                <dump_group_fun, ProductModel::Features, true, Features, true>
                (fun, model.features, features);
            groups_stream.write_stream(message);
            message.Clear();
        }
    }
}

template<bool cached>
struct ProductModel::Mixture<cached>::infer_hypers_fun
{
    const protobuf::ProductModel_HyperPrior & hyper_prior;
    ProductModel::Features & shared_features;
    const Features & mixture_features;
    size_t featureid;
    rng_t & rng;

    template<class T>
    bool operator() (T * t)
    {
        // TODO use more efficient mixture type
        typedef typename T::CachedMixture Mixture;

        auto & shareds = shared_features[t];
        if (auto maybe_pos = shareds.try_find_pos(featureid)) {
            size_t i = maybe_pos.value();
            auto & mixtures = mixture_features[t];
            LOOM_ASSERT_EQ(shareds.size(), mixtures.size());
            auto & shared = shareds[i];
            const Mixture & mixture = mixtures[i];

            InferShared<Mixture> infer_shared(shared, mixture, rng);
            const auto & grid_prior =
                protobuf::GridPriors<typename T::Shared>::get(hyper_prior);
            distributions::for_each_gridpoint(grid_prior, infer_shared);

            return true;
        } else {
            return false;
        }
    }

    bool operator() (DPD * t)
    {
        auto & shareds = shared_features[t];
        if (shareds.try_find_pos(featureid)) {
            // TODO implement DPD inference
            return true;
        } else {
            return false;
        }
    }
};

template<bool cached>
void ProductModel::Mixture<cached>::infer_feature_hypers (
        ProductModel & model,
        const protobuf::ProductModel_HyperPrior & hyper_prior,
        size_t featureid,
        rng_t & rng) const
{
    infer_hypers_fun fun = {
        hyper_prior,
        model.features,
        features,
        featureid,
        rng};
    bool found = for_some_feature_type(fun);
    LOOM_ASSERT(found, "feature not found: " << featureid);
}

template<bool cached>
void ProductModel::Mixture<cached>::infer_clustering_hypers (
        ProductModel &,
        const protobuf::ProductModel_HyperPrior &,
        rng_t &) const
{
    // TODO infer clustering hypers
}

template<class SourceMixture, class DestinMixture>
struct ProductModel::move_groups_to_fun
{
    const size_t featureid;
    typename SourceMixture::Features & source_features;
    typename DestinMixture::Features & destin_features;

    template<class T>
    bool operator() (T * t)
    {
        auto & sources = source_features[t];

        if (sources.try_find_pos(featureid)) {

            auto & destins = destin_features[t];
            auto & source = sources.find(featureid);
            auto & destin = destins.find_or_insert(featureid);
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
    move_groups_to_fun<SourceMixture, DestinMixture> fun = {
        featureid,
        source.features,
        destin.features};
    bool found = for_some_feature_type(fun);
    LOOM_ASSERT(found, "feature not found: " << featureid);
}

template<class MixtureT>
struct ProductModel::move_shared_to_fun
{
    const size_t featureid;
    ProductModel::Features & source_shared_features;
    typename MixtureT::Features & source_mixture_features;
    ProductModel::Features & destin_shared_features;
    typename MixtureT::Features & destin_mixture_features;
    rng_t & rng;

    template<class T>
    bool operator() (T * t)
    {
        typedef typename T::Shared Shared;

        auto & source_shareds = source_shared_features[t];

        if (source_shareds.try_find_pos(featureid)) {

            auto & destin_shareds = destin_shared_features[t];
            auto & source_mixtures = source_mixture_features[t];
            auto & destin_mixtures = destin_mixture_features[t];

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
void ProductModel::move_shared_to (
        size_t featureid,
        ProductModel & source_model, MixtureT & source_mixture,
        ProductModel & destin_model, MixtureT & destin_mixture,
        rng_t & rng)
{
    move_shared_to_fun<MixtureT> fun = {
        featureid,
        source_model.features, source_mixture.features,
        destin_model.features, destin_mixture.features,
        rng};

    bool found = for_some_feature_type(fun);
    LOOM_ASSERT(found, "feature not found: " << featureid);

    source_model.update_schema();
    destin_model.update_schema();
}

} // namespace loom
