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
            fun(t, i, x[i], y[i]);
        }
    }
};

template<class Fun, class X, bool X_const, class Y, bool Y_const>
inline void for_each_feature_ (
        Fun & fun,
        typename Reference<X, X_const>::t xs,
        typename Reference<Y, Y_const>::t ys)
{
    for_each_feature_fun<Fun, X, X_const, Y, Y_const> loop = {fun, xs, ys};
    for_each_feature_type(loop);
}

template<class Fun, class X, class Y>
inline void for_each_feature (Fun & fun, X & x, Y & y)
{
    for_each_feature_<Fun, X, false, Y, false>(fun, x, y);
}
template<class Fun, class X, class Y>
inline void for_each_feature (Fun & fun, const X & x, Y & y)
{
    for_each_feature_<Fun, X, true, Y, false>(fun, x, y);
}
template<class Fun, class X, class Y>
inline void for_each_feature (Fun & fun, X & x, const Y & y)
{
    for_each_feature_<Fun, X, false, Y, true>(fun, x, y);
}
template<class Fun, class X, class Y>
inline void for_each_feature (Fun & fun, const X & x, const Y & y)
{
    for_each_feature_<Fun, X, true, Y, true>(fun, x, y);
}


template<class Fun, class X, bool X_const>
struct for_one_feature_fun
{
    Fun & fun;
    typename Reference<X, X_const>::t xs;
    size_t featureid;

    template<class T>
    bool operator() (T * t)
    {
        auto & x = xs[t];
        if (auto maybe_pos = x.try_find_pos(featureid)) {
            size_t i = maybe_pos.value();
            fun(t, i, x[i]);
            return true;
        } else {
            return false;
        }
    }
};

template<class Fun, class X, bool X_const>
inline void for_one_feature_ (
        Fun & fun,
        typename Reference<X, X_const>::t xs,
        size_t featureid)
{
    for_one_feature_fun<Fun, X, X_const> search = {fun, xs, featureid};
    bool found = for_some_feature_type(search);
    LOOM_ASSERT(found, "feature not found: " << featureid);
}

template<class Fun, class X>
inline void for_one_feature (Fun & fun, X & x, size_t featureid)
{
    for_one_feature_<Fun, X, false>(fun, x, featureid);
}
template<class Fun, class X>
inline void for_one_feature (Fun & fun, const X & x, size_t featureid)
{
    for_one_feature_<Fun, X, true>(fun, x, featureid);
}

//----------------------------------------------------------------------------
// Product Model

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
    void dump (protobuf::ProductModel_Shared & message) const;

    void update_schema ();

    void extend (const ProductModel & other);

    template<bool cached> struct Mixture;
    typedef Mixture<false> SimpleMixture;
    typedef Mixture<true> CachedMixture;

private:

    struct dump_fun;
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

    void init_unobserved (
            const ProductModel & model,
            const std::vector<int> & counts,
            rng_t & rng);

    void load (
            const ProductModel & model,
            const char * filename,
            rng_t & rng,
            size_t empty_group_count = 1);

    void dump (const ProductModel & model, const char * filename) const;

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

    float score_data (
            const ProductModel & model,
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
            rng_t & rng);

    template<class OtherMixture>
    void move_feature_to (
            size_t featureid,
            ProductModel & source_model, OtherMixture & source_mixture,
            ProductModel & destin_model, OtherMixture & destin_mixture,
            bool init_cache,
            rng_t & rng);

    void validate (const ProductModel & model) const;

    size_t count_rows () const
    {
        const auto & counts = clustering.counts();
        return std::accumulate(counts.begin(), counts.end(), 0);
    }


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

    struct validate_fun;
    struct load_group_fun;
    struct init_fun;
    struct init_unobserved_fun;
    struct dump_group_fun;
    struct add_group_fun;
    struct add_value_fun;
    struct remove_group_fun;
    struct remove_value_fun;
    struct score_value_fun;
    struct score_feature_fun;
    struct score_data_fun;
    struct sample_fun;
    struct infer_hypers_fun;

    template<class OtherMixture>
    struct move_feature_to_fun;
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

    size_t absolute_pos = 0;

    if (value.booleans_size()) {
        TODO("implement bb");
    } else {
        absolute_pos += 0;
    }

    if (value.counts_size()) {
        size_t packed_pos = 0;
        for (size_t i = 0, size = features.dd16.size(); i < size; ++i) {
            if (value.observed(absolute_pos++)) {
                fun(model.features.dd16[i],
                    features.dd16[i],
                    value.counts(packed_pos++));
            }
        }
        for (size_t i = 0, size = features.dd256.size(); i < size; ++i) {
            if (value.observed(absolute_pos++)) {
                fun(model.features.dd256[i],
                    features.dd256[i],
                    value.counts(packed_pos++));
            }
        }
        for (size_t i = 0, size = features.dpd.size(); i < size; ++i) {
            if (value.observed(absolute_pos++)) {
                fun(model.features.dpd[i],
                    features.dpd[i],
                    value.counts(packed_pos++));
            }
        }
        for (size_t i = 0, size = features.gp.size(); i < size; ++i) {
            if (value.observed(absolute_pos++)) {
                fun(model.features.gp[i],
                    features.gp[i],
                    value.counts(packed_pos++));
            }
        }
    } else {
        absolute_pos +=
            features.dd16.size() +
            features.dd256.size() +
            features.dpd.size() +
            features.gp.size();
    }

    if (value.reals_size()) {
        size_t packed_pos = 0;
        for (size_t i = 0, size = features.nich.size(); i < size; ++i) {
            if (value.observed(absolute_pos++)) {
                fun(model.features.nich[i],
                    features.nich[i],
                    value.reals(packed_pos++));
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

    size_t absolute_pos = 0;

    if (value.booleans_size()) {
        TODO("implement bb");
    } else {
        absolute_pos += 0;
    }

    if (value.counts_size()) {
        size_t packed_pos = 0;
        for (size_t i = 0, size = features.dd16.size(); i < size; ++i) {
            if (value.observed(absolute_pos++)) {
                value.set_counts(
                    packed_pos++,
                    fun(model.features.dd16[i], features.dd16[i]));
            }
        }
        for (size_t i = 0, size = features.dd256.size(); i < size; ++i) {
            if (value.observed(absolute_pos++)) {
                value.set_counts(
                    packed_pos++,
                    fun(model.features.dd256[i], features.dd256[i]));
            }
        }
        for (size_t i = 0, size = features.dpd.size(); i < size; ++i) {
            if (value.observed(absolute_pos++)) {
                value.set_counts(
                    packed_pos++,
                    fun(model.features.dpd[i], features.dpd[i]));
            }
        }
        for (size_t i = 0, size = features.gp.size(); i < size; ++i) {
            if (value.observed(absolute_pos++)) {
                value.set_counts(
                    packed_pos++,
                    fun(model.features.gp[i], features.gp[i]));
            }
        }
    } else {
        absolute_pos +=
            features.dd16.size() +
            features.dd256.size() +
            features.dpd.size() +
            features.gp.size();
    }

    if (value.reals_size()) {
        size_t packed_pos = 0;
        for (size_t i = 0, size = features.nich.size(); i < size; ++i) {
            if (value.observed(absolute_pos++)) {
                value.set_reals(
                    packed_pos++,
                    fun(model.features.nich[i], features.nich[i]));
            }
        }
    }
}

template<bool cached>
struct ProductModel::Mixture<cached>::validate_fun
{
    const size_t group_count;

    template<class T, class Mixture>
    void operator() (
            T *,
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
        for_each_feature(fun, model.features, features);
        LOOM_ASSERT_EQ(id_tracker.packed_size(), group_count);
    }
}

template<bool cached>
struct ProductModel::Mixture<cached>::add_group_fun
{
    rng_t & rng;

    template<class T, class Mixture>
    void operator() (
            T *,
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
        for_each_feature(fun, model.features, features);
        id_tracker.add_group();
        validate(model);
    }
}

template<bool cached>
struct ProductModel::Mixture<cached>::remove_group_fun
{
    const size_t groupid;

    template<class T, class Mixture>
    void operator() (
            T *,
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
        for_each_feature(fun, model.features, features);
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
struct ProductModel::Mixture<cached>::score_data_fun
{
    float & score;
    rng_t & rng;

    template<class T, class Mixture>
    void operator() (
            T *,
            size_t,
            const typename Mixture::Shared & shared,
            const Mixture & mixture)
    {
        score += mixture.score_data(shared, rng);
    }
};

template<bool cached>
inline float ProductModel::Mixture<cached>::score_data (
        const ProductModel & model,
        rng_t & rng) const
{
    float score = clustering.score_data(model.clustering);

    score_data_fun fun = {score, rng};
    for_each_feature(fun, model.features, features);

    return score;
}

template<bool cached>
struct ProductModel::Mixture<cached>::score_feature_fun
{
    const Features & mixtures;
    rng_t & rng;
    float score;

    template<class T>
    void operator() (
            T * t,
            size_t i,
            const typename T::Shared & shared)
    {
        score = mixtures[t][i].score_data(shared, rng);
    }
};

template<bool cached>
inline float ProductModel::Mixture<cached>::score_feature (
        const ProductModel & model,
        size_t featureid,
        rng_t & rng) const
{
    score_feature_fun fun = {features, rng, NAN};
    for_one_feature(fun, model.features, featureid);
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
struct ProductModel::Mixture<cached>::init_unobserved_fun
{
    size_t group_count;
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
            mixture.groups().resize(group_count);
            for (auto & group : mixture.groups()) {
                group.init(shared, rng);
            }
            mixture.init(shared, rng);
        }
    }
};

template<bool cached>
void ProductModel::Mixture<cached>::init_unobserved (
        const ProductModel & model,
        const std::vector<int> & counts,
        rng_t & rng)
{
    clustering.init(model.clustering, counts);

    init_unobserved_fun fun = {counts.size(), model.features, features, rng};
    for_each_feature_type(fun);

    id_tracker.init(counts.size());

    validate(model);
}

template<bool cached>
struct ProductModel::Mixture<cached>::load_group_fun
{
    size_t groupid;
    const protobuf::ProductModel::Group & message;
    protobuf::ModelCounts model_counts;

    template<class T, class Mixture>
    void operator() (
            T * t,
            size_t,
            const typename Mixture::Shared & shared,
            Mixture & mixture)
    {
        size_t offset = model_counts[t]++;
        mixture.groups().resize(mixture.groups().size() + 1);
        distributions::group_load(
            shared,
            mixture.groups(groupid),
            protobuf::Groups<T>::get(message).Get(offset));
    }
};

template<bool cached>
struct ProductModel::Mixture<cached>::init_fun
{
    rng_t & rng;

    template<class T, class Mixture>
    void operator() (
            T *,
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
    const std::vector<int> counts(empty_group_count, 0);
    init_unobserved(model, counts, rng);
    protobuf::InFile groups(filename);
    protobuf::ProductModel::Group message;
    for (size_t groupid = 0; groups.try_read_stream(message); ++groupid) {
        clustering.add_value(model.clustering, groupid, message.count());
        load_group_fun fun = {groupid, message, protobuf::ModelCounts()};
        for_each_feature(fun, model.features, features);
    }
    init_fun fun = {rng};
    for_each_feature(fun, model.features, features);
    id_tracker.init(clustering.counts().size());
    validate(model);
}

template<bool cached>
struct ProductModel::Mixture<cached>::dump_group_fun
{
    size_t groupid;
    protobuf::ProductModel::Group & message;

    template<class T, class Mixture>
    void operator() (
            T *,
            size_t,
            const typename Mixture::Shared & shared,
            const Mixture & mixture)
    {
        distributions::group_dump(
            shared,
            mixture.groups(groupid),
            * protobuf::Groups<T>::get(message).Add());
    }
};

template<bool cached>
void ProductModel::Mixture<cached>::dump (
        const ProductModel & model,
        const char * filename) const
{
    protobuf::OutFile groups_stream(filename);
    protobuf::ProductModel::Group message;
    const size_t group_count = clustering.counts().size();
    for (size_t i = 0; i < group_count; ++i) {
        bool group_is_not_empty = clustering.counts(i);
        if (group_is_not_empty) {
            message.set_count(clustering.counts(i));
            dump_group_fun fun = {i, message};
            for_each_feature(fun, model.features, features);
            groups_stream.write_stream(message);
            message.Clear();
        }
    }
}

template<bool cached>
struct ProductModel::Mixture<cached>::infer_hypers_fun
{
    const protobuf::ProductModel_HyperPrior & hyper_prior;
    Features & mixtures;
    rng_t & rng;

    template<class T>
    void operator() (
            T * t,
            size_t i,
            typename T::Shared & shared)
    {
        // TODO optimize mixture to cache score_data(...)
        typedef typename T::template Mixture<cached>::t Mixture;
        Mixture & mixture = mixtures[t][i];
        InferShared<Mixture> infer_shared(shared, mixture, rng);
        const auto & grid_prior = protobuf::GridPriors<T>::get(hyper_prior);
        distributions::for_each_gridpoint(grid_prior, infer_shared);
        mixture.init(shared, rng);
    }

    void operator() (
        DPD * t,
        size_t i,
        DPD::Shared & shared)
    {
        // TODO implement DPD inference
        auto & mixture = mixtures[t][i];
        mixture.init(shared, rng);
    }
};

template<bool cached>
void ProductModel::Mixture<cached>::infer_feature_hypers (
        ProductModel & model,
        const protobuf::ProductModel_HyperPrior & hyper_prior,
        size_t featureid,
        rng_t & rng)
{
    infer_hypers_fun fun = {hyper_prior, features, rng};
    for_one_feature(fun, model.features, featureid);
}

template<bool cached>
inline void ProductModel::Mixture<cached>::infer_clustering_hypers (
        ProductModel & model,
        const protobuf::ProductModel_HyperPrior & hyper_prior,
        rng_t & rng) const
{
    const auto & grid_prior = hyper_prior.clustering();
    if (grid_prior.size()) {
        const auto & counts = clustering.counts();
        model.clustering = sample_clustering_posterior(grid_prior, counts, rng);
    }
}

template<bool cached>
template<class OtherMixture>
struct ProductModel::Mixture<cached>::move_feature_to_fun
{
    const size_t featureid;
    ProductModel::Features & source_shareds;
    typename OtherMixture::Features & source_mixtures;
    ProductModel::Features & destin_shareds;
    typename OtherMixture::Features & destin_mixtures;
    const bool init_cache;
    rng_t & rng;

    template<class T>
    void operator() (
            T * t,
            size_t,
            typename T::template Mixture<cached>::t & temp_mixture)
    {
        typedef typename T::Shared Shared;
        Shared & source_shared = source_shareds[t].find(featureid);
        Shared & destin_shared = destin_shareds[t].insert(featureid);
        destin_shared = std::move(source_shared);
        source_shareds[t].remove(featureid);

        source_mixtures[t].remove(featureid);
        auto & destin_mixture = destin_mixtures[t].insert(featureid);
        destin_mixture.groups() = std::move(temp_mixture.groups());

        if (init_cache) {
            destin_mixture.init(destin_shared, rng);
        }
    }
};

template<bool cached>
template<class OtherMixture>
void ProductModel::Mixture<cached>::move_feature_to (
        size_t featureid,
        ProductModel & source_model, OtherMixture & source_mixture,
        ProductModel & destin_model, OtherMixture & destin_mixture,
        bool init_cache,
        rng_t & rng)
{
    if (LOOM_DEBUG_LEVEL >= 2) {
        LOOM_ASSERT_EQ(
            destin_mixture.clustering.counts(),
            clustering.counts());
    } else if (LOOM_DEBUG_LEVEL >= 1) {
        LOOM_ASSERT_EQ(
            destin_mixture.clustering.counts().size(),
            clustering.counts().size());
    }

    move_feature_to_fun<OtherMixture> fun = {
        featureid,
        source_model.features, source_mixture.features,
        destin_model.features, destin_mixture.features,
        init_cache,
        rng};
    for_one_feature(fun, features, featureid);

    source_model.update_schema();
    destin_model.update_schema();
}

} // namespace loom
