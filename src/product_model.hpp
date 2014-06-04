#pragma once

#include <vector>
#include <distributions/io/protobuf.hpp>
#include <loom/common.hpp>
#include <loom/indexed_vector.hpp>
#include <loom/protobuf.hpp>
#include <loom/models.hpp>

namespace distributions {
// Kludge because ProductModel::sample_value masks this lookup
// otherwise. Once we refactor ProductModel to fit the same pattern,
// these go away.
using beta_bernoulli::sample_value;
using dirichlet_discrete::sample_value;
using dirichlet_process_discrete::sample_value;
using gamma_poisson::sample_value;
using beta_negative_binomial::sample_value;
using normal_inverse_chi_sq::sample_value;
}

namespace loom
{

//----------------------------------------------------------------------------
// Generics

template<class, bool> struct Reference;
template<class T> struct Reference<T, false> { typedef T & t; };
template<class T> struct Reference<T, true> { typedef const T & t; };


template<class Fun, class X, bool X_const>
struct for_each_feature_fun
{
    Fun & fun;
    typename Reference<X, X_const>::t xs;

    template<class T>
    void operator() (T * t)
    {
        auto & x = xs[t];
        for (size_t i = 0, size = x.size(); i < size; ++i) {
            fun(t, i, x[i]);
        }
    }
};

template<class Fun, class X, bool X_const>
inline void for_each_feature_ (
        Fun & fun,
        typename Reference<X, X_const>::t xs)
{
    for_each_feature_fun<Fun, X, X_const> loop = {fun, xs};
    for_each_feature_type(loop);
}

template<class Fun, class X>
inline void for_each_feature (Fun & fun, X & x)
{
    for_each_feature_<Fun, X, false>(fun, x);
}
template<class Fun, class X>
inline void for_each_feature (Fun & fun, const X & x)
{
    for_each_feature_<Fun, X, true>(fun, x);
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

template<class Feature, class Fun>
inline void read_sparse_value (
        Fun & fun,
        const protobuf::SparseValueSchema & value_schema,
        const ForEachFeatureType<Feature> & model_schema,
        const protobuf::ProductModel::SparseValue & value)
{
    if (LOOM_DEBUG_LEVEL >= 2) {
        value_schema.validate(value);
    }

    size_t absolute_pos = 0;

    if (value.booleans_size()) {
        size_t packed_pos = 0;
        for (size_t i = 0, size = model_schema.bb.size(); i < size; ++i) {
            if (value.observed(absolute_pos++)) {
                fun(BB::null(), i, value.booleans(packed_pos++));
            }
        }
    } else {
        absolute_pos += model_schema.bb.size();
    }

    if (value.counts_size()) {
        size_t packed_pos = 0;
        for (size_t i = 0, size = model_schema.dd16.size(); i < size; ++i) {
            if (value.observed(absolute_pos++)) {
                fun(DD16::null(), i, value.counts(packed_pos++));
            }
        }
        for (size_t i = 0, size = model_schema.dd256.size(); i < size; ++i) {
            if (value.observed(absolute_pos++)) {
                fun(DD256::null(), i, value.counts(packed_pos++));
            }
        }
        for (size_t i = 0, size = model_schema.dpd.size(); i < size; ++i) {
            if (value.observed(absolute_pos++)) {
                fun(DPD::null(), i, value.counts(packed_pos++));
            }
        }
        for (size_t i = 0, size = model_schema.gp.size(); i < size; ++i) {
            if (value.observed(absolute_pos++)) {
                fun(GP::null(), i, value.counts(packed_pos++));
            }
        }
        for (size_t i = 0, size = model_schema.bnb.size(); i < size; ++i) {
            if (value.observed(absolute_pos++)) {
                fun(BNB::null(), i, value.counts(packed_pos++));
            }
        }
    } else {
        absolute_pos +=
            model_schema.dd16.size() +
            model_schema.dd256.size() +
            model_schema.dpd.size()  +
            model_schema.gp.size()+
            model_schema.bnb.size();
    }

    if (value.reals_size()) {
        size_t packed_pos = 0;
        for (size_t i = 0, size = model_schema.nich.size(); i < size; ++i) {
            if (value.observed(absolute_pos++)) {
                fun(NICH::null(), i, value.reals(packed_pos++));
            }
        }
    }
}

template<class Feature, class Fun>
inline void write_sparse_value (
        Fun & fun,
        const protobuf::SparseValueSchema & value_schema,
        const ForEachFeatureType<Feature> & model_schema,
        protobuf::ProductModel::SparseValue & value)
{
    size_t absolute_pos = 0;

    value.clear_booleans();
    for (size_t i = 0, size = model_schema.bb.size(); i < size; ++i) {
        if (value.observed(absolute_pos++)) {
            value.add_booleans(fun(BB::null(), i));
        }
    }

    value.clear_counts();
    for (size_t i = 0, size = model_schema.dd16.size(); i < size; ++i) {
        if (value.observed(absolute_pos++)) {
            value.add_counts(fun(DD16::null(), i));
        }
    }
    for (size_t i = 0, size = model_schema.dd256.size(); i < size; ++i) {
        if (value.observed(absolute_pos++)) {
            value.add_counts(fun(DD256::null(), i));
        }
    }
    for (size_t i = 0, size = model_schema.dpd.size(); i < size; ++i) {
        if (value.observed(absolute_pos++)) {
            value.add_counts(fun(DPD::null(), i));
        }
    }
    for (size_t i = 0, size = model_schema.gp.size(); i < size; ++i) {
        if (value.observed(absolute_pos++)) {
            value.add_counts(fun(GP::null(), i));
        }
    }
    for (size_t i = 0, size = model_schema.bnb.size(); i < size; ++i) {
        if (value.observed(absolute_pos++)) {
            value.add_counts(fun(BNB::null(), i));
        }
    }

    value.clear_reals();
    for (size_t i = 0, size = model_schema.nich.size(); i < size; ++i) {
        if (value.observed(absolute_pos++)) {
            value.add_reals(fun(NICH::null(), i));
        }
    }

    if (LOOM_DEBUG_LEVEL >= 2) {
        value_schema.validate(value);
    }
}

//----------------------------------------------------------------------------
// Product Model

struct ProductModel
{
    typedef protobuf::ProductModel::SparseValue Value;
    struct Feature
    {
        template<class T>
        struct Container { typedef IndexedVector<typename T::Shared> t; };
    };
    typedef ForEachFeatureType<Feature> Features;

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
    struct Feature
    {
        template<class T>
        struct Container
        {
            typedef IndexedVector<typename T::template Mixture<cached>::t> t;
        };
    };
    typedef ForEachFeatureType<Feature> Features;

    typename Clustering::Mixture<cached>::t clustering;
    Features features;
    distributions::MixtureIdTracker id_tracker;

    void init_unobserved (
            const ProductModel & model,
            const std::vector<int> & counts,
            rng_t & rng);

    void load_step_1_of_2 (
            const ProductModel & model,
            const char * filename,
            size_t empty_group_count);

    void load_step_2_of_2 (
            const ProductModel & model,
            size_t featureid,
            size_t empty_group_count,
            rng_t & rng);

    void dump (
            const ProductModel & model,
            const char * filename,
            const std::vector<uint32_t> & sorted_to_global) const;

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
            rng_t & rng) const;

    float score_feature (
            const ProductModel & model,
            size_t featureid,
            rng_t & rng) const;

    float score_data (
            const ProductModel & model,
            rng_t & rng) const;

    size_t sample_value (
            const ProductModel & model,
            const VectorFloat & probs,
            Value & value,
            rng_t & rng) const;

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

    struct validate_fun;
    struct clear_fun;
    struct load_group_fun;
    struct init_groups_fun;
    struct init_unobserved_fun;
    struct sort_groups_fun;
    struct dump_group_fun;
    struct add_group_fun;
    struct add_value_fun;
    struct remove_group_fun;
    struct remove_value_fun;
    struct score_value_fun;
    struct score_feature_fun;
    struct score_data_fun;
    struct sample_fun;

    template<class OtherMixture>
    struct move_feature_to_fun;
};

template<bool cached>
struct ProductModel::Mixture<cached>::validate_fun
{
    const size_t group_count;
    const ProductModel::Features & models;
    const Features & mixtures;

    template<class T>
    void operator() (T * t)
    {
        LOOM_ASSERT_EQ(models[t].size(), mixtures[t].size());
        for (const auto & mixture : mixtures[t]) {
            LOOM_ASSERT_EQ(mixture.groups().size(), group_count);
        }
    }
};

template<bool cached>
inline void ProductModel::Mixture<cached>::validate (
        const ProductModel & model) const
{
    if (LOOM_DEBUG_LEVEL >= 2) {
        const size_t group_count = clustering.counts().size();
        validate_fun fun = {group_count, model.features, features};
        for_each_feature_type(fun);
        LOOM_ASSERT_EQ(id_tracker.packed_size(), group_count);
    }
}

template<bool cached>
struct ProductModel::Mixture<cached>::add_group_fun
{
    Features & mixtures;
    rng_t & rng;

    template<class T>
    void operator() (
            T * t,
            size_t i,
            const typename T::Shared & shared)
    {
        mixtures[t][i].add_group(shared, rng);
    }
};

template<bool cached>
struct ProductModel::Mixture<cached>::add_value_fun
{
    Features & mixtures;
    const ProductModel::Features & shareds;
    const size_t groupid;
    rng_t & rng;

    template<class T>
    void operator() (
        T * t,
        size_t i,
        const typename T::Value & value)
    {
        mixtures[t][i].add_value(shareds[t][i], groupid, value, rng);
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
    add_value_fun fun = {features, model.features, groupid, rng};
    read_sparse_value(fun, model.schema, features, value);

    if (LOOM_UNLIKELY(add_group)) {
        add_group_fun fun = {features, rng};
        for_each_feature(fun, model.features);
        id_tracker.add_group();
        validate(model);
    }
}

template<bool cached>
struct ProductModel::Mixture<cached>::remove_group_fun
{
    Features & mixtures;
    const size_t groupid;

    template<class T>
    void operator() (
            T * t,
            size_t i,
            const typename T::Shared & shared)
    {
        mixtures[t][i].remove_group(shared, groupid);
    }
};

template<bool cached>
struct ProductModel::Mixture<cached>::remove_value_fun
{
    Features & mixtures;
    const ProductModel::Features & shareds;
    const size_t groupid;
    rng_t & rng;

    template<class T>
    void operator() (
            T * t,
            size_t i,
            const typename T::Value & value)
    {
        mixtures[t][i].remove_value(shareds[t][i], groupid, value, rng);
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
    remove_value_fun fun = {features, model.features, groupid, rng};
    read_sparse_value(fun, model.schema, features, value);

    if (LOOM_UNLIKELY(remove_group)) {
        remove_group_fun fun = {features, groupid};
        for_each_feature(fun, model.features);
        id_tracker.remove_group(groupid);
        validate(model);
    }
}

template<bool cached>
struct ProductModel::Mixture<cached>::score_value_fun
{
    const Features & mixtures;
    const ProductModel::Features & shareds;
    VectorFloat & scores;
    rng_t & rng;

    template<class T>
    void operator() (
            T * t,
            size_t i,
            const typename T::Value & value)
    {
        mixtures[t][i].score_value(shareds[t][i], value, scores, rng);
    }
};

template<bool cached>
inline void ProductModel::Mixture<cached>::score_value (
        const ProductModel & model,
        const Value & value,
        VectorFloat & scores,
        rng_t & rng) const
{
    scores.resize(clustering.counts().size());
    clustering.score_value(model.clustering, scores);
    score_value_fun fun = {features, model.features, scores, rng};
    read_sparse_value(fun, model.schema, features, value);
}

template<bool cached>
struct ProductModel::Mixture<cached>::score_data_fun
{
    const Features & mixtures;
    rng_t & rng;
    float & score;

    template<class T>
    void operator() (
            T * t,
            size_t i,
            const typename T::Shared & shared)
    {
        score += mixtures[t][i].score_data(shared, rng);
    }
};

template<bool cached>
inline float ProductModel::Mixture<cached>::score_data (
        const ProductModel & model,
        rng_t & rng) const
{
    float score = clustering.score_data(model.clustering);

    score_data_fun fun = {features, rng, score};
    for_each_feature(fun, model.features);

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
    const Features & mixtures;
    const ProductModel::Features & shareds;
    const size_t groupid;
    rng_t & rng;

    template<class T>
    typename T::Value operator() (T * t, size_t i)
    {
        return distributions::sample_value(
            shareds[t][i],
            mixtures[t][i].groups(groupid),
            rng);
    }
};

template<bool cached>
inline size_t ProductModel::Mixture<cached>::sample_value (
        const ProductModel & model,
        const VectorFloat & probs,
        Value & value,
        rng_t & rng) const
{
    size_t groupid = distributions::sample_from_probs(rng, probs);
    sample_fun fun = {features, model.features, groupid, rng};
    write_sparse_value(fun, model.schema, features, value);
    return groupid;
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
    clustering.counts() = counts;
    clustering.init(model.clustering);

    init_unobserved_fun fun = {counts.size(), model.features, features, rng};
    for_each_feature_type(fun);

    id_tracker.init(counts.size());

    validate(model);
}

template<bool cached>
struct ProductModel::Mixture<cached>::clear_fun
{
    const ProductModel::Features & shareds;
    Features & mixtures;

    template<class T>
    void operator() (T * t)
    {
        mixtures[t].clear();
        for (auto featureid : shareds[t].index()) {
            mixtures[t].insert(featureid);
        }
    }
};

template<bool cached>
struct ProductModel::Mixture<cached>::load_group_fun
{
    const ProductModel::Features & shareds;
    const protobuf::ProductModel::Group & messages;
    protobuf::ModelCounts model_counts;

    template<class T>
    void operator() (
            T * t,
            size_t i,
            typename T::template Mixture<cached>::t & mixture)
    {
        auto & groups = mixture.groups();
        groups.resize(groups.size() + 1);
        size_t offset = model_counts[t]++;
        const auto & message = protobuf::Fields<T>::get(messages).Get(offset);
        distributions::group_load(shareds[t][i], groups.back(), message);
    }
};

template<bool cached>
void ProductModel::Mixture<cached>::load_step_1_of_2 (
        const ProductModel & model,
        const char * filename,
        size_t empty_group_count)
{
    clear_fun fun = {model.features, features};
    for_each_feature_type(fun);
    auto & counts = clustering.counts();
    counts.clear();

    protobuf::InFile groups(filename);
    protobuf::ProductModel::Group message;
    while (groups.try_read_stream(message)) {
        counts.push_back(message.count());
        load_group_fun fun = {model.features, message, protobuf::ModelCounts()};
        for_each_feature(fun, features);
    }

    counts.resize(counts.size() + empty_group_count, 0);
    clustering.init(model.clustering);
    id_tracker.init(counts.size());
}

template<bool cached>
struct ProductModel::Mixture<cached>::init_groups_fun
{
    const ProductModel::Features & shareds;
    const size_t empty_group_count;
    rng_t & rng;

    template<class T>
    void operator() (
            T * t,
            size_t i,
            typename T::template Mixture<cached>::t & mixture)
    {
        const typename T::Shared & shared = shareds[t][i];
        std::vector<typename T::Group> & groups = mixture.groups();
        const size_t nonempty_group_count = groups.size();
        const size_t group_count = nonempty_group_count + empty_group_count;
        groups.resize(groups.size() + empty_group_count);
        for (size_t i = nonempty_group_count; i < group_count; ++i) {
            groups[i].init(shared, rng);
        }
        mixture.init(shared, rng);
    }
};

template<bool cached>
void ProductModel::Mixture<cached>::load_step_2_of_2 (
        const ProductModel & model,
        size_t featureid,
        size_t empty_group_count,
        rng_t & rng)
{
    init_groups_fun fun = {model.features, empty_group_count, rng};
    for_one_feature(fun, features, featureid);
}

template<bool cached>
struct ProductModel::Mixture<cached>::dump_group_fun
{
    size_t groupid;
    const Features & mixtures;
    protobuf::ProductModel::Group & message;

    template<class T>
    void operator() (
            T * t,
            size_t i,
            const typename T::Shared & shared)
    {
        distributions::group_dump(
            shared,
            mixtures[t][i].groups(groupid),
            * protobuf::Fields<T>::get(message).Add());
    }
};

template<bool cached>
void ProductModel::Mixture<cached>::dump (
        const ProductModel & model,
        const char * filename,
        const std::vector<uint32_t> & sorted_to_global) const
{
    const size_t group_count = clustering.counts().size();
    LOOM_ASSERT_LT(sorted_to_global.size(), group_count);
    protobuf::OutFile groups_stream(filename);
    protobuf::ProductModel::Group message;
    for (auto global : sorted_to_global) {
        auto packed = id_tracker.global_to_packed(global);
        if (LOOM_DEBUG_LEVEL >= 1) {
            LOOM_ASSERT_LT(packed, group_count);
            LOOM_ASSERT_LT(0, clustering.counts(packed));
        }
        message.set_count(clustering.counts(packed));
        dump_group_fun fun = {packed, features, message};
        for_each_feature(fun, model.features);
        groups_stream.write_stream(message);
        message.Clear();
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
