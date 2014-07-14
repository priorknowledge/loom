// Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// - Neither the name of Salesforce.com nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <vector>
#include <distributions/io/protobuf.hpp>
#include <loom/common.hpp>
#include <loom/indexed_vector.hpp>
#include <loom/protobuf.hpp>
#include <loom/models.hpp>
#include <loom/product_value.hpp>

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

//----------------------------------------------------------------------------
// Product Model

struct ProductModel
{
    typedef protobuf::ProductValue Value;
    struct Feature
    {
        template<class T>
        struct Container { typedef IndexedVector<typename T::Shared> t; };
    };
    typedef ForEachFeatureType<Feature> Features;

    ValueSchema schema;
    Clustering::Shared clustering;
    Features features;

    void clear ();

    void load (
            const protobuf::ProductModel_Shared & message,
            const std::vector<size_t> & featureids);
    void dump (protobuf::ProductModel_Shared & message) const;

    void update_schema ();

    void extend (const ProductModel & other);

    void add_value (const Value & value, rng_t & rng);
    void remove_value (const Value & value, rng_t & rng);
    void realize (rng_t & rng);

    template<bool cached> struct Mixture;
    typedef Mixture<false> SmallMixture;
    typedef Mixture<true> FastMixture;

private:

    struct dump_fun;
    struct add_value_fun;
    struct remove_value_fun;
    struct realize_fun;
    struct extend_fun;
    struct clear_fun;
};

struct ProductModel::add_value_fun
{
    Features & shareds;
    rng_t & rng;

    template<class T>
    void operator() (
        T * t,
        size_t i,
        const typename T::Value & value)
    {
        shareds[t][i].add_value(value, rng);
    }
};

inline void ProductModel::add_value (
        const Value & value,
        rng_t & rng)
{
    add_value_fun fun = {features, rng};
    read_value(fun, schema, features, value);
}

struct ProductModel::remove_value_fun
{
    Features & shareds;
    rng_t & rng;

    template<class T>
    void operator() (
        T * t,
        size_t i,
        const typename T::Value & value)
    {
        shareds[t][i].remove_value(value, rng);
    }
};

inline void ProductModel::remove_value (
        const Value & value,
        rng_t & rng)
{
    remove_value_fun fun = {features, rng};
    read_value(fun, schema, features, value);
}

struct ProductModel::realize_fun
{
    rng_t & rng;

    template<class T>
    void operator() (
            T *,
            size_t,
            typename T::Shared & shared)
    {
        shared.realize(rng);
    }
};

inline void ProductModel::realize (rng_t & rng)
{
    realize_fun fun = {rng};
    for_each_feature(fun, features);
}

//----------------------------------------------------------------------------
// Mixture

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

    void remove_unobserved_value (
            const ProductModel & model,
            size_t groupid);

    void add_diff (
            const ProductModel & model,
            size_t groupid,
            const ProductValue::Diff & diff,
            rng_t & rng);

    void add_tare (
            const ProductModel & model,
            const Value & tare,
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
    struct add_tare_fun;
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
        for (size_t i = 0, size = models[t].size(); i < size; ++i) {
            const auto & model = models[t][i];
            const auto & mixture = mixtures[t][i];
            LOOM_ASSERT_EQ(mixture.groups().size(), group_count);
            mixture.validate(model);
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
    read_value(fun, model.schema, features, value);

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
    read_value(fun, model.schema, features, value);

    if (LOOM_UNLIKELY(remove_group)) {
        remove_group_fun fun = {features, groupid};
        for_each_feature(fun, model.features);
        id_tracker.remove_group(groupid);
        validate(model);
    }
}

template<bool cached>
inline void ProductModel::Mixture<cached>::remove_unobserved_value (
        const ProductModel & model,
        size_t groupid)
{
    bool remove_group = clustering.remove_value(model.clustering, groupid);

    if (LOOM_UNLIKELY(remove_group)) {
        remove_group_fun fun = {features, groupid};
        for_each_feature(fun, model.features);
        id_tracker.remove_group(groupid);
        validate(model);
    }
}

template<bool cached>
inline void ProductModel::Mixture<cached>::add_diff (
        const ProductModel & model,
        size_t groupid,
        const ProductValue::Diff & diff,
        rng_t & rng)
{
    bool add_group = clustering.add_value(model.clustering, groupid);
    {
        add_value_fun fun = {features, model.features, groupid, rng};
        read_value(fun, model.schema, features, diff.pos());
    }
    {
        remove_value_fun fun = {features, model.features, groupid, rng};
        read_value(fun, model.schema, features, diff.neg());
    }

    if (LOOM_UNLIKELY(add_group)) {
        add_group_fun fun = {features, rng};
        for_each_feature(fun, model.features);
        id_tracker.add_group();
        validate(model);
    }
}

template<bool cached>
struct ProductModel::Mixture<cached>::add_tare_fun
{
    Features & mixtures;
    const ProductModel::Features & shareds;
    const std::vector<int> & counts;
    rng_t & rng;

    template<class T>
    void operator() (
        T * t,
        size_t i,
        const typename T::Value & tare)
    {
        static_assert(not cached, "cached mixtures are not supported");
        const auto & shared = shareds[t][i];
        auto group = mixtures[t][i].groups().begin();
        for (auto count : counts) {
            if (count) {
                group->add_repeated_value(shared, tare, count, rng);
            }
            ++group;
        }
    }
};

template<bool cached>
inline void ProductModel::Mixture<cached>::add_tare (
        const ProductModel & model,
        const Value & tare,
        rng_t & rng)
{
    add_tare_fun fun = {features, model.features, clustering.counts(), rng};
    read_value(fun, model.schema, features, tare);
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
    read_value(fun, model.schema, features, value);
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
        return mixtures[t][i].groups(groupid).sample_value(shareds[t][i], rng);
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
    write_value(fun, model.schema, features, value);
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
    const protobuf::ProductModel::Group & messages;
    protobuf::ModelCounts model_counts;

    template<class T>
    void operator() (
            T * t,
            size_t,
            typename T::template Mixture<cached>::t & mixture)
    {
        auto & groups = mixture.groups();
        groups.resize(groups.size() + 1);
        size_t offset = model_counts[t]++;
        const auto & message = protobuf::Fields<T>::get(messages).Get(offset);
        groups.back().protobuf_load(message);
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
        load_group_fun fun = {message, protobuf::ModelCounts()};
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
    protobuf::ProductModel::Group & message;

    template<class T>
    void operator() (
            T *,
            size_t,
            const typename T::template Mixture<cached>::t & mixture)
    {
        const auto & group = mixture.groups(groupid);
        group.protobuf_dump(* protobuf::Fields<T>::get(message).Add());
    }
};

template<bool cached>
void ProductModel::Mixture<cached>::dump (
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
        dump_group_fun fun = {packed, message};
        for_each_feature(fun, features);
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
