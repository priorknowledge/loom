#pragma once

#include <unordered_set>
#include <distributions/vector.hpp>
#include "common.hpp"
#include "protobuf.hpp"
#include "product_model.hpp"

namespace loom
{

struct CrossCat
{
    typedef protobuf::ProductModel::SparseValue Value;
    struct Kind
    {
        ProductModel model;
        ProductModel::CachedMixture mixture;
        std::unordered_set<size_t> featureids;
    };

    protobuf::SparseValueSchema schema;
    protobuf::CrossCat_HyperPrior hyper_prior;
    Clustering::Shared feature_clustering;
    distributions::Packed_<Kind> kinds;
    std::vector<uint32_t> featureid_to_kindid;

    void model_load (const char * filename);
    void model_dump (const char * filename) const;

    void mixture_init_empty (
            size_t empty_group_count,
            rng_t & rng);
    void mixture_load (
            const char * dirname,
            size_t empty_group_count,
            rng_t & rng);
    void mixture_dump (
            const char * dirname,
            const std::vector<std::vector<uint32_t>> & sorted_to_globals) const;

    std::vector<std::vector<uint32_t>> get_sorted_groupids () const;

    void value_split (
            const Value & full_value,
            std::vector<Value> & partial_values) const;

    struct ValueJoiner;
    void value_join (
            Value & full_value,
            const std::vector<Value> & partial_values) const;

    void value_resize (Value & value) const;

    void infer_hypers (rng_t & rng, bool parallel);

    float score_data (rng_t & rng) const;

    void validate () const;

private:

    void infer_clustering_hypers (rng_t & rng);

    std::string get_mixture_filename (
            const char * dirname,
            size_t kindid) const;

    struct value_split_fun;
    struct value_join_fun;
    struct value_resize_fun;
};

inline void CrossCat::mixture_init_empty (size_t empty_group_count, rng_t & rng)
{
    const std::vector<int> counts(empty_group_count, 0);
    for (auto & kind : kinds) {
        kind.mixture.init_unobserved(kind.model, counts, rng);
    }
}

struct CrossCat::value_split_fun
{
    const CrossCat & cross_cat;
    const Value & full_value;
    std::vector<Value> & partial_values;
    size_t absolute_pos;

    template<class FieldType>
    inline void operator() (FieldType *, size_t size)
    {
        typedef protobuf::Fields<FieldType> Fields;
        const auto & full_fields = Fields::get(full_value);
        for (size_t i = 0, packed_pos = 0; i < size; ++i, ++absolute_pos) {
            auto kindid = cross_cat.featureid_to_kindid[absolute_pos];
            auto & partial_value = partial_values[kindid];
            bool observed = full_value.observed(absolute_pos);
            partial_value.add_observed(observed);
            if (observed) {
                Fields::get(partial_value).Add(full_fields.Get(packed_pos++));
            }
        }
    }
};

inline void CrossCat::value_split (
        const Value & full_value,
        std::vector<Value> & partial_values) const
{
    if (LOOM_DEBUG_LEVEL >= 1) {
        LOOM_ASSERT_EQ(partial_values.size(), kinds.size());
    }

    for (auto & partial_value : partial_values) {
        partial_value.Clear();
    }

    value_split_fun fun = {* this, full_value, partial_values, 0};
    schema.for_each_datatype(fun);
}

struct CrossCat::value_join_fun
{
    const CrossCat & cross_cat;
    std::vector<size_t> & packed_pos_list;
    Value & full_value;
    const std::vector<Value> & partial_values;
    size_t absolute_pos;

    template<class FieldType>
    void operator() (FieldType *, size_t size)
    {
        typedef protobuf::Fields<FieldType> Fields;
        auto & full_observed = * full_value.mutable_observed();
        auto & full_fields = Fields::get(full_value);
        packed_pos_list.clear();
        packed_pos_list.resize(cross_cat.kinds.size(), 0);
        for (size_t i = 0; i < size; ++i, ++absolute_pos) {
            auto kindid = cross_cat.featureid_to_kindid[absolute_pos];
            const auto & partial_value = partial_values[kindid];
            auto & packed_pos = packed_pos_list[kindid];
            bool observed = partial_value.observed(packed_pos);
            full_observed.Add(observed);
            if (observed) {
                full_fields.Add(Fields::get(partial_value).Get(packed_pos++));
            }
        }
    }
};

struct CrossCat::ValueJoiner
{
    ValueJoiner (const CrossCat & cross_cat) : cross_cat_(cross_cat) {}

    void operator() (
            Value & full_value,
            const std::vector<Value> & partial_values)
    {
        full_value.Clear();

        CrossCat::value_join_fun fun =
            {cross_cat_, packed_pos_list_, full_value, partial_values, 0};
        cross_cat_.schema.for_each_datatype(fun);
    }

private:

    const CrossCat & cross_cat_;
    std::vector<size_t> packed_pos_list_;
};

inline void CrossCat::value_join (
        Value & full_value,
        const std::vector<Value> & partial_values) const
{
    ValueJoiner(* this)(full_value, partial_values);
}

struct CrossCat::value_resize_fun
{
    Value & value;
    size_t absolute_pos;


    template<class FieldType>
    void operator() (FieldType *, size_t size)
    {
        auto & fields = protobuf::Fields<FieldType>::get(value);
        fields.Clear();
        for (size_t i = 0; i < size; ++i) {
            if (value.observed(absolute_pos++)) {
                fields.Add();
            }
        }
    }
};

inline void CrossCat::value_resize (Value & value) const
{
    value_resize_fun fun = {value, 0};
    schema.for_each_datatype(fun);
}

inline void CrossCat::validate () const
{
    if (LOOM_DEBUG_LEVEL >= 1) {
        LOOM_ASSERT_LT(0, schema.total_size());
        protobuf::SparseValueSchema expected_schema;
        for (const auto & kind : kinds) {
            kind.mixture.validate(kind.model);
            expected_schema += kind.model.schema;
        }
        LOOM_ASSERT_EQ(schema, expected_schema);
    }
    if (LOOM_DEBUG_LEVEL >= 2) {
        for (size_t f = 0; f < featureid_to_kindid.size(); ++f) {
            size_t k = featureid_to_kindid[f];
            const auto & featureids = kinds[k].featureids;
            LOOM_ASSERT(
                featureids.find(f) != featureids.end(),
                "kind.featureids is missing " << f);
        }
        for (size_t k = 0; k < kinds.size(); ++k) {
            for (size_t f : kinds[k].featureids) {
                LOOM_ASSERT_EQ(featureid_to_kindid[f], k);
            }
        }
    }
    if (LOOM_DEBUG_LEVEL >= 3) {
        std::vector<size_t> row_counts;
        for (const auto & kind : kinds) {
            row_counts.push_back(kind.mixture.count_rows());
        }
        for (size_t k = 1; k < kinds.size(); ++k) {
            LOOM_ASSERT_EQ(row_counts[k], row_counts[0]);
        }
    }
}

} // namespace loom
