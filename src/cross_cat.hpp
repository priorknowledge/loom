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

    void mixture_load (const char * dirname, rng_t & rng);
    void mixture_dump (const char * dirname);
    void mixture_init_empty (size_t empty_group_count, rng_t & rng);

    void value_split (
            const Value & product,
            std::vector<Value> & factors) const;

    struct ValueJoiner;
    void value_join (
            Value & product,
            const std::vector<Value> & factors) const;

    void value_resize (Value & value) const;

    void infer_hypers (rng_t & rng);

    float total_score (rng_t & rng) const;

    void validate () const;

private:

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
    const Value & product;
    std::vector<Value> & factors;
    size_t absolute_pos;

    template<class FieldType>
    inline void operator() (FieldType *, size_t size)
    {
        typedef protobuf::Fields<FieldType> Fields;
        const auto & product_fields = Fields::get(product);
        for (size_t i = 0, packed_pos = 0; i < size; ++i, ++absolute_pos) {
            auto kindid = cross_cat.featureid_to_kindid[absolute_pos];
            auto & factor = factors[kindid];
            bool observed = product.observed(absolute_pos);
            factor.add_observed(observed);
            if (observed) {
                Fields::get(factor).Add(product_fields.Get(packed_pos++));
            }
        }
    }
};

inline void CrossCat::value_split (
        const Value & product,
        std::vector<Value> & factors) const
{
    if (LOOM_DEBUG_LEVEL >= 1) {
        LOOM_ASSERT_EQ(factors.size(), kinds.size());
    }

    for (auto & factor : factors) {
        factor.Clear();
    }

    value_split_fun fun = {* this, product, factors, 0};
    schema.for_each_datatype(fun);
}

struct CrossCat::value_join_fun
{
    const CrossCat & cross_cat;
    std::vector<size_t> & packed_pos_list;
    Value & product;
    const std::vector<Value> & factors;
    size_t absolute_pos;

    template<class FieldType>
    void operator() (FieldType *, size_t size)
    {
        typedef protobuf::Fields<FieldType> Fields;
        auto & product_observed = * product.mutable_observed();
        auto & product_fields = Fields::get(product);
        packed_pos_list.clear();
        packed_pos_list.resize(cross_cat.kinds.size(), 0);
        for (size_t i = 0; i < size; ++i, ++absolute_pos) {
            auto kindid = cross_cat.featureid_to_kindid[absolute_pos];
            const auto & factor = factors[kindid];
            auto & packed_pos = packed_pos_list[kindid];
            bool observed = factor.observed(packed_pos);
            product_observed.Add(observed);
            if (observed) {
                product_fields.Add(Fields::get(factor).Get(packed_pos++));
            }
        }
    }
};

struct CrossCat::ValueJoiner
{
    ValueJoiner (const CrossCat & cross_cat) : cross_cat_(cross_cat) {}

    void operator() (
            Value & product,
            const std::vector<Value> & factors)
    {
        product.Clear();

        CrossCat::value_join_fun fun =
            {cross_cat_, packed_pos_list_, product, factors, 0};
        cross_cat_.schema.for_each_datatype(fun);
    }

private:

    const CrossCat & cross_cat_;
    std::vector<size_t> packed_pos_list_;
};

inline void CrossCat::value_join (
        Value & product,
        const std::vector<Value> & factors) const
{
    ValueJoiner(* this)(product, factors);
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
