#pragma once

#include "common.hpp"
#include "protobuf.hpp"
#include "product_model.hpp"

namespace loom
{

struct CrossCat
{
    typedef protobuf::ProductModel::SparseValue Value;
    typedef distributions::Clustering<int>::PitmanYor Clustering;
    struct Kind
    {
        ProductModel model;
        ProductModel::Mixture mixture;
        std::vector<size_t> featureids;
    };

    protobuf::SparseValueSchema schema;
    std::vector<Kind> kinds;
    distributions::Clustering<int>::PitmanYor clustering;
    //Clustering clustering;
    std::vector<size_t> featureid_to_kindid;

    void load (const protobuf::CrossCatModel & message);

    void mixture_load (const char * dirname) { TODO("load mixtures"); }
    void mixture_dump (const char * dirname);
    void mixture_init (rng_t & rng);

    void value_split (
            const Value & product,
            std::vector<Value> & factors) const;

    struct ValueJoiner;
    void value_join (
            Value & product,
            const std::vector<Value> & factors) const;

private:

    struct value_split_fun;
    struct value_join_fun;
};

inline void CrossCat::mixture_init (rng_t & rng)
{
    for (auto & kind : kinds) {
        kind.model.mixture_init(kind.mixture, rng);
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
        if (size) {
            typedef protobuf::Fields<FieldType> Fields;
            const auto & product_fields = Fields::get(product);
            for (size_t i = 0, packed_pos = 0; i < size; ++i, ++absolute_pos) {
                size_t kindid = cross_cat.featureid_to_kindid[absolute_pos];
                auto & factor = factors[kindid];
                bool observed = product.observed(absolute_pos);
                factor.add_observed(observed);
                if (observed) {
                    Fields::get(factor).Add(product_fields.Get(packed_pos++));
                }
            }
        }
    }
};

inline void CrossCat::value_split (
        const Value & product,
        std::vector<Value> & factors) const
{
    factors.resize(kinds.size());
    for (auto & factor : factors) {
        factor.Clear();
    }

    value_split_fun fun = {* this, product, factors, 0};
    schema.for_each_datatype(fun);
}

struct CrossCat::value_join_fun
{
    const CrossCat & cross_cat;
    std::vector<size_t> packed_pos_list;
    Value & product;
    const std::vector<Value> & factors;
    size_t absolute_pos;

    template<class FieldType>
    void operator() (FieldType *, size_t size)
    {
        if (size) {
            typedef protobuf::Fields<FieldType> Fields;
            auto & product_observed = * product.mutable_observed();
            auto & product_fields = Fields::get(product);
            packed_pos_list.clear();
            packed_pos_list.resize(cross_cat.kinds.size(), 0);
            for (size_t i = 0; i < size; ++i, ++absolute_pos) {
                size_t kindid = cross_cat.featureid_to_kindid[absolute_pos];
                const auto & factor = factors[kindid];
                auto & packed_pos = packed_pos_list[kindid];
                bool observed = factor.observed(packed_pos);
                product_observed.Add(observed);
                if (observed) {
                    product_fields.Add(Fields::get(factor).Get(packed_pos++));
                }
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

} // namespace loom
