#pragma once

#include "common.hpp"
#include "cross_cat.hpp"

namespace loom
{

struct Algorithm8
{
    typedef CrossCat::Value Value;
    typedef CrossCat::Clustering Clustering;
    struct Kind
    {
        ProductModel::Mixture mixture;
        std::vector<size_t> featureids;
    };

    protobuf::SparseValueSchema schema;
    ProductModel model;
    std::vector<Kind> kinds;
    Clustering clustering;
    std::vector<size_t> featureid_to_kindid;
    ::google::protobuf::RepeatedField<bool> unobserved;

    void clear ();

    void model_load (CrossCat & cross_cat);
    void model_dump (CrossCat & cross_cat);

    void mixture_init_empty (rng_t & rng, size_t ephemeral_kind_count);
    void mixture_dump (CrossCat & cross_cat);

    void value_split (
            const Value & product,
            std::vector<Value> & factors) const;

private:

    struct value_split_fun;
};

struct Algorithm8::value_split_fun
{
    const Algorithm8 & algorithm8;
    const Value & product;
    std::vector<Value> & factors;
    size_t absolute_pos;

    template<class FieldType>
    inline void operator() (FieldType *, size_t size)
    {
        typedef protobuf::Fields<FieldType> Fields;
        const auto & product_fields = Fields::get(product);
        for (size_t i = 0, packed_pos = 0; i < size; ++i, ++absolute_pos) {
            size_t kindid = algorithm8.featureid_to_kindid[absolute_pos];
            auto & factor = factors[kindid];
            bool observed = product.observed(absolute_pos);
            factor.set_observed(i, observed);
            if (observed) {
                Fields::get(factor).Add(product_fields.Get(packed_pos++));
            }
        }
    }
};

inline void Algorithm8::value_split (
        const Value & product,
        std::vector<Value> & factors) const
{
    if (LOOM_DEBUG_LEVEL >= 1) {
        LOOM_ASSERT_EQ(factors.size(), kinds.size());
    }

    for (auto & factor : factors) {
        factor.Clear();
        * factor.mutable_observed() = unobserved;
    }

    value_split_fun fun = {* this, product, factors, 0};
    schema.for_each_datatype(fun);
}

} // namespace loom
