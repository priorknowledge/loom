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

    std::vector<Kind> kinds;
    distributions::Clustering<int>::PitmanYor clustering;
    //Clustering clustering;
    std::vector<size_t> featureid_to_kindid;
    protobuf::SparseValueSchema schema;

    void load (const protobuf::CrossCatModel & message);

    void mixture_load (const char * dirname) { TODO("load mixtures"); }
    void mixture_dump (const char * dirname);
    void mixture_init (rng_t & rng);

    void value_split (
            const Value & product,
            std::vector<Value> & factors) const;

    struct ValueJoiner;

private:

    template<class FieldType>
    void _value_split (
            const Value & product,
            std::vector<Value> & factors,
            size_t & absolute_pos) const;
};

inline void CrossCat::mixture_init (rng_t & rng)
{
    for (auto & kind : kinds) {
        kind.model.mixture_init(kind.mixture, rng);
    }
}

template<class FieldType>
inline void CrossCat::_value_split (
        const Value & product,
        std::vector<Value> & factors,
        size_t & absolute_pos) const
{
    if (const size_t size = schema.size<FieldType>()) {
        typedef protobuf::Fields<FieldType> Fields;
        const auto & product_fields = Fields::get(product);
        for (size_t i = 0, packed_pos = 0; i < size; ++i, ++absolute_pos) {
            auto & factor = factors[featureid_to_kindid[absolute_pos]];
            bool observed = product.observed(absolute_pos);
            factor.add_observed(observed);
            if (observed) {
                Fields::get(factor).Add(product_fields.Get(packed_pos++));
            }
        }
    }
}

inline void CrossCat::value_split (
        const Value & product,
        std::vector<Value> & factors) const
{
    factors.resize(kinds.size());
    for (auto & factor : factors) {
        factor.Clear();
    }

    size_t absolute_pos = 0;
    _value_split<bool>(product, factors, absolute_pos);
    _value_split<uint32_t>(product, factors, absolute_pos);
    _value_split<float>(product, factors, absolute_pos);
}

class CrossCat::ValueJoiner
{
    ValueJoiner (const CrossCat & cross_cat) : cross_cat_(cross_cat) {}

    void operator() (
            Value & product,
            const std::vector<Value> & factors)
    {
        product.Clear();

        size_t absolute_pos = 0;
        _value_join<bool>(product, factors, absolute_pos);
        _value_join<uint32_t>(product, factors, absolute_pos);
        _value_join<float>(product, factors, absolute_pos);
    }

private:

    template<class FieldType>
    void _value_join (
            Value & product,
            const std::vector<Value> & factors,
            size_t & absolute_pos)
    {
        if (const size_t size = cross_cat_.schema.size<FieldType>()) {
            typedef protobuf::Fields<FieldType> Fields;
            auto & product_observed = * product.mutable_observed();
            auto & product_fields = Fields::get(product);
            packed_pos_.clear();
            packed_pos_.resize(cross_cat_.kinds.size(), 0);
            for (size_t i = 0; i < size; ++i, ++absolute_pos) {
                size_t kindid = cross_cat_.featureid_to_kindid[absolute_pos];
                const auto & factor = factors[kindid];
                auto & packed_pos = packed_pos_[kindid];
                bool observed = factor.observed(packed_pos);
                product_observed.Add(observed);
                if (observed) {
                    product_fields.Add(Fields::get(factor).Get(packed_pos++));
                }
            }
        }
    }

    const CrossCat & cross_cat_;
    std::vector<size_t> packed_pos_;
};

} // namespace loom
