#pragma once

#include "common.hpp"
#include "protobuf.hpp"
#include "product_model.hpp"

namespace loom
{

struct Schema
{
    size_t booleans_size;
    size_t counts_size;
    size_t reals_size;

    void clear ()
    {
        booleans_size = 0;
        counts_size = 0;
        reals_size = 0;
    }

    template<class T>
    size_t size () const { return size(static_cast<T *>(nullptr)); }
    size_t size (bool *) const { return booleans_size; }
    size_t size (uint32_t *) const { return counts_size; }
    size_t size (float *) const { return reals_size; }
};

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
    Schema schema;

    void load (const protobuf::CrossCatModel & message);

    void mixture_load (const char * dirname) { TODO("load mixtures"); }
    void mixture_dump (const char * dirname);
    void mixture_init (rng_t & rng);

    void value_split (
            const Value & product,
            std::vector<Value> & factors) const;
    void value_join (
            Value & product,
            const std::vector<Value> & factors) const;

private:

    template<class T>
    void _value_split (
            const Value & product,
            std::vector<Value> & factors,
            size_t & absolute_pos) const;

    template<class T>
    void _value_join (
            Value & product,
            const std::vector<Value> & factors,
            size_t & absolute_pos) const;
};

inline void CrossCat::mixture_init (rng_t & rng)
{
    for (auto & kind : kinds) {
        kind.model.mixture_init(kind.mixture, rng);
    }
}

template<class T>
inline void CrossCat::_value_split (
        const Value & product,
        std::vector<Value> & factors,
        size_t & absolute_pos) const
{
    const auto & product_fields = protobuf::Fields<T>::get(product);
    if (const size_t size = schema.size<T>()) {
        for (size_t i = 0, packed_pos = 0; i < size; ++i, ++absolute_pos) {
            auto & factor = factors[featureid_to_kindid[absolute_pos]];
            bool observed = product.observed(absolute_pos);
            factor.add_observed(observed);
            if (observed) {
                const auto & value = product_fields.Get(packed_pos++);
                protobuf::Fields<T>::get(factor).Add(value);
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

//template<class T>
//inline void CrossCat::_value_join (
//        Value & product,
//        const std::vector<Value> & factors,
//        size_t & absolute_pos) const
//{
//    auto & product_fields = protobuf::Fields<T>::get(product);
//    if (const size_t size = product_fields.size()) {
//        for (size_t i = 0, packed_pos = 0; i < size; ++i, ++absolute_pos) {
//            auto & factor = factors[featureid_to_kindid[absolute_pos]];
//            bool observed = product.observed(absolute_pos);
//            factor.add_observed(observed);
//            if (observed) {
//                const auto & value = product_fields.Get(packed_pos++);
//                protobuf::Fields<T>::get(factor).Add(value);
//            }
//        }
//    }
//}
//
//inline void CrossCat::value_join (
//        Value & product,
//        const std::vector<Value> & factors) const
//{
//    product.Clear();
//
//    size_t absolute_pos = 0;
//    _value_join<bool>(product, factors, absolute_pos);
//    _value_join<uint32_t>(product, factors, absolute_pos);
//    _value_join<float>(product, factors, absolute_pos);
//}

} // namespace loom
