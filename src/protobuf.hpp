#pragma once

#include <distributions/io/protobuf.hpp>
#include <distributions/io/protobuf_stream.hpp>
#include "common.hpp"
#include "schema.pb.h"

namespace loom
{
namespace protobuf
{

using namespace ::protobuf::loom;
using namespace ::protobuf::distributions;
using namespace ::distributions::protobuf;


template<class Value>
struct Fields;

template<>
struct Fields<bool>
{
    static auto get (ProductModel_SparseValue & value)
        -> decltype(* value.mutable_booleans())
    {
        return * value.mutable_booleans();
    }

    static const auto get (const ProductModel_SparseValue & value)
        -> decltype(value.booleans())
    {
        return value.booleans();
    }
};

template<>
struct Fields<uint32_t>
{
    static auto get (ProductModel_SparseValue & value)
        -> decltype(* value.mutable_counts())
    {
        return * value.mutable_counts();
    }

    static const auto get (const ProductModel_SparseValue & value)
        -> decltype(value.counts())
    {
        return value.counts();
    }
};

template<>
struct Fields<float>
{
    static auto get (ProductModel_SparseValue & value)
        -> decltype(* value.mutable_reals())
    {
        return * value.mutable_reals();
    }

    static const auto get (const ProductModel_SparseValue & value)
        -> decltype(value.reals())
    {
        return value.reals();
    }
};


struct SparseValueSchema
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

    void operator+= (const SparseValueSchema & other)
    {
        booleans_size += other.booleans_size;
        counts_size += other.counts_size;
        reals_size += other.reals_size;
    }

    void validate (const ProductModel_SparseValue & value) const
    {
        LOOM_ASSERT_LE(value.booleans_size(), booleans_size);
        LOOM_ASSERT_LE(value.counts_size(), counts_size);
        LOOM_ASSERT_LE(value.reals_size(), reals_size);
        LOOM_ASSERT_EQ(
            value.observed_size(),
            booleans_size + counts_size + reals_size);
    }

    template<class Fun>
    void for_each_datatype (Fun & fun) const
    {
        fun(static_cast<bool *>(nullptr), booleans_size);
        fun(static_cast<uint32_t *>(nullptr), counts_size);
        fun(static_cast<float *>(nullptr), reals_size);
    }
};

} // namespace protobuf
} // namespace loom
