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

} // namespace protobuf
} // namespace loom
