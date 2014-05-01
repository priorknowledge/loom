#pragma once

#include <distributions/io/protobuf.hpp>
#include <distributions/models_fwd.hpp>
#include "protobuf_stream.hpp"
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

    size_t total_size () const
    {
        return booleans_size + counts_size + reals_size;
    }

    static size_t total_size (const ProductModel_SparseValue & value)
    {
        return value.booleans_size()
            + value.counts_size()
            + value.reals_size();
    }

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
        LOOM_ASSERT_EQ(value.observed_size(), total_size());
    }

    bool is_valid (const ProductModel_SparseValue & value) const
    {
        return value.booleans_size() <= booleans_size
            and value.counts_size() <= counts_size
            and value.reals_size() <= reals_size
            and value.observed_size() == total_size();
    }

    template<class Fun>
    void for_each_datatype (Fun & fun) const
    {
        if (booleans_size) {
            fun(static_cast<bool *>(nullptr), booleans_size);
        }
        if (counts_size) {
            fun(static_cast<uint32_t *>(nullptr), counts_size);
        }
        if (reals_size) {
            fun(static_cast<float *>(nullptr), reals_size);
        }
    }
};


template<class Model>
struct Groups;

template<int max_dim>
struct Groups<distributions::dirichlet_discrete::Group<max_dim>>
{
    static auto get (ProductModel_Group & value)
        -> decltype(* value.mutable_dd())
    {
        return * value.mutable_dd();
    }

    static const auto get (const ProductModel_Group & value)
        -> decltype(value.dd())
    {
        return value.dd();
    }
};

template<>
struct Groups<distributions::dirichlet_process_discrete::Group>
{
    static auto get (ProductModel_Group & value)
        -> decltype(* value.mutable_dpd())
    {
        return * value.mutable_dpd();
    }

    static const auto get (const ProductModel_Group & value)
        -> decltype(value.dpd())
    {
        return value.dpd();
    }
};

template<>
struct Groups<distributions::gamma_poisson::Group>
{
    static auto get (ProductModel_Group & value)
        -> decltype(* value.mutable_gp())
    {
        return * value.mutable_gp();
    }

    static const auto get (const ProductModel_Group & value)
        -> decltype(value.gp())
    {
        return value.gp();
    }
};

template<>
struct Groups<distributions::normal_inverse_chi_sq::Group>
{
    static auto get (ProductModel_Group & value)
        -> decltype(* value.mutable_nich())
    {
        return * value.mutable_nich();
    }

    static const auto get (const ProductModel_Group & value)
        -> decltype(value.nich())
    {
        return value.nich();
    }
};


template<class Model>
struct GridPriors;

template<int max_dim>
struct GridPriors<distributions::dirichlet_discrete::Shared<max_dim>>
{
    static const auto get (const ProductModel_HyperPrior & value)
        -> decltype(value.dd())
    {
        return value.dd();
    }
};

template<>
struct GridPriors<distributions::dirichlet_process_discrete::Shared>
{
    static const auto get (const ProductModel_HyperPrior & value)
        -> decltype(value.dpd())
    {
        return value.dpd();
    }
};

template<>
struct GridPriors<distributions::gamma_poisson::Shared>
{
    static const auto get (const ProductModel_HyperPrior & value)
        -> decltype(value.gp())
    {
        return value.gp();
    }
};

template<>
struct GridPriors<distributions::normal_inverse_chi_sq::Shared>
{
    static const auto get (const ProductModel_HyperPrior & value)
        -> decltype(value.nich())
    {
        return value.nich();
    }
};



} // namespace protobuf
} // namespace loom
