#pragma once

#include <distributions/io/protobuf.hpp>
#include <distributions/models_fwd.hpp>
#include <loom/common.hpp>
#include <loom/protobuf_stream.hpp>
#include <loom/models.hpp>
#include <loom/schema.pb.h>


namespace loom
{

inline std::ostream & operator<< (
        std::ostream & os,
        const ::google::protobuf::Message & message)
{
    return os << message.ShortDebugString();
}


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

    SparseValueSchema () :
        booleans_size(0),
        counts_size(0),
        reals_size(0)
    {
    }

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

    static size_t observed_count (const ProductModel_SparseValue & value)
    {
        size_t count = 0;
        for (size_t i = 0, size = value.observed_size(); i < size; ++i) {
            if (value.observed(i)) {
                count += 1;
            }
        }
        return count;
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
        LOOM_ASSERT_EQ(value.observed_size(), total_size());
        LOOM_ASSERT_LE(value.booleans_size(), booleans_size);
        LOOM_ASSERT_LE(value.counts_size(), counts_size);
        LOOM_ASSERT_LE(value.reals_size(), reals_size);
        LOOM_ASSERT_EQ(observed_count(value), total_size(value));
    }

    bool is_valid (const ProductModel_SparseValue & value) const
    {
        return value.observed_size() == total_size()
            and value.booleans_size() <= booleans_size
            and value.counts_size() <= counts_size
            and value.reals_size() <= reals_size
            and observed_count(value) == total_size(value);
    }

    template<class Fun>
    void for_each_datatype (Fun & fun) const
    {
        fun(static_cast<bool *>(nullptr), booleans_size);
        fun(static_cast<uint32_t *>(nullptr), counts_size);
        fun(static_cast<float *>(nullptr), reals_size);
    }

    bool operator== (const SparseValueSchema & other) const
    {
        return booleans_size == other.booleans_size
            and counts_size == other.counts_size
            and reals_size == other.reals_size;
    }

    friend std::ostream & operator<< (
        std::ostream & os,
        const SparseValueSchema & schema)
    {
        return os << "{" <<
            schema.booleans_size << ", " <<
            schema.counts_size << ", " <<
            schema.reals_size << "}";
    }
};


// This accounts for the many-to-one C++-to-protobuf model mapping,
// e.g. DirichletDiscrete<N> maps to DirichletDiscrete for all N.
struct ModelCounts
{
    size_t bb;
    size_t dd;
    size_t dpd;
    size_t gp;
    size_t nich;

    ModelCounts () :
        bb(0),
        dd(0),
        dpd(0),
        gp(0),
        nich(0)
    {}

    size_t & operator[] (BetaBernoulli *) { return bb; }
    template<int max_dim>
    size_t & operator[] (DirichletDiscrete<max_dim> *) { return dd; }
    size_t & operator[] (DirichletProcessDiscrete *) { return dpd; }
    size_t & operator[] (GammaPoisson *) { return gp; }
    size_t & operator[] (NormalInverseChiSq *) { return nich; }
};


template<class Model>
struct Shareds;

template<>
struct Shareds<BetaBernoulli>
{
    static auto get (ProductModel_Shared & value)
        -> decltype(* value.mutable_bb())
    {
        return * value.mutable_bb();
    }

    static const auto get (const ProductModel_Shared & value)
        -> decltype(value.bb())
    {
        return value.bb();
    }
};

template<int max_dim>
struct Shareds<DirichletDiscrete<max_dim>>
{
    static auto get (ProductModel_Shared & value)
        -> decltype(* value.mutable_dd())
    {
        return * value.mutable_dd();
    }

    static const auto get (const ProductModel_Shared & value)
        -> decltype(value.dd())
    {
        return value.dd();
    }
};

template<>
struct Shareds<DirichletProcessDiscrete>
{
    static auto get (ProductModel_Shared & value)
        -> decltype(* value.mutable_dpd())
    {
        return * value.mutable_dpd();
    }

    static const auto get (const ProductModel_Shared & value)
        -> decltype(value.dpd())
    {
        return value.dpd();
    }
};

template<>
struct Shareds<GammaPoisson>
{
    static auto get (ProductModel_Shared & value)
        -> decltype(* value.mutable_gp())
    {
        return * value.mutable_gp();
    }

    static const auto get (const ProductModel_Shared & value)
        -> decltype(value.gp())
    {
        return value.gp();
    }
};

template<>
struct Shareds<NormalInverseChiSq>
{
    static auto get (ProductModel_Shared & value)
        -> decltype(* value.mutable_nich())
    {
        return * value.mutable_nich();
    }

    static const auto get (const ProductModel_Shared & value)
        -> decltype(value.nich())
    {
        return value.nich();
    }
};


template<class Model>
struct Groups;

template<>
struct Groups<BetaBernoulli>
{
    static auto get (ProductModel_Group & value)
        -> decltype(* value.mutable_bb())
    {
        return * value.mutable_bb();
    }

    static const auto get (const ProductModel_Group & value)
        -> decltype(value.bb())
    {
        return value.bb();
    }
};

template<int max_dim>
struct Groups<DirichletDiscrete<max_dim>>
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
struct Groups<DirichletProcessDiscrete>
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
struct Groups<GammaPoisson>
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
struct Groups<NormalInverseChiSq>
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

template<>
struct GridPriors<BetaBernoulli>
{
    static const auto get (const ProductModel_HyperPrior & value)
        -> decltype(value.bb())
    {
        return value.bb();
    }
};

template<int max_dim>
struct GridPriors<DirichletDiscrete<max_dim>>
{
    static const auto get (const ProductModel_HyperPrior & value)
        -> decltype(value.dd())
    {
        return value.dd();
    }
};

template<>
struct GridPriors<DirichletProcessDiscrete>
{
    static const auto get (const ProductModel_HyperPrior & value)
        -> decltype(value.dpd())
    {
        return value.dpd();
    }
};

template<>
struct GridPriors<GammaPoisson>
{
    static const auto get (const ProductModel_HyperPrior & value)
        -> decltype(value.gp())
    {
        return value.gp();
    }
};

template<>
struct GridPriors<NormalInverseChiSq>
{
    static const auto get (const ProductModel_HyperPrior & value)
        -> decltype(value.nich())
    {
        return value.nich();
    }
};

} // namespace protobuf
} // namespace loom
