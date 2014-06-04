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

//----------------------------------------------------------------------------
// Datatypes

template<class Value> struct Fields;

#define DECLARE_DATATYPE(Typename, fieldname)                       \
template<>                                                          \
struct Fields<Typename>                                             \
{                                                                   \
    static auto get (ProductModel_SparseValue & value)              \
        -> decltype(* value.mutable_ ## fieldname())                \
    {                                                               \
        return * value.mutable_ ## fieldname();                     \
    }                                                               \
    static const auto get (const ProductModel_SparseValue & value)  \
        -> decltype(value.fieldname())                              \
    {                                                               \
        return value.fieldname();                                   \
    }                                                               \
};

DECLARE_DATATYPE(bool, booleans)
DECLARE_DATATYPE(uint32_t, counts)
DECLARE_DATATYPE(float, reals)

#undef DECLARE_DATATYPE


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

//----------------------------------------------------------------------------
// Models

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


template<class Model> struct Shareds;
template<class Model> struct Groups;
template<class Model> struct GridPriors;

#define DECLARE_MODEL(template_, Typename, fieldname)               \
template_                                                           \
struct Shareds<Typename>                                            \
{                                                                   \
    static auto get (ProductModel_Shared & value)                   \
        -> decltype(* value.mutable_ ## fieldname())                \
    {                                                               \
        return * value.mutable_ ## fieldname();                     \
    }                                                               \
    static const auto get (const ProductModel_Shared & value)       \
        -> decltype(value.fieldname())                              \
    {                                                               \
        return value.fieldname();                                   \
    }                                                               \
};                                                                  \
template_                                                           \
struct Groups<Typename>                                             \
{                                                                   \
    static auto get (ProductModel_Group & value)                    \
        -> decltype(* value.mutable_ ## fieldname())                \
    {                                                               \
        return * value.mutable_ ## fieldname();                     \
    }                                                               \
    static const auto get (const ProductModel_Group & value)        \
        -> decltype(value.fieldname())                              \
    {                                                               \
        return value.fieldname();                                   \
    }                                                               \
};                                                                  \
template_                                                           \
struct GridPriors<Typename>                                         \
{                                                                   \
    static const auto get (const ProductModel_HyperPrior & value)   \
        -> decltype(value.fieldname())                              \
    {                                                               \
        return value.fieldname();                                   \
    }                                                               \
};

DECLARE_MODEL(template<>, BetaBernoulli, bb)
DECLARE_MODEL(template<int max_dim>, DirichletDiscrete<max_dim>, dd)
DECLARE_MODEL(template<>, DirichletProcessDiscrete, dpd)
DECLARE_MODEL(template<>, GammaPoisson, gp)
DECLARE_MODEL(template<>, NormalInverseChiSq, nich)

#undef DECLARE_MODEL

} // namespace protobuf
} // namespace loom
