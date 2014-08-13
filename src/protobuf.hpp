// Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// - Neither the name of Salesforce.com nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <distributions/io/protobuf.hpp>
#include <loom/common.hpp>
#include <loom/protobuf_stream.hpp>
#include <loom/models.hpp>
#include <loom/schema.pb.h>


namespace loom
{
namespace protobuf
{

using namespace ::protobuf::loom;
using namespace ::protobuf::distributions;
using namespace ::distributions::protobuf;

template<class Typename> struct Fields;

//----------------------------------------------------------------------------
// Values

#define DECLARE_FIELDS(Typename, fieldname)                         \
template<>                                                          \
struct Fields<Typename>                                             \
{                                                                   \
    static auto get (ProductValue & value)                          \
        -> decltype(* value.mutable_ ## fieldname())                \
    {                                                               \
        return * value.mutable_ ## fieldname();                     \
    }                                                               \
    static const auto get (const ProductValue & value)              \
        -> decltype(value.fieldname())                              \
    {                                                               \
        return value.fieldname();                                   \
    }                                                               \
};

DECLARE_FIELDS(bool, booleans)
DECLARE_FIELDS(uint32_t, counts)
DECLARE_FIELDS(float, reals)

#undef DECLARE_FIELDS

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


#define DECLARE_FIELDS(template_, Typename, fieldname)              \
template_                                                           \
struct Fields<Typename>                                             \
{                                                                   \
    static auto get (ProductModel::Shared & value)                  \
        -> decltype(* value.mutable_ ## fieldname())                \
    {                                                               \
        return * value.mutable_ ## fieldname();                     \
    }                                                               \
    static const auto get (const ProductModel::Shared & value)      \
        -> decltype(value.fieldname())                              \
    {                                                               \
        return value.fieldname();                                   \
    }                                                               \
    static auto get (ProductModel::Group & value)                   \
        -> decltype(* value.mutable_ ## fieldname())                \
    {                                                               \
        return * value.mutable_ ## fieldname();                     \
    }                                                               \
    static const auto get (const ProductModel::Group & value)       \
        -> decltype(value.fieldname())                              \
    {                                                               \
        return value.fieldname();                                   \
    }                                                               \
    static const auto get (const HyperPrior & value)                \
        -> decltype(value.fieldname())                              \
    {                                                               \
        return value.fieldname();                                   \
    }                                                               \
};

DECLARE_FIELDS(template<>, BetaBernoulli, bb)
DECLARE_FIELDS(template<int max_dim>, DirichletDiscrete<max_dim>, dd)
DECLARE_FIELDS(template<>, DirichletProcessDiscrete, dpd)
DECLARE_FIELDS(template<>, GammaPoisson, gp)
DECLARE_FIELDS(template<>, NormalInverseChiSq, nich)

#undef DECLARE_FIELDS

} // namespace protobuf
} // namespace loom
