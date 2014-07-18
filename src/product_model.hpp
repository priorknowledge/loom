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

#include <vector>
#include <distributions/io/protobuf.hpp>
#include <loom/common.hpp>
#include <loom/indexed_vector.hpp>
#include <loom/models.hpp>
#include <loom/products.hpp>
#include <loom/product_value.hpp>

namespace loom
{

struct ProductModel
{
    typedef protobuf::ProductValue Value;
    struct Feature
    {
        template<class T>
        struct Container { typedef IndexedVector<typename T::Shared> t; };
    };
    typedef ForEachFeatureType<Feature> Features;

    ValueSchema schema;
    Clustering::Shared clustering;
    Features features;
    std::vector<Value> tares;

    void clear ();

    void load (
            const protobuf::ProductModel_Shared & message,
            const std::vector<size_t> & featureids);
    void dump (protobuf::ProductModel_Shared & message) const;

    void extend (const ProductModel & other);

    void add_value (const Value & value, rng_t & rng);
    void remove_value (const Value & value, rng_t & rng);
    void add_diff (const Value::Diff & diff, rng_t & rng);
    void remove_diff (const Value::Diff & diff, rng_t & rng);
    void realize (rng_t & rng);

    void validate () const;

private:

    struct dump_fun;
    struct add_value_fun;
    struct remove_value_fun;
    struct realize_fun;
    struct extend_fun;
    struct clear_fun;
};

inline void ProductModel::validate () const
{
    if (LOOM_DEBUG_LEVEL >= 1) {
        schema.validate(features);
        for (const auto & tare : tares) {
            schema.validate(tare);
        }
    }
}

struct ProductModel::add_value_fun
{
    Features & shareds;
    rng_t & rng;

    template<class T>
    void operator() (
        T * t,
        size_t i,
        const typename T::Value & value)
    {
        shareds[t][i].add_value(value, rng);
    }
};

inline void ProductModel::add_value (
        const Value & value,
        rng_t & rng)
{
    add_value_fun fun = {features, rng};
    read_value(fun, schema, features, value);
}

struct ProductModel::remove_value_fun
{
    Features & shareds;
    rng_t & rng;

    template<class T>
    void operator() (
        T * t,
        size_t i,
        const typename T::Value & value)
    {
        shareds[t][i].remove_value(value, rng);
    }
};

inline void ProductModel::remove_value (
        const Value & value,
        rng_t & rng)
{
    remove_value_fun fun = {features, rng};
    read_value(fun, schema, features, value);
}

inline void ProductModel::add_diff (
        const Value::Diff & diff,
        rng_t & rng)
{
    add_value(diff.pos(), rng);
}

inline void ProductModel::remove_diff (
        const Value::Diff & diff,
        rng_t & rng)
{
    remove_value(diff.pos(), rng);
}

struct ProductModel::realize_fun
{
    rng_t & rng;

    template<class T>
    void operator() (
            T *,
            size_t,
            typename T::Shared & shared)
    {
        shared.realize(rng);
    }
};

inline void ProductModel::realize (rng_t & rng)
{
    realize_fun fun = {rng};
    for_each_feature(fun, features);
}

} // namespace loom
