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

#include <mutex>
#include <loom/common.hpp>
#include <loom/protobuf.hpp>
#include <loom/models.hpp>

namespace loom
{

typedef protobuf::ProductValue ProductValue;

inline std::ostream & operator<< (
        std::ostream & os,
        const ProductValue::Observed::Sparsity & sparsity)
{
    return os << ProductValue::Observed::Sparsity_Name(sparsity);
}

inline bool operator== (
        const ProductValue::Observed & x,
        const ProductValue::Observed & y)
{
    return LOOM_LIKELY(x.sparsity() == y.sparsity())
        and LOOM_LIKELY(x.dense() == y.dense())
        and LOOM_LIKELY(x.sparse() == y.sparse());
}

inline bool operator== (
        const ProductValue & x,
        const ProductValue & y)
{
    return LOOM_LIKELY(x.observed() == y.observed())
        and LOOM_LIKELY(x.booleans() == y.booleans())
        and LOOM_LIKELY(x.counts() == y.counts())
        and LOOM_LIKELY(x.reals() == y.reals());
}

class BlockIterator
{
    size_t begin_;
    size_t end_;

public:

    BlockIterator () : begin_(0), end_(0) {}
    operator bool () const { return end_ != begin_; }
    BlockIterator & operator() (size_t size)
    {
        begin_ = end_;
        end_ += size;
        return * this;
    }
    bool ok (size_t i) const { return i < end_; }
    size_t get (size_t i) const { return i - begin_; }
    size_t begin () const { return begin_; }
    size_t end () const { return end_; }
};

template<class Derived>
class ForEachDataType
{
    typedef typename Derived::template Container<bool>::t Booleans;
    typedef typename Derived::template Container<uint32_t>::t Counts;
    typedef typename Derived::template Container<float>::t Reals;

public:

    Booleans booleans;
    Counts counts;
    Reals reals;

    Booleans & operator[] (bool *) { return booleans; }
    Counts & operator[] (uint32_t *) { return counts; }
    Reals & operator[] (float *) { return reals; }

    const Booleans & operator[] (bool *) const { return booleans; }
    const Counts & operator[] (uint32_t *) const { return counts; }
    const Reals & operator[] (float *) const { return reals; }
};

//----------------------------------------------------------------------------
// Schema

struct ValueSchema
{
    size_t booleans_size;
    size_t counts_size;
    size_t reals_size;

    ValueSchema () :
        booleans_size(0),
        counts_size(0),
        reals_size(0)
    {
    }

    void load (const ProductValue & value)
    {
        booleans_size = value.booleans_size();
        counts_size = value.counts_size();
        reals_size = value.reals_size();
    }

    template<class Derived>
    void load (const ForEachFeatureType<Derived> & features)
    {
        clear();
        booleans_size += features.bb.size();
        counts_size += features.dd16.size();
        counts_size += features.dd256.size();
        counts_size += features.dpd.size();
        counts_size += features.gp.size();
        reals_size += features.nich.size();
    }

    void dump (ProductValue & value) const
    {
        clear(value);
        value.mutable_observed()->set_sparsity(ProductValue::Observed::ALL);

        for (size_t i = 0; i < booleans_size; ++i) {
            value.add_booleans(false);
        }
        for (size_t i = 0; i < counts_size; ++i) {
            value.add_counts(0);
        }
        for (size_t i = 0; i < reals_size; ++i) {
            value.add_reals(0.f);
        }
    }

    size_t total_size () const
    {
        return booleans_size + counts_size + reals_size;
    }

    static size_t total_size (const ProductValue & value)
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

    static void clear (ProductValue::Observed & observed)
    {
        observed.set_sparsity(ProductValue::Observed::NONE);
        observed.clear_dense();
        observed.clear_sparse();
    }

    static void clear (ProductValue & value)
    {
        clear(* value.mutable_observed());
        value.clear_booleans();
        value.clear_counts();
        value.clear_reals();
    }

    static void clear (ProductValue::Diff & diff)
    {
        clear(* diff.mutable_pos());
        clear(* diff.mutable_neg());
        diff.clear_tares();
    }

    void fill_data_with_zeros (ProductValue & value) const;

    void operator+= (const ValueSchema & other)
    {
        booleans_size += other.booleans_size;
        counts_size += other.counts_size;
        reals_size += other.reals_size;
    }

    size_t observed_count (const ProductValue::Observed & observed) const
    {
        switch (observed.sparsity()) {
            case ProductValue::Observed::ALL:
                return total_size();

            case ProductValue::Observed::DENSE:
                return std::count_if(
                    observed.dense().begin(),
                    observed.dense().end(),
                    [](bool i){ return i; });

            case ProductValue::Observed::SPARSE:
                return observed.sparse_size();

            case ProductValue::Observed::NONE:
                return 0;
        }

        return 0;  // pacify gcc
    }

    bool sparse_is_valid (const ProductValue::Observed & observed) const
    {
        const auto & sparse = observed.sparse();
        if (const size_t size = sparse.size()) {
            for (size_t i = 1; i < size; ++i) {
                if (LOOM_UNLIKELY(sparse.Get(i - 1) >= sparse.Get(i))) {
                    return false;
                }
            }
            return sparse.Get(size - 1) < total_size();
        }
        return true;
    }

    void validate (const ProductValue::Observed & observed) const
    {
        switch (observed.sparsity()) {
            case ProductValue::Observed::ALL:
                LOOM_ASSERT_EQ(observed.dense_size(), 0);
                LOOM_ASSERT_EQ(observed.sparse_size(), 0);
                return;

            case ProductValue::Observed::DENSE:
                LOOM_ASSERT_EQ(observed.dense_size(), total_size());
                LOOM_ASSERT_EQ(observed.sparse_size(), 0);
                return;

            case ProductValue::Observed::SPARSE:
                LOOM_ASSERT_EQ(observed.dense_size(), 0);
                LOOM_ASSERT(
                    sparse_is_valid(observed),
                    "invalid sparse: " << observed.sparse() <<
                    ", total_size = " << total_size());
                return;

            case ProductValue::Observed::NONE:
                LOOM_ASSERT_EQ(observed.dense_size(), 0);
                LOOM_ASSERT_EQ(observed.sparse_size(), 0);
                return;
        }
    }

    bool is_valid (const ProductValue::Observed & observed) const
    {
        switch (observed.sparsity()) {
            case ProductValue::Observed::ALL:
                return observed.dense_size() == 0
                    and observed.sparse_size() == 0;

            case ProductValue::Observed::DENSE:
                return observed.dense_size() == total_size()
                    and observed.sparse_size() == 0;

            case ProductValue::Observed::SPARSE:
                return observed.dense_size() == 0
                    and sparse_is_valid(observed);

            case ProductValue::Observed::NONE:
                return observed.dense_size() == 0
                    and observed.sparse_size() == 0;
        }

        return false;  // pacify gcc
    }

    void validate (const ProductValue & value) const
    {
        const auto & observed = value.observed();
        validate(observed);
        switch (observed.sparsity()) {
            case ProductValue::Observed::ALL:
                LOOM_ASSERT_EQ(value.booleans_size(), booleans_size);
                LOOM_ASSERT_EQ(value.counts_size(), counts_size);
                LOOM_ASSERT_EQ(value.reals_size(), reals_size);
                return;

            case ProductValue::Observed::DENSE:
                LOOM_ASSERT_LE(value.booleans_size(), booleans_size);
                LOOM_ASSERT_LE(value.counts_size(), counts_size);
                LOOM_ASSERT_LE(value.reals_size(), reals_size);
                LOOM_ASSERT_LE(observed_count(observed), total_size(value));
                return;

            case ProductValue::Observed::SPARSE:
                LOOM_ASSERT_LE(value.booleans_size(), booleans_size);
                LOOM_ASSERT_LE(value.counts_size(), counts_size);
                LOOM_ASSERT_LE(value.reals_size(), reals_size);
                LOOM_ASSERT_LE(observed_count(observed), total_size(value));
                return;

            case ProductValue::Observed::NONE:
                LOOM_ASSERT_EQ(value.booleans_size(), 0);
                LOOM_ASSERT_EQ(value.counts_size(), 0);
                LOOM_ASSERT_EQ(value.reals_size(), 0);
                return;
        }
    }

    bool is_valid (const ProductValue & value) const
    {
        const auto & observed = value.observed();
        if (LOOM_UNLIKELY(not is_valid(observed))) {
            return false;
        }
        switch (observed.sparsity()) {
            case ProductValue::Observed::ALL:
                return value.booleans_size() == booleans_size
                    and value.counts_size() == counts_size
                    and value.reals_size() == reals_size;

            case ProductValue::Observed::DENSE:
                return value.booleans_size() <= booleans_size
                    and value.counts_size() <= counts_size
                    and value.reals_size() <= reals_size
                    and observed_count(observed) <= total_size(value);

            case ProductValue::Observed::SPARSE:
                return value.booleans_size() <= booleans_size
                    and value.counts_size() <= counts_size
                    and value.reals_size() <= reals_size
                    and observed_count(observed) <= total_size(value);

            case ProductValue::Observed::NONE:
                return value.booleans_size() == 0
                    and value.counts_size() == 0
                    and value.reals_size() == 0;
        }

        return false;  // pacify gcc
    }

    void validate (const ProductValue::Diff & diff) const
    {
        validate(diff.pos());
        validate(diff.neg());
        LOOM_ASSERT(
            diff.tares_size() or not total_size(diff.neg()),
            "diff has neg parts but no tares");
    }

    bool is_valid (const ProductValue::Diff & diff) const
    {
        return is_valid(diff.pos())
            and is_valid(diff.neg())
            and (diff.tares_size() or not total_size(diff.neg()));
    }

    template<class Derived>
    void validate (const ForEachFeatureType<Derived> & features) const
    {
        LOOM_ASSERT_EQ(booleans_size,
            features.bb.size());
        LOOM_ASSERT_EQ(counts_size,
            features.dd16.size() +
            features.dd256.size() +
            features.dpd.size() +
            features.gp.size());
        LOOM_ASSERT_EQ(reals_size,
            features.nich.size());
    }

    void normalize_small (
            ProductValue::Observed & observed,
            float sparse_threshold = 0.1) const
    {
        switch (observed.sparsity()) {
            case ProductValue::Observed::ALL:
                break;

            case ProductValue::Observed::DENSE: {
                const size_t size = total_size();
                const size_t count = observed_count(observed);
                if (count == 0) {
                    observed.set_sparsity(ProductValue::Observed::NONE);
                    observed.clear_dense();
                } else if (count == size) {
                    observed.set_sparsity(ProductValue::Observed::ALL);
                    observed.clear_dense();
                } else if (count < sparse_threshold * size) {
                    observed.set_sparsity(ProductValue::Observed::SPARSE);
                    for (size_t i = 0; i < size; ++i) {
                        if (LOOM_UNLIKELY(observed.dense(i))) {
                            observed.add_sparse(i);
                        }
                    }
                    observed.clear_dense();
                }
            } break;

            case ProductValue::Observed::SPARSE: {
                const size_t size = total_size();
                const size_t count = observed.sparse_size();
                if (count == 0) {
                    observed.set_sparsity(ProductValue::Observed::NONE);
                } else if (count == size) {
                    observed.set_sparsity(ProductValue::Observed::ALL);
                    observed.clear_sparse();
                } else if (count >= sparse_threshold * size) {
                    observed.set_sparsity(ProductValue::Observed::DENSE);
                    for (size_t i = 0; i < size; ++i) {
                        observed.add_dense(false);
                    }
                    for (auto i : observed.sparse()) {
                        observed.set_dense(i, true);
                    }
                    observed.clear_sparse();
                }
            } break;

            case ProductValue::Observed::NONE:
                break;
        }

        if (LOOM_DEBUG_LEVEL >= 2) {
            validate(observed);
        }
    }

    void normalize_small (ProductValue & value) const
    {
        normalize_small(* value.mutable_observed());
    }

    void normalize_small (ProductValue::Diff & diff) const
    {
        normalize_small(* diff.mutable_pos());
        normalize_small(* diff.mutable_neg());
    }

    void normalize_dense (ProductValue::Observed & observed) const
    {
        auto & dense = * observed.mutable_dense();
        const size_t size = total_size();
        dense.Reserve(size);
        switch (observed.sparsity()) {
            case ProductValue::Observed::ALL: {
                observed.set_sparsity(ProductValue::Observed::DENSE);
                for (size_t i = 0; i < size; ++i) {
                    dense.AddAlreadyReserved(true);
                }
            } break;

            case ProductValue::Observed::DENSE:
                break;

            case ProductValue::Observed::SPARSE: {
                observed.set_sparsity(ProductValue::Observed::DENSE);
                for (size_t i = 0; i < size; ++i) {
                    dense.AddAlreadyReserved(false);
                }
                for (auto i : observed.sparse()) {
                    observed.set_dense(i, true);
                }
                observed.clear_sparse();
            } break;

            case ProductValue::Observed::NONE: {
                observed.set_sparsity(ProductValue::Observed::DENSE);
                for (size_t i = 0; i < size; ++i) {
                    dense.AddAlreadyReserved(false);
                }
            } break;
        }

        if (LOOM_DEBUG_LEVEL >= 2) {
            validate(observed);
        }
    }

    template<class Fun>
    void for_each_datatype (Fun & fun) const
    {
        fun(static_cast<bool *>(nullptr), booleans_size);
        fun(static_cast<uint32_t *>(nullptr), counts_size);
        fun(static_cast<float *>(nullptr), reals_size);
    }

    bool operator== (const ValueSchema & other) const
    {
        return booleans_size == other.booleans_size
            and counts_size == other.counts_size
            and reals_size == other.reals_size;
    }

    friend std::ostream & operator<< (
        std::ostream & os,
        const ValueSchema & schema)
    {
        return os << "{" <<
            schema.booleans_size << ", " <<
            schema.counts_size << ", " <<
            schema.reals_size << "}";
    }
};

//----------------------------------------------------------------------------
// Read

template<class Feature, class Fun>
inline void read_value_all (
        Fun & fun,
        const ForEachFeatureType<Feature> & model_schema,
        const ProductValue & value)
{
    if (value.booleans_size()) {
        auto packed = value.booleans().begin();
        for (size_t i = 0, size = model_schema.bb.size(); i < size; ++i) {
            fun(BB::null(), i, *packed++);
        }
        LOOM_ASSERT2(packed == value.booleans().end(), "programmer error");
    }

    if (value.counts_size()) {
        auto packed = value.counts().begin();
        for (size_t i = 0, size = model_schema.dd16.size(); i < size; ++i) {
            fun(DD16::null(), i, *packed++);
        }
        for (size_t i = 0, size = model_schema.dd256.size(); i < size; ++i) {
            fun(DD256::null(), i, *packed++);
        }
        for (size_t i = 0, size = model_schema.dpd.size(); i < size; ++i) {
            fun(DPD::null(), i, *packed++);
        }
        for (size_t i = 0, size = model_schema.gp.size(); i < size; ++i) {
            fun(GP::null(), i, *packed++);
        }
        LOOM_ASSERT2(packed == value.counts().end(), "programmer error");
    }

    if (value.reals_size()) {
        auto packed = value.reals().begin();
        for (size_t i = 0, size = model_schema.nich.size(); i < size; ++i) {
            fun(NICH::null(), i, *packed++);
        }
        LOOM_ASSERT2(packed == value.reals().end(), "programmer error");
    }
}

template<class Feature, class Fun>
inline void read_value_dense (
        Fun & fun,
        const ForEachFeatureType<Feature> & model_schema,
        const ProductValue & value)
{
    auto observed = value.observed().dense().begin();
    const auto end = value.observed().dense().end();

    if (value.booleans_size()) {
        auto packed = value.booleans().begin();
        for (size_t i = 0, size = model_schema.bb.size(); i < size; ++i) {
            if (*observed++) {
                fun(BB::null(), i, *packed++);
            }
        }
        LOOM_ASSERT2(packed == value.booleans().end(), "programmer error");
    } else {
        observed += model_schema.bb.size();
    }

    if (value.counts_size()) {
        auto packed = value.counts().begin();
        for (size_t i = 0, size = model_schema.dd16.size(); i < size; ++i) {
            if (*observed++) {
                fun(DD16::null(), i, *packed++);
            }
        }
        for (size_t i = 0, size = model_schema.dd256.size(); i < size; ++i) {
            if (*observed++) {
                fun(DD256::null(), i, *packed++);
            }
        }
        for (size_t i = 0, size = model_schema.dpd.size(); i < size; ++i) {
            if (*observed++) {
                fun(DPD::null(), i, *packed++);
            }
        }
        for (size_t i = 0, size = model_schema.gp.size(); i < size; ++i) {
            if (*observed++) {
                fun(GP::null(), i, *packed++);
            }
        }
        LOOM_ASSERT2(packed == value.counts().end(), "programmer error");
    } else {
        observed +=
            model_schema.dd16.size() +
            model_schema.dd256.size() +
            model_schema.dpd.size() +
            model_schema.gp.size();
    }

    if (value.reals_size()) {
        auto packed = value.reals().begin();
        for (size_t i = 0, size = model_schema.nich.size(); i < size; ++i) {
            if (*observed++) {
                fun(NICH::null(), i, *packed++);
            }
        }
        LOOM_ASSERT2(packed == value.reals().end(), "programmer error");
    } else {
        observed += model_schema.nich.size();
    }

    LOOM_ASSERT2(observed == end, "programmer error");
}

template<class Feature, class Fun>
inline void read_value_sparse (
        Fun & fun,
        const ForEachFeatureType<Feature> & model_schema,
        const ProductValue & value)
{
    auto i = value.observed().sparse().begin();
    const auto end = value.observed().sparse().end();
    BlockIterator block;

    {
        auto packed = value.booleans().begin();
        for (block(model_schema.bb.size()); i != end and block.ok(*i); ++i) {
            fun(BB::null(), block.get(*i), *packed++);
        }
        LOOM_ASSERT2(packed == value.booleans().end(), "programmer error");
    }
    {
        auto packed = value.counts().begin();
        for (block(model_schema.dd16.size()); i != end and block.ok(*i); ++i) {
            fun(DD16::null(), block.get(*i), *packed++);
        }
        for (block(model_schema.dd256.size()); i != end and block.ok(*i); ++i) {
            fun(DD256::null(), block.get(*i), *packed++);
        }
        for (block(model_schema.dpd.size()); i != end and block.ok(*i); ++i) {
            fun(DPD::null(), block.get(*i), *packed++);
        }
        for (block(model_schema.gp.size()); i != end and block.ok(*i); ++i) {
            fun(GP::null(), block.get(*i), *packed++);
        }
        LOOM_ASSERT2(packed == value.counts().end(), "programmer error");
    }
    {
        auto packed = value.reals().begin();
        for (block(model_schema.nich.size()); i != end and block.ok(*i); ++i) {
            fun(NICH::null(), block.get(*i), *packed++);
        }
        LOOM_ASSERT2(packed == value.reals().end(), "programmer error");
    }
}

template<class Feature, class Fun>
inline void read_value (
        Fun & fun,
        const ValueSchema & value_schema,
        const ForEachFeatureType<Feature> & model_schema,
        const ProductValue & value)
{
    try {
        if (LOOM_DEBUG_LEVEL >= 2) {
            value_schema.validate(model_schema);
            value_schema.validate(value);
        }

        switch (value.observed().sparsity()) {
            case ProductValue::Observed::ALL:
                read_value_all(fun, model_schema, value);
                break;

            case ProductValue::Observed::DENSE:
                read_value_dense(fun, model_schema, value);
                break;

            case ProductValue::Observed::SPARSE:
                read_value_sparse(fun, model_schema, value);
                break;

            case ProductValue::Observed::NONE:
                break;
        }
    } catch (google::protobuf::FatalException e) {
        LOOM_ERROR(e.what());
    }
}

//----------------------------------------------------------------------------
// Write

template<class Feature, class Fun>
inline void write_value_all (
        Fun & fun,
        const ForEachFeatureType<Feature> & model_schema,
        ProductValue & value)
{
    value.clear_booleans();
    for (size_t i = 0, size = model_schema.bb.size(); i < size; ++i) {
        value.add_booleans(fun(BB::null(), i));
    }

    value.clear_counts();
    for (size_t i = 0, size = model_schema.dd16.size(); i < size; ++i) {
        value.add_counts(fun(DD16::null(), i));
    }
    for (size_t i = 0, size = model_schema.dd256.size(); i < size; ++i) {
        value.add_counts(fun(DD256::null(), i));
    }
    for (size_t i = 0, size = model_schema.dpd.size(); i < size; ++i) {
        value.add_counts(fun(DPD::null(), i));
    }
    for (size_t i = 0, size = model_schema.gp.size(); i < size; ++i) {
        value.add_counts(fun(GP::null(), i));
    }

    value.clear_reals();
    for (size_t i = 0, size = model_schema.nich.size(); i < size; ++i) {
        value.add_reals(fun(NICH::null(), i));
    }
}

template<class Feature, class Fun>
inline void write_value_dense (
        Fun & fun,
        const ForEachFeatureType<Feature> & model_schema,
        ProductValue & value)
{
    auto observed = value.observed().dense().begin();

    value.clear_booleans();
    for (size_t i = 0, size = model_schema.bb.size(); i < size; ++i) {
        if (*observed++) {
            value.add_booleans(fun(BB::null(), i));
        }
    }

    value.clear_counts();
    for (size_t i = 0, size = model_schema.dd16.size(); i < size; ++i) {
        if (*observed++) {
            value.add_counts(fun(DD16::null(), i));
        }
    }
    for (size_t i = 0, size = model_schema.dd256.size(); i < size; ++i) {
        if (*observed++) {
            value.add_counts(fun(DD256::null(), i));
        }
    }
    for (size_t i = 0, size = model_schema.dpd.size(); i < size; ++i) {
        if (*observed++) {
            value.add_counts(fun(DPD::null(), i));
        }
    }
    for (size_t i = 0, size = model_schema.gp.size(); i < size; ++i) {
        if (*observed++) {
            value.add_counts(fun(GP::null(), i));
        }
    }

    value.clear_reals();
    for (size_t i = 0, size = model_schema.nich.size(); i < size; ++i) {
        if (*observed++) {
            value.add_reals(fun(NICH::null(), i));
        }
    }

    LOOM_ASSERT2(
        observed == value.observed().dense().end(),
        "programmer error");
}

template<class Feature, class Fun>
inline void write_value_sparse (
        Fun & fun,
        const ForEachFeatureType<Feature> & model_schema,
        ProductValue & value)
{
    auto i = value.observed().sparse().begin();
    const auto end = value.observed().sparse().end();
    BlockIterator block;

    value.clear_booleans();
    for (block(model_schema.bb.size()); i != end and block.ok(*i); ++i) {
       value.add_booleans(fun(BB::null(), block.get(*i)));
    }

    value.clear_counts();
    for (block(model_schema.dd16.size()); i != end and block.ok(*i); ++i) {
       value.add_counts(fun(DD16::null(), block.get(*i)));
    }
    for (block(model_schema.dd256.size()); i != end and block.ok(*i); ++i) {
       value.add_counts(fun(DD256::null(), block.get(*i)));
    }
    for (block(model_schema.dpd.size()); i != end and block.ok(*i); ++i) {
       value.add_counts(fun(DPD::null(), block.get(*i)));
    }
    for (block(model_schema.gp.size()); i != end and block.ok(*i); ++i) {
       value.add_counts(fun(GP::null(), block.get(*i)));
    }

    value.clear_reals();
    for (block(model_schema.nich.size()); i != end and block.ok(*i); ++i) {
        value.add_reals(fun(NICH::null(), block.get(*i)));
    }
}

inline void write_value_none (ProductValue & value)
{
    value.clear_booleans();
    value.clear_counts();
    value.clear_reals();
}

template<class Feature, class Fun>
inline void write_value (
        Fun & fun,
        const ValueSchema & value_schema,
        const ForEachFeatureType<Feature> & model_schema,
        ProductValue & value)
{
    try {
        if (LOOM_DEBUG_LEVEL >= 2) {
            value_schema.validate(model_schema);
        }

        switch (value.observed().sparsity()) {
            case ProductValue::Observed::ALL:
                write_value_all(fun, model_schema, value);
                break;

            case ProductValue::Observed::DENSE:
                write_value_dense(fun, model_schema, value);
                break;

            case ProductValue::Observed::SPARSE:
                write_value_sparse(fun, model_schema, value);
                break;

            case ProductValue::Observed::NONE:
                write_value_none(value);
                break;
        }

        if (LOOM_DEBUG_LEVEL >= 2) {
            value_schema.validate(value);
        }
    } catch (google::protobuf::FatalException e) {
        LOOM_ERROR(e.what());
    }
}

//----------------------------------------------------------------------------
// ValueSpliter

struct ValueSplitter : noncopyable
{

    void init (
            const ValueSchema & schema,
            const std::vector<uint32_t> & full_to_partid,
            size_t part_count);

    void validate (
            const ValueSchema & schema,
            const std::vector<uint32_t> & full_to_partid,
            size_t part_count) const;

    void split (
            const ProductValue & full_value,
            std::vector<ProductValue> & partial_values) const;

    template<class Getter>
    void split (
            const ProductValue & full_value,
            std::vector<ProductValue> & temp_values,
            const Getter & get) const
    {
        split(full_value, temp_values);
        const size_t part_count = part_schemas_.size();
        for (size_t i = 0; i < part_count; ++i) {
            temp_values[i].Swap(get(i));
        }
    }

    void join (
            ProductValue & full_value,
            const std::vector<ProductValue> & partial_values) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        unsafe_join(full_value, partial_values);
    }

    template<class Getter>
    void join (
            ProductValue & full_value,
            const Getter & get) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t part_count = part_schemas_.size();
        temp_values_.resize(part_count);
        for (size_t i = 0; i < part_count; ++i) {
            temp_values_[i] = * get(i);
        }
        unsafe_join(full_value, temp_values_);
    }

private:

    struct Map
    {
        template<class T>
        struct Container
        {
            typedef std::vector<std::pair<uint32_t, T>> t;
        };
    };
    typedef ForEachDataType<Map> Maps;

    ValueSchema schema_;
    std::vector<ValueSchema> part_schemas_;
    std::vector<uint32_t> full_to_partid_;
    std::vector<uint32_t> full_to_part_;
    std::vector<std::vector<uint32_t>> part_to_full_;
    mutable std::mutex mutex_;
    mutable std::vector<size_t> absolute_pos_list_;
    mutable std::vector<size_t> packed_pos_list_;
    mutable std::vector<ProductValue> temp_values_;
    mutable Maps temp_maps_;

    void unsafe_join (
            ProductValue & full_value,
            const std::vector<ProductValue> & partial_values) const;

    void validate (const ProductValue & full_value) const;
    void validate (const std::vector<ProductValue> & partial_values) const;

    struct split_value_all_fun;
    struct split_value_dense_fun;
    struct split_value_sparse_fun;
    struct join_value_all_fun;
    struct join_value_dense_fun;
    struct join_value_sparse_fun;
};

inline void ValueSplitter::validate (
        const ValueSchema & schema,
        const std::vector<uint32_t> & full_to_partid,
        size_t part_count) const
{
    LOOM_ASSERT_EQ(schema_, schema);
    LOOM_ASSERT_EQ(full_to_partid_, full_to_partid);
    LOOM_ASSERT_EQ(part_schemas_.size(), part_count);
}

inline void ValueSplitter::validate (const ProductValue & full_value) const
{
    if (LOOM_DEBUG_LEVEL >= 2) {
        schema_.validate(full_value);
    }
}

inline void ValueSplitter::validate (
        const std::vector<ProductValue> & partial_values) const
{
    if (LOOM_DEBUG_LEVEL >= 2) {
        const size_t part_count = part_schemas_.size();
        LOOM_ASSERT_EQ(partial_values.size(), part_count);
        const auto sparsity0 = partial_values[0].observed().sparsity();
        for (size_t i = 0; i < part_count; ++i) {
            const auto sparsity = partial_values[i].observed().sparsity();
            LOOM_ASSERT_EQ(sparsity, sparsity0);
            part_schemas_[i].validate(partial_values[i]);
        }
    }
}

} // namespace loom
