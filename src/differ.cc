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

#include <loom/differ.hpp>

namespace loom
{

namespace
{
inline protobuf::ProductValue get_blank (const ValueSchema & schema)
{
    ProductValue value;
    auto & observed = * value.mutable_observed();
    observed.set_sparsity(ProductValue::Observed::DENSE);
    for (size_t i = 0; i < schema.total_size(); ++i) {
        observed.add_dense(false);
    }
    return value;
}
inline protobuf::ProductValue::Observed get_full (const ValueSchema & schema)
{
    ProductValue::Observed observed;
    observed.set_sparsity(ProductValue::Observed::DENSE);
    for (size_t i = 0; i < schema.total_size(); ++i) {
        observed.add_dense(true);
    }
    return observed;
}
} // anonymous namespace

Differ::Differ (const ValueSchema & schema) :
    schema_(schema),
    blank_(get_blank(schema)),
    full_(get_full(schema)),
    row_count_(0),
    booleans_(schema.booleans_size),
    counts_(schema.counts_size),
    small_tare_(),
    dense_tare_()
{
    set_tare(blank_);
}

Differ::Differ (
        const ValueSchema & schema,
        const ProductValue & tare) :
    schema_(schema),
    blank_(get_blank(schema)),
    full_(get_full(schema)),
    row_count_(0),
    booleans_(schema.booleans_size),
    counts_(schema.counts_size),
    small_tare_(),
    dense_tare_()
{
    set_tare(tare);
}

void Differ::set_tare (const ProductValue & tare)
{
    schema_.validate(tare);
    small_tare_ = tare;
    dense_tare_ = tare;
    schema_.normalize_small(* small_tare_.mutable_observed());
    schema_.normalize_dense(* dense_tare_.mutable_observed());
}

void Differ::add_rows (const char * rows_in)
{
    protobuf::InFile rows(rows_in);
    protobuf::Row row;
    while (rows.try_read_stream(row)) {
        LOOM_ASSERT(not row.diff().tares_size(), "row is already sparsified");
        const auto & value = row.diff().pos();
        LOOM_ASSERT_EQ(
            value.observed().sparsity(),
            ProductValue::Observed::DENSE);

        auto observed = value.observed().dense().begin();
        {
            auto fields = value.booleans().begin();
            for (auto & summary : booleans_) {
                if (*observed++) {
                    summary.add(*fields++);
                }
            }
        }
        {
            auto fields = value.counts().begin();
            for (auto & summary : counts_) {
                if (*observed++) {
                    summary.add(*fields++);
                }
            }
        }
        // do not sparsify reals
        ++row_count_;
    }

    _make_tare();
}

void Differ::_make_tare ()
{
    ProductValue tare;
    auto & observed = * tare.mutable_observed();
    observed.set_sparsity(ProductValue::Observed::DENSE);

    _make_tare_type(observed, booleans_, * tare.mutable_booleans());
    _make_tare_type(observed, counts_, * tare.mutable_counts());

    size_t ignored = schema_.reals_size;
    for (size_t i = 0; i < ignored; ++i) {
        observed.add_dense(false);
    }

    set_tare(tare);
}

inline void Differ::_compress (ProductValue & data) const
{
    schema_.normalize_small(* data.mutable_observed());
}

inline void Differ::_compress (ProductValue::Diff & diff) const
{
    _compress(* diff.mutable_pos());
    _compress(* diff.mutable_neg());
}

void Differ::compress_rows (
        const char * rows_in,
        const char * diffs_out) const
{
    protobuf::InFile rows(rows_in);
    if (rows.is_file()) {
        LOOM_ASSERT(
            std::string(rows_in) != std::string(diffs_out),
            "in-place sparsify is not supported");
    }
    protobuf::OutFile diffs(diffs_out);
    protobuf::Row abs;
    protobuf::Row rel;
    ProductValue actual;
    if (schema_.total_size(dense_tare_)) {
        while (rows.try_read_stream(abs)) {
            rel.set_id(abs.id());
            ProductValue & data = * abs.mutable_diff()->mutable_pos();
            ProductValue::Diff & diff = * rel.mutable_diff();
            _abs_to_rel(data, diff);
            _compress(* rel.mutable_diff());
            diffs.write_stream(rel);
            if (LOOM_DEBUG_LEVEL >= 3) {
                _rel_to_abs(actual, diff);
                LOOM_ASSERT_EQ(actual, data);
            }
        }
    } else {
        while (rows.try_read_stream(abs)) {
            _compress(* abs.mutable_diff());
            diffs.write_stream(abs);
        }
    }
}

template<class Summaries, class Values>
inline void Differ::_make_tare_type (
        ProductValue::Observed & observed,
        const Summaries & summaries,
        Values & values) const
{
    const float count_threshold = 0.5 * row_count_;
    for (const auto & summary : summaries) {
        const auto mode = summary.get_mode();
        bool is_dense = (summary.get_count(mode) > count_threshold);
        observed.add_dense(is_dense);
        if (is_dense) {
            values.Add(mode);
        }
    }
}

inline void Differ::_build_temporaries (ProductValue & value) const
{
    // Ensure observed.has_dense(), even if observed.sparsity() != DENSE.
    auto & observed = * value.mutable_observed();
    auto & dense = * observed.mutable_dense();
    switch (observed.sparsity()) {
        case ProductValue::Observed::ALL: {
            dense = full_.dense();
        } break;

        case ProductValue::Observed::DENSE:
            break;

        case ProductValue::Observed::SPARSE: {
            dense = blank_.observed().dense();
            for (auto i : observed.sparse()) {
                dense.Set(i, true);
            }
        } break;

        case ProductValue::Observed::NONE: {
            dense = blank_.observed().dense();
        } break;
    }
    if (LOOM_DEBUG_LEVEL >= 1) {
        LOOM_ASSERT_EQ(dense.size(), schema_.total_size());
    }
}

inline void Differ::_clean_temporaries (ProductValue & value) const
{
    auto & observed = * value.mutable_observed();
    if (LOOM_DEBUG_LEVEL >= 1) {
        LOOM_ASSERT_EQ(observed.dense().size(), schema_.total_size());
    }
    if (observed.sparsity() != ProductValue::Observed::DENSE) {
        observed.clear_dense();
    }
}

template<class T>
inline void Differ::_abs_to_rel_type (
        const ProductValue & data,
        ProductValue & pos,
        ProductValue & neg,
        const BlockIterator & block) const
{
    const size_t begin = block.begin();
    const size_t end = block.end();
    auto tare_observed = dense_tare_.observed().dense().begin() + begin;
    const auto tare_observed_end = dense_tare_.observed().dense().begin() + end;
    auto data_observed = data.observed().dense().begin() + begin;
    auto pos_observed =
        pos.mutable_observed()->mutable_dense()->begin() + begin;
    auto neg_observed =
        neg.mutable_observed()->mutable_dense()->begin() + begin;
    auto tare_value = protobuf::Fields<T>::get(dense_tare_).begin();
    auto data_value = protobuf::Fields<T>::get(data).begin();
    auto & pos_values = protobuf::Fields<T>::get(pos);
    auto & neg_values = protobuf::Fields<T>::get(neg);

    pos_values.Reserve(end - begin);
    neg_values.Reserve(end - begin);
    while (tare_observed != tare_observed_end) {
        if (*tare_observed) {
            if (LOOM_LIKELY(*data_observed)) {
                if (LOOM_UNLIKELY(*data_value != *tare_value)) {
                    *pos_observed = true;
                    *neg_observed = true;
                    pos_values.AddAlreadyReserved(*data_value);
                    neg_values.AddAlreadyReserved(*tare_value);
                }
                ++data_value;
            } else {
                *neg_observed = true;
                neg_values.AddAlreadyReserved(*tare_value);
            }
            ++tare_value;
        } else {
            if (*data_observed) {
                *pos_observed = true;
                pos_values.AddAlreadyReserved(*data_value);
                ++data_value;
            }
        }
        ++tare_observed;
        ++data_observed;
        ++pos_observed;
        ++neg_observed;
    }
}

template<class T>
inline void Differ::_rel_to_abs_type (
        ProductValue & data,
        const ProductValue & pos,
        const ProductValue & neg,
        const BlockIterator & block) const
{
    const size_t begin = block.begin();
    const size_t end = block.end();
    auto tare_observed = dense_tare_.observed().dense().begin() + begin;
    const auto tare_observed_end = dense_tare_.observed().dense().begin() + end;
    auto data_observed =
        data.mutable_observed()->mutable_dense()->begin() + begin;
    auto pos_observed = pos.observed().dense().begin() + begin;
    auto neg_observed = neg.observed().dense().begin() + begin;
    auto tare_value = protobuf::Fields<T>::get(dense_tare_).begin();
    auto & data_values = protobuf::Fields<T>::get(data);
    auto pos_value = protobuf::Fields<T>::get(pos).begin();

    data_values.Reserve(end - begin);
    while (tare_observed != tare_observed_end) {
        if (*pos_observed) {
            *data_observed = true;
            data_values.AddAlreadyReserved(*pos_value);
            ++pos_value;
        }
        if (*tare_observed) {
            if (LOOM_LIKELY(not *neg_observed and not *pos_observed)) {
                *data_observed = true;
                data_values.AddAlreadyReserved(*tare_value);
            }
            ++tare_value;
        }
        ++tare_observed;
        ++data_observed;
        ++pos_observed;
        ++neg_observed;
    }
}

inline void Differ::_validate_diff (
        const ProductValue & data,
        const ProductValue::Diff & diff) const
{
    if (LOOM_DEBUG_LEVEL >= 3) {
        const auto & tare_dense = dense_tare_.observed().dense();
        const auto & data_dense = data.observed().dense();
        const auto & pos_dense = diff.pos().observed().dense();
        const auto & neg_dense = diff.neg().observed().dense();
        for (size_t i = 0, size = schema_.total_size(); i < size; ++i) {
            int tare = tare_dense.Get(i);
            int data = data_dense.Get(i);
            int pos = pos_dense.Get(i);
            int neg = neg_dense.Get(i);
            LOOM_ASSERT(
                data == tare + pos - neg,
                data  << " != " << tare << " + " << pos << " - " << neg);
        }
    }
}

inline void Differ::_abs_to_rel (
        ProductValue & data,
        ProductValue::Diff & diff) const
{
    if (LOOM_DEBUG_LEVEL >= 2) {
        schema_.validate(data);
    }

    ProductValue & pos = * diff.mutable_pos();
    ProductValue & neg = * diff.mutable_neg();

    _build_temporaries(data);
    pos = blank_;
    neg = blank_;
    diff.clear_tares();
    diff.add_tares(0);

    {
        BlockIterator block;
        if (block(schema_.booleans_size)) {
            _abs_to_rel_type<bool>(data, pos, neg, block);
        }
        if (block(schema_.counts_size)) {
            _abs_to_rel_type<uint32_t>(data, pos, neg, block);
        }
        if (block(schema_.reals_size)) {
            _abs_to_rel_type<float>(data, pos, neg, block);
        }
    }

    _validate_diff(data, diff);
    _clean_temporaries(data);

    if (LOOM_DEBUG_LEVEL >= 2) {
        schema_.validate(data);
        schema_.validate(diff);
    }
}

inline void Differ::_rel_to_abs (
        ProductValue & data,
        ProductValue::Diff & diff) const
{
    if (LOOM_DEBUG_LEVEL >= 2) {
        schema_.validate(diff);
    }

    ProductValue & pos = * diff.mutable_pos();
    ProductValue & neg = * diff.mutable_neg();

    data = blank_;
    _build_temporaries(pos);
    _build_temporaries(neg);
    LOOM_ASSERT1(diff.tares_size(), "diff has no tares");

    {
        BlockIterator block;
        if (block(schema_.booleans_size)) {
            _rel_to_abs_type<bool>(data, pos, neg, block);
        }
        if (block(schema_.counts_size)) {
            _rel_to_abs_type<uint32_t>(data, pos, neg, block);
        }
        if (block(schema_.reals_size)) {
            _rel_to_abs_type<float>(data, pos, neg, block);
        }
    }

    _validate_diff(data, diff);
    _clean_temporaries(pos);
    _clean_temporaries(neg);

    if (LOOM_DEBUG_LEVEL >= 2) {
        schema_.validate(data);
        schema_.validate(diff);
    }
}

} // namespace loom
