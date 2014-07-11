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

#include <loom/protobuf.hpp>
#include <loom/protobuf_stream.hpp>
#include <loom/product_value.hpp>

namespace loom
{

class Differ
{
public:

    Differ (const ValueSchema & schema);
    Differ (const ValueSchema & schema, const ProductValue & tare);

    void add_rows (const char * rows_in);
    const ProductValue & get_tare () const { return small_tare_; }
    void set_tare (const ProductValue & tare);
    bool has_tare () const { return has_tare_; }

    void compress (protobuf::Row & row) const;
    void fill_in (protobuf::Row & row) const;
    void compress_rows (const char * rows_in, const char * diffs_out) const;

private:

    struct BooleanSummary
    {
        typedef bool Value;
        size_t counts[2];

        BooleanSummary () : counts({0, 0}) {}
        void add (Value value) { ++counts[value]; }
        Value get_mode () const { return counts[1] > counts[0]; }
        size_t get_count (Value value) const { return counts[value]; }
    };

    struct CountSummary
    {
        enum { max_count = 16 };  // assume mode lies in [0, max_count)

        typedef uint32_t Value;
        size_t counts[max_count];

        CountSummary () { std::fill(counts, counts + max_count, 0); }

        void add (Value value)
        {
            if (value < max_count) {
                ++counts[value];
            }
        }

        Value get_mode () const
        {
            Value value = 0;
            for (size_t i = 0; i < max_count; ++i) {
                if (counts[i] > counts[value]) {
                    value = i;
                }
            }
            return value;
        }

        size_t get_count (Value value) const
        {
            LOOM_ASSERT_LT(value, max_count);
            return counts[value];
        }
    };

    void _make_tare ();

    template<class Summaries, class Values>
    void _make_tare_type (
            ProductValue::Observed & observed,
            const Summaries & summaries,
            Values & values) const;

    void _compress (protobuf::Row & row) const;
    void _fill_in (protobuf::Row & row) const;
    void _validate_diff (const protobuf::Row & row) const;
    void _validate_compressed (const protobuf::Row & row) const;
    void _validate_filled_in (const protobuf::Row & row) const;
    void _build_temporaries (ProductValue & value) const;
    void _clean_temporaries (ProductValue & value) const;

    template<class T>
    void _compress_type (
            const ProductValue & abs,
            ProductValue & pos,
            ProductValue & neg,
            const BlockIterator & block) const;

    template<class T>
    void _fill_in_type (
            ProductValue & abs,
            const ProductValue & pos,
            ProductValue & neg,
            const BlockIterator & block) const;

    const ValueSchema & schema_;
    const protobuf::ProductValue blank_;
    size_t row_count_;
    std::vector<BooleanSummary> booleans_;
    std::vector<CountSummary> counts_;
    protobuf::ProductValue small_tare_;
    protobuf::ProductValue dense_tare_;
    bool has_tare_;
};

inline void Differ::compress (protobuf::Row & row) const
{
    LOOM_ASSERT(row.has_data(), "row has no data");
    if (has_tare()) {
        _compress(row);
    } else {
        schema_.normalize_small(* row.mutable_data()->mutable_observed());
        row.clear_diff();
    }
}

inline void Differ::fill_in (protobuf::Row & row) const
{
    if (has_tare()) {
        LOOM_ASSERT(not row.has_data(), "row is already filled in");
        LOOM_ASSERT(row.has_diff(), "row has nether data nor diff");
        _fill_in(row);
    } else {
        LOOM_ASSERT(row.has_data(), "tare is empty, but row has no data");
        LOOM_ASSERT(not row.has_diff(), "tare is empty, but row has diff");
    }
}

} // namespace loom
