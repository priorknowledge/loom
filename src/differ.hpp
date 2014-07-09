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
};

inline void Differ::compress (protobuf::Row & row) const
{
    if (small_tare_.observed().sparsity() == ProductValue::Observed::NONE) {
        schema_.normalize_small(* row.mutable_data()->mutable_observed());
    } else {
        _compress(row);
    }
}

inline void Differ::fill_in (protobuf::Row & row) const
{
    if (not row.has_data()) {
        LOOM_ASSERT(row.has_diff(), "row has nether data nor diff");
        _fill_in(row);
    }
}

} // namespace loom
