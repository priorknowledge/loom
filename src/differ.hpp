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

    void add_rows (const char * rows_in);

    const ProductValue & get_tare () const { return tare_; }
    void set_tare (const ProductValue & tare);

    void absolute_to_relative (
            const ProductValue & abs,
            ProductValue & pos,
            ProductValue & neg) const;

    void relative_to_absolute (
            ProductValue & abs,
            const ProductValue & pos,
            const ProductValue & neg) const;

    void sparsify_rows (
            const protobuf::Config::Sparsify & config,
            const char * absolute_rows_in,
            const char * relative_rows_out) const;

private:

    struct BooleanSummary
    {
        typedef uint32_t Value;
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

        void add (uint32_t value)
        {
            if (value < max_count) {
                ++counts[value];
            }
        }

        uint32_t get_mode () const
        {
            uint32_t value = 0;
            for (size_t i = 0; i < max_count; ++i) {
                if (counts[i] > counts[value]) {
                    value = i;
                }
            }
            return value;
        }

        uint32_t get_count (Value value) const
        {
            LOOM_ASSERT_LT(value, max_count);
            return counts[value];
        }
    };

    static protobuf::ProductValue::Observed get_unobserved (
            const ValueSchema & schema);

    void make_tare ();

    template<class Summaries, class Values>
    void make_tare_type (const Summaries & summaries, Values & values);

    template<class T>
    void _copy (
            const ProductValue & source,
            ProductValue & destin,
            const BlockIterator & block) const;

    template<class T>
    void _abs_to_rel (
            const ProductValue & row,
            ProductValue & pos,
            ProductValue & neg,
            const BlockIterator & block) const;

    template<class T>
    void _rel_to_abs (
            ProductValue & row,
            const ProductValue & pos,
            const ProductValue & neg,
            const BlockIterator & block) const;

    const ValueSchema & schema_;
    const protobuf::ProductValue::Observed unobserved_;
    size_t row_count_;
    std::vector<BooleanSummary> booleans_;
    std::vector<CountSummary> counts_;
    protobuf::ProductValue tare_;
};

} // namespace loom
