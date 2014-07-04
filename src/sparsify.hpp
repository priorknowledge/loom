#pragma once

#include <loom/product_value.hpp>

namespace loom
{

struct Sparsifier
{
    enum { max_count = 16 };

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

    const ValueSchema schema;
    size_t row_count;
    std::vector<BooleanSummary> booleans;
    std::vector<CountSummary> counts;
    protobuf::ProductValue tare;

    Sparsifier (const ValueSchema & schema_in) :
        schema(schema_in),
        row_count(0),
        booleans(schema.booleans_size),
        counts(schema.counts_size)
    {
    }

    void add_rows (const char * rows_in)
    {
        protobuf::InFile rows(rows_in);
        protobuf::Row row;
        while (rows.try_read_stream(row)) {
            LOOM_ASSERT(not row.has_diff(), "row is already sparsified");
            const auto & value = row.data();
            LOOM_ASSERT_EQ(
                value.observed().sparsity(),
                ProductValue::Observed::DENSE);

            auto observed = value.observed().dense().begin();
            for (size_t i = 0; i < schema.booleans_size; ++i) {
                if (*observed++) {
                    booleans[i].add(value.booleans(i));
                }
            }
            for (size_t i = 0; i < schema.counts_size; ++i) {
                if (*observed++) {
                    counts[i].add(value.counts(i));
                }
            }
            ++row_count;
        }
    }

    void set_tare ()
    {
        tare.Clear();
        tare.mutable_observed()->set_sparsity(ProductValue::Observed::DENSE);

        TODO("set tare");
    }

    protobuf::ProductValue::Observed get_unobserved () const
    {
        protobuf::ProductValue::Observed unobserved;
        unobserved.set_sparsity(ProductValue::Observed::DENSE);
        for (size_t i = 0; i < schema.total_size(); ++i) {
            unobserved.add_dense(false);
        }
        return unobserved;
    }

    void sparsify_rows (
        const char * rows_in,
        const char * diffs_out)
    {
        LOOM_ASSERT(
            std::string(rows_in) != std::string(diffs_out),
            "in-place sparsify is not supported");
        protobuf::InFile rows(rows_in);
        protobuf::OutFile diffs(diffs_out);
        protobuf::Row row;
        protobuf::Row diff;
        const auto unobserved = get_unobserved();
        while (rows.try_read_stream(row)) {
            diff.Clear();
            diff.set_id(row.id());
            * diff.mutable_data()->mutable_observed() = unobserved;
            * diff.mutable_diff()->mutable_observed() = unobserved;

            TODO("compute diff");

            schema.normalize_small(* diff.mutable_data());
            schema.normalize_small(* diff.mutable_diff());
            diffs.write_stream(diff);
        }
    }
};

} // namespace loom
