#pragma once

#include <loom/protobuf.hpp>
#include <loom/protobuf_stream.hpp>
#include <loom/product_value.hpp>

namespace loom
{

class Sparsifier
{
public:

    Sparsifier (
            const protobuf::Config::Sparsify & config,
            const ValueSchema & schema) :
        config_(config),
        schema_(schema),
        unobserved_(get_unobserved(schema)),
        row_count_(0),
        booleans_(schema.booleans_size),
        counts_(schema.counts_size),
        tare_()
    {
        LOOM_ASSERT(config.run(), "sparsify is not configured to run");
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
            for (size_t i = 0; i < schema_.booleans_size; ++i) {
                if (*observed++) {
                    booleans_[i].add(value.booleans(i));
                }
            }
            for (size_t i = 0; i < schema_.counts_size; ++i) {
                if (*observed++) {
                    counts_[i].add(value.counts(i));
                }
            }
            ++row_count_;
        }
    }

    const ProductValue & tare ()
    {
        tare_.Clear();
        tare_.mutable_observed()->set_sparsity(ProductValue::Observed::DENSE);

        tare_type(booleans_, * tare_.mutable_booleans());
        tare_type(counts_, * tare_.mutable_counts());

        return tare_;
    }

    void sparsify_rows (
        const char * rows_in,
        const char * diffs_out) const
    {
        LOOM_ASSERT(
            std::string(rows_in) != std::string(diffs_out),
            "in-place sparsify is not supported");
        protobuf::InFile rows(rows_in);
        protobuf::OutFile diffs(diffs_out);
        protobuf::Row row;
        protobuf::Row diff;
        const float sparse_threshold = config_.sparse_threshold();
        while (rows.try_read_stream(row)) {
            diff.Clear();
            diff.set_id(row.id());
            * diff.mutable_data()->mutable_observed() = unobserved_;
            * diff.mutable_diff()->mutable_observed() = unobserved_;

            TODO("compute diff");

            schema_.normalize_small(* diff.mutable_data(), sparse_threshold);
            schema_.normalize_small(* diff.mutable_diff(), sparse_threshold);
            diffs.write_stream(diff);
        }
    }

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
            const ValueSchema & schema)
    {
        protobuf::ProductValue::Observed unobserved;
        unobserved.set_sparsity(ProductValue::Observed::DENSE);
        for (size_t i = 0; i < schema.total_size(); ++i) {
            unobserved.add_dense(false);
        }
        return unobserved;
    }

    template<class Summaries, class Values>
    void tare_type (const Summaries & summaries, Values & values)
    {
        const float count_threshold = config_.tare_threshold() * row_count_;
        for (const auto & summary : summaries) {
            const auto mode = summary.get_mode();
            bool is_dense = (summary.get_count(mode) > count_threshold);
            tare_.mutable_observed()->add_dense(is_dense);
            if (is_dense) {
                values.Add(mode);
            }
        }
    }

    const protobuf::Config::Sparsify & config_;
    const ValueSchema & schema_;
    const protobuf::ProductValue::Observed unobserved_;
    size_t row_count_;
    std::vector<BooleanSummary> booleans_;
    std::vector<CountSummary> counts_;
    protobuf::ProductValue tare_;
};

} // namespace loom
