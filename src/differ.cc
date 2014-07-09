#include <loom/differ.hpp>

namespace loom
{

namespace
{
inline protobuf::ProductValue get_blank (const ValueSchema & schema)
{
    ProductValue blank;
    auto & observed = * blank.mutable_observed();
    observed.set_sparsity(ProductValue::Observed::DENSE);
    for (size_t i = 0; i < schema.total_size(); ++i) {
        observed.add_dense(false);
    }
    return blank;
}
} // anonymous namespace

Differ::Differ (const ValueSchema & schema) :
    schema_(schema),
    blank_(get_blank(schema)),
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
    protobuf::Row row;
    if (LOOM_DEBUG_LEVEL >= 3) {
        while (rows.try_read_stream(row)) {
            ProductValue expected = row.data();
            _compress(row);
            diffs.write_stream(row);
            _fill_in(row);
            ProductValue actual = row.data();
            schema_.normalize_dense(* expected.mutable_observed());
            schema_.normalize_dense(* actual.mutable_observed());
            LOOM_ASSERT_EQ(actual, expected);
        }
    } else {
        while (rows.try_read_stream(row)) {
            _compress(row);
            diffs.write_stream(row);
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
    const size_t size = schema_.total_size();
    dense.Reserve(size);
    switch (observed.sparsity()) {
        case ProductValue::Observed::ALL: {
            for (size_t i = 0; i < size; ++i) {
                dense.Add(true);
            }
        } break;

        case ProductValue::Observed::DENSE:
            break;

        case ProductValue::Observed::SPARSE: {
            for (size_t i = 0; i < size; ++i) {
                dense.Add(false);
            }
            for (auto i : observed.sparse()) {
                dense.Set(i, true);
            }
        } break;

        case ProductValue::Observed::NONE: {
            for (size_t i = 0; i < size; ++i) {
                dense.Add(false);
            }
        } break;
    }

    LOOM_ASSERT1(dense.size() == size, "programmer error");
}

inline void Differ::_clean_temporaries (ProductValue & value) const
{
    auto & observed = * value.mutable_observed();
    if (observed.sparsity() != ProductValue::Observed::DENSE) {
        observed.clear_dense();
    }
}

template<class T>
inline void Differ::_compress_type (
        const ProductValue & data,
        ProductValue & pos,
        ProductValue & neg,
        const BlockIterator & block) const
{
    const auto & tare_dense = dense_tare_.observed().dense();
    const auto & data_dense = data.observed().dense();
    auto & pos_dense = * pos.mutable_observed()->mutable_dense();
    auto & neg_dense = * neg.mutable_observed()->mutable_dense();

    const auto & tare_values = protobuf::Fields<T>::get(dense_tare_);
    const auto & data_values = protobuf::Fields<T>::get(data);
    auto & pos_values = protobuf::Fields<T>::get(pos);

    size_t tare_pos = 0;
    size_t data_pos = 0;
    for (size_t i = block.begin(); i < block.end(); ++i) {
        const bool tare_observed = tare_dense.Get(i);
        const bool data_observed = data_dense.Get(i);
        if (tare_observed) {
            const auto tare_value = tare_values.Get(tare_pos++);
            if (LOOM_LIKELY(data_observed)) {
                const auto data_value = data_values.Get(data_pos++);
                if (LOOM_UNLIKELY(data_value != tare_value)) {
                    pos_dense.Set(i, true);
                    pos_values.Add(data_value);
                    neg_dense.Set(i, true);
                }
            } else {
                neg_dense.Set(i, true);
            }
        } else {
            if (data_observed) {
                const auto data_value = data_values.Get(data_pos++);
                pos_dense.Set(i, true);
                pos_values.Add(data_value);
            }
        }
    }
}

template<class T>
inline void Differ::_fill_in_type (
        ProductValue & data,
        const ProductValue & pos,
        ProductValue & neg,
        const BlockIterator & block) const
{
    const auto & tare_dense = dense_tare_.observed().dense();
    auto & data_dense = * data.mutable_observed()->mutable_dense();
    const auto & pos_dense = pos.observed().dense();
    const auto & neg_dense = neg.observed().dense();

    const auto & tare_values = protobuf::Fields<T>::get(dense_tare_);
    auto & data_values = protobuf::Fields<T>::get(data);
    const auto & pos_values = protobuf::Fields<T>::get(pos);
    auto & neg_values = protobuf::Fields<T>::get(neg);

    size_t tare_pos = 0;
    size_t pos_pos = 0;
    for (size_t i = block.begin(); i < block.end(); ++i) {
        const bool tare_observed = tare_dense.Get(i);
        const bool pos_observed = pos_dense.Get(i);
        if (pos_observed) {
            const auto pos_value = pos_values.Get(pos_pos++);
            data_dense.Set(i, true);
            data_values.Add(pos_value);
        }
        if (tare_observed) {
            const auto tare_value = tare_values.Get(tare_pos++);
            if (LOOM_UNLIKELY(neg_dense.Get(i))) {
                neg_values.Add(tare_value);
            } else {
                LOOM_ASSERT1(not pos_observed, "tare, pos, and neg disagree");
                data_dense.Set(i, true);
                data_values.Add(tare_value);
            }
        }
    }
}

inline void Differ::_validate_compressed (const protobuf::Row & row) const
{
    if (LOOM_DEBUG_LEVEL >= 2) {
        LOOM_ASSERT(not row.has_data(), "compressed row has data");
        LOOM_ASSERT(row.has_diff(), "compressed row has no diff");
        schema_.validate(row.diff().pos());
        schema_.validate(row.diff().neg().observed());
        LOOM_ASSERT_EQ(schema_.total_size(row.diff().neg()), 0);
    }
}

inline void Differ::_validate_filled_in (const protobuf::Row & row) const
{
    if (LOOM_DEBUG_LEVEL >= 2) {
        LOOM_ASSERT(row.has_data(), "filled-in row has no data");
        LOOM_ASSERT(row.has_diff(), "filled-in row has no diff");
        schema_.validate(row.data());
        schema_.validate(row.diff().pos());
        schema_.validate(row.diff().neg());  // FIXME this fails in tests
    }
}

inline void Differ::_compress (protobuf::Row & row) const
{
    LOOM_ASSERT(row.has_data(), "row has no data");

    ProductValue & data = * row.mutable_data();
    ProductValue & pos = * row.mutable_diff()->mutable_pos();
    ProductValue & neg = * row.mutable_diff()->mutable_neg();

    _build_temporaries(data);
    pos = blank_;
    neg = blank_;

    {
        BlockIterator block;

        block(schema_.booleans_size);
        _compress_type<bool>(data, pos, neg, block);

        block(schema_.counts_size);
        _compress_type<uint32_t>(data, pos, neg, block);

        block(schema_.reals_size);
        _compress_type<float>(data, pos, neg, block);
    }

    row.clear_data();
    schema_.normalize_small(* pos.mutable_observed());
    schema_.normalize_small(* neg.mutable_observed());

    _validate_compressed(row);
}

void Differ::_fill_in (protobuf::Row & row) const
{
    _validate_compressed(row);

    ProductValue & data = * row.mutable_data();
    ProductValue & pos = * row.mutable_diff()->mutable_pos();
    ProductValue & neg = * row.mutable_diff()->mutable_neg();

    data = blank_;
    _build_temporaries(pos);
    _build_temporaries(neg);

    {
        BlockIterator block;

        block(schema_.booleans_size);
        _fill_in_type<bool>(data, pos, neg, block);

        block(schema_.counts_size);
        _fill_in_type<uint32_t>(data, pos, neg, block);

        block(schema_.reals_size);
        _fill_in_type<float>(data, pos, neg, block);
    }

    schema_.normalize_small(* data.mutable_observed());
    _clean_temporaries(pos);
    _clean_temporaries(neg);

    _validate_filled_in(row);
}

} // namespace loom
