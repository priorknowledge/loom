#include "loom.hpp"

namespace loom
{

using ::distributions::sample_from_scores_overwrite;

Loom::Loom (
        rng_t & rng,
        const char * model_in,
        const char * groups_in,
        const char * assign_in) :
    cross_cat_(model_in),
    kind_count_(cross_cat_.kinds.size()),
    assignments_(kind_count_),
    value_join_(cross_cat_),
    factors_(kind_count_),
    scores_()
{
    LOOM_ASSERT(kind_count_, "no kinds, loom is empty");

    if (groups_in) {
        cross_cat_.mixture_load(groups_in, rng);
    } else {
        cross_cat_.mixture_init_empty(rng);
    }

    if (assign_in) {
        assignments_.load(assign_in);
        for (const auto & kind : cross_cat_.kinds) {
            LOOM_ASSERT_LE(
                assignments_.size(),
                kind.mixture.clustering.sample_size);
        }
    }
}

//----------------------------------------------------------------------------
// High level operations

void Loom::dump (
        const char * groups_out,
        const char * assign_out)
{
    if (groups_out) {
        cross_cat_.mixture_dump(groups_out);
    }

    if (assign_out) {
        assignments_.dump(assign_out);
    }
}

void Loom::infer_single_pass (
        rng_t & rng,
        const char * rows_in,
        const char * assign_out)
{
    protobuf::InFile rows(rows_in);
    protobuf::SparseRow row;

    if (assign_out) {

        protobuf::OutFile assignments(assign_out);
        protobuf::Assignment assignment;

        while (rows.try_read_stream(row)) {
            add_row(rng, row, assignment);
            assignments.write_stream(assignment);
        }

    } else {

        while (rows.try_read_stream(row)) {
            add_row_noassign(rng, row);
        }
    }
}

void Loom::infer_multi_pass (
        rng_t & rng,
        const char * rows_in,
        double extra_passes)
{
    protobuf::InFile rows_to_add(rows_in);
    protobuf::InFile rows_to_remove(rows_in);
    protobuf::SparseRow row;

    // Set positions of both read heads,
    // assuming at least some rows are unassigned.
    if (assignments_.size()) {
        protobuf::SparseRow row_to_remove;
        protobuf::SparseRow row_to_add;

        // find any unassigned row
        do {
            rows_to_remove.cyclic_read_stream(row_to_remove);
            rows_to_add.cyclic_read_stream(row_to_add);
        } while (assignments_.try_find(row_to_remove.id()));

        // find the first assigned row
        do {
            rows_to_remove.cyclic_read_stream(row_to_remove);
            rows_to_add.cyclic_read_stream(row_to_add);
        } while (not assignments_.try_find(row_to_remove.id()));

        // find the first unassigned row
        do {
            rows_to_add.cyclic_read_stream(row_to_add);
        } while (not assignments_.try_find(row_to_remove.id()));

        // consume one row at each head
        bool added = try_add_row(rng, row_to_add);
        LOOM_ASSERT(added, "failed to add first row");
        remove_row(rng, row_to_remove);
    }

    AnnealingSchedule schedule(extra_passes);
    while (true) {
        if (schedule.next_action_is_add()) {

            rows_to_add.cyclic_read_stream(row);
            bool all_rows_assigned = not try_add_row(rng, row);
            if (LOOM_UNLIKELY(all_rows_assigned)) {
                break;
            }

        } else {

            rows_to_remove.cyclic_read_stream(row);
            remove_row(rng, row);
        }
    }
}

void Loom::predict (
        rng_t & rng,
        const char * queries_in,
        const char * results_out)
{
    protobuf::InFile query_stream(queries_in);
    protobuf::OutFile result_stream(results_out);
    protobuf::PreQL::Predict::Query query;
    protobuf::PreQL::Predict::Result result;

    while (query_stream.try_read_stream(query)) {
        predict_row(rng, query, result);
        result_stream.write_stream(result);
        result_stream.flush();
    }
}

//----------------------------------------------------------------------------
// Low level operations

inline void Loom::add_row_noassign (
        rng_t & rng,
        const protobuf::SparseRow & row)
{
    cross_cat_.value_split(row.data(), factors_);

    for (size_t i = 0; i < kind_count_; ++i) {
        const auto & value = factors_[i];
        auto & kind = cross_cat_.kinds[i];
        const ProductModel & model = kind.model;
        ProductModel::Mixture & mixture = kind.mixture;

        mixture.score(model, value, scores_, rng);
        size_t groupid = sample_from_scores_overwrite(rng, scores_);
        mixture.add_value(model, groupid, value, rng);
    }
}

inline void Loom::add_row (
        rng_t & rng,
        const protobuf::SparseRow & row,
        protobuf::Assignment & assignment)
{
    cross_cat_.value_split(row.data(), factors_);
    assignment.set_rowid(row.id());
    assignment.clear_groupids();

    for (size_t i = 0; i < kind_count_; ++i) {
        const auto & value = factors_[i];
        auto & kind = cross_cat_.kinds[i];
        const ProductModel & model = kind.model;
        ProductModel::Mixture & mixture = kind.mixture;

        mixture.score(model, value, scores_, rng);
        size_t groupid = sample_from_scores_overwrite(rng, scores_);
        mixture.add_value(model, groupid, value, rng);
        assignment.add_groupids(groupid);
    }
}

inline bool Loom::try_add_row (
        rng_t & rng,
        const protobuf::SparseRow & row)
{
    cross_cat_.value_split(row.data(), factors_);
    auto * global_groupids = assignments_.try_add(row.id());

    bool already_added = (global_groupids == nullptr);
    if (LOOM_UNLIKELY(already_added)) {
        return false;
    }

    for (size_t i = 0; i < kind_count_; ++i) {
        const auto & value = factors_[i];
        auto & kind = cross_cat_.kinds[i];
        const ProductModel & model = kind.model;
        ProductModel::Mixture & mixture = kind.mixture;

        mixture.score(model, value, scores_, rng);
        size_t groupid = sample_from_scores_overwrite(rng, scores_);
        mixture.add_value(model, groupid, value, rng);
        global_groupids[i] = mixture.id_tracker.packed_to_global(groupid);
    }

    return true;
}

inline void Loom::remove_row (
        rng_t & rng,
        const protobuf::SparseRow & row)
{
    cross_cat_.value_split(row.data(), factors_);
    auto self_destructing = assignments_.remove(row.id());
    const auto * global_groupids = self_destructing.value;

    for (size_t i = 0; i < kind_count_; ++i) {
        const auto & value = factors_[i];
        auto & kind = cross_cat_.kinds[i];
        const ProductModel & model = kind.model;
        ProductModel::Mixture & mixture = kind.mixture;

        auto groupid = mixture.id_tracker.global_to_packed(global_groupids[i]);
        mixture.remove_value(model, groupid, value, rng);
    }
}

inline void Loom::predict_row (
        rng_t & rng,
        const protobuf::PreQL::Predict::Query & query,
        protobuf::PreQL::Predict::Result & result)
{
    result.Clear();
    result.set_id(query.id());
    if (not cross_cat_.schema.is_valid(query.data())) {
        result.set_error("invalid query data");
        return;
    }
    if (query.data().observed_size() != query.to_predict_size()) {
        result.set_error("observed size != to_predict size");
        return;
    }
    const size_t sample_count = query.sample_count();
    if (sample_count == 0) {
        return;
    }

    cross_cat_.value_split(query.data(), factors_);
    std::vector<std::vector<ProductModel::Value>> result_factors(1);
    {
        ProductModel::Value sample;
        * sample.mutable_observed() = query.to_predict();
        cross_cat_.value_resize(sample);
        cross_cat_.value_split(sample, result_factors[0]);
        result_factors.resize(sample_count, result_factors[0]);
    }

    for (size_t i = 0; i < kind_count_; ++i) {
        if (protobuf::SparseValueSchema::total_size(result_factors[0][i])) {
            const auto & value = factors_[i];
            auto & kind = cross_cat_.kinds[i];
            const ProductModel & model = kind.model;
            ProductModel::Mixture & mixture = kind.mixture;

            mixture.score(model, value, scores_, rng);
            float total = distributions::scores_to_likelihoods(scores_);
            distributions::vector_scale(
                scores_.size(),
                scores_.data(),
                1.f / total);
            const VectorFloat & probs = scores_;

            for (auto & result_values : result_factors) {
                mixture.sample_value(model, probs, result_values[i], rng);
            }
        }
    }

    for (const auto & result_values : result_factors) {
        value_join_(* result.add_samples(), result_values);
    }
}

} // namespace loom
