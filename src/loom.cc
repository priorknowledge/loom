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
    factors_(kind_count_),
    scores_()
{
    LOOM_ASSERT(kind_count_, "no kinds, loom is empty");

    if (groups_in) {
        cross_cat_.mixture_init_empty(rng);
    } else {
        cross_cat_.mixture_load(groups_in, rng);
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
            add_row(rng, row);
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

    if (assignments_.size()) {
        LOOM_ERROR("TODO advance rows files to correct positions");
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

inline void Loom::add_row (
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

} // namespace loom
