#include "common.hpp"
#include "cross_cat.hpp"
#include "assignments.hpp"
#include "annealing_schedule.hpp"

namespace loom
{

void infer_single_pass (
        CrossCat & cross_cat,
        const char * rows_in,
        rng_t & rng)
{
    const size_t kind_count = cross_cat.kinds.size();
    protobuf::SparseRow row;
    std::vector<ProductModel::Value> factors(kind_count);
    VectorFloat scores;

    protobuf::InFile rows(rows_in);

    while (rows.try_read_stream(row)) {
        cross_cat.value_split(row.data(), factors);

        for (size_t i = 0; i < kind_count; ++i) {
            const auto & value = factors[i];
            auto & kind = cross_cat.kinds[i];
            const ProductModel & model = kind.model;
            ProductModel::Mixture & mixture = kind.mixture;

            mixture.score(model, value, scores, rng);
            size_t groupid =
                distributions::sample_from_scores_overwrite(rng, scores);
            mixture.add_value(model, groupid, value, rng);
        }
    }
}

void infer_multi_pass (
        CrossCat & cross_cat,
        const char * assign_in,
        const char * rows_in,
        double extra_passes,
        rng_t & rng)
{
    const size_t kind_count = cross_cat.kinds.size();
    protobuf::SparseRow row;
    std::vector<ProductModel::Value> factors(kind_count);
    VectorFloat scores;

    Assignments assignments(kind_count);
    protobuf::InFile rows_to_insert(rows_in);
    protobuf::InFile rows_to_remove(rows_in);

    if (assign_in) {
        assignments.load(assign_in);
        if (assignments.size()) {
            for (auto & kind : cross_cat.kinds) {
                LOOM_ASSERT_LE(
                    assignments.size(),
                    kind.mixture.clustering.sample_size);
            }
            LOOM_ERROR("TODO advance rows files to correct positions");
        }
    }

    AnnealingSchedule schedule(extra_passes);

    while (true) {

        if (schedule.next_action_is_add()) {

            rows_to_insert.cyclic_read_stream(row);
            cross_cat.value_split(row.data(), factors);
            auto * global_groupids = assignments.try_insert(row.id());

            bool already_inserted = (global_groupids == nullptr);
            if (LOOM_UNLIKELY(already_inserted)) {
                break;
            }

            for (size_t i = 0; i < kind_count; ++i) {
                const auto & value = factors[i];
                auto & kind = cross_cat.kinds[i];
                const ProductModel & model = kind.model;
                ProductModel::Mixture & mixture = kind.mixture;
                auto & id_tracker = mixture.id_tracker;

                mixture.score(model, value, scores, rng);
                auto groupid =
                    distributions::sample_from_scores_overwrite(rng, scores);
                mixture.add_value(model, groupid, value, rng);
                global_groupids[i] = id_tracker.packed_to_global(groupid);
            }

        } else {

            rows_to_remove.cyclic_read_stream(row);
            cross_cat.value_split(row.data(), factors);
            auto self_destructing = assignments.remove(row.id());
            const auto * global_groupids = self_destructing.value;

            for (size_t i = 0; i < kind_count; ++i) {
                const auto & value = factors[i];
                auto & kind = cross_cat.kinds[i];
                const ProductModel & model = kind.model;
                ProductModel::Mixture & mixture = kind.mixture;
                auto & id_tracker = mixture.id_tracker;

                auto groupid = id_tracker.global_to_packed(global_groupids[i]);
                mixture.remove_value(model, groupid, value, rng);
            }
        }
    }
}

} // namespace loom
