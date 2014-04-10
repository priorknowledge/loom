#include <cstdlib>
#include <distributions/random.hpp>
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
        const char * rows_in,
        double extra_passes,
        rng_t & rng)
{
    const size_t kind_count = cross_cat.kinds.size();
    protobuf::SparseRow row;
    std::vector<ProductModel::Value> factors(kind_count);
    VectorFloat scores;

    Assignments<uint64_t, uint32_t> assignments(kind_count);
    AnnealingSchedule schedule(extra_passes);

    protobuf::InFile rows_to_insert(rows_in);
    protobuf::InFile rows_to_remove(rows_in);

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

const char * help_message =
"Usage: loom MODEL_IN GROUPS_IN ROWS_IN GROUPS_OUT [EXTRA_PASSES]"
"\nArguments:"
"\n  MODEL_IN      filename of model (e.g. model.pb.gz)"
"\n  GROUPS_IN     dirname containing per-kind group files,"
"\n                or --empty for empty initialization"
"\n  ROWS_IN       filename of input dataset (e.g. rows.pbs.gz)"
"\n  GROUPS_OUT    dirname to contain per-kind group files"
"\n  EXTRA_PASSES  number of extra learning passes over data,"
"\n                any positive real number"
"\nNotes:"
"\n  Any filename can end with .gz to indicate gzip compression."
"\n  Any filename can be '-' or '-.gz' to indicate stdin/stdout."
;

int main (int argc, char ** argv)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    if (argc < 6 or 6 < argc) {
        std::cerr << help_message << std::endl;
        exit(1);
    }

    const char * model_in = argv[1];
    const char * groups_in = argv[2];
    const char * rows_in = argv[3];
    const char * groups_out = argv[4];
    const double extra_passes = argc > 5 ? atof(argv[5]) : 0.0;

    distributions::rng_t rng;

    loom::CrossCat cross_cat;
    cross_cat.model_load(model_in);
    LOOM_ASSERT(cross_cat.kinds.size(), "no kinds, nothing to do");
    if (strcmp(groups_in, "--empty") == 0) {
        cross_cat.mixture_init_empty(rng);
    } else {
        cross_cat.mixture_load(groups_in, rng);
    }

    if (extra_passes == 0.0) {
        loom::infer_single_pass(cross_cat, rows_in, rng);
    } else if (0.0 < extra_passes and extra_passes < INFINITY) {
        loom::infer_multi_pass(cross_cat, rows_in, extra_passes, rng);
    } else {
        LOOM_ERROR("bad annealing iters: " << extra_passes);
    }

    cross_cat.mixture_dump(groups_out);

    return 0;
}
