#include <cstdlib>
#include <distributions/random.hpp>
#include "common.hpp"
#include "cross_cat.hpp"
#include "assignments.hpp"
#include "annealing_schedule.hpp"

namespace loom
{

void infer_greedy (
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

void infer_annealing (
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

            if (LOOM_UNLIKELY(global_groupids == nullptr)) {
                break;
            }

            for (size_t i = 0; i < kind_count; ++i) {
                const auto & value = factors[i];
                auto & kind = cross_cat.kinds[i];
                const ProductModel & model = kind.model;
                ProductModel::Mixture & mixture = kind.mixture;

                mixture.score(model, value, scores, rng);
                auto groupid =
                    distributions::sample_from_scores_overwrite(rng, scores);
                mixture.add_value(model, groupid, value, rng);
                global_groupids[i] =
                    mixture.assignments.packed_to_global(groupid);
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

                auto groupid =
                    mixture.assignments.global_to_packed(global_groupids[i]);
                mixture.remove_value(model, groupid, value, rng);
            }
        }
    }
}

} // namespace loom

const char * help_message =
"Usage: loom MODEL_IN GROUPS_IN ROWS_IN GROUPS_OUT [EXTRA_PASSES]"
"\nArguments:"
"\n  MODEL_IN   filename of model (e.g. model.pb.gz)"
"\n  GROUPS_IN  dirname containing per-kind group files"
"\n  ROWS_IN    filename of input"
"\nNotes:"
"\n  Any filename can end with .gz to indicate gzip compression."
"\n  Any input/output can be named '-' or '-.gz' to indicate stdin/stdout."
"\n  GROUPS_IN can be named '--empty' to indicate empty initialization."
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
        loom::infer_greedy(cross_cat, rows_in, rng);
    } else if (0.0 < extra_passes and extra_passes < INFINITY) {
        loom::infer_annealing(cross_cat, rows_in, extra_passes, rng);
    } else {
        LOOM_ERROR("bad annealing iters: " << extra_passes);
    }

    cross_cat.mixture_dump(groups_out);

    return 0;
}
