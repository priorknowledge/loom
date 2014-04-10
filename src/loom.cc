#include <distributions/random.hpp>
#include "args.hpp"
#include "loom.hpp"

const char * help_message =
"Usage: loom MODEL_IN GROUPS_IN ASSIGN_IN ROWS_IN GROUPS_OUT [EXTRA_PASSES]"
"\nArguments:"
"\n  MODEL_IN      filename of model (e.g. model.pb.gz)"
"\n  GROUPS_IN     dirname containing per-kind group files,"
"\n                or --empty for empty group initialization"
"\n  ASSIGN_IN     filename of assignments stream (e.g. assignments.pbs.gz)"
"\n                or --empty for empty assignments initialization"
"\n  ROWS_IN       filename of input dataset stream (e.g. rows.pbs.gz)"
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

    Args args(argc, argv, help_message);
    const char * model_in = args.get();
    const char * groups_in = args.get();
    const char * assign_in = args.get();
    const char * rows_in = args.get();
    const char * groups_out = args.get();
    const double extra_passes = args.get_default(0.0);

    if (groups_in == std::string("--empty")) {
        groups_in = nullptr;
    }
    if (assign_in == std::string("--empty")) {
        assign_in = nullptr;
    }

    distributions::rng_t rng;

    loom::CrossCat cross_cat;
    cross_cat.model_load(model_in);
    LOOM_ASSERT(cross_cat.kinds.size(), "no kinds, nothing to do");
    if (groups_in) {
        cross_cat.mixture_init_empty(rng);
    } else {
        cross_cat.mixture_load(groups_in, rng);
    }

    if (extra_passes == 0.0) {
        loom::infer_single_pass(cross_cat, rows_in, rng);
    } else {
        loom::infer_multi_pass(
            cross_cat,
            assign_in,
            rows_in,
            extra_passes,
            rng);
    }

    cross_cat.mixture_dump(groups_out);

    return 0;
}
