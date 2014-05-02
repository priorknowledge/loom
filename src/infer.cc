#ifdef _OPENMP
#    include <omp.h>
#endif // _OPENMP

#include "args.hpp"
#include "loom.hpp"

const char * help_message =
"Usage: infer MODEL_IN GROUPS_IN ASSIGN_IN ROWS_IN GROUPS_OUT ASSIGN_OUT\\"
"\n  [EXTRA_PASSES=0] [KIND_COUNT=0] [KIND_ITERS=32]"
"\nArguments:"
"\n  MODEL_IN      filename of model (e.g. model.pb.gz)"
"\n  GROUPS_IN     dirname containing per-kind group files,"
"\n                or --none for empty group initialization"
"\n  ASSIGN_IN     filename of assignments stream (e.g. assign.pbs.gz)"
"\n                or --none for empty assignments initialization"
"\n  ROWS_IN       filename of input dataset stream (e.g. rows.pbs.gz)"
"\n  GROUPS_OUT    dirname to contain per-kind group files"
"\n  ASSIGN_OUT    filename of assignments stream (e.g. assign.pbs.gz)"
"\n                or --none for empty assignments initialization"
"\n  EXTRA_PASSES  number of extra learning passes over data,"
"\n                any positive real number"
"\n  KIND_COUNT    if nonzero, run kind inference with this many"
"\n                ephemeral kinds; otherwise assume fixed kind structure"
"\n  KIND_ITERS    if running kind inference, run inner loop of algorithm8"
"\n                for this many iterations"
"\nNotes:"
"\n  Any filename can end with .gz to indicate gzip compression."
"\n  Any filename can be '-' or '-.gz' to indicate stdin/stdout."
;

int main (int argc, char ** argv)
{
#ifdef _OPENMP
    omp_set_num_threads(1 + omp_get_num_procs());
#endif // _OPENMP
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    Args args(argc, argv, help_message);
    const char * model_in = args.pop();
    const char * groups_in = args.pop();
    const char * assign_in = args.pop();
    const char * rows_in = args.pop();
    const char * groups_out = args.pop();
    const char * assign_out = args.pop();
    const double extra_passes = args.pop_default(0.0);
    const double kind_count = args.pop_default(0);
    const double kind_iters = args.pop_default(32);
    args.done();

    if (groups_in == std::string("--none")) {
        groups_in = nullptr;
    }
    if (assign_in == std::string("--none")) {
        assign_in = nullptr;
    }
    if (assign_out == std::string("--none")) {
        assign_out = nullptr;
    }

    loom::rng_t rng;
    loom::Loom engine(rng, model_in, groups_in, assign_in);

    if (extra_passes == 0.0) {
        engine.infer_single_pass(rng, rows_in, assign_out);
        engine.dump(groups_out);
    } else if (kind_count == 0) {
        engine.infer_multi_pass(rng, rows_in, extra_passes);
        engine.dump(groups_out, assign_out);
    } else {
        engine.infer_kind_structure(
            rng,
            rows_in,
            extra_passes,
            kind_count,
            kind_iters);
        engine.dump(groups_out, assign_out);
    }

    return 0;
}
