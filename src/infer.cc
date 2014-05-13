#include "args.hpp"
#include "loom.hpp"
#include "logger.hpp"

const char * help_message =
"Usage: infer MODEL_IN GROUPS_IN ASSIGN_IN ROWS_IN"
"\n  MODEL_OUT GROUPS_OUT ASSIGN_OUT LOG_OUT"
"\n  [CAT_PASSES=9.0] [KIND_PASSES=90.0] [KIND_COUNT=32] [KIND_ITERS=32]"
"\n  [MAX_REJECT_ITERS=100]"
"\nArguments:"
"\n  MODEL_IN          filename of model (e.g. model.pb.gz)"
"\n  GROUPS_IN         dirname containing per-kind group files,"
"\n                    or --none for empty group initialization"
"\n  ASSIGN_IN         filename of assignments stream (e.g. assign.pbs.gz)"
"\n                    or --none for empty assignments initialization"
"\n  ROWS_IN           filename of input dataset stream (e.g. rows.pbs.gz)"
"\n  MODEL_OUT         filename of model to write, or --none to discard groups"
"\n  GROUPS_OUT        dirname to contain per-kind group files"
"\n                    or --none to discard groups"
"\n  ASSIGN_OUT        filename of assignments stream (e.g. assign.pbs.gz)"
"\n                    or --none to discard assignments"
"\n  LOG_OUT           filename of log (e.g. log.pbs.gz)"
"\n  CAT_PASSES        number of extra category-learning passes over data,"
"\n                    any positive real number"
"\n  KIND_PASSES       number of extra kind-learning passes over data,"
"\n                    any positive real number"
"\n  KIND_COUNT        if nonzero, run kind inference with this many"
"\n                    ephemeral kinds; otherwise assume fixed kind structure"
"\n  KIND_ITERS        if running kind inference, run inner loop of algorithm8"
"\n                    for this many iterations"
"\n  MAX_REJECT_ITERS  stop running kind inference after this many rejected proposals"
"\nNotes:"
"\n  Any filename can end with .gz to indicate gzip compression."
"\n  Any filename can be '-' or '-.gz' to indicate stdin/stdout."
"\n  If running kind inference and GROUPS_IN is provided,"
"\n    then all data in groups must be accounted for in ASSIGN_IN."
;

inline const char * optional_file (const char * arg)
{
    return (arg == std::string("--none")) ? nullptr : arg;
}

int main (int argc, char ** argv)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    Args args(argc, argv, help_message);
    const char * model_in = args.pop();
    const char * groups_in = optional_file(args.pop());
    const char * assign_in = optional_file(args.pop());
    const char * rows_in = args.pop();
    const char * model_out = optional_file(args.pop());
    const char * groups_out = optional_file(args.pop());
    const char * assign_out = optional_file(args.pop());
    const char * log_out = optional_file(args.pop());
    const double cat_passes = args.pop_default(9.0);
    const double kind_passes = args.pop_default(90.0);
    const int kind_count = args.pop_default(32);
    const int kind_iters = args.pop_default(32);
    const int max_reject_iters = args.pop_default(100);
    args.done();

    LOOM_ASSERT_LE(0, cat_passes);
    LOOM_ASSERT_LE(0, kind_passes);
    LOOM_ASSERT_LE(0, kind_count);
    LOOM_ASSERT_LE(0, kind_iters);
    LOOM_ASSERT_LE(0, max_reject_iters);

    if (log_out) {
        loom::global_logger.open(log_out);
    }

    loom::rng_t rng;
    loom::Loom engine(rng, model_in, groups_in, assign_in);

    if (kind_passes + cat_passes > 0) {

        engine.infer_multi_pass(
            rng,
            rows_in,
            cat_passes,
            kind_passes,
            kind_count,
            kind_iters,
            max_reject_iters);
        engine.dump(model_out, groups_out, assign_out);

    } else {

        engine.infer_single_pass(rng, rows_in, assign_out);
        engine.dump(model_out, groups_out);
    }

    return 0;
}
