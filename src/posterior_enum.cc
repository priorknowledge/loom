#include "args.hpp"
#include "loom.hpp"

const char * help_message =
"Usage: posterior_enum MODEL_IN GROUPS_IN ASSIGN_IN ROWS_IN SAMPLES_OUT"
"\n  [SAMPLE_COUNT=100] [SAMPLE_SKIP=10] [KIND_COUNT=32] [KIND_ITERS=32]"
"\nArguments:"
"\n  MODEL_IN      filename of model (e.g. model.pb.gz)"
"\n  GROUPS_IN     dirname containing per-kind group files,"
"\n                or --none for empty group initialization"
"\n  ASSIGN_IN     filename of assignments stream (e.g. assign.pbs.gz)"
"\n                or --none for empty assignments initialization"
"\n  ROWS_IN       filename of input dataset stream (e.g. rows.pbs.gz)"
"\n  SAMPLES_OUT   filename of samples stream (e.g. samples.pbs.gz)"
"\n  SAMPLE_COUNT  number of samples to output"
"\n  SAMPLE_SKIP  number of samples to output"
"\n  KIND_COUNT    if nonzero, run kind inference with this many"
"\n                ephemeral kinds; otherwise assume fixed kind structure"
"\n  KIND_ITERS    if running kind inference, run inner loop of algorithm8"
"\n                for this many iterations"
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
    const char * samples_out = args.pop();
    const int sample_count = args.pop_default(100);
    const int sample_skip = args.pop_default(10);
    const int kind_count = args.pop_default(32);
    const int kind_iters = args.pop_default(32);
    args.done();

    LOOM_ASSERT_LE(1, sample_count);
    LOOM_ASSERT_LE(0, sample_skip);
    LOOM_ASSERT(sample_skip > 0 or sample_count == 1, "zero diversity");
    LOOM_ASSERT_LE(0, kind_count);
    LOOM_ASSERT_LE(0, kind_iters);

    loom::rng_t rng;
    loom::Loom engine(rng, model_in, groups_in, assign_in);

    if (kind_count == 0) {

        engine.posterior_enum(
            rng,
            rows_in,
            samples_out,
            sample_count,
            sample_skip);

    } else {

        engine.posterior_enum(
            rng,
            rows_in,
            samples_out,
            sample_count,
            sample_skip,
            kind_count,
            kind_iters);
    }

    return 0;
}
