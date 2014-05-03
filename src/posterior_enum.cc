#include "args.hpp"
#include "loom.hpp"

const char * help_message =
"Usage: infer_many MODEL_IN ROWS_IN SAMPLES_OUT"
"\n  [SAMPLE_COUNT=100] [EXTRA_PASSES=0] [KIND_COUNT=0] [KIND_ITERS=32]"
"\nArguments:"
"\n  MODEL_IN      filename of model (e.g. model.pb.gz)"
"\n  ROWS_IN       filename of input dataset stream (e.g. rows.pbs.gz)"
"\n  SAMPLES_OUT   filename of samples stream (e.g. samples.pbs.gz)"
"\n  SAMPLE_COUNT  number of samples to output"
"\n  KIND_COUNT    if nonzero, run kind inference with this many"
"\n                ephemeral kinds; otherwise assume fixed kind structure"
"\n  KIND_ITERS    if running kind inference, run inner loop of algorithm8"
"\n                for this many iterations"
;

int main (int argc, char ** argv)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    Args args(argc, argv, help_message);
    const char * model_in = args.pop();
    const char * rows_in = args.pop();
    const char * samples_out = args.pop();
    const int sample_count = args.pop_default(100);
    const int kind_count = args.pop_default(0);
    const int kind_iters = args.pop_default(32);
    args.done();

    LOOM_ASSERT_LE(0, sample_count);
    LOOM_ASSERT_LE(0, kind_count);

    loom::rng_t rng;
    loom::Loom engine(rng, model_in);

    if (kind_count == 0) {

        engine.posterior_enum(
            rng,
            rows_in,
            samples_out,
            sample_count);

    } else {

        engine.posterior_enum(
            rng,
            rows_in,
            samples_out,
            sample_count,
            kind_count,
            kind_iters);
    }

    return 0;
}
