#include "args.hpp"
#include "loom.hpp"

const char * help_message =
"Usage: predict MODEL_IN GROUPS_IN QUERIES_IN RESULTS_OUT"
"\nArguments:"
"\n  MODEL_IN      filename of model (e.g. model.pb.gz)"
"\n  GROUPS_IN     dirname containing per-kind group files"
"\n  QUERIES_IN    filename of queries stream (e.g. queries.pbs.gz)"
"\n  RESULTS_OUT   filename of results stream (e.g. results.pbs.gz)"
"\nNotes:"
"\n  Any filename can end with .gz to indicate gzip compression."
"\n  Any filename can be '-' or '-.gz' to indicate stdin/stdout."
;

int main (int argc, char ** argv)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    Args args(argc, argv, help_message);
    const char * model_in = args.pop();
    const char * groups_in = args.pop();
    const char * queries_in = args.pop();
    const char * results_out = args.pop();

    loom::rng_t rng;
    loom::Loom engine(rng, model_in, groups_in);

    engine.predict(rng, queries_in, results_out);

    return 0;
}
