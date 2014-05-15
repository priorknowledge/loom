#include <loom/args.hpp>
#include <loom/protobuf_stream.hpp>
#include <loom/loom.hpp>

const char * help_message =
"Usage: posterior_enum CONFIG_IN MODEL_IN GROUPS_IN ASSIGN_IN ROWS_IN"
"\n  SAMPLES_OUT"
"\nArguments:"
"\n  CONFIG_IN     filename of config (e.g. config.pb.gz)"
"\n  MODEL_IN      filename of model (e.g. model.pb.gz)"
"\n  GROUPS_IN     dirname containing per-kind group files,"
"\n                or --none for empty group initialization"
"\n  ASSIGN_IN     filename of assignments stream (e.g. assign.pbs.gz)"
"\n                or --none for empty assignments initialization"
"\n  ROWS_IN       filename of input dataset stream (e.g. rows.pbs.gz)"
"\n  SAMPLES_OUT   filename of samples stream (e.g. samples.pbs.gz)"
"\nNotes:"
"\n  Any filename can end with .gz to indicate gzip compression."
"\n  Any filename can be '-' or '-.gz' to indicate stdin/stdout."
"\n  If running kind inference and GROUPS_IN is provided,"
"\n    then all data in groups must be accounted for in ASSIGN_IN."
;

int main (int argc, char ** argv)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    Args args(argc, argv, help_message);
    const char * config_in = args.pop();
    const char * model_in = args.pop();
    const char * groups_in = args.pop_optional_file();
    const char * assign_in = args.pop_optional_file();
    const char * rows_in = args.pop();
    const char * samples_out = args.pop();
    args.done();

    const auto config = loom::protobuf_load<loom::protobuf::Config>(config_in);
    loom::rng_t rng(config.seed());
    loom::Loom engine(rng, config, model_in, groups_in, assign_in);

    engine.posterior_enum(rng, rows_in, samples_out);

    return 0;
}
