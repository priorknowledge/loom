#include <loom/args.hpp>
#include <loom/protobuf_stream.hpp>
#include <loom/loom.hpp>

const char * help_message =
"Usage: predict CONFIG_IN MODEL_IN ROWS_OUT [ROW_COUNT=1]"
"\nArguments:"
"\n  CONFIG_IN     filename of config (e.g. config.pb.gz)"
"\n  MODEL_IN      filename of model (e.g. model.pb.gz)"
"\n  ROWS_OUT      filename of output dataset stream (e.g. rows.pbs.gz)"
"\nNotes:"
"\n  Any filename can end with .gz to indicate gzip compression."
"\n  Any filename can be '-' or '-.gz' to indicate stdin/stdout."
;

int main (int argc, char ** argv)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    Args args(argc, argv, help_message);
    const char * config_in = args.pop();
    const char * model_in = args.pop();
    const char * rows_out = args.pop();
    args.done();

    const auto config = loom::protobuf_load<loom::protobuf::Config>(config_in);
    loom::rng_t rng(config.seed());
    loom::Loom engine(rng, config, model_in);

    engine.generate(rng, rows_out);

    return 0;
}
