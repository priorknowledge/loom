#include <loom/args.hpp>
#include <loom/protobuf_stream.hpp>
#include <loom/logger.hpp>
#include <loom/loom.hpp>

const char * help_message =
"Usage: predict CONFIG_IN MODEL_IN GROUPS_IN QUERIES_IN RESULTS_OUT LOG_OUT"
"\nArguments:"
"\n  CONFIG_IN     filename of config (e.g. config.pb.gz)"
"\n  MODEL_IN      filename of model (e.g. model.pb.gz)"
"\n  GROUPS_IN     dirname containing per-kind group files"
"\n  QUERIES_IN    filename of queries stream (e.g. queries.pbs.gz)"
"\n  RESULTS_OUT   filename of results stream (e.g. results.pbs.gz)"
"\n  LOG_OUT       filename of log (e.g. log.pbs.gz)"
"\n                or --none to not log"
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
    const char * groups_in = args.pop();
    const char * queries_in = args.pop();
    const char * results_out = args.pop();
    const char * log_out = args.pop_optional_file();
    args.done();

    if (log_out) {
        loom::logger.open(log_out);
    }

    const auto config = loom::protobuf_load<loom::protobuf::Config>(config_in);
    loom::rng_t rng(config.seed());
    loom::Loom engine(rng, config, model_in, groups_in);

    engine.predict(rng, queries_in, results_out);

    return 0;
}
