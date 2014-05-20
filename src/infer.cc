#include <loom/args.hpp>
#include <loom/protobuf_stream.hpp>
#include <loom/logger.hpp>
#include <loom/loom.hpp>

const char * help_message =
"Usage: infer CONFIG_IN ROWS_IN MODEL_IN GROUPS_IN ASSIGN_IN"
"\n  MODEL_OUT GROUPS_OUT ASSIGN_OUT LOG_OUT"
"\nArguments:"
"\n  CONFIG_IN         filename of config (e.g. config.pb.gz)"
"\n  ROWS_IN           filename of input dataset stream (e.g. rows.pbs.gz)"
"\n  MODEL_IN          filename of model (e.g. model.pb.gz)"
"\n  GROUPS_IN         dirname containing per-kind group files,"
"\n                    or --none for empty group initialization"
"\n  ASSIGN_IN         filename of assignments stream (e.g. assign.pbs.gz)"
"\n                    or --none for empty assignments initialization"
"\n  CHECKPOINT_IN     filename of checkpoint state (e.g. checkpoint.pb.gz)"
"\n                    or --none if not running from checkpoint"
"\n  MODEL_OUT         filename of model to write, or --none to discard groups"
"\n  GROUPS_OUT        dirname to contain per-kind group files"
"\n                    or --none to discard groups"
"\n  ASSIGN_OUT        filename of assignments stream (e.g. assign.pbs.gz)"
"\n                    or --none to discard assignments"
"\n  CHECKPOINT_OUT    filename of checkpoint state (e.g. checkpoint.pb.gz)"
"\n                    or --none if not running from checkpoint"
"\n  LOG_OUT           filename of log (e.g. log.pbs.gz)"
"\n                    or --none to not log"
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
    const char * rows_in = args.pop();
    const char * model_in = args.pop();
    const char * groups_in = args.pop_optional_file();
    const char * assign_in = args.pop_optional_file();
    const char * checkpoint_in = args.pop_optional_file();
    const char * model_out = args.pop_optional_file();
    const char * groups_out = args.pop_optional_file();
    const char * assign_out = args.pop_optional_file();
    const char * checkpoint_out = args.pop_optional_file();
    const char * log_out = args.pop_optional_file();
    args.done();

    if (log_out) {
        loom::logger.open(log_out);
    }

    const auto config = loom::protobuf_load<loom::protobuf::Config>(config_in);
    loom::rng_t rng(config.seed());
    loom::Loom engine(rng, config, model_in, groups_in, assign_in);

    if (config.schedule().extra_passes() > 0) {

        engine.infer_multi_pass(rng, rows_in, checkpoint_in, checkpoint_out);
        engine.dump(model_out, groups_out, assign_out);

    } else {

        engine.infer_single_pass(rng, rows_in, assign_out);
        engine.dump(model_out, groups_out);
    }

    return 0;
}
