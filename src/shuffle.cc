#include <vector>
#include <algorithm>
#include "protobuf_stream.hpp"
#include "args.hpp"

const char * help_message =
"Usage: shuffle ROWS_IN ROWS_OUT [SEED=0]"
"\nArguments:"
"\n  ROWS_IN       filename of input dataset stream (e.g. rows.pbs.gz)"
"\n  ROWS_OUT      filename of output dataset stream (e.g. rows_out.pbs.gz)"
"\n  SEED          random seed"
"\nNotes:"
"\n  Any filename can end with .gz to indicate gzip compression."
"\n  Any filename can be '-' or '-.gz' to indicate stdin/stdout."
;

int main (int argc, char ** argv)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    Args args(argc, argv, help_message);
    const char * rows_in = args.pop();
    const char * rows_out = args.pop();
    const long seed = args.pop_default(0L);
    args.done();

    auto rows = loom::protobuf_stream_load<std::vector<char>>(rows_in);
    std::shuffle(rows.begin(), rows.end(), loom::rng_t(seed));
    loom::protobuf_stream_dump(rows, rows_out);

    return 0;
}
