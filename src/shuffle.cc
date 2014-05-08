#include <vector>
#include <algorithm>
#include "protobuf_stream.hpp"
#include "args.hpp"

void load_rows (
        const char * filename,
        std::vector<std::vector<char>> & rows)
{
    rows.resize(1);
    loom::protobuf::InFile file(filename);
    while (file.try_read_stream(rows.back())) {
        rows.resize(rows.size() + 1);
    }
    rows.pop_back();
}

void dump_rows (
        const char * filename,
        const std::vector<std::vector<char>> & rows)
{
    loom::protobuf::OutFile file(filename);
    for (const auto & row : rows) {
        file.write_stream(row);
    }
}


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

    std::vector<std::vector<char>> rows;
    load_rows(rows_in, rows);

    loom::rng_t rng(seed);
    std::shuffle(rows.begin(), rows.end(), rng);

    dump_rows(rows_out, rows);

    return 0;
}
