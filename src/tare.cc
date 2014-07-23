// Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// - Neither the name of Salesforce.com nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <loom/args.hpp>
#include <loom/differ.hpp>

const char * help_message =
"Usage: tare SCHEMA_ROW_IN ROWS_IN TARES_OUT"
"\nArguments:"
"\n  SCHEMA_ROW_IN filename of schema row (e.g. schema.pb.gz)"
"\n  ROWS_IN       filename of input dataset stream (e.g. rows.pbs.gz)"
"\n  TARES_OUT     filename of output tare rows (e.g. tares.pbs.gz)"
"\nNotes:"
"\n  Any filename can end with .gz to indicate gzip compression."
"\n  Any filename can be '-' or '-.gz' to indicate stdin/stdout."
;

int main (int argc, char ** argv)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    Args args(argc, argv, help_message);
    const char * schema_row_in = args.pop();
    const char * rows_in = args.pop();
    const char * tares_out = args.pop();
    args.done();

    loom::ProductValue value;
    loom::protobuf::InFile(schema_row_in).read(value);
    loom::ValueSchema schema;
    schema.load(value);

    loom::Differ differ(schema);
    differ.add_rows(rows_in);

    loom::protobuf::OutFile tares(tares_out);
    const auto & tare = differ.get_tare();
    if (schema.total_size(tare)) {
        tares.write_stream(tare);
    }

    return 0;
}
