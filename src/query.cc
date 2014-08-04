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
#include <loom/protobuf_stream.hpp>
#include <loom/logger.hpp>
#include <loom/multi_loom.hpp>
#include <loom/query_server.hpp>

const char * help_message =
"Usage: query ROOT_IN REQUESTS_IN RESPONSES_OUT LOG_OUT"
"\nArguments:"
"\n  ROOT_IN         root dirname of dataset in loom store"
"\n  REQUESTS_IN     filename of requests stream (e.g. requests.pbs.gz)"
"\n  RESPONSES_OUT   filename of responses stream (e.g. responses.pbs.gz)"
"\n  LOG_OUT         filename of log (e.g. log.pbs.gz)"
"\n                  or --none to not log"
"\nNotes:"
"\n  Any filename can end with .gz to indicate gzip compression."
"\n  Any filename can be '-' or '-.gz' to indicate stdin/stdout."
;

int main (int argc, char ** argv)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    Args args(argc, argv, help_message);
    const char * root_in = args.pop();
    const char * requests_in = args.pop();
    const char * responses_out = args.pop();
    const char * log_out = args.pop_optional_file();
    args.done();

    if (log_out) {
        loom::logger.append(log_out);
    }

    const bool load_groups = true;
    const bool load_assign = false;
    const bool load_tares = true;
    loom::MultiLoom engine(root_in, load_groups, load_assign, load_tares);
    loom::QueryServer server(engine.cross_cats());
    loom::rng_t rng;

    server.serve(rng, requests_in, responses_out);

    return 0;
}
