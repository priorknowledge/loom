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
#include <loom/loom.hpp>

const char * help_message =
"Usage: query CONFIG_IN MODEL_IN GROUPS_IN REQUESTS_IN RESPONSES_OUT LOG_OUT"
"\nArguments:"
"\n  CONFIG_IN       filename of config (e.g. config.pb.gz)"
"\n  MODEL_IN        filename of model (e.g. model.pb.gz)"
"\n  GROUPS_IN       dirname containing per-kind group files"
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
    const char * config_in = args.pop();
    const char * model_in = args.pop();
    const char * groups_in = args.pop();
    const char * requests_in = args.pop();
    const char * responses_out = args.pop();
    const char * log_out = args.pop_optional_file();
    args.done();

    if (log_out) {
        loom::logger.open(log_out);
    }

    const auto config = loom::protobuf_load<loom::protobuf::Config>(config_in);
    loom::rng_t rng(config.seed());
    loom::Loom engine(rng, config, model_in, groups_in);

    engine.query(rng, requests_in, responses_out);

    return 0;
}
