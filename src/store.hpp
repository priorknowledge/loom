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

# pragma once

#include <sstream>
#include <fstream>
#include <loom/common.hpp>

namespace loom
{
namespace store
{

struct Paths
{
    struct Ingest
    {
        std::string tares;
    };

    struct Sample
    {
        std::string config;
        std::string model;
        std::string groups;
        std::string assign;
    };

    Ingest ingest;
    std::vector<Sample> samples;
};

inline std::string get_mixture_path (
        const std::string & groups_path,
        size_t kindid)
{
    std::ostringstream filename;
    filename << groups_path << "/mixture." << kindid << ".pbs.gz";
    return filename.str();
}

inline std::string get_sample_path (
        const std::string & root,
        size_t seed)
{
    std::ostringstream filename;
    filename << root << "/samples/sample." << seed;
    return filename.str();
}

inline Paths get_paths (const std::string & root)
{
    Paths paths;
    paths.ingest.tares = root + "/ingest/tares.pbs.gz";
    for (size_t seed = 0;; ++seed) {
        const std::string sample_root = get_sample_path(root, seed);
        if (std::ifstream(sample_root)) {
            paths.samples.resize(seed + 1);
            auto & sample = paths.samples.back();
            sample.config = sample_root + "/config.pb.gz";
            sample.model = sample_root + "/model.pb.gz";
            sample.groups = sample_root + "/groups";
            sample.assign = sample_root + "/assign.pbs.gz";
        } else {
            break;
        }
    }
    return paths;
}

} // namespace store
} // namespace loom
