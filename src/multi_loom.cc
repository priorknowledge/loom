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

#include <fstream>
#include <loom/store.hpp>
#include <loom/multi_loom.hpp>

namespace loom
{

struct MultiLoom::Sample
{
    protobuf::Config config;
    rng_t rng;
    Loom loom;

    Sample (const store::Paths::Sample & paths,
            bool load_groups,
            bool load_assign,
            const char * tares_in) :
        config(protobuf_load<protobuf::Config>(paths.config.c_str())),
        rng(config.seed()),
        loom(
            rng,
            config,
            paths.model.c_str(),
            load_groups ? paths.groups.c_str() : nullptr,
            load_assign ? paths.assign.c_str() : nullptr,
            tares_in)
    {
    }
};

MultiLoom::MultiLoom (
        const char * root_in,
        bool load_groups,
        bool load_assign,
        bool load_tares)
{
    const auto paths = store::get_paths(root_in);
    const char * tares_in = paths.ingest.tares.c_str();
    if (not (load_tares and std::ifstream(tares_in))) {
        tares_in = nullptr;
    }
    for (const auto & sample_paths : paths.samples) {
        samples_.push_back(
            new Sample(sample_paths, load_groups, load_assign, tares_in));
    }
    LOOM_ASSERT(not samples_.empty(), "no samples were found at " << root_in);
}

MultiLoom::~MultiLoom ()
{
    for (auto * sample : samples_) {
        delete sample;
    }
}

const std::vector<const CrossCat *> MultiLoom::cross_cats () const
{
    std::vector<const CrossCat *> result;
    for (const auto * sample : samples_) {
        result.push_back(& sample->loom.cross_cat());
    }
    return result;
}

} // namespace loom
