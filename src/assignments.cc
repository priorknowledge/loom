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

#include <loom/assignments.hpp>
#include <unordered_map>
#include <distributions/trivial_hash.hpp>
#include <loom/protobuf.hpp>

namespace loom
{

void Assignments::init (size_t kind_count)
{
    clear();
    values_.resize(kind_count);
}

void Assignments::clear ()
{
    keys_.clear();
    for (auto & values : values_) {
        values.clear();
    }
}

void Assignments::load (const char * filename)
{
    clear();

    protobuf::InFile file(filename);
    protobuf::Assignment assignment;

    const size_t kind_count = this->kind_count();
    while (file.try_read_stream(assignment)) {
        LOOM_ASSERT_EQ(assignment.groupids_size(), kind_count);
        keys_.push(assignment.rowid());
        for (size_t i = 0; i < kind_count; ++i) {
            values_[i].push(assignment.groupids(i));
        }
    }
}

void Assignments::dump (
        const char * filename,
        const std::vector<std::vector<uint32_t>> & sorted_to_globals) const
{
    const size_t row_count = this->row_count();
    const size_t kind_count = this->kind_count();

    typedef distributions::TrivialHash<Value> Hash;
    typedef std::unordered_map<Value, Value, Hash> Map;
    std::vector<Map> global_to_sorteds(kind_count);
    for (size_t k = 0; k < kind_count; ++k) {
        Map & global_to_sorted = global_to_sorteds[k];
        const auto & sorted_to_global = sorted_to_globals[k];
        const size_t group_count = sorted_to_global.size();
        for (size_t g = 0; g < group_count; ++g) {
            global_to_sorted[sorted_to_global[g]] = g;
        }
    }

    protobuf::OutFile file(filename);
    protobuf::Assignment assignment;
    for (size_t r = 0; r < row_count; ++r) {
        assignment.clear_groupids();
        assignment.set_rowid(keys_[r]);
        for (size_t k = 0; k < kind_count; ++k) {
            const Map & global_to_sorted = global_to_sorteds[k];
            uint32_t global = values_[k][r];
            auto i = global_to_sorted.find(global);
            LOOM_ASSERT1(i != global_to_sorted.end(), "bad id: " << global);
            assignment.add_groupids(i->second);
        }
        file.write_stream(assignment);
    }
}

} // namespace loom
