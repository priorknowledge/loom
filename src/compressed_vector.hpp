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

#pragma once

#include <vector>
#include <unordered_map>
#include <loom/common.hpp>

namespace loom
{

template<class Value>
class CompressedVector
{
    typedef uint32_t id_t;
    typedef std::unordered_map<std::string, id_t> Map;

    Map string_to_id_;
    std::vector<const std::string *> id_to_string_;
    std::vector<id_t> pos_to_id_;

    bool is_initialized () const
    {
        return id_to_string_.size() == string_to_id_.size();
    }

public:

    void push_back (const Value & value)
    {
        // never freed
        static thread_local Map::value_type * to_insert = nullptr;
        construct_if_null(to_insert);

        value.SerializeToString(const_cast<std::string *>(& to_insert->first));
        auto inserted = string_to_id_.insert(*to_insert);
        id_t & id = inserted.first->second;
        if (LOOM_UNLIKELY(inserted.second)) {
            id = string_to_id_.size() - 1;
        }
        pos_to_id_.push_back(id);
    }

    void init_index ()
    {
        id_to_string_.resize(string_to_id_.size());
        for (const auto & pair : string_to_id_) {
            id_to_string_[pair.second] = &pair.first;
        }

        if (LOOM_DEBUG_LEVEL >= 1) {
            LOOM_ASSERT(is_initialized(), "index is not initialized");
        }

    }

    size_t unique_count () const
    {
        if (LOOM_DEBUG_LEVEL >= 1) {
            LOOM_ASSERT(is_initialized(), "index is not initialized");
        }

        return id_to_string_.size();
    }

    void unique_value (size_t id, Value & value) const
    {
        if (LOOM_DEBUG_LEVEL >= 1) {
            LOOM_ASSERT(is_initialized(), "index is not initialized");
            LOOM_ASSERT_LT(id, id_to_string_.size());
        }

        value.ParseFromString(*id_to_string_[id]);
    }

    id_t unique_id (size_t pos) const
    {
        if (LOOM_DEBUG_LEVEL >= 1) {
            LOOM_ASSERT(is_initialized(), "index is not initialized");
            LOOM_ASSERT_LT(pos, pos_to_id_.size());
        }

        return pos_to_id_[pos];
    }
};

} // namespace loom
