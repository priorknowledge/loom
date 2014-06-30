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

#include <limits>
#include <algorithm>
#include <loom/common.hpp>
#include <loom/protobuf_stream.hpp>

namespace loom
{

inline void shuffle_stream (
        const char * messages_in,
        const char * shuffled_out,
        long seed,
        double target_mem_bytes)
{
    typedef std::vector<char> Message;
    typedef uint32_t pos_t;

    LOOM_ASSERT(
        std::string(messages_in) != std::string(shuffled_out),
        "cannot shuffle file in-place: " << messages_in);
    const auto stats = protobuf::InFile::stream_stats(messages_in);
    LOOM_ASSERT(stats.is_file, "shuffle input is not a file: " << messages_in);
    const uint64_t max_message_count = std::numeric_limits<pos_t>::max();
    LOOM_ASSERT(stats.message_count, max_message_count);
    const size_t message_count = stats.message_count;

    double index_bytes = sizeof(pos_t) * message_count;
    double target_chunk_size = std::max(1.0, std::min(double(message_count),
        (target_mem_bytes - index_bytes) / stats.max_message_size));
    size_t chunk_size = static_cast<size_t>(std::round(target_chunk_size));
    //LOOM_DEBUG("chunk_size = " << chunk_size);

    std::vector<pos_t> index(message_count);
    for (size_t i = 0; i < message_count; ++i) {
        index[i] = i;
    }
    std::shuffle(index.begin(), index.end(), loom::rng_t(seed));

    Message message;
    std::vector<Message> chunk;
    protobuf::OutFile shuffled(shuffled_out);
    for (size_t begin = 0; begin < message_count; begin += chunk_size) {
        size_t end = std::min(begin + chunk_size, message_count);
        chunk.resize(end - begin);
        protobuf::InFile messages(messages_in);
        for (size_t i : index) {
            messages.try_read_stream(message);
            if (begin <= i and i < end) {
                std::swap(message, chunk[i - begin]);
            }
        }
        for (const auto & message : chunk) {
            shuffled.write_stream(message);
        }
    }
}

} // namespace loom
