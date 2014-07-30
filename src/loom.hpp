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

#include <thread>
#include <vector>
#include <unordered_map>
#include <loom/common.hpp>
#include <loom/cross_cat.hpp>
#include <loom/assignments.hpp>
#include <loom/timer.hpp>
#include <loom/logger.hpp>
#include <loom/schedules.hpp>

namespace loom
{

class StreamInterval;

class Loom : noncopyable
{
public:

    typedef protobuf::Checkpoint Checkpoint;

    Loom (
            rng_t & rng,
            const protobuf::Config & config,
            const char * model_in,
            const char * groups_in = nullptr,
            const char * assign_in = nullptr,
            const char * tares_in = nullptr);

    void dump (
            const char * model_out = nullptr,
            const char * groups_out = nullptr,
            const char * assign_out = nullptr) const;

    void infer_single_pass (
            rng_t & rng,
            const char * rows_in,
            const char * assign_out = nullptr);

    void infer_multi_pass (
            rng_t & rng,
            const char * rows_in,
            const char * checkpoint_in = nullptr,
            const char * checkpoint_out = nullptr);

    void posterior_enum (
            rng_t & rng,
            const char * rows_in,
            const char * samples_out);

    void generate (
            rng_t & rng,
            const char * rows_out);

    void mix (
            rng_t & rng,
            const char * rows_in);

    void query (
            rng_t & rng,
            const char * requests_in,
            const char * responses_out);

private:

    bool infer_kind_structure_sequential (
            StreamInterval & rows,
            Checkpoint & checkpoint,
            CombinedSchedule & schedule,
            rng_t & rng);

    bool infer_kind_structure_parallel (
            StreamInterval & rows,
            Checkpoint & checkpoint,
            CombinedSchedule & schedule,
            rng_t & rng);

    bool infer_kind_structure (
            StreamInterval & rows,
            Checkpoint & checkpoint,
            CombinedSchedule & schedule,
            rng_t & rng);

    bool infer_cat_structure_sequential (
            StreamInterval & rows,
            Checkpoint & checkpoint,
            CombinedSchedule & schedule,
            rng_t & rng);

    bool infer_cat_structure_parallel (
            StreamInterval & rows,
            Checkpoint & checkpoint,
            CombinedSchedule & schedule,
            rng_t & rng);

    bool infer_cat_structure (
            StreamInterval & rows,
            Checkpoint & checkpoint,
            CombinedSchedule & schedule,
            rng_t & rng);

    void log_metrics (Logger::Message & message);

    void dump_posterior_enum (
            protobuf::PosteriorEnum::Sample & message,
            rng_t & rng);

    const protobuf::Config & config_;
    CrossCat cross_cat_;
    Assignments assignments_;
};

inline bool Loom::infer_kind_structure (
        StreamInterval & rows,
        Checkpoint & checkpoint,
        CombinedSchedule & schedule,
        rng_t & rng)
{
    if (config_.kernels().kind().row_queue_capacity()) {
        return infer_kind_structure_parallel(rows, checkpoint, schedule, rng);
    } else {
        return infer_kind_structure_sequential(rows, checkpoint, schedule, rng);
    }
}

inline bool Loom::infer_cat_structure (
        StreamInterval & rows,
        Checkpoint & checkpoint,
        CombinedSchedule & schedule,
        rng_t & rng)
{
    if (config_.kernels().cat().row_queue_capacity()) {
        return infer_cat_structure_parallel(rows, checkpoint, schedule, rng);
    } else {
        return infer_cat_structure_sequential(rows, checkpoint, schedule, rng);
    }
}

} // namespace loom
