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
#include <loom/common.hpp>
#include <loom/cross_cat.hpp>
#include <loom/assignments.hpp>
#include <loom/stream_interval.hpp>
#include <loom/kind_kernel.hpp>
#include <loom/pipeline.hpp>
#include <loom/atomic_array.hpp>

namespace loom
{

class KindPipeline
{
public:

    KindPipeline (
            const protobuf::Config::Kernels::Kind & config,
            CrossCat & cross_cat,
            StreamInterval & rows,
            Assignments & assignments,
            KindKernel & kind_kernel,
            rng_t & rng);

    void add_row ()
    {
        pipeline_.start([](Task & task){ task.add = true; });
    }

    void remove_row ()
    {
        pipeline_.start([](Task & task){ task.add = false; });
    }

    void wait ()
    {
        pipeline_.wait();
    }

    bool try_run ()
    {
        bool changed = kind_kernel_.try_run();
        if (changed) {
            start_kind_threads();
            pipeline_.validate();
        }
        return changed;
    }

    void update_hypers ()
    {
        kind_kernel_.update_hypers();
    }

    void log_metrics (Logger::Message & message)
    {
        kind_kernel_.log_metrics(message);
    }

private:

    struct Task
    {
        std::vector<char> raw;
        protobuf::Row row;
        std::vector<protobuf::ProductModel::Value> partial_values;
        AtomicArray<uint_fast64_t> groupids;
        bool add;
    };

    struct ThreadState
    {
        rng_t rng;
        VectorFloat scores;
        size_t position;
    };

    template<class Fun>
    void add_thread (size_t stage_number, const Fun & fun);

    void start_threads (size_t parser_threads);
    void start_kind_threads ();

    const bool proposer_stage_;
    const size_t stage_count_;
    Pipeline<Task, ThreadState> pipeline_;
    CrossCat & cross_cat_;
    StreamInterval & rows_;
    Assignments & assignments_;
    KindKernel & kind_kernel_;
    size_t kind_count_;
    rng_t & rng_;
};

} // namespace loom
