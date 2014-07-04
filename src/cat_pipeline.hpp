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
#include <loom/cat_kernel.hpp>
#include <loom/pipeline.hpp>

namespace loom
{

class CatPipeline
{
public:

    enum { stage_count = 3 };

    CatPipeline (
            const protobuf::Config::Kernels::Cat & config,
            CrossCat & cross_cat,
            StreamInterval & rows,
            Assignments & assignments,
            CatKernel & cat_kernel,
            rng_t & rng);

    void add_row ()
    {
        pipeline_.start([](Task & task){ task.add = true; });
    }

    void remove_row ()
    {
        pipeline_.start([](Task & task){ task.add = false; });
    }

    void wait () { pipeline_.wait(); }

private:

    struct Task
    {
        bool add;
        std::vector<char> raw;
        protobuf::Row row;
        std::vector<ProductModel::Value> partial_values;
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

    Pipeline<Task, ThreadState> pipeline_;
    CrossCat & cross_cat_;
    StreamInterval & rows_;
    Assignments & assignments_;
    CatKernel & cat_kernel_;
    rng_t & rng_;
};

} // namespace loom
