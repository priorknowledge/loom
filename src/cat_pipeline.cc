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

#include <loom/cat_pipeline.hpp>

namespace loom
{

CatPipeline::CatPipeline (
        const protobuf::Config::Kernels::Cat & config,
        CrossCat & cross_cat,
        StreamInterval & rows,
        Assignments & assignments,
        CatKernel & cat_kernel,
        rng_t & rng) :
    pipeline_(config.row_queue_capacity(), stage_count),
    cross_cat_(cross_cat),
    rows_(rows),
    assignments_(assignments),
    cat_kernel_(cat_kernel),
    rng_(rng)
{
    start_threads(config.parser_threads());
}

template<class Fun>
inline void CatPipeline::add_thread (
        size_t stage_number,
        const Fun & fun)
{
    ThreadState thread;
    thread.rng.seed(rng_());
    pipeline_.unsafe_add_thread(stage_number, thread, fun);
}

void CatPipeline::start_threads (size_t parser_threads)
{
    // unzip
    add_thread(0, [this](Task & task, const ThreadState &){
        if (task.add) {
            task.parsed.clear();
            rows_.read_unassigned(task.raw);
        }
    });
    add_thread(0, [this](Task & task, const ThreadState &){
        if (not task.add) {
            task.parsed.clear();
            rows_.read_assigned(task.raw);
        }
    });

    // parse
    LOOM_ASSERT_LT(0, parser_threads);
    for (size_t i = 0; i < parser_threads; ++i) {
        add_thread(1,
            [i, this, parser_threads](Task & task, ThreadState & thread){
            if (not task.parsed.test_and_set()) {
                task.row.ParseFromArray(task.raw.data(), task.raw.size());
                cross_cat_.diff_split(
                    task.row.diff(),
                    task.partial_diffs,
                    thread.temp_values);
            }
        });
    }

    // add/remove
    auto & rowids = assignments_.rowids();
    add_thread(2, [&rowids](const Task & task, ThreadState &){
        if (task.add) {
            bool ok = rowids.try_push(task.row.id());
            LOOM_ASSERT1(ok, "duplicate row: " << task.row.id());
        } else {
            const auto rowid = rowids.pop();
            if (LOOM_DEBUG_LEVEL >= 1) {
                LOOM_ASSERT_EQ(rowid, task.row.id());
            }
        }
    });
    LOOM_ASSERT(not cross_cat_.kinds.empty(), "no kinds");
    for (size_t i = 0; i < cross_cat_.kinds.size(); ++i) {
        auto & kind = cross_cat_.kinds[i];
        auto & groupids = assignments_.groupids(i);
        add_thread(2,
            [i, this, &kind, &groupids]
            (const Task & task, ThreadState & thread)
        {
            if (task.add) {
                cat_kernel_.process_add_task(
                    kind,
                    task.partial_diffs[i],
                    thread.scores,
                    groupids,
                    thread.rng);
            } else {
                cat_kernel_.process_remove_task(
                    kind,
                    task.partial_diffs[i],
                    groupids,
                    thread.rng);
            }
        });
    }

    pipeline_.validate();
}

} // namespace loom
