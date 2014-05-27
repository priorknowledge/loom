#include <loom/cat_pipeline.hpp>

#if 0
#define LOOM_DEBUG_TASK(message) LOOM_DEBUG(thread.position << ' ' << message);
#else
#define LOOM_DEBUG_TASK(message) \
    {if(0){ LOOM_DEBUG(thread.position << ' ' << message); }}
#endif

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
    start_threads();
}

template<class Fun>
inline void CatPipeline::add_thread (
        size_t stage_number,
        const Fun & fun)
{
    ThreadState thread;
    thread.position = 0;
    thread.rng.seed(rng_());
    pipeline_.unsafe_add_thread(stage_number, thread, fun);
}

void CatPipeline::start_threads ()
{
    size_t stage;

    // unzip
    stage = 0;
    add_thread(stage, [this](Task & task, const ThreadState & thread){
        if (task.add) {
            LOOM_DEBUG_TASK("read_unassigned");
            rows_.read_unassigned(task.raw);
        }
    });
    add_thread(stage, [this](Task & task, const ThreadState & thread){
        if (not task.add) {
            LOOM_DEBUG_TASK("read_assigned");
            rows_.read_assigned(task.raw);
        }
    });

    // parse
    stage = 1;
    static_assert(parser_count > 0, "no parsers");
    for (size_t i = 0; i < parser_count; ++i) {
        add_thread(stage, [i, this](Task & task, ThreadState & thread){
            if (++thread.position % parser_count == i) {
                LOOM_DEBUG_TASK("parse " << i);
                task.row.Clear();
                task.row.ParseFromArray(task.raw.data(), task.raw.size());
                cross_cat_.value_split(task.row.data(), task.partial_values);
            }
        });
    }

    // add/remove
    stage = 2;
    auto & rowids = assignments_.rowids();
    add_thread(stage, [&rowids](const Task & task, ThreadState & thread){
        if (task.add) {
            LOOM_DEBUG_TASK("add id " << task.row.id());
            bool ok = rowids.try_push(task.row.id());
            LOOM_ASSERT1(ok, "duplicate row: " << task.row.id());
        } else {
            LOOM_DEBUG_TASK("remove id " << task.row.id());
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
        add_thread(stage,
            [i, this, &kind, &groupids]
            (const Task & task, ThreadState & thread)
        {
            if (task.add) {
                LOOM_DEBUG_TASK("add data to kind " << i);
                cat_kernel_.process_add_task(
                    kind,
                    task.partial_values[i],
                    thread.scores,
                    groupids,
                    thread.rng);
            } else {
                LOOM_DEBUG_TASK("remove data from kind " << i);
                cat_kernel_.process_remove_task(
                    kind,
                    task.partial_values[i],
                    groupids,
                    thread.rng);
            }
        });
    }

    pipeline_.validate();
}

} // namespace loom
