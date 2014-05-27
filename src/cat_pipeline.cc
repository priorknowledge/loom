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
    thread.position = 0;
    thread.rng.seed(rng_());
    pipeline_.unsafe_add_thread(stage_number, thread, fun);
}

void CatPipeline::start_threads (size_t parser_threads)
{
    // unzip
    add_thread(0, [this](Task & task, const ThreadState &){
        if (task.add) {
            rows_.read_unassigned(task.raw);
        }
    });
    add_thread(0, [this](Task & task, const ThreadState &){
        if (not task.add) {
            rows_.read_assigned(task.raw);
        }
    });

    // parse
    LOOM_ASSERT_LT(0, parser_threads);
    for (size_t i = 0; i < parser_threads; ++i) {
        add_thread(1,
            [i, this, parser_threads](Task & task, ThreadState & thread){
            if (++thread.position % parser_threads == i) {
                task.row.ParseFromArray(task.raw.data(), task.raw.size());
                cross_cat_.value_split(task.row.data(), task.partial_values);
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
                    task.partial_values[i],
                    thread.scores,
                    groupids,
                    thread.rng);
            } else {
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
