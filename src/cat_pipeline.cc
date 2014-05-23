#include <loom/cat_pipeline.hpp>

#if 1
#define LOOM_DEBUG_TASK(message) LOOM_DEBUG(thread.position << ' ' << message);
#else
#define LOOM_DEBUG_TASK(message)
#endif

#if 1
#define LOOM_DEBUG_MUTEX std::lock_guard<std::mutex> debug_lock(debug_mutex_);
#else
#define LOOM_DEBUG_MUTEX
#endif

namespace loom
{

template<class Fun>
inline void CatPipeline::add_thread (size_t stage_number, const Fun & fun)
{
    auto seed = rng_();
    threads_.push_back(std::thread([this, stage_number, seed, &fun](){
        ThreadState thread;
        thread.rng.seed(seed);
        thread.position = 0;
        for (bool alive = true; LOOM_LIKELY(alive);) {
            queue_.consume(stage_number, thread.position, [&](Task & task){
                if (LOOM_UNLIKELY(task.exit)) {
                    alive = false;
                } else {
                    LOOM_DEBUG_MUTEX
                    fun(task, thread);
                }
            });
            ++thread.position;
        }
    }));
}

CatPipeline::CatPipeline (
        const protobuf::Config::Kernels::Cat & config,
        CrossCat & cross_cat,
        StreamInterval & rows,
        Assignments & assignments,
        CatKernel & cat_kernel,
        rng_t & rng) :
    queue_(
        config.row_queue_capacity(),
        {2, parser_count, splitter_count, cross_cat.kinds.size()}),
    threads_(),
    rng_(rng)
{
    threads_.reserve(queue_.thread_count());

    // unzip
    add_thread(0, [&rows](Task & task, const ThreadState & thread){
        if (task.add) {
            LOOM_DEBUG_TASK("read_unassigned");
            rows.read_unassigned(task.raw);
        }
    });
    add_thread(0, [&rows](Task & task, const ThreadState & thread){
        if (not task.add) {
            LOOM_DEBUG_TASK("read_assigned");
            rows.read_assigned(task.raw);
        }
    });

    // parse
    static_assert(parser_count > 0, "no parsers");
    for (size_t i = 0; i < parser_count; ++i) {
        add_thread(1, [i](Task & task, const ThreadState & thread){
            if (thread.position % parser_count == 0) {
                LOOM_DEBUG_TASK("parse " << i);
                task.row.Clear();
                task.row.ParseFromArray(task.raw.data(), task.raw.size());
            }
        });
    }

    // split
    static_assert(splitter_count > 0, "no splitters");
    for (size_t i = 0; i < splitter_count; ++i) {
        add_thread(2, [i, &cross_cat](Task & task, const ThreadState & thread){
            if (thread.position % splitter_count == 0) {
                LOOM_DEBUG_TASK("split " << i);
                cross_cat.value_split(task.row.data(), task.partial_values);
            }
        });
    }

    // add/remove
    LOOM_ASSERT(not cross_cat.kinds.empty(), "no kinds");
    for (size_t i = 0; i < cross_cat.kinds.size(); ++i) {
        auto & kind = cross_cat.kinds[i];
        auto & rowids = assignments.rowids();
        auto & groupids = assignments.groupids(i);
        add_thread(3,
            [i, &kind, &rowids, &groupids, &cat_kernel]
            (const Task & task, ThreadState & thread)
        {
            if (task.add) {
                LOOM_DEBUG_TASK("add " << i);

                bool ok = rowids.try_push(task.row.id());
                LOOM_ASSERT1(ok, "duplicate row: " << task.row.id());

                cat_kernel.process_add_task(
                    kind,
                    task.partial_values[i],
                    thread.scores,
                    groupids,
                    thread.rng);

            } else {
                LOOM_DEBUG_TASK("add " << i);

                const auto rowid = rowids.pop();
                if (LOOM_DEBUG_LEVEL >= 1) {
                    LOOM_ASSERT_EQ(rowid, task.row.id());
                }

                cat_kernel.process_remove_task(
                    kind,
                    task.partial_values[i],
                    groupids,
                    thread.rng);
            }
        });
    }
}

CatPipeline::~CatPipeline ()
{
    queue_.produce([](Task & task){ task.exit = true; });
    queue_.wait();
    for (auto & thread : threads_) {
        thread.join();
    }
}

} // namespace loom
