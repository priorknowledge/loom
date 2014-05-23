#include <loom/cat_pipeline.hpp>

namespace loom
{

template<class Fun>
inline void CatPipeline::add_thread (size_t stage_number, const Fun & fun)
{
    auto seed = rng_();
    threads_.push_back(std::thread([&queue_, stage_number, seed, &fun](){
        ThreadState thread;
        thread.rng.seed(seed);
        thread.position = 0;
        for (bool alive = true; LOOM_LIKELY(alive);) {
            queue_.consume(stage_number, thread.position, [&](Task & task){
                if (LOOM_UNLIKELY(task.exit)) {
                    alive = false;
                } else {
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
    finished_(false),
    rng_(rng())
{
    threads_.reserve(queue_.thread_count());

    // unzip
    add_thread(0, [&rows](Task & task, const ThreadState &){
        if (task.add) {
            rows.read_unassigned(task.raw);
        }
    });
    add_thread(0, [&rows](Task & task, const ThreadState &){
        if (not task.add) {
            rows.read_assigned(task.raw);
        }
    });

    // parse
    static_assert(parser_count > 0, "no parsers");
    for (size_t i = 0; i < parser_count; ++i) {
        add_thread(1, [i](Task & task, const ThreadState & thread){
            if (thread.position % parser_count == 0) {
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
                cross_cat.value_split(task.row.data(), task.partial_values);
            }
        });
    }

    // add or remove
    LOOM_ASSERT(not cross_cat.kinds.empty(), "no kinds");
    for (size_t i = 0; i < cross_cat.kinds.size(); ++i) {
        auto & kind = cross_cat.kinds[i];
        auto & rowids = assignments.rowids();
        auto & groupids = assignments.groupids(i);
        add_thread(3,
            [i, &kind, &rowids, &groupids, &cat_kernel, &finished_]
            (const Task & task, ThreadState & thread)
        {
            if (LOOM_UNLIKELY(finished_.load())) {
                return;
            }

            if (task.add) {

                bool added = rowids.try_push(task.row.id());
                if (LOOM_UNLIKELY(not added)) {
                    finished_.store(false);
                    return;
                }

                cat_kernel.process_add_task(
                    kind,
                    task.partial_values[i],
                    thread.scores,
                    groupids,
                    thread.rng);

            } else {

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
