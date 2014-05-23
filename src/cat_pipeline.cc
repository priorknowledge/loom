#include <loom/cat_pipeline.hpp>

#if 0
#define LOOM_DEBUG_TASK(message) LOOM_DEBUG(thread.position << ' ' << message);
#else
#define LOOM_DEBUG_TASK(message) \
    {if(0){ LOOM_DEBUG(thread.position << ' ' << message); }}
#endif

#if 0
#define LOOM_DEBUG_MUTEX std::lock_guard<std::mutex> debug_lock(debug_mutex_);
#else
#define LOOM_DEBUG_MUTEX
#endif

namespace loom
{

struct CatPipeline::ThreadState
{
    size_t position;
    VectorFloat scores;
    rng_t rng;
};

template<class Fun>
inline void CatPipeline::add_thread (size_t stage_number, const Fun & fun)
{
    auto seed = rng_();
    threads_.push_back(std::thread([this, stage_number, seed, fun](){
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
        {2, parser_count, splitter_count, 1, cross_cat.kinds.size()}),
    threads_(),
    cross_cat_(cross_cat),
    rows_(rows),
    assignments_(assignments),
    cat_kernel_(cat_kernel),
    rng_(rng)
{
    start_threads();
}

void CatPipeline::start_threads ()
{
    threads_.reserve(queue_.thread_count());

    // unzip
    add_thread(0, [this](Task & task, const ThreadState & thread){
        if (task.add) {
            LOOM_DEBUG_TASK("read_unassigned");
            rows_.read_unassigned(task.raw);
        }
    });
    add_thread(0, [this](Task & task, const ThreadState & thread){
        if (not task.add) {
            LOOM_DEBUG_TASK("read_assigned");
            rows_.read_assigned(task.raw);
        }
    });

    // parse
    static_assert(parser_count > 0, "no parsers");
    for (size_t i = 0; i < parser_count; ++i) {
        add_thread(1, [i](Task & task, const ThreadState & thread){
            if (thread.position % parser_count == i) {
                LOOM_DEBUG_TASK("parse " << i);
                task.row.Clear();
                task.row.ParseFromArray(task.raw.data(), task.raw.size());
            }
        });
    }

    // split
    static_assert(splitter_count > 0, "no splitters");
    for (size_t i = 0; i < splitter_count; ++i) {
        add_thread(2, [i, this](Task & task, const ThreadState & thread){
            if (thread.position % splitter_count == i) {
                LOOM_DEBUG_TASK("split " << i);
                cross_cat_.value_split(task.row.data(), task.partial_values);
            }
        });
    }

    // add/remove id
    auto & rowids = assignments_.rowids();
    add_thread(3, [&rowids](const Task & task, ThreadState & thread){
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

    // add/remove data
    LOOM_ASSERT(not cross_cat_.kinds.empty(), "no kinds");
    for (size_t i = 0; i < cross_cat_.kinds.size(); ++i) {
        auto & kind = cross_cat_.kinds[i];
        auto & groupids = assignments_.groupids(i);
        add_thread(4,
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
