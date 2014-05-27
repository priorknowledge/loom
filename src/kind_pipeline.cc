#include <loom/kind_pipeline.hpp>

namespace loom
{

KindPipeline::KindPipeline (
        const protobuf::Config::Kernels::Kind & config,
        CrossCat & cross_cat,
        StreamInterval & rows,
        Assignments & assignments,
        KindKernel & kind_kernel,
        rng_t & rng) :
    pipeline_(config.row_queue_capacity(), stage_count),
    cross_cat_(cross_cat),
    rows_(rows),
    assignments_(assignments),
    kind_kernel_(kind_kernel),
    kind_count_(0),
    rng_(rng)
{
    start_threads();
}

template<class Fun>
inline void KindPipeline::add_thread (
        size_t stage_number,
        const Fun & fun)
{
    ThreadState thread;
    thread.position = 0;
    thread.rng.seed(rng_());
    pipeline_.unsafe_add_thread(stage_number, thread, fun);
}

void KindPipeline::start_threads ()
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
    static_assert(parser_count > 0, "no parsers");
    for (size_t i = 0; i < parser_count; ++i) {
        add_thread(1, [i, this](Task & task, ThreadState & thread){
            if (++thread.position % parser_count == i) {
                task.row.Clear();
                task.row.ParseFromArray(task.raw.data(), task.raw.size());
                cross_cat_.value_split(task.row.data(), task.partial_values);
            }
        });
    }

    // cat kernel
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

    start_kind_threads();

    pipeline_.validate();
}

void KindPipeline::start_kind_threads ()
{
    size_t start_count = kind_count_;
    size_t target_count = cross_cat_.kinds.size();
    for (size_t i = start_count; i < target_count; ++i, ++kind_count_) {

        // cat kernel
        add_thread(2, [i, this](Task & task, ThreadState & thread){
            if (i < cross_cat_.kinds.size()) {
                if (task.add) {
                    task.groupid = kind_kernel_.add_to_cross_cat(
                        i,
                        task.partial_values[i],
                        thread.scores,
                        thread.rng);
                } else {
                    task.groupid = kind_kernel_.remove_from_cross_cat(
                        i,
                        task.partial_values[i],
                        thread.rng);
                }
            }
        });

        // kind proposer
        add_thread(3, [i, this](const Task & task, ThreadState & thread){
            if (i < cross_cat_.kinds.size()) {
                if (task.add) {
                    kind_kernel_.add_to_kind_proposer(
                        i,
                        task.groupid,
                        task.row.data(),
                        thread.rng);
                } else {
                    kind_kernel_.remove_from_kind_proposer(
                        i,
                        task.groupid,
                        thread.rng);
                }
            }
        });
    }
}

} // namespace loom
