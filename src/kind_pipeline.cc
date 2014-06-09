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
    start_threads(config.parser_threads());
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

void KindPipeline::start_threads (size_t parser_threads)
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

    // proposer model add
    add_thread(2, [this](const Task & task, ThreadState & thread){
        if (task.add) {
            std::unique_lock<shared_mutex> lock(proposer_model_mutex_);
            kind_kernel_.add_to_kind_proposer(task.row.data(), thread.rng);
        }
    });

    // mixture add/remove
    auto & rowids = assignments_.rowids();
    add_thread(3, [&rowids](const Task & task, ThreadState &){
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

    // proposer model remove
    add_thread(4, [this](const Task & task, ThreadState & thread){
        if (not task.add) {
            std::unique_lock<shared_mutex> lock(proposer_model_mutex_);
            kind_kernel_.remove_from_kind_proposer(task.row.data(), thread.rng);
        }
    });

    pipeline_.validate();
}

void KindPipeline::start_kind_threads ()
{
    while (kind_count_ < cross_cat_.kinds.size()) {
        size_t i = kind_count_++;

        // mixture add/remove
        add_thread(3, [i, this](const Task & task, ThreadState & thread){
            if (LOOM_LIKELY(i < cross_cat_.kinds.size())) {
                if (task.add) {

                    auto groupid = kind_kernel_.add_to_cross_cat(
                        i,
                        task.partial_values[i],
                        thread.scores,
                        thread.rng);

                    shared_lock<shared_mutex> lock(proposer_model_mutex_);
                    kind_kernel_.add_to_kind_proposer(
                        i,
                        groupid,
                        task.row.data(),
                        thread.rng);

                } else {

                    auto groupid = kind_kernel_.remove_from_cross_cat(
                        i,
                        task.partial_values[i],
                        thread.rng);

                    shared_lock<shared_mutex> lock(proposer_model_mutex_);
                    kind_kernel_.remove_from_kind_proposer(
                        i,
                        groupid,
                        thread.rng);
                }
            }
        });
    }
}

} // namespace loom
