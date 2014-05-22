#include <loom/cat_kernel.hpp>

namespace loom
{

CatKernel::CatKernel (
        const protobuf::Config::Kernels::Cat & config,
        CrossCat & cross_cat) :
    cross_cat_(cross_cat),
    task_queue_(config.row_queue_capacity(), {0}),
    workers_(),
    partial_values_(),
    scores_(),
    timer_()
{
    LOOM_ASSERT_LT(0, config.empty_group_count());
}

CatKernel::~CatKernel ()
{
    if (not workers_.empty()) {
        task_queue_.produce([&](Task & task){
            task.action = Task::exit;
        });
        task_queue_.wait();
        for (auto & worker : workers_) {
            worker.join();
        }
    }
}

void CatKernel::wait (Assignments & assignments, rng_t & rng)
{
    if (workers_.empty()) {

        size_t queue_size = task_queue_.size();
        bool can_parallelize = (queue_size > 0);
        bool worth_parallelizing = (assignments.row_count() > queue_size);
        if (can_parallelize and worth_parallelizing) {
            const size_t consumer_position = 0;
            const size_t kind_count = cross_cat_.kinds.size();
            task_queue_.wait();
            task_queue_.unsafe_set_consumer_count(0, kind_count);
            workers_.reserve(kind_count);
            for (size_t k = 0; k < kind_count; ++k) {
                rng_t::result_type seed = rng();
                workers_.push_back(
                    std::thread(
                        &CatKernel::process_tasks,
                        this,
                        k,
                        consumer_position,
                        seed,
                        &assignments));
            }
        }

    } else {

        task_queue_.wait();
    }
}

void CatKernel::process_tasks (
        const size_t kindid,
        size_t consumer_position,
        rng_t::result_type seed,
        Assignments * assignments)
{
    auto & kind = cross_cat_.kinds[kindid];
    VectorFloat scores;
    auto & groupids = assignments->groupids(kindid);
    rng_t rng(seed);

    for (bool alive = true; LOOM_LIKELY(alive);) {
        task_queue_.consume(0, consumer_position++, [&](const Task & task) {
            const Value & partial_value = task.partial_values[kindid];
            switch (task.action) {
                case Task::add:
                    process_add_task(
                        kind,
                        partial_value,
                        scores,
                        groupids,
                        rng);
                    break;

                case Task::remove:
                    process_remove_task(kind, partial_value, groupids, rng);
                    break;

                case Task::exit:
                    alive = false;
                    break;
            }
        });
    }
}

} // namespace loom
