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
    struct Task
    {
        enum Action { exit, log_metrics, add, remove };
        Action action;
        std::vector<char> raw;
        protobuf::SparseRow row;
        std::vector<protobuf::ProductModel::SparseValue> partial_values;
    };

    void produce (Task::Action action)
    {
        queue_.produce([action](Task & task){ task.action = action; });
    }

public:

    enum {
        parser_count = 3,
        splitter_count = 3
    };

    CatPipeline (
            const protobuf::Config::Kernels::Cat & config,
            CrossCat & cross_cat,
            StreamInterval & rows,
            Assignments & assignments,
            CatKernel & cat_kernel,
            rng_t & rng);

    ~CatPipeline ();

    void add_row () { produce(Task::add); }
    void remove_row () { produce(Task::remove); }
    void wait () { queue_.wait(); }

    void log_metrics (Logger::Message & message);

private:

    struct ThreadState;

    template<class Fun>
    void add_thread (size_t stage_number, const Fun & fun);

    void start_threads ();

    PipelineQueue<Task> queue_;
    std::vector<std::thread> threads_;
    CrossCat & cross_cat_;
    StreamInterval & rows_;
    Assignments & assignments_;
    CatKernel & cat_kernel_;
    rng_t & rng_;

    std::vector<std::pair<usec_t, size_t>> times_;
    std::mutex times_mutex_;
};

inline void CatPipeline::log_metrics (Logger::Message & message)
{
    produce(Task::log_metrics);
    wait();

    auto & status = * message.mutable_kernel_status()->mutable_parcat();
    std::unique_lock<std::mutex> lock(times_mutex_);
    for (auto pair : times_) {
        status.add_times(pair.first);
        status.add_counts(pair.second);
    }
    times_.clear();
}

} // namespace loom
