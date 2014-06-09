#pragma once

#include <thread>
#include <loom/common.hpp>
#include <loom/mutex.hpp>
#include <loom/cross_cat.hpp>
#include <loom/assignments.hpp>
#include <loom/stream_interval.hpp>
#include <loom/kind_kernel.hpp>
#include <loom/pipeline.hpp>

namespace loom
{

class KindPipeline
{
public:

    enum { stage_count = 5 };

    KindPipeline (
            const protobuf::Config::Kernels::Kind & config,
            CrossCat & cross_cat,
            StreamInterval & rows,
            Assignments & assignments,
            KindKernel & kind_kernel,
            rng_t & rng);

    void add_row ()
    {
        pipeline_.start([](Task & task){ task.add = true; });
    }

    void remove_row ()
    {
        pipeline_.start([](Task & task){ task.add = false; });
    }

    void wait ()
    {
        pipeline_.wait();
    }

    bool try_run ()
    {
        bool changed = kind_kernel_.try_run();
        if (changed) {
            start_kind_threads();
            pipeline_.validate();
        }
        return changed;
    }

    void update_hypers ()
    {
        kind_kernel_.update_hypers();
    }

    void log_metrics (Logger::Message & message)
    {
        kind_kernel_.log_metrics(message);
    }

private:

    struct Task
    {
        bool add;
        std::vector<char> raw;
        protobuf::SparseRow row;
        std::vector<protobuf::ProductModel::SparseValue> partial_values;
    };

    struct ThreadState
    {
        rng_t rng;
        VectorFloat scores;
        size_t position;
    };

    template<class Fun>
    void add_thread (size_t stage_number, const Fun & fun);

    void start_threads (size_t parser_threads);
    void start_kind_threads ();

    Pipeline<Task, ThreadState> pipeline_;
    CrossCat & cross_cat_;
    StreamInterval & rows_;
    Assignments & assignments_;
    KindKernel & kind_kernel_;
    size_t kind_count_;
    shared_mutex proposer_model_mutex_;
    rng_t & rng_;
};

} // namespace loom
