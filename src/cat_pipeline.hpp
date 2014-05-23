#pragma once

#include <atomic>
#include <thread>
#include <loom/common.hpp>
#include <loom/cross_cat.hpp>
#include <loom/assignments.hpp>
#include <loom/stream_interval.hpp>
#include <loom/cat_kernel.hpp>
#include <loom/shared_queue.hpp>

namespace loom
{

class CatPipeline
{
    struct Task
    {
        bool exit;
        bool add;
        std::vector<char> raw;
        protobuf::SparseRow row;
        std::vector<protobuf::ProductModel::SparseValue> partial_values;

        Task () : exit(false) {}
    };

    struct ThreadState
    {
        size_t position;
        VectorFloat scores;
        rng_t rng;
    };

    pipeline::SharedQueue<Task> queue_;
    std::vector<std::thread> threads_;
    rng_t & rng_;
    std::mutex debug_mutex_;

    template<class Fun>
    void add_thread (size_t stage_number, const Fun & fun);

public:

    enum { parser_count = 3, splitter_count = 3 };

    CatPipeline (
            const protobuf::Config::Kernels::Cat & config,
            CrossCat & cross_cat,
            StreamInterval & rows,
            Assignments & assignments,
            CatKernel & cat_kernel,
            rng_t & rng);

    ~CatPipeline ();

    void add_row ()
    {
        queue_.produce([](Task & task){ task.add = true; });
    }

    void remove_row ()
    {
        queue_.produce([](Task & task){ task.add = false; });
    }

    void wait () { queue_.wait(); }
};

} // namespace loom
