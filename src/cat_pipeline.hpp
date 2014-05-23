#pragma once

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

    void add_row () { queue_.produce([](Task & task){ task.add = true; }); }
    void remove_row () { queue_.produce([](Task & task){ task.add = false; }); }
    void wait () { queue_.wait(); }

private:

    struct ThreadState;

    template<class Fun>
    void add_thread (size_t stage_number, const Fun & fun);

    void start_threads ();

    pipeline::SharedQueue<Task> queue_;
    std::vector<std::thread> threads_;
    CrossCat & cross_cat_;
    StreamInterval & rows_;
    Assignments & assignments_;
    CatKernel & cat_kernel_;
    rng_t & rng_;
    std::mutex debug_mutex_;
};

} // namespace loom
