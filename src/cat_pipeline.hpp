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
public:

    enum { stage_count = 3 };

    CatPipeline (
            const protobuf::Config::Kernels::Cat & config,
            CrossCat & cross_cat,
            StreamInterval & rows,
            Assignments & assignments,
            CatKernel & cat_kernel,
            rng_t & rng);

    void add_row ()
    {
        pipeline_.start([](Task & task){ task.add = true; });
    }

    void remove_row ()
    {
        pipeline_.start([](Task & task){ task.add = false; });
    }

    void wait () { pipeline_.wait(); }

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

    Pipeline<Task, ThreadState> pipeline_;
    CrossCat & cross_cat_;
    StreamInterval & rows_;
    Assignments & assignments_;
    CatKernel & cat_kernel_;
    rng_t & rng_;
};

} // namespace loom
