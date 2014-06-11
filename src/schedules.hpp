#pragma once

#include <limits>
#include <loom/common.hpp>
#include <loom/assignments.hpp>
#include <loom/protobuf.hpp>
#include <loom/timer.hpp>

namespace loom
{

//----------------------------------------------------------------------------
// Annealing Schedule
//
// Let N be the number of extra_passes, i.e.
// the number of passes through data beyond a single greedy append-only pass.
// Then the final ratio of ADD to REMOVE actions is given by
//
//   "total dataset size" = ADD = (1 + N) (ADD - REMOVE)
//
// whence
//
//             ADD            N
//   REMOVE = ----- - ADD = ----- ADD
//            1 + N         1 + N  
//
// yielding relative rates
//
//   REMOVE     N
//   ------ = -----
//    ADD     1 + N

class AnnealingSchedule
{
    double add_rate_;
    double remove_rate_;
    double state_;

public:

    enum { max_extra_passes = 1000000 };

    void set_extra_passes (double extra_passes)
    {
        LOOM_ASSERT_LE(0, extra_passes);
        LOOM_ASSERT_LE(extra_passes, max_extra_passes);
        add_rate_ = 1.0 + extra_passes;
        remove_rate_ = extra_passes;
        LOOM_ASSERT(remove_rate_ < add_rate_, "underflow");
    }

    AnnealingSchedule (const protobuf::Config::Schedule & config)
    {
        set_extra_passes(config.extra_passes());
        state_ = add_rate_;
    }

    void load (const protobuf::Checkpoint::Schedule & checkpoint)
    {
        state_ = checkpoint.annealing_state();
    }

    void dump (protobuf::Checkpoint::Schedule & checkpoint)
    {
        checkpoint.set_annealing_state(state_);
    }

    bool next_action_is_add ()
    {
        if (state_ >= 0) {
            state_ -= remove_rate_;
            return true;
        } else {
            state_ += add_rate_;
            return false;
        }
    }
};

//----------------------------------------------------------------------------
/** Accelerating Annealing Schedule
 *
 * This is a piecewise linear schedule in log(row count)-log(passes) space:
 *
 *          passes  ^
 *                  |
 *                  |
 * 1 + extra_passes +======*
 *                  |       \\
 *                  |        \\
 *                  |         \\
 *                1 +------+----+=====> row count
 *                  |      |    |
 *                  1    small big
 *                        1e3  1e9 typically
 */

class AcceleratingSchedule
{
    const double extra_passes_;
    const double small_data_size_;
    const double big_data_size_;

public:

    AcceleratingSchedule (const protobuf::Config::Schedule & config) :
        extra_passes_(config.extra_passes()),
        small_data_size_(config.small_data_size()),
        big_data_size_(config.big_data_size())
    {
        LOOM_ASSERT_LE(0, small_data_size_);
        LOOM_ASSERT_LE(small_data_size_, big_data_size_);
        LOOM_ASSERT(
            std::isfinite(big_data_size_) or std::isinf(small_data_size_),
            "big_data_size infinite but small_data_size finite");
    }

    void load (const protobuf::Checkpoint::Schedule &) {}
    void dump (protobuf::Checkpoint::Schedule &) {}

    double extra_passes (size_t row_count)
    {
        if (row_count <= small_data_size_) {
            return extra_passes_;
        } else if (row_count <= big_data_size_) {
            double power = log(big_data_size_ / row_count)
                         / log(big_data_size_ / small_data_size_);
            double passes = pow(1.0 + extra_passes_, power);
            if (passes > 2.0) {  // avoid passes barely greater than 1.0
                return passes - 1.0;
            } else {
                return 0.0;
            }
        } else {
            return 0.0;
        }
    }
};

//----------------------------------------------------------------------------
// Batching Schedule
//
// Batch processes whenever data is completely fresh.

class BatchingSchedule
{
    size_t stale_count_;
    size_t fresh_count_;

public:

    BatchingSchedule (const protobuf::Config::Schedule &) :
        stale_count_(0),
        fresh_count_(0)
    {}

    void load (const protobuf::Checkpoint::Schedule & checkpoint)
    {
        stale_count_ = checkpoint.row_count();
        fresh_count_ = 0;
    }

    void dump (protobuf::Checkpoint::Schedule & checkpoint)
    {
        checkpoint.set_row_count(stale_count_ + fresh_count_);
    }

    void add () { ++fresh_count_; }

    void remove () { --stale_count_; }

    bool test ()
    {
        if (LOOM_UNLIKELY(not stale_count_) and LOOM_LIKELY(fresh_count_)) {
            stale_count_ = fresh_count_;
            fresh_count_ = 0;
            return true;
        } else {
            return false;
        }
    }
};

//----------------------------------------------------------------------------
// Kernel Disabling Schedule

class KernelDisablingSchedule
{
    const size_t max_reject_iters_;
    size_t reject_iters_;

public:

    KernelDisablingSchedule (const protobuf::Config::Schedule & config) :
        max_reject_iters_(config.max_reject_iters()),
        reject_iters_(0)
    {
    }

    void load (const protobuf::Checkpoint::Schedule & checkpoint)
    {
        reject_iters_ = checkpoint.reject_iters();
    }

    void dump (protobuf::Checkpoint::Schedule & checkpoint)
    {
        checkpoint.set_reject_iters(reject_iters_);
    }

    void run (bool accepted)
    {
        if (accepted) {
            reject_iters_ = 0;
        } else {
            reject_iters_ += 1;
        }
    }

    bool test () const
    {
        return reject_iters_ <= max_reject_iters_;
    }
};

//----------------------------------------------------------------------------
// Checkpointing Schedule

class CheckpointingSchedule
{
    const usec_t stop_usec_;

public:

    CheckpointingSchedule (const protobuf::Config::Schedule & config) :
        stop_usec_(
            current_time_usec() +
            static_cast<usec_t>(config.checkpoint_period_sec() * 1e6))
    {
    }

    void load (const protobuf::Checkpoint::Schedule &) {}
    void dump (protobuf::Checkpoint::Schedule &) {}

    bool test () const
    {
        return current_time_usec() >= stop_usec_;
    }
};

//----------------------------------------------------------------------------
// Combined Schedule

struct CombinedSchedule
{
    AnnealingSchedule annealing;
    AcceleratingSchedule accelerating;
    BatchingSchedule batching;
    KernelDisablingSchedule disabling;
    CheckpointingSchedule checkpointing;

    CombinedSchedule (
            const protobuf::Config::Schedule & config) :
        annealing(config),
        accelerating(config),
        batching(config),
        disabling(config),
        checkpointing(config)
    {
    }

    void load (const protobuf::Checkpoint::Schedule & checkpoint)
    {
        annealing.load(checkpoint);
        accelerating.load(checkpoint);
        batching.load(checkpoint);
        disabling.load(checkpoint);
        checkpointing.load(checkpoint);
    }

    void dump (protobuf::Checkpoint::Schedule & checkpoint)
    {
        annealing.dump(checkpoint);
        accelerating.dump(checkpoint);
        batching.dump(checkpoint);
        disabling.dump(checkpoint);
        checkpointing.dump(checkpoint);
    }
};


} // namepace loom
