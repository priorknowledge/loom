#pragma once

#include <loom/common.hpp>
#include <loom/assignments.hpp>

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
public:

    enum { max_extra_passes = 1000000 };

    AnnealingSchedule (
            double extra_passes) :
        add_rate_(1.0 + extra_passes),
        remove_rate_(extra_passes),
        state_(add_rate_)
    {
        LOOM_ASSERT_LE(0, extra_passes);
        LOOM_ASSERT_LE(extra_passes, max_extra_passes);
        LOOM_ASSERT(remove_rate_ < add_rate_, "underflow");
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

private:

    const double add_rate_;
    const double remove_rate_;
    double state_;
};


//----------------------------------------------------------------------------
// Batched Annealing Schedule
//
// Batch processes whenever data is completely fresh.

class BatchedAnnealingSchedule
{
public:

    BatchedAnnealingSchedule (
            double extra_passes,
            size_t initial_assigned_count) :
        schedule_(extra_passes),
        stale_count_(initial_assigned_count),
        fresh_count_(0)
    {
    }

    enum Action { add, remove, process_batch };

    Action next_action ()
    {
        if (LOOM_UNLIKELY(stale_count_ == 0) and
            LOOM_LIKELY(fresh_count_ > 0))
        {
            stale_count_ = fresh_count_;
            fresh_count_ = 0;
            return process_batch;
        } else if (schedule_.next_action_is_add()) {
            ++fresh_count_;
            return add;
        } else {
            LOOM_ASSERT1(stale_count_, "programmer error");
            --stale_count_;
            return remove;
        }
    }

private:

    AnnealingSchedule schedule_;
    size_t stale_count_;
    size_t fresh_count_;
};

} // namepace loom
