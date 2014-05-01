#pragma once

#include "common.hpp"
#include "assignments.hpp"

namespace loom
{

// Annealing Schedule.
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
        LOOM_ASSERT_LT(0, extra_passes);
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

class FlushingAnnealingSchedule
{
public:

    FlushingAnnealingSchedule (
            double extra_passes,
            size_t initial_assigned_count) :
        schedule_(extra_passes),
        pending_count_(initial_assigned_count),
        flushed_count_(0)
    {
    }

    bool next_action_is_add ()
    {
        if (schedule_.next_action_is_add()) {
            ++pending_count_;
            return true;
        } else {
            if (flushed_count_) {
                --flushed_count_;
            }
            return false;
        }
    }

    bool time_to_flush ()
    {
        if (DIST_UNLIKELY(flushed_count_ == 0) and
            DIST_LIKELY(pending_count_ > 0))
        {
            flushed_count_ = pending_count_;
            pending_count_ = 0;
            return true;
        } else {
            return false;
        }
    }

private:

    AnnealingSchedule schedule_;
    size_t pending_count_;
    size_t flushed_count_;
};

} // namepace loom
