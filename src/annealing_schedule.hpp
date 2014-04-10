#pragma once

#include "common.hpp"

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

    AnnealingSchedule (double extra_passes) :
        add_rate_(1.0 + extra_passes),
        remove_rate_(extra_passes),
        state_(add_rate_)
    {
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

} // namepace loom
