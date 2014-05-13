#pragma once

#include <sys/time.h>

namespace loom
{

typedef uint64_t usec_t;

inline double get_time_sec (const timeval & t)
{
    return t.tv_sec + 1e-6 * t.tv_usec;
}

inline usec_t get_time_usec (timeval & t)
{
    return 1000000L * t.tv_sec + t.tv_usec;
}

inline usec_t current_time_usec ()
{
    timeval t;
    gettimeofday(&t, NULL);
    return get_time_usec(t);
}

// This is a C++ port of kmetrics.metrics.Timer
class Timer
{
    usec_t started_at_;
    usec_t elapsed_;
    usec_t total_;

public:

    Timer () :
        started_at_(current_time_usec()),
        elapsed_(0),
        total_(0)
    {
    }

    void start ()
    {
        started_at_ = current_time_usec();
    }

    void stop ()
    {
        if (LOOM_DEBUG_LEVEL >= 1) {
            LOOM_ASSERT(started_at_, "called stop but not start");
        }
        elapsed_ = current_time_usec() - started_at_;
        total_ += elapsed_;
        if (LOOM_DEBUG_LEVEL >= 1) {
            started_at_ = 0;
        }
    }

    class Scope
    {
        Timer & timer_;
    public:
        Scope  (Timer & timer) : timer_(timer) { timer_.start(); }
        ~Scope () { timer_.stop(); }
    };

    double elapsed () const { return elapsed_ * 1e-6; }
    double total () const { return total_ * 1e-6; }
};

} // namespace loom
