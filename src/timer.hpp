// Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// - Neither the name of Salesforce.com nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <sys/time.h>
#include <loom/common.hpp>

namespace loom
{

typedef uint64_t usec_t;

inline double get_time_sec (const timeval & t)
{
    return t.tv_sec + 1e-6 * t.tv_usec;
}

inline usec_t get_time_usec (timeval & t)
{
    return 1000000UL * t.tv_sec + t.tv_usec;
}

inline usec_t current_time_usec ()
{
    timeval t;
    gettimeofday(&t, NULL);
    return get_time_usec(t);
}

class TimedScope
{
    usec_t & time_;

public:

    TimedScope  (usec_t & time) :
        time_(time)
    {
        time_ -= current_time_usec();
    }

    ~TimedScope ()
    {
        time_ += current_time_usec();
    }
};

class Timer
{
    usec_t total_;

public:

    Timer () : total_(0) {}

    void clear () { total_ = 0; }
    void start () { total_ -= current_time_usec(); }
    void stop () { total_ += current_time_usec(); }
    usec_t total () const { return total_; }

    class Scope
    {
        Timer & timer_;
    public:
        Scope  (Timer & timer) : timer_(timer) { timer_.start(); }
        ~Scope () { timer_.stop(); }
    };
};

} // namespace loom
