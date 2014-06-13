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

#include <pthread.h>
#include <loom/common.hpp>

namespace loom
{

template<class Mutex>
struct shared_lock
{
    Mutex & mutex_;
public:
    shared_lock (Mutex & mutex) : mutex_(mutex) { mutex_.lock_shared(); }
    ~shared_lock () { mutex_.unlock_shared(); }
};

// This wraps pthread_wrlock, which is smaller & faster than
// boost::shared_mutex.
//
// adapted from:
// http://boost.2283326.n4.nabble.com/boost-shared-mutex-performance-td2659061.html
class shared_mutex
{
    pthread_rwlock_t rwlock_;

public:

    shared_mutex ()
    {
        int status = pthread_rwlock_init(&rwlock_, nullptr);
        LOOM_ASSERT1(status == 0, "pthread_rwlock_init failed");
    }

    ~shared_mutex ()
    {
        int status = pthread_rwlock_destroy(&rwlock_);
        LOOM_ASSERT1(status == 0, "pthread_rwlock_destroy failed");
    }

    void lock ()
    {
        int status = pthread_rwlock_wrlock(&rwlock_);
        LOOM_ASSERT1(status == 0, "pthread_rwlock_wrlock failed");
    }

    // glibc seems to be buggy; don't unlock more often than it has been locked
    // see http://sourceware.org/bugzilla/show_bug.cgi?id=4825
    void unlock ()
    {
        int status = pthread_rwlock_unlock(&rwlock_);
        LOOM_ASSERT1(status == 0, "pthread_rwlock_unlock failed");
    }

    void lock_shared ()
    {
        int status = pthread_rwlock_rdlock(&rwlock_);
        LOOM_ASSERT1(status == 0, "pthread_rwlock_rdlock failed");
    }

    void unlock_shared ()
    {
        unlock();
    }
};

} // namespace loom
