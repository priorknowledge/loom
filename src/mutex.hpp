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
