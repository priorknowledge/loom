#pragma once

#include <vector>
#include <atomic>
#include <tbb/concurrent_queue.h>
#include "common.hpp"

namespace loom
{

//----------------------------------------------------------------------------
// Single Producer Single Consumer Queue

template<class Message>
class RecyclingQueue
{
public:

    RecyclingQueue () : capacity_(0) {}

    ~RecyclingQueue ()
    {
        LOOM_ASSERT1(inactive(), "queue is active at destruction");
        Message * message;
        while (freed_.try_pop(message)) {
            delete message;
        }
    }

    bool inactive () const
    {
        return queue_.size() == 0 and freed_.size() == capacity_;
    }

    void unsafe_set_capacity (size_t capacity)
    {
        LOOM_ASSERT1(inactive(), "cannot set capacity when queue is active");
        while (capacity_ > capacity) {
            delete freed_.pop();
            --capacity_;
        }
        freed_.set_capacity(capacity);
        queue_.set_capacity(capacity);
        while (capacity_ < capacity) {
            freed_.push(new Message());
            ++capacity_;
        }
    }

    Message * producer_alloc ()
    {
        LOOM_ASSERT1(capacity_, "cannot use zero-capacity queue");
        Message * message;
        freed_.pop(message);
        return message;
    }

    void producer_send (Message * message)
    {
        queue_.push(message);
    }

    Message * consumer_recv ()
    {
        Message * message;
        queue_.pop(message);
        return message;
    }

    void consumer_free (Message * message)
    {
        freed_.push(message);
    }

private:

    typedef tbb::concurrent_bounded_queue<Message *> Queue_;
    Queue_ queue_;
    Queue_ freed_;
    size_t capacity_;
};

//----------------------------------------------------------------------------
// Single Producer Parallel Consumer Queue

template<class Message>
class ParallelRecyclingQueue
{
public:

    struct Envelope : noncopyable
    {
        Envelope () : ref_count(0) {}
        Message message;
    private:
        std::atomic<uint_fast64_t> ref_count;
    };

    ParallelRecyclingQueue () : capacity_(0) {}

    ~ParallelRecyclingQueue ()
    {
        LOOM_ASSERT1(inactive(), "queue is active at destruction");
        Envelope * envelope;
        while (freed_.try_pop(envelope)) {
            delete envelope;
        }
    }

    bool inactive () const
    {
        if (freed_.size() != capacity_) {
            return false;
        }
        for (const auto & queue : queues_) {
            if (queue.size() != 0) {
                return false;
            }
        }
        return true;
    }

    size_t size () const { return queues_.size(); }

    void unsafe_resize (size_t size)
    {
        LOOM_ASSERT1(inactive(), "cannot resize when queue is active");
        queues_.resize(size);
    }

    void unsafe_set_capacity (size_t capacity)
    {
        LOOM_ASSERT1(inactive(), "cannot set capacity when queue is active");
        while (capacity_ > capacity) {
            delete freed_.pop();
            --capacity_;
        }
        freed_.set_capacity(capacity);
        for (auto & queue : queues_) {
            queue.set_capacity(capacity);
        }
        while (capacity_ < capacity) {
            freed_.push(new Message());
            ++capacity_;
        }
    }

    Envelope * producer_alloc ()
    {
        LOOM_ASSERT1(capacity_, "cannot use zero-capacity queue");
        Envelope * envelope;
        freed_.pop(envelope);
        if (LOOM_DEBUG_LEVEL >= 1) {
            auto ref_count = envelope->ref_count.load();
            LOOM_ASSERT_EQ(ref_count, 0);
        }
        return envelope;
    }

    void producer_send (Envelope * envelope)
    {
        envelope.ref_count.store(queues_.size(), std::memory_order_acq_rel);
        for (auto & queue : queues_) {
            queue.push(envelope);
        }
    }

    Envelope * consumer_recv (size_t i)
    {
        LOOM_ASSERT1(i < queues_.size(), "out of bounds: " << i);
        Envelope * envelope;
        queues_[i].pop(envelope);
        return envelope;
    }

    void consumer_free (size_t i, Envelope * envelope)
    {
        LOOM_ASSERT1(i < queues_.size(), "out of bounds: " << i);
        if (envelope->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            freed_.push(envelope);
        }
    }

private:

    typedef tbb::concurrent_bounded_queue<Envelope *> Queue_;
    std::vector<Queue_> queues_;
    Queue_ freed_;
    size_t capacity_;
};

} // namespace loom
