#pragma once

#include <vector>
#include <atomic>
#include <tbb/concurrent_queue.h>
#include "common.hpp"

namespace loom
{

template<class Message>
class ParallelQueue
{
public:

    struct Envelope : noncopyable
    {
        Envelope () : ref_count(0) {}
        Message message;
    private:
        std::atomic<uint_fast64_t> ref_count;
        friend class ParallelQueue<Message>;
    };

    ParallelQueue () : capacity_(0) {}

    ~ParallelQueue ()
    {
        assert_inactive();
        Envelope * envelope;
        while (freed_.try_pop(envelope)) {
            delete envelope;
        }
    }

    void assert_inactive () const
    {
        if (LOOM_DEBUG_LEVEL >= 2) {
            LOOM_ASSERT_EQ(freed_.size(), capacity_);
            for (const auto & queue : queues_) {
                LOOM_ASSERT_LE(queue.size(), 0); // FIXME assert == -1?
            }
        }
    }

    size_t size () const { return queues_.size(); }

    void unsafe_resize (size_t size)
    {
        assert_inactive();
        queues_.resize(size);
    }

    void unsafe_set_capacity (size_t capacity)
    {
        assert_inactive();
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
        LOOM_ASSERT2(capacity_, "cannot use zero-capacity queue");
        Envelope * envelope;
        freed_.pop(envelope);
        if (LOOM_DEBUG_LEVEL >= 1) {
            auto ref_count = envelope->ref_count.load();
            LOOM_ASSERT_EQ(ref_count, 0);
        }
        return envelope;
    }

    void producer_send (Envelope * envelope, size_t worker_count)
    {
        LOOM_ASSERT2(
            worker_count <= queues_.size(),
            "too many workers " << worker_count);
        LOOM_ASSERT2(envelope, "got null envelope from producer");
        envelope.ref_count.store(worker_count, std::memory_order_acq_rel);
        for (size_t i = 0; i < worker_count; ++i) {
            queues_[i].push(envelope);
        }
    }

    void producer_hangup (size_t i)
    {
        LOOM_ASSERT2(i < queues_.size(), "out of bounds: " << i);
        queues_[i].push(nullptr);
    }

    const Envelope * consumer_receive (size_t i)
    {
        LOOM_ASSERT2(i < queues_.size(), "out of bounds: " << i);
        Envelope * envelope;
        queues_[i].pop(envelope);
        return envelope;
    }

    void consumer_free (const Envelope * const_envelope)
    {
        Envelope * envelope = const_cast<Envelope *>(const_envelope);
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
