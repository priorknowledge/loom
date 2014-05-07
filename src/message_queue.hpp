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
        assert_ready();
        Envelope * envelope;
        while (freed_.try_pop(envelope)) {
            delete envelope;
        }
    }

    size_t size () const { return queues_.size(); }

    void unsafe_resize (size_t size)
    {
        assert_ready();
        queues_.resize(size);
        for (auto & queue : queues_) {
            queue.set_capacity(capacity_);
        }
    }

    size_t pending_count () const { return capacity_ - freed_.size(); }

    void assert_ready () const
    {
        if (LOOM_DEBUG_LEVEL >= 2) {
            LOOM_ASSERT_EQ(pending_count(), 0);
            for (const auto & queue : queues_) {
                LOOM_ASSERT_LE(queue.size(), 0); // FIXME assert == -1?
                LOOM_ASSERT_EQ(queue.capacity(), capacity_);
            }
        }
    }

    void unsafe_set_capacity (size_t capacity)
    {
        assert_ready();
        while (capacity_ > capacity) {
            Envelope * envelope;
            freed_.pop(envelope);
            delete envelope;
            --capacity_;
        }
        freed_.set_capacity(capacity);
        for (auto & queue : queues_) {
            queue.set_capacity(capacity_);
        }
        while (capacity_ < capacity) {
            freed_.push(new Envelope());
            ++capacity_;
        }
    }

    Envelope * producer_alloc ()
    {
        LOOM_ASSERT2(capacity_, "cannot use zero-capacity queue");

        Envelope * envelope;
        freed_.pop(envelope);
        if (LOOM_DEBUG_LEVEL >= 2) {
            auto ref_count = envelope->ref_count.load();
            LOOM_ASSERT_EQ(ref_count, 0);
        }
        return envelope;
    }

    void producer_send (Envelope * envelope, size_t consumer_count)
    {
        LOOM_ASSERT2(consumer_count, "message sent to zero consumers");
        LOOM_ASSERT2(
            consumer_count <= queues_.size(),
            "too many consumers " << consumer_count);
        LOOM_ASSERT2(envelope, "got null envelope from producer");

        envelope->ref_count.store(consumer_count, std::memory_order_acq_rel);
        for (size_t i = 0; i < consumer_count; ++i) {
            queues_[i].push(envelope);
        }
    }

    void producer_wait ()
    {
        if (LOOM_DEBUG_LEVEL >= 2) {
            LOOM_ASSERT_EQ(freed_.size(), 0);
        }

        if (pending_count()) {
            Envelope * envelope;
            for (size_t i = 0; i < capacity_; ++i) {
                freed_.pop(envelope);
                ready_.push_back(envelope);
            }
            for (size_t i = 0; i < capacity_; ++i) {
                freed_.push(ready_.back());
                ready_.pop_back();
            }
        }

        assert_ready();
    }

    void producer_hangup (size_t i)
    {
        if (LOOM_DEBUG_LEVEL >= 2) {
            LOOM_ASSERT_LT(i, queues_.size());
        }

        queues_[i].push(nullptr);
    }

    const Envelope * consumer_receive (size_t i)
    {
        if (LOOM_DEBUG_LEVEL >= 2) {
            LOOM_ASSERT_LT(i, queues_.size());
        }

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
    std::vector<Envelope *> ready_;
    size_t capacity_;
};

} // namespace loom
