#pragma once

#include <vector>
#include <atomic>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_vector.h>
#include <loom/common.hpp>

#ifdef LOOM_ASSUME_X86
#  define load_barrier() asm volatile("lfence":::"memory")
#  define store_barrier() asm volatile("sfence" ::: "memory")
#else // LOOM_ASSUME_X86
#  warn "defaulting to full memory barriers"
#  define load_barrier() __sync_synchronize()
#  define store_barrier() __sync_synchronize()
#endif // LOOM_ASSUME_X86

#if 0
#define LOOM_DEBUG_QUEUE(message) \
    LOOM_DEBUG(freed_.size() << " " << sizes() << " " << message);
#else
#define LOOM_DEBUG_QUEUE(message)
#endif

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

    ParallelQueue (size_t capacity) :
        capacity_(capacity)
    {
        queues_.reserve(64);
        freed_.set_capacity(capacity_);
        for (size_t i = 0; i < capacity_; ++i) {
            freed_.push(new Envelope());
        }
    }

    ~ParallelQueue ()
    {
        assert_ready();
        Envelope * envelope;
        while (freed_.try_pop(envelope)) {
            delete envelope;
        }
    }

    size_t size () const { return queues_.size(); }
    size_t capacity () const { return capacity_; }

    void unsafe_resize (size_t size)
    {
        LOOM_DEBUG_QUEUE("unsafe_resize(" << size << ")");
        assert_ready();
        queues_.grow_to_at_least(size);
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

    Envelope * producer_alloc ()
    {
        LOOM_DEBUG_QUEUE("start producer_alloc");
        LOOM_ASSERT2(capacity_, "cannot use zero-capacity queue");

        Envelope * envelope;
        freed_.pop(envelope);
        if (LOOM_DEBUG_LEVEL >= 2) {
            auto ref_count = envelope->ref_count.load();
            LOOM_ASSERT_EQ(ref_count, 0);
        }
        LOOM_DEBUG_QUEUE("done producer_alloc");
        return envelope;
    }

    void producer_send (Envelope * envelope, size_t consumer_count)
    {
        LOOM_DEBUG_QUEUE("producer_send(" << consumer_count << ")");
        LOOM_ASSERT2(consumer_count, "message sent to zero consumers");
        LOOM_ASSERT2(
            consumer_count <= queues_.size(),
            "too many consumers " << consumer_count);
        LOOM_ASSERT2(envelope, "got null envelope from producer");

        envelope->ref_count.store(consumer_count, std::memory_order_acq_rel);
        store_barrier();
        for (size_t i = 0; i < consumer_count; ++i) {
            queues_[i].push(envelope);
        }
        LOOM_DEBUG_QUEUE("queues_[-].push(" << consumer_count << ")");
    }

    void producer_wait ()
    {
        LOOM_DEBUG_QUEUE("producer_wait");
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
        LOOM_DEBUG_QUEUE("producer_hangup(" << i << ")");
        if (LOOM_DEBUG_LEVEL >= 2) {
            LOOM_ASSERT_LT(i, queues_.size());
        }

        queues_[i].push(nullptr);
    }

    const Envelope * consumer_receive (size_t i)
    {
        LOOM_DEBUG_QUEUE("start consumer_receive(" << i << ")");
        if (LOOM_DEBUG_LEVEL >= 2) {
            LOOM_ASSERT_LT(i, queues_.size());
        }

        Envelope * envelope;
        queues_[i].pop(envelope);
        LOOM_DEBUG_QUEUE("done consumer_receive(" << i << ")");
        load_barrier();
        return envelope;
    }

    void consumer_free (const Envelope * const_envelope)
    {
        LOOM_DEBUG_QUEUE("consumer_free");
        Envelope * envelope = const_cast<Envelope *>(const_envelope);
        if (envelope->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            LOOM_DEBUG_QUEUE("free_.push");
            freed_.push(envelope);
        }
    }

private:

    std::vector<int> sizes () const
    {
        std::vector<int> result;
        for (const auto & queue : queues_) {
            result.push_back(queue.size());
        }
        return result;
    }

    typedef tbb::concurrent_bounded_queue<Envelope *> Queue_;
    tbb::concurrent_vector<Queue_> queues_;
    Queue_ freed_;  // this should really be a stack
    std::vector<Envelope *> ready_;
    const size_t capacity_;
};

} // namespace loom
