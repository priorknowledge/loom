#pragma once

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <loom/common.hpp>

#ifdef LOOM_ASSUME_X86
#  define load_barrier() asm volatile("lfence":::"memory")
#  define store_barrier() asm volatile("sfence":::"memory")
#else // LOOM_ASSUME_X86
#  warn "defaulting to full memory barriers"
#  define load_barrier() __sync_synchronize()
#  define store_barrier() __sync_synchronize()
#endif // LOOM_ASSUME_X86

#if 0
#define LOOM_DEBUG_QUEUE(message) LOOM_DEBUG(pendings() << ' ' << message);
#else
#define LOOM_DEBUG_QUEUE(message)
#endif

namespace loom
{

template<class Message>
class SharedQueue
{
    struct Envelope
    {
        Message message;
        std::atomic<uint_fast64_t> pending;

        Envelope () : pending(0) {}
    };

    struct Guard
    {
        std::mutex mutex;
        std::condition_variable cond_variable;

        template<class Predicate>
        void wait (
            std::atomic<uint_fast64_t> & variable,
            const Predicate & predicate)
        {
            if (not predicate(variable.load(std::memory_order_acquire))) {
                std::unique_lock<std::mutex> lock(mutex);
                cond_variable.wait(lock, [&](){
                    return predicate(variable.load(std::memory_order_acquire));
                });
            }
        }
    };

    Envelope * const envelopes_;
    const size_t size_plus_one_;
    size_t position_;
    Guard front_;
    Guard back_;

    Envelope & envelopes (size_t position)
    {
        return envelopes_[position % size_plus_one_];  // TODO use mask
    }

    std::vector<size_t> pendings () const
    {
        std::vector<size_t> counts;
        counts.reserve(size_plus_one_);
        for (size_t i = 0; i < size_plus_one_; ++i) {
            counts.push_back(envelopes_[i].pending.load());
        }
        return counts;
    }

public:

    SharedQueue (size_t size) :
        envelopes_(new Envelope[size + 1]),
        size_plus_one_(size + 1),
        position_(0)
    {
        assert_ready();
    }

    ~SharedQueue ()
    {
        assert_ready();
        delete[] envelopes_;
    }

    size_t size () const { return size_plus_one_ - 1; }

    void assert_ready () const
    {
        if (LOOM_DEBUG_LEVEL >= 2) {
            for (size_t i = 0; i < size_plus_one_; ++i) {
                LOOM_ASSERT_EQ(envelopes_[i].pending.load(), 0);
            }
        }
    }

    size_t unsafe_position ()
    {
        assert_ready();
        return position_;
    }

    void producer_wait ()
    {
        LOOM_DEBUG_QUEUE("wait at " << (position_ % size_plus_one_));
        Envelope & last_to_finish = envelopes(position_ + size_plus_one_ - 1);
        back_.wait(last_to_finish.pending, [](size_t count) {
            return count == 0;
        });
        assert_ready();
    }

    template<class Producer>
    void produce (const Producer & producer)
    {
        LOOM_DEBUG_QUEUE("produce " << (position_ % size_plus_one_));
        LOOM_ASSERT2(size_plus_one_ > 1, "cannot use zero-length queue");

        Envelope & fence = envelopes(position_ + 1);
        back_.wait(fence.pending, [](size_t count) { return count == 0; });

        Envelope & envelope = envelopes(position_);
        position_ += 1;
        size_t consumer_count = producer(envelope.message);
        LOOM_ASSERT2(consumer_count, "message sent to no consumers");
        store_barrier();

        std::unique_lock<std::mutex> lock(front_.mutex);
        envelope.pending.store(consumer_count, std::memory_order_release);
        front_.cond_variable.notify_all();
    }

    template<class Consumer>
    void consume (size_t position, const Consumer & consumer)
    {
        LOOM_DEBUG_QUEUE("consume " << (position % size_plus_one_));
        LOOM_ASSERT2(size_plus_one_ > 1, "cannot use zero-length queue");

        Envelope & envelope = envelopes(position);
        front_.wait(envelope.pending, [](size_t count){ return count != 0; });

        load_barrier();
        consumer(const_cast<const Message &>(envelope.message));

        if (envelope.pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::unique_lock<std::mutex> lock(back_.mutex);
            back_.cond_variable.notify_one();
        }
    }
};

} // namespace loom
