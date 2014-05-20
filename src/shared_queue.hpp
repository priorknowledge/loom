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
    typedef uint_fast64_t count_t;

    struct Envelope
    {
        Message message;
        std::atomic<count_t> pending;

        Envelope () : pending(0) {}
    };

    class Guard
    {
        std::mutex mutex_;
        std::condition_variable cond_variable_;

    public:

        void producer_wait (const std::atomic<count_t> & pending)
        {
            if (pending.load(std::memory_order_acquire) != 0) {
                std::unique_lock<std::mutex> lock(mutex_);
                cond_variable_.wait(lock, [&](){
                    return pending.load(std::memory_order_acquire) == 0;
                });
            }
        }

        void consumer_wait (const std::atomic<count_t> & pending)
        {
            if (pending.load(std::memory_order_acquire) == 0) {
                std::unique_lock<std::mutex> lock(mutex_);
                cond_variable_.wait(lock, [&](){
                    return pending.load(std::memory_order_acquire) != 0;
                });
            }
        }

        void produce (
            std::atomic<count_t> & pending,
            const count_t & consumer_count)
        {
            LOOM_ASSERT2(consumer_count, "message sent to no consumers");
            pending.store(consumer_count, std::memory_order_release);
            std::unique_lock<std::mutex> lock(mutex_);
            cond_variable_.notify_all();
        }

        void consume (std::atomic<count_t> & pending)
        {
            if (pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::unique_lock<std::mutex> lock(mutex_);
                cond_variable_.notify_one();
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

    std::vector<count_t> pendings () const
    {
        std::vector<count_t> counts;
        counts.reserve(size_plus_one_);
        for (count_t i = 0; i < size_plus_one_; ++i) {
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
        back_.producer_wait(last_to_finish.pending);
        assert_ready();
    }

    template<class Producer>
    void produce (const Producer & producer)
    {
        LOOM_DEBUG_QUEUE("produce " << (position_ % size_plus_one_));
        LOOM_ASSERT2(size_plus_one_ > 1, "cannot use zero-length queue");

        const Envelope & fence = envelopes(position_ + 1);
        back_.producer_wait(fence.pending);

        Envelope & envelope = envelopes(position_);
        position_ += 1;
        count_t consumer_count = producer(envelope.message);
        store_barrier();

        front_.produce(envelope.pending, consumer_count);
    }

    template<class Consumer>
    void consume (size_t position, const Consumer & consumer)
    {
        LOOM_DEBUG_QUEUE("consume " << (position % size_plus_one_));
        LOOM_ASSERT2(size_plus_one_ > 1, "cannot use zero-length queue");

        Envelope & envelope = envelopes(position);
        front_.consumer_wait(envelope.pending);

        load_barrier();
        consumer(const_cast<const Message &>(envelope.message));

        back_.consume(envelope.pending);
    }
};

} // namespace loom
