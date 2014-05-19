#pragma once

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <loom/common.hpp>

#ifdef LOOM_ASSUME_X86
#  define load_barrier() asm volatile("lfence":::"memory")
#  define store_barrier() asm volatile("sfence" ::: "memory")
#else // LOOM_ASSUME_X86
#  warn "defaulting to full memory barriers"
#  define load_barrier() __sync_synchronize()
#  define store_barrier() __sync_synchronize()
#endif // LOOM_ASSUME_X86

namespace loom
{

template<class Message>
class SharedQueue
{
    struct Envelope
    {
        Message message;
        std::atomic<uint_fast64_t> pending_count;

        Envelope () : pending_count(0) {}
    };

    struct Position
    {
        std::mutex mutex;
        std::condition_variable cond_variable;
        size_t position;

        Position () : position(0) {}

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
    const size_t size_;
    Position front_;
    Position back_;

    const Envelope & envelopes (size_t position) const
    {
        return envelopes_[position % size_];  // TODO use mask
    }
    Envelope & envelopes (size_t position)
    {
        return envelopes_[position % size_];  // TODO use mask
    }

public:

    SharedQueue (size_t size) :
        envelopes_(size ? new Envelope[size] : nullptr),
        size_(size),
        front_(),
        back_()
    {
    }

    ~SharedQueue ()
    {
        assert_ready();
        delete[] envelopes_;
    }

    size_t size () const { return size_; }

    void assert_ready () const
    {
        if (LOOM_DEBUG_LEVEL >= 2) {
            if (size_) {
                load_barrier();
                LOOM_ASSERT_EQ(back_.position, front_.position);
                const Envelope & envelope = envelopes(back_.position);
                LOOM_ASSERT_EQ(envelope.pending_count.load(), 0);
            }
        }
    }

    size_t unsafe_position ()
    {
        assert_ready();

        load_barrier();
        return front_.position;
    }

    void producer_wait ()
    {
        std::unique_lock<std::mutex> back_lock(back_.mutex);
        front_.cond_variable.wait(back_lock, [&](){
            return back_.position == front_.position;
        });
    }

    template<class Producer>
    void produce (const Producer & producer)
    {
        LOOM_ASSERT2(size_, "cannot use zero-length queue");

        Envelope & envelope = envelopes(front_.position);
        back_.wait(envelope.pending_count, [](size_t count) {
            return count == 0;
        });

        size_t consumer_count = producer(envelope.message);
        front_.position += 1;
        store_barrier();

        std::unique_lock<std::mutex> lock(front_.mutex);
        envelope.pending_count.store(
            consumer_count,
            std::memory_order_release);
        front_.cond_variable.notify_all();
    }

    template<class Consumer>
    void consume (size_t position, const Consumer & consumer)
    {
        if (LOOM_DEBUG_LEVEL >= 2) {
            load_barrier();
            LOOM_ASSERT(size_, "cannot use zero-length queue");
            LOOM_ASSERT_LE(position, front_.position);
            LOOM_ASSERT_LE(back_.position, position);
        }

        Envelope & envelope = envelopes(position);
        front_.wait(envelope.pending_count, [](size_t count){
            return count != 0;
        });

        load_barrier();
        consumer(const_cast<const Message &>(envelope.message));

        bool at_back =
            envelope.pending_count.fetch_sub(1, std::memory_order_acq_rel) == 1;
        if (at_back) {
            std::unique_lock<std::mutex> lock(back_.mutex);
            back_.position += 1;
            back_.cond_variable.notify_one();
        }
    }

private:
};

} // namespace loom
