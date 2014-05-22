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
#define LOOM_DEBUG_QUEUE(message) LOOM_DEBUG(counts() << ' ' << message);
#else
#define LOOM_DEBUG_QUEUE(message)
#endif

namespace loom
{
namespace pipeline
{

class State
{
    std::atomic<uint_fast64_t> pair_;

public:

    enum { max_stage_count = 32 };

    State () : pair_(0) {}

    typedef uint_fast64_t stage_t;
    typedef uint_fast64_t count_t;
    typedef uint_fast64_t pair_t;

    static pair_t create_state (uint_fast64_t stage_number, count_t count)
    {
        LOOM_ASSERT_LT(stage_number, max_stage_count);
        LOOM_ASSERT_LE(count, 0xFFFFUL);
        return (0x10000UL << stage_number) | count;
    }

    static stage_t get_stage (const pair_t & pair)
    {
        return pair & 0xFFFF0000UL;
    }

    static count_t get_count (const pair_t & pair)
    {
        return pair & 0x0000FFFFUL;
    }

    stage_t load_stage () const
    {
        return get_stage(pair_.load(std::memory_order_acquire));
    }

    count_t load_count () const
    {
        return get_count(pair_.load(std::memory_order_acquire));
    }

    void store (const pair_t & pair)
    {
        pair_.store(pair, std::memory_order_release);
    }

    count_t decrement_count ()
    {
        return get_count(pair_.fetch_sub(1, std::memory_order_acq_rel));
    }
};

class Guard
{
    State::pair_t state_;
    State::stage_t stage_;
    std::mutex mutex_;
    std::condition_variable cond_variable_;

public:

    void init (size_t stage_number, size_t count)
    {
        state_ = State::create_state(stage_number, count);
        stage_ = State::create_state(stage_number, 0);
    }

    size_t get_count () { return State::get_count(state_); }

    void acquire (const State & state)
    {
        if (state.load_stage() != stage_) {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_variable_.wait(lock, [&](){
                return state.load_stage() == stage_;
            });
        }
    }

    void release (State & state)
    {
        if (state.decrement_count() == 1) {
            state.store(state_);
            std::unique_lock<std::mutex> lock(mutex_);
            cond_variable_.notify_all();
        }
    }

    void unsafe_set_ready (State & state) const
    {
        state.store(state_);
    }

    void assert_ready (const State & state) const
    {
        LOOM_ASSERT2(state.load_stage() == stage_, "state is not ready");
    }
};

template<class Message>
class SharedQueue
{
    struct Envelope
    {
        Message message;
        State state;
    };

    Envelope * const envelopes_;
    const size_t size_plus_one_;
    const size_t consumer_count_;
    size_t position_;
    Guard guards_[State::max_stage_count];

    Envelope & envelopes (size_t position)
    {
        return envelopes_[position % size_plus_one_];  // TODO use mask
    }

    std::vector<size_t> counts () const
    {
        std::vector<size_t> counts;
        counts.reserve(size_plus_one_);
        for (size_t i = 0; i < size_plus_one_; ++i) {
            counts.push_back(envelopes_[i].state.load_count());
        }
        return counts;
    }

public:

    SharedQueue (size_t size, std::vector<size_t> consumer_counts) :
        envelopes_(new Envelope[size + 1]),
        size_plus_one_(size + 1),
        consumer_count_(consumer_counts.size()),
        position_(0),
        guards_()
    {
        LOOM_ASSERT_LE(1, consumer_count_);
        LOOM_ASSERT_LE(1 + consumer_count_, State::max_stage_count);

        for (size_t i = 0; i < consumer_count_; ++i) {
            guards_[i].init(i, consumer_counts[i]);
        }
        guards_[consumer_count_].init(consumer_count_, 1);

        Guard & guard = guards_[consumer_count_];
        for (size_t i = 0; i < size_plus_one_; ++i) {
            guard.unsafe_set_ready(envelopes_[i].state);
        }

        assert_ready();
    }

    ~SharedQueue ()
    {
        assert_ready();
        delete[] envelopes_;
    }

    size_t size () const { return size_plus_one_ - 1; }

    size_t thread_count ()
    {
        size_t count = 0;
        for (size_t i = 0; i < consumer_count_; ++i) {
            count += guards_[i].get_count();
        }
        return count;
    }

    void assert_ready () const
    {
        if (LOOM_DEBUG_LEVEL >= 2) {
            const Guard & guard = guards_[consumer_count_];
            for (size_t i = 0; i < size_plus_one_; ++i) {
                guard.assert_ready(envelopes_[i].state);
            }
        }
    }

    void unsafe_set_consumer_count (size_t stage_number, size_t count)
    {
        LOOM_ASSERT_LT(stage_number, consumer_count_);

        assert_ready();
        guards_[stage_number].init(stage_number, count);
    }

    size_t unsafe_position ()
    {
        assert_ready();
        return position_;
    }

    void wait ()
    {
        LOOM_DEBUG_QUEUE("wait at " << (position_ % size_plus_one_));
        Envelope & last_to_finish = envelopes(position_ + size_plus_one_ - 1);
        guards_[consumer_count_].acquire(last_to_finish.state);
        assert_ready();
    }

    template<class Producer>
    void produce (const Producer & producer)
    {
        LOOM_DEBUG_QUEUE("produce " << (position_ % size_plus_one_));
        LOOM_ASSERT2(size_plus_one_ > 1, "cannot use zero-length queue");

        const Envelope & fence = envelopes(position_ + 1);
        guards_[consumer_count_].acquire(fence.state);

        Envelope & envelope = envelopes(position_);
        position_ += 1;

        load_barrier();
        producer(envelope.message);
        store_barrier();

        guards_[0].release(envelope.state);
    }

    template<class Consumer>
    void consume (
        size_t stage_number,
        size_t position,
        const Consumer & consumer)
    {
        LOOM_DEBUG_QUEUE("consume " << stage_number << " "
                                    << (position % size_plus_one_));
        LOOM_ASSERT2(size_plus_one_ > 1, "cannot use zero-length queue");
        LOOM_ASSERT2(stage_number < consumer_count_,
            "bad stage number: " << stage_number);

        Envelope & envelope = envelopes(position);
        guards_[stage_number].acquire(envelope.state);

        load_barrier();
        consumer(envelope.message);
        store_barrier();

        guards_[stage_number + 1].release(envelope.state);
    }
};

} // namespace pipeline
} // namespace loom
