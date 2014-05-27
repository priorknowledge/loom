#pragma once

#include <atomic>
#include <mutex>
#include <thread>
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

class PipelineState
{
    std::atomic<uint_fast64_t> pair_;

public:

    enum {
        max_stage_count = 48,
        max_consumer_count = 65535
    };

    typedef uint_fast64_t stage_t;
    typedef uint_fast64_t count_t;
    typedef uint_fast64_t pair_t;

    static pair_t create_state (uint_fast64_t stage_number, count_t count)
    {
        LOOM_ASSERT_LT(stage_number, max_stage_count);
        LOOM_ASSERT_LE(count, 0xFFFFUL);
        return _state(stage_number, count);
    }

    static constexpr stage_t get_stage (const pair_t & pair)
    {
        return pair & 0xFFFFFFFFFFFF0000UL;
    }

    static constexpr count_t get_count (const pair_t & pair)
    {
        return pair & 0xFFFFUL;
    }

    PipelineState () : pair_(0)
    {
        static_test();
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

private:

    static constexpr pair_t _state (uint_fast64_t stage_number, count_t count)
    {
        return (0x10000UL << stage_number) | count;
    }

    static void static_test ()
    {
        static_assert(get_count(_state(0, 1234)) == 1234, "fail");
        static_assert(get_count(_state(1, 1234)) == 1234, "fail");
        static_assert(get_count(_state(2, 1234)) == 1234, "fail");
        static_assert(get_count(_state(3, 1234)) == 1234, "fail");
        static_assert(get_count(_state(4, 1234)) == 1234, "fail");

        static_assert(get_stage(_state(0, 1234)) ==
                      get_stage(_state(0, 5679)), "fail");
        static_assert(get_stage(_state(1, 1234)) ==
                      get_stage(_state(1, 5679)), "fail");
        static_assert(get_stage(_state(2, 1234)) ==
                      get_stage(_state(2, 5679)), "fail");
        static_assert(get_stage(_state(3, 1234)) ==
                      get_stage(_state(3, 5679)), "fail");

        static_assert(_state(0, 1234) != _state(1, 1234), "fail");
        static_assert(_state(0, 1234) != _state(2, 1234), "fail");
        static_assert(_state(0, 1234) != _state(3, 1234), "fail");
        static_assert(_state(0, 1234) != _state(4, 1234), "fail");
        static_assert(_state(1, 1234) != _state(2, 1234), "fail");
        static_assert(_state(1, 1234) != _state(3, 1234), "fail");
        static_assert(_state(1, 1234) != _state(4, 1234), "fail");
        static_assert(_state(2, 1234) != _state(3, 1234), "fail");
        static_assert(_state(2, 1234) != _state(4, 1234), "fail");
        static_assert(_state(3, 1234) != _state(4, 1234), "fail");
    }
};

class PipelineGuard
{
    PipelineState::pair_t state_;
    PipelineState::stage_t stage_;
    std::mutex mutex_;
    std::condition_variable cond_variable_;

public:

    void init (size_t stage_number, size_t count)
    {
        state_ = PipelineState::create_state(stage_number, count);
        stage_ = PipelineState::create_state(stage_number, 0);
    }

    size_t get_count () { return PipelineState::get_count(state_); }

    void acquire (const PipelineState & state)
    {
        if (state.load_stage() != stage_) {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_variable_.wait(lock, [&](){
                return state.load_stage() == stage_;
            });
        }
        load_barrier();
    }

    void release (PipelineState & state)
    {
        store_barrier();
        if (state.decrement_count() == 1) {
            state.store(state_);
            std::unique_lock<std::mutex> lock(mutex_);
            cond_variable_.notify_all();
        }
    }

    void unsafe_set_ready (PipelineState & state) const
    {
        state.store(state_);
    }

    void assert_ready (const PipelineState & state) const
    {
        LOOM_ASSERT2(state.load_stage() == stage_, "state is not ready");
    }
};

template<class Message>
class PipelineQueue
{
    struct Envelope
    {
        Message message;
        PipelineState state;
    };

    Envelope * const envelopes_;
    const size_t size_plus_one_;
    const size_t stage_count_;
    std::vector<size_t> consumer_counts_;
    size_t position_;
    PipelineGuard guards_[PipelineState::max_stage_count];

    Envelope & envelopes (size_t position)
    {
        return envelopes_[position % size_plus_one_];
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

    PipelineQueue (size_t size, size_t stage_count) :
        envelopes_(new Envelope[size + 1]),
        size_plus_one_(size + 1),
        stage_count_(stage_count),
        consumer_counts_(stage_count, 0),
        position_(0),
        guards_()
    {
        LOOM_ASSERT_LE(1, stage_count_);
        LOOM_ASSERT_LE(1 + stage_count_, PipelineState::max_stage_count);

        for (size_t i = 0; i < stage_count_; ++i) {
            guards_[i].init(i, 0);
        }
        guards_[stage_count_].init(stage_count_, 1);

        PipelineGuard & guard = guards_[stage_count_];
        for (size_t i = 0; i < size_plus_one_; ++i) {
            guard.unsafe_set_ready(envelopes_[i].state);
        }

        assert_ready();
    }

    ~PipelineQueue ()
    {
        assert_ready();
        delete[] envelopes_;
    }

    size_t size () const { return size_plus_one_ - 1; }
    size_t stage_count () const { return stage_count_; }

    void assert_ready () const
    {
        if (LOOM_DEBUG_LEVEL >= 2) {
            const PipelineGuard & guard = guards_[stage_count_];
            for (size_t i = 0; i < size_plus_one_; ++i) {
                guard.assert_ready(envelopes_[i].state);
            }
        }
    }

    void unsafe_add_consumer (size_t stage_number)
    {
        LOOM_ASSERT_LT(stage_number, stage_count_);
        assert_ready();
        size_t count = ++consumer_counts_[stage_number];
        guards_[stage_number].init(stage_number, count);
        assert_ready();
    }

    void validate () const
    {
        assert_ready();
        for (size_t i = 0; i < stage_count_; ++i) {
            LOOM_ASSERT(consumer_counts_[i], "no threads in stage " << i);
        }
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
        guards_[stage_count_].acquire(last_to_finish.state);
        assert_ready();
    }

    template<class Producer>
    void produce (const Producer & producer)
    {
        LOOM_DEBUG_QUEUE("produce " << (position_ % size_plus_one_));
        LOOM_ASSERT2(size_plus_one_ > 1, "cannot use zero-length queue");

        const Envelope & fence = envelopes(position_ + 1);
        guards_[stage_count_].acquire(fence.state);
        Envelope & envelope = envelopes(position_);
        producer(envelope.message);
        guards_[0].release(envelope.state);

        position_ += 1;
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
        LOOM_ASSERT2(stage_number < stage_count_,
            "bad stage number: " << stage_number);

        Envelope & envelope = envelopes(position);
        guards_[stage_number].acquire(envelope.state);
        consumer(envelope.message);
        guards_[stage_number + 1].release(envelope.state);
    }
};

template<class Task, class ThreadState>
class Pipeline
{
    struct PipelineTask
    {
        Task task;
        bool exit;
        PipelineTask () : exit(false) {}
    };

    PipelineQueue<PipelineTask> queue_;
    std::vector<std::thread> threads_;

public:

    Pipeline (size_t capacity, size_t stage_count) :
        queue_(capacity, stage_count),
        threads_()
    {
    }

    template<class Fun>
    void unsafe_add_thread (
            size_t stage_number,
            const ThreadState & init_thread,
            const Fun & fun)
    {
        queue_.unsafe_add_consumer(stage_number);
        size_t init_position = queue_.unsafe_position();
        threads_.push_back(std::thread(
                [this, stage_number, init_thread, init_position, fun](){
            ThreadState thread = init_thread;
            size_t position = init_position;
            for (bool alive = true; LOOM_LIKELY(alive);) {
                queue_.consume(stage_number, position, [&](PipelineTask & task){
                    if (LOOM_UNLIKELY(task.exit)) {
                        alive = false;
                    } else {
                        fun(task.task, thread);
                    }
                });
                ++position;
            }
        }));
    }

    void validate ()
    {
        queue_.validate();
    }

    template<class Fun>
    void start (const Fun & fun)
    {
        queue_.produce([fun](PipelineTask & task){ fun(task.task); });
    }

    void wait ()
    {
        queue_.wait();
    }

    ~Pipeline ()
    {
        queue_.produce([](PipelineTask & task) { task.exit = true; });
        queue_.wait();
        for (auto & thread : threads_) {
            thread.join();
        }
    }
};

} // namespace loom
