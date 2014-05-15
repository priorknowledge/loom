#pragma once

#include <deque>
#include <utility>
#include <distributions/vector.hpp>
#include <loom/common.hpp>

namespace loom
{

class Assignments : noncopyable
{
public:

    template<class T>
    class Queue
    {
    public:

        bool empty () const { return queue_.empty(); }
        size_t size () const { return queue_.size(); }

        const T & front () const { return queue_.front(); }
        const T & back () const { return queue_.back(); }
        const T & operator[] (size_t i) const { return queue_[i]; }

        void clear () { queue_.clear(); }

        void push (const T & t) { queue_.push_back(t); }

        bool try_push (const T & t)
        {
            if (LOOM_UNLIKELY(empty()) or LOOM_LIKELY(t != front())) {
                queue_.push_back(t);
                return true;
            } else {
                return false;
            }
        }

        T pop ()
        {
            LOOM_ASSERT1(not empty(), "cannot pop from empty queue");
            const T t = front();
            queue_.pop_front();
            return t;
        }

        typedef typename std::deque<T>::const_iterator iterator;
        iterator begin () const { return queue_.begin(); }
        iterator end () const { return queue_.end(); }

    private:

        std::deque<T> queue_;
    };

    typedef uint64_t Key;
    typedef uint32_t Value;

    void init (size_t kind_count);
    void clear ();
    void load (const char * filename);
    void dump (
            const char * filename,
            const std::vector<std::vector<uint32_t>> & sorted_to_globals) const;
    Queue<Value> & packed_add () { return values_.packed_add(); }
    void packed_remove (size_t i) { values_.packed_remove(i); }

    size_t row_count () const { return keys_.size(); }
    size_t kind_count () const { return values_.size(); }

    Queue<Key> & rowids () { return keys_; }
    const Queue<Key> & rowids () const { return keys_; }
    Queue<Value> & groupids (size_t i) { return values_[i]; }
    const Queue<Value> & groupids (size_t i) const { return values_[i]; }

    void validate () const
    {
        for (const auto & values : values_) {
            LOOM_ASSERT_EQ(values.size(), keys_.size());
        }
    }

private:

    Queue<Key> keys_;
    distributions::Packed_<Queue<Value>> values_;
};

} // namespace loom
