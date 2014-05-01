#pragma once

#include <deque>
#include <utility>
#include <distributions/vector.hpp>
#include "common.hpp"

namespace loom
{

class Assignments : noncopyable
{
public:

    template<class T>
    class Queue
    {
    public:

        size_t size () const { return queue_.size(); }

        const T & front () const { return queue_.front(); }
        const T & back () const { return queue_.back(); }
        const T & operator[] (size_t i) const { return queue_[i]; }

        void clear () { queue_.clear(); }

        void push (const T & t) { queue_.push_back(t); }

        bool try_push (const T & t)
        {
            if (LOOM_LIKELY(t != queue_.front())) {
                queue_.push_back(t);
                return true;
            } else {
                return false;
            }
        }

        T pop ()
        {
            const T t = queue_.front();
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

    void init (size_t dim);
    void clear ();
    void load (const char * filename);
    void dump (const char * filename) const;
    Queue<Value> & packed_add () { return values_.packed_add(); }
    void packed_remove (size_t i) { values_.packed_remove(i); }

    size_t dim () const { return values_.size(); }
    size_t size () const { return keys_.size(); }

    Queue<Key> & rowids () { return keys_; }
    Queue<Value> & groupids (size_t i) { return values_[i]; }

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
