// Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// - Neither the name of Salesforce.com nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
