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

#include <atomic>
#include <loom/common.hpp>

namespace loom
{

template<class T, size_t default_bytes = 64>
class AtomicArray : noncopyable
{
    std::atomic<T> * data_;
    size_t capacity_;

public:

    enum { default_capacity = (default_bytes + sizeof(T) - 1) / sizeof(T) };

    AtomicArray (size_t capacity = default_capacity) :
        data_(new std::atomic<T>[capacity]),
        capacity_(capacity)
    {
    }

    ~AtomicArray ()
    {
        delete[] data_;
    }

    void clear_and_resize (size_t capacity)
    {
        if (LOOM_UNLIKELY(capacity > capacity_)) {
            do {
                capacity_ *= 2;
            } while (capacity_ < capacity);
            delete[] data_;
            data_ = new std::atomic<T>[capacity_];
        }
    }

    T load (size_t pos) const
    {
        LOOM_ASSERT3(pos < capacity_, "out of bounds: " << pos);
        return data_[pos].load(std::memory_order_acquire);
    }

    void store (size_t pos, T value)
    {
        LOOM_ASSERT3(pos < capacity_, "out of bounds: " << pos);
        data_[pos].store(value, std::memory_order_release);
    }
};

} // namespace loom
