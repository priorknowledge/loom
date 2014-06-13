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

#include <iostream>
#include <sstream>
#include <vector>
#include <utility>
#include <algorithm>
#include <distributions/random_fwd.hpp>
#include <distributions/vector.hpp>
#include <distributions/sparse.hpp>


#ifdef __GNUG__
#  define LOOM_LIKELY(x) __builtin_expect(bool(x), true)
#  define LOOM_UNLIKELY(x) __builtin_expect(bool(x), false)
#else // __GNUG__
#  warning "ignoring LOOM_LIKELY(-), LOOM_UNLIKELY(-)"
#  define LOOM_LIKELY(x) (x)
#  define LOOM_UNLIKELY(x) (x)
#endif // __GNUG__


#define LOOM_ERROR(message) {                           \
    std::ostringstream PRIVATE_message;                 \
    PRIVATE_message                                     \
        << "ERROR " << message << "\n\t"                \
        << __FILE__ << " : " << __LINE__ << "\n\t"      \
        << __PRETTY_FUNCTION__ << '\n';                 \
    std::cerr << PRIVATE_message.str() << std::flush;   \
    abort(); }

#define LOOM_DEBUG(message) {                           \
    std::ostringstream PRIVATE_message;                 \
    PRIVATE_message << "DEBUG " << message << '\n';     \
    std::cout << PRIVATE_message.str() << std::flush; }

#define LOOM_ASSERT(cond, message) \
    { if (LOOM_UNLIKELY(not (cond))) LOOM_ERROR(message) }

#define LOOM_ASSERT_EQ(x, y) \
    LOOM_ASSERT((x) == (y), \
            "expected " #x " == " #y "; actual " << (x) << " vs " << (y))
#define LOOM_ASSERT_LE(x, y) \
    LOOM_ASSERT((x) <= (y), \
            "expected " #x " <= " #y "; actual " << (x) << " vs " << (y))
#define LOOM_ASSERT_LT(x, y) \
    LOOM_ASSERT((x) < (y), \
            "expected " #x " < " #y "; actual " << (x) << " vs " << (y))
#define LOOM_ASSERT_NE(x, y) \
    LOOM_ASSERT((x) != (y), \
            "expected " #x " != " #y "; actual " << (x) << " vs " << (y))

#ifndef LOOM_DEBUG_LEVEL
#  define LOOM_DEBUG_LEVEL 0
#endif // LOOM_DEBUG_LEVEL

#define LOOM_ASSERT_(level, cond, message) \
    { if (LOOM_DEBUG_LEVEL >= (level)) LOOM_ASSERT(cond, message) }

#define LOOM_ASSERT1(cond, message) LOOM_ASSERT_(1, cond, message)
#define LOOM_ASSERT2(cond, message) LOOM_ASSERT_(2, cond, message)
#define LOOM_ASSERT3(cond, message) LOOM_ASSERT_(3, cond, message)

#define TODO(message) LOOM_ERROR("TODO " << message)


namespace loom
{

using distributions::rng_t;
using distributions::VectorFloat;

class noncopyable
{
    noncopyable (const noncopyable &) = delete;
    void operator= (const noncopyable &) = delete;
public:
    noncopyable () {}
};

template<class Value, typename... Args>
void inplace_destroy_and_construct (Value & value, Args... args)
{
    value->~Value();
    new (& value) Value(args...);
}

//----------------------------------------------------------------------------
// Debug printing of common data structures

template<class T, class Alloc>
inline std::ostream & operator<< (
        std::ostream & os,
        const std::vector<T, Alloc> & vect)
{
    if (vect.empty()) {
        return os << "[]";
    } else {
        os << '[' << vect[0];
        for (size_t i = 1; i < vect.size(); ++i) {
            os << ',' << vect[i];
        }
        return os << ']';
    }
}

template<class T1, class T2>
inline std::ostream & operator<< (
        std::ostream & os,
        const std::pair<T1, T2> & pair)
{
    return os << '(' << pair.first << ',' << pair.second << ')';
}

template<class Map>
inline std::ostream & print_map (
        std::ostream & os,
        const Map & map)
{
    std::vector<std::pair<typename Map::key_t, typename Map::value_t>> sorted;
    for (auto & i : map) {
        sorted.push_back(std::make_pair(i.first, i.second));
    }
    if (sorted.empty()) {
        return os << "{}";
    } else {
        std::sort(sorted.begin(), sorted.end());
        os << '{' << sorted[0].first << ':' << sorted[0].second;
        for (size_t i = 1; i < sorted.size(); ++i) {
            os << ',' << sorted[i].first << ':' << sorted[i].second;
        }
        return os << '}';
    }
}

template<class Key, class Value>
inline std::ostream & operator<< (
        std::ostream & os,
        const std::unordered_map<Key, Value> & map)
{
    return print_map(os, map);
}

template<class Key, class Value>
inline std::ostream & operator<< (
        std::ostream & os,
        const distributions::Sparse_<Key, Value> & map)
{
    return print_map(os, map);
}

template<class Key, class Value>
inline std::ostream & operator<< (
        std::ostream & os,
        const distributions::SparseCounter<Key, Value> & map)
{
    return print_map(os, map);
}

} // namespace loom
