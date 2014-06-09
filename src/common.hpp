#pragma once

#include <iostream>
#include <sstream>
#include <distributions/random_fwd.hpp>
#include <distributions/vector.hpp>


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
            os << ", " << vect[i];
        }
        return os << ']';
    }
}

} // namespace loom
