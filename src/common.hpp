#pragma once

#include <iostream>


#ifdef __GNUG__
#  define LOOM_LIKELY(x) __builtin_expect(bool(x), true)
#  define LOOM_UNLIKELY(x) __builtin_expect(bool(x), false)
#else // __GNUG__
#  warning "ignoring LOOM_LIKELY(-), LOOM_UNLIKELY(-)"
#  define LOOM_LIKELY(x) (x)
#  define LOOM_UNLIKELY(x) (x)
#endif // __GNUG__


#ifndef LOOM_DEBUG_LEVEL
#  define LOOM_DEBUG_LEVEL 0
#endif // LOOM_DEBUG_LEVEL


#define LOOM_ERROR(message) {\
    std::cerr << "ERROR " << message << "\n\t"\
              << __FILE__ << " : " << __LINE__ << "\n\t"\
              << __PRETTY_FUNCTION__ << std::endl; \
    abort(); }

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

#define LOOM_ASSERT_(level, cond, message) \
    { if (LOOM_DEBUG_LEVEL >= (level)) LOOM_ASSERT(cond, message) }

#define LOOM_ASSERT1(cond, message) LOOM_ASSERT_(1, cond, message)
#define LOOM_ASSERT2(cond, message) LOOM_ASSERT_(2, cond, message)
#define LOOM_ASSERT3(cond, message) LOOM_ASSERT_(3, cond, message)

#define TODO(message) LOOM_ERROR("TODO " << message)


namespace loom
{

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

} // namespace loom
