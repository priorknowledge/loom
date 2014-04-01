#pragma once

#include <iostream>

#define LOOM_ERROR(message) {\
    std::cerr << "ERROR " << message << "\n\t"\
              << __FILE__ << " : " << __LINE__ << "\n\t"\
              << __PRETTY_FUNCTION__ << std::endl; \
    abort(); }

#ifndef LOOM_DEBUG_LEVEL
#  define LOOM_DEBUG_LEVEL 0
#endif // LOOM_DEBUG_LEVEL

#define LOOM_ASSERT(cond, message) { if (not (cond)) LOOM_ERROR(message) }

#define LOOM_ASSERT_(level, cond, message) \
    { if (LOOM_DEBUG_LEVEL >= (level)) LOOM_ASSERT(cond, message) }

#define LOOM_ASSERT1(cond, message) LOOM_ASSERT_(1, cond, message)
#define LOOM_ASSERT2(cond, message) LOOM_ASSERT_(2, cond, message)
#define LOOM_ASSERT3(cond, message) LOOM_ASSERT_(3, cond, message)
