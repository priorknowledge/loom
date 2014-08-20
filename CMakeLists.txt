cmake_minimum_required(VERSION 2.8)
project(loom)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++0x")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-sign-compare -Wno-strict-aliasing -Wno-unknown-pragmas")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -funsafe-math-optimizations")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLOOM_ASSUME_X86")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDIST_DISALLOW_SLOW_FALLBACKS")

# use tcmalloc
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")

if(DEFINED ENV{CXX_FLAGS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} $ENV{CXX_FLAGS}")
endif()
message("CXX_FLAGS =${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -DDIST_DEBUG_LEVEL=3 -DLOOM_DEBUG_LEVEL=3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} -DNDEBUG=1")

include_directories("${CMAKE_SOURCE_DIR}/include")

# germ of a future FindDistributions.cmake
find_path(DISTRIBUTIONS_INCLUDE_DIR distributions/random.hpp)
if(NOT DISTRIBUTIONS_INCLUDE_DIR)
  message(FATAL_ERROR
    "distributions include dir not found, try setting CMAKE_PREFIX_PATH")
else()
  message(STATUS
    "using distributions include dir ${DISTRIBUTIONS_INCLUDE_DIR}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  find_library(DISTRIBUTIONS_LIBRARIES distributions_shared_release)
else()
  find_library(DISTRIBUTIONS_LIBRARIES distributions_shared_debug)
endif()
if(NOT DISTRIBUTIONS_LIBRARIES)
  message(FATAL_ERROR
    "distributions libraries not found, try setting CMAKE_PREFIX_PATH")
endif()

add_subdirectory(src)

set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_FILE_NAME "loom")
include(CPack)
