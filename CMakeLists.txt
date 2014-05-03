cmake_minimum_required(VERSION 2.8)
project(loom)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g -std=c++0x")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-sign-compare -Wno-strict-aliasing -Wno-unknown-pragmas")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -funsafe-math-optimizations")
# TODO get Debug and RelWithDebInfo versions working
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLOOM_DEBUG_LEVEL=3")
if(DEFINED ENV{CXX_FLAGS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} $ENV{CXX_FLAGS}")
endif()
message("CXX_FLAGS =${CMAKE_CXX_FLAGS}")


# distributions
include(ExternalProject)
if(DEFINED ENV{DISTRIBUTIONS_PATH})
  ExternalProject_Add(distributions
    PREFIX distributions
    DOWNLOAD_COMMAND ""
    SOURCE_DIR $ENV{DISTRIBUTIONS_PATH}
    BINARY_DIR $ENV{DISTRIBUTIONS_PATH}/build
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
  )
else()
  ExternalProject_Add(distributions
    PREFIX distributions
    GIT_REPOSITORY https://github.com/forcedotcom/distributions.git
    INSTALL_COMMAND ""
  )
endif()
ExternalProject_Get_Property(distributions SOURCE_DIR)
ExternalProject_Get_Property(distributions BINARY_DIR)
include_directories("${SOURCE_DIR}/include")
link_directories("${BINARY_DIR}/src")
set(DISTRIBUTIONS_PATH ${SOURCE_DIR})


add_subdirectory(src)
