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

#include <loom/models.hpp>
#include <loom/protobuf.hpp>

#define LOOM_ASSERT_CLOSE(x, y) \
    LOOM_ASSERT(loom::are_close((x), (y)), \
        "expected " #x " close to " #y "; actual " << (x) << " vs " << (y))

namespace loom
{

static const float assert_close_tol = 1e-1f;

template<class T>
inline bool are_close (const T & x, const T & y)
{
    return x == y;
}

template<>
inline bool are_close (const float & x, const float & y)
{
    return fabs(x - y) <= (1 + fabs(x) + fabs(y)) * assert_close_tol;
}

template<>
inline bool are_close (const double & x, const double & y)
{
    return fabs(x - y) <= (1 + fabs(x) + fabs(y)) * assert_close_tol;
}

template<>
inline bool are_close (
        const distributions::protobuf::DirichletProcessDiscrete::Group & x,
        const distributions::protobuf::DirichletProcessDiscrete::Group & y)
{
    if (x.keys_size() != y.keys_size()) {
        return false;
    }
    const size_t size = x.keys_size();
    std::vector<std::pair<uint32_t, uint32_t>> sorted_x(size);
    std::vector<std::pair<uint32_t, uint32_t>> sorted_y(size);
    for (size_t i = 0; i < size; ++i) {
        sorted_x[i].first = x.keys(i);
        sorted_x[i].second = x.values(i);
        sorted_y[i].first = y.keys(i);
        sorted_y[i].second = y.values(i);
    }
    std::sort(sorted_x.begin(), sorted_x.end());
    std::sort(sorted_y.begin(), sorted_y.end());
    return sorted_x == sorted_y;
}

template<>
inline bool are_close (
        const distributions::protobuf::GammaPoisson::Group & x,
        const distributions::protobuf::GammaPoisson::Group & y)
{
    return x.count() == y.count()
        and x.sum() == y.sum()
        and are_close(x.log_prod(), y.log_prod());
}

template<>
inline bool are_close (
        const distributions::protobuf::NormalInverseChiSq::Group & x,
        const distributions::protobuf::NormalInverseChiSq::Group & y)
{
    return x.count() == y.count()
        and are_close(x.mean(), y.mean())
        and are_close(x.count_times_variance(), y.count_times_variance());
}

template<>
inline bool are_close (
        const google::protobuf::Message & x,
        const google::protobuf::Message & y)
{
    // TODO use protobuf reflection to recurse through message structure
    // as in the python distributions.test.util.assert_close
    return x == y;
}

} // namespace loom
