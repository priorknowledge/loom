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
    return x == y;
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
