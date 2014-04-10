#pragma once

#include "common.hpp"
#include <unordered_map>
#include <utility>

namespace loom
{

template<class Key = uint64_t, class Value = uint32_t>
class Assignments
{
    typedef std::unordered_map<Key, Value *> Map;

public:

    Assignments (size_t dim) : dim_(dim)
    {
        LOOM_ASSERT(dim, "assignments dim = 0");
    }

    ~Assignments ()
    {
        for (auto & pair : map_) {
            delete[] pair.second;
        }
    }

    Value * insert (const Key & key)
    {
        Value * value = new Value[dim_];
        auto pair = map_.insert(typename Map::value_type(key, value));
        LOOM_ASSERT1(pair.second, "duplicate key in insert");
        return value;
    }

    Value * try_insert (const Key & key)
    {
        Value * value = new Value[dim_];
        auto pair = map_.insert(typename Map::value_type(key, value));
        if (LOOM_LIKELY(pair.second)) {
            return value;
        } else {
            delete[] value;
            return nullptr;
        }
    }

    Value * find (const Key & key) const
    {
        auto i = map_.find(key);
        LOOM_ASSERT1(i != map_.end(), "missing key in find");
        return i->second;
    }

    struct SelfDestructing
    {
        Value * const value;

        SelfDestructing (Value * v) : value(v) {}
        ~SelfDestructing () { delete[] value; }
    };

    SelfDestructing remove (const Key & key)
    {
        auto i = map_.find(key);
        LOOM_ASSERT1(i != map_.end(), "missing key in  remove");
        map_.erase(i);
        return SelfDestructing(i->second);
    }

private:

    Map map_;
    const size_t dim_;
};

} // namespace loom
