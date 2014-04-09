#pragma once

#include <unordered_map>
#include <utility>

namespace loom
{

template<class Key = size_t, class Value = int>
class Assignments
{
    struct TrivialHash
    {
        typedef Key argument_type;
        typedef size_t result_type;

        size_t operator() (const Key & key) const { return key; }
    };

    typedef std::unordered_map<Key, Value *, TrivialHash> Map;

public:

    ~Assignments ()
    {
        clear();
    }

    Value * insert (const Key & key, size_t dim)
    {
        Value * value = new Value[dim];
        auto pair = map_.insert(Map::value_type(key, value));
        LOOM_ASSERT1(pair.second, "duplicate key in insert");
        return value;
    }

    Value * find (const Key & key) const
    {
        auto i = map_.find(key);
        LOOM_ASSERT1(i != map_.end(), "missing key in find");
        return i->second;
    }

    Value * try_find (const Key & key) const
    {
        auto i = map_.find(key);
        if (i == map_.end()) {
            return nullptr;
        } else {
            return i->second;
        }
    }

    struct SelfDestructingValue
    {
        const Value * const value;

        SelfDestructingValue (Value * v) : value(v) {}
        ~SelfDestructingValue () { delete[] value; }
    };

    SelfDestructingValue remove (const Key & key)
    {
        auto i = map_.find(key);
        LOOM_ASSERT1(i != map_.end(), "missing key in  remove");
        map_.erase(i);
        return SelfDestructingValue(i->second);
    }

    SelfDestructingValue try_remove (const Key & key)
    {
        auto i = map_.find(key);
        if (i == map_.end()) {
            return SelfDestructingValue(nullptr);
        } else {
            map_.erase(i);
            return SelfDestructingValue(i->second);
        }
    }

    void clear ()
    {
        for (auto & pair : map_) {
            delete[] pair.second;
        }
        map_.clear();
    }

private:

    Map map_;
};

} // namespace loom
