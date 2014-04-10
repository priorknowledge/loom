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

template<class Value = uint32_t>
class AssignmentsMixture
{
public:

    void load (size_t group_count)
    {
        packed_to_global_.clear();
        global_to_packed_.clear();
        for (size_t i = 0; i < group_count; ++i) {
            add_group();
        }
    }

    void init_empty ()
    {
        load(1);
    }

    void add_group ()
    {
        const Value packed = packed_to_global_.size();
        const Value global = global_to_packed_.size();
        packed_to_global_.push_back(global);
        global_to_packed_.push_back(packed);
    }

    void remove_group (const Value & packed)
    {
        if (LOOM_DEBUG_LEVEL) {
            LOOM_ASSERT(packed < packed_size(), "bad packed id: " << packed);
            global_to_packed_[packed_to_global_[packed]] = ~Value(0);
        }
        if (packed != packed_size()) {
            const Value global = packed_to_global_.back();
            packed_to_global_[packed] = global;
            global_to_packed_[global] = packed;
        }
        packed_to_global_.pop_back();
    }

    Value packed_to_global (const Value & packed) const
    {
        LOOM_ASSERT1(packed < packed_size(), "bad packed id: " << packed);
        Value global = packed_to_global_[packed];
        LOOM_ASSERT1(global < global_size(), "bad global id: " << global);
        return global;
    }

    Value global_to_packed (const Value & global) const
    {
        LOOM_ASSERT1(global < global_size(), "bad global id: " << global);
        Value packed = global_to_packed_[global];
        LOOM_ASSERT1(packed < packed_size(), "bad packed id: " << packed);
        return packed;
    }

private:

    size_t packed_size () const { return packed_to_global_.size(); }
    size_t global_size () const { return global_to_packed_.size(); }

    std::vector<Value> packed_to_global_;
    std::vector<Value> global_to_packed_;
};

} // namespace loom
