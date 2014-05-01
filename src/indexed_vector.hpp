#pragma once

#include <vector>
#include <algorithm>

namespace loom
{

template<class Value>
class IndexedVector
{
public:

    //------------------------------------------------------------------------
    // slow structural interface

    typedef uint32_t Id;

    Id index (size_t pos) const { return index_.at(pos); }

    Value & insert (Id id)
    {
        size_t pos = find(id);
        index_.insert(index_.begin() + pos, id);
        values_.insert(values_.begin() + pos, Value());
        return values_[pos];
    }

    void move_to (Id id, IndexedVector<Value> & destin)
    {
        size_t pos = find(id);
        destin.insert(id) = std::move(values_[pos]);
        index_.erase(index_.begin() + pos);
        values_.erase(values_.begin() + pos);
    }

    void clear ()
    {
        index_.clear();
        values_.clear();
    }

    //------------------------------------------------------------------------
    // zero-overhead element-wise interface

    size_t size () const { return values_.size(); }
    Value & operator[] (size_t pos) { return values_[pos]; }
    const Value & operator[] (size_t pos) const { return values_[pos]; }

private:

    size_t find (Id id)
    {
        auto i = std::lower_bound(index_.begin(), index_.end(), id);
        LOOM_ASSERT(i == index_.end() or *i != id, "duplicate id: " << id);
        return i - index_.begin();
    }

    std::vector<Value> values_;
    std::vector<Id> index_;
};

} // namespace loom
