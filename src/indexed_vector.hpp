#pragma once

#include <vector>
#include <algorithm>

namespace loom
{

template<class Value>
class Maybe
{
public:

    Maybe () : whether_(false), value_() {}
    Maybe (const Value & value) : whether_(true), value_(value) {}

    operator bool () const { return whether_; }
    const Value & value () const { return value_; }

private:

    const bool whether_;
    const Value value_;
};

template<class Value>
class IndexedVector
{
public:

    //------------------------------------------------------------------------
    // slow structural interface

    typedef uint32_t Id;

    Id index (size_t pos) const { return index_.at(pos); }

    Maybe<Id> try_find (Id id) const
    {
        size_t pos = lower_bound(id);
        if (pos == size() or index_[pos] != id) {
            return Maybe<Id>();
        } else {
            return Maybe<Id>(pos);
        }
    }

    Value & insert (Id id)
    {
        size_t pos = lower_bound(id);
        LOOM_ASSERT(pos == size() or index(pos) != id, "duplicate id: " << id);
        index_.insert(index_.begin() + pos, id);
        values_.insert(values_.begin() + pos, Value());
        return values_[pos];
    }

    void move_to (Id id, IndexedVector<Value> & destin)
    {
        size_t pos = lower_bound(id);
        LOOM_ASSERT(pos != size() and index(pos) == id, "missing id: " << id);
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

    size_t lower_bound (Id id) const
    {
        auto i = std::lower_bound(index_.begin(), index_.end(), id);
        return i - index_.begin();
    }

    std::vector<Value> values_;
    std::vector<Id> index_;
};

} // namespace loom
