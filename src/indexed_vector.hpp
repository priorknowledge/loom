#pragma once

#include <vector>
#include <algorithm>

namespace loom
{

//----------------------------------------------------------------------------
// Indexed Vector
//
// Design Goals:
//  * Maintain an std::vector<Value> sorted by Id.
//  * Provide fast vector operations: operator[] and iterators.
//  * Provide slow insert and remove operations.

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

    const std::vector<Id> & index () const { return index_; }
    Id index (size_t pos) const { return index_.at(pos); }

    Maybe<Id> try_find_pos (Id id) const
    {
        size_t pos = lower_bound(id);
        if (pos == size() or index_[pos] != id) {
            return Maybe<Id>();
        } else {
            return Maybe<Id>(pos);
        }
    }

    Value & find (Id id)
    {
        size_t pos = lower_bound(id);
        LOOM_ASSERT(pos != size() and index(pos) == id, "missing id: " << id);
        return values_[pos];
    }

    Value & insert (Id id)
    {
        size_t pos = lower_bound(id);
        LOOM_ASSERT(pos == size() or index(pos) != id, "duplicate id: " << id);
        index_.insert(index_.begin() + pos, id);
        values_.insert(values_.begin() + pos, Value());
        return values_[pos];
    }

    Value & find_or_insert (Id id)
    {
        size_t pos = lower_bound(id);
        if (LOOM_UNLIKELY(pos == size() or index(pos) != id)) {
            index_.insert(index_.begin() + pos, id);
            values_.insert(values_.begin() + pos, Value());
        }
        return values_[pos];
    }

    void extend (const IndexedVector & other)
    {
        for (size_t i = 0, size = other.size(); i < size; ++i) {
            insert(other.index_[i]) = other.values_[i];
        }
    }

    void remove (Id id)
    {
        size_t pos = lower_bound(id);
        LOOM_ASSERT(pos != size() and index(pos) == id, "missing id: " << id);
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

    bool empty () const { return values_.empty(); }
    size_t size () const { return values_.size(); }

    Value & operator[] (size_t pos)
    {
        LOOM_ASSERT2(pos < size(), "out of bounds: " << pos);
        return values_[pos];
    }
    const Value & operator[] (size_t pos) const
    {
        LOOM_ASSERT2(pos < size(), "out of bounds: " << pos);
        return values_[pos];
    }

    typedef typename std::vector<Value>::iterator iterator;
    iterator begin () { return values_.begin(); }
    iterator end () { return values_.end(); }

    typedef typename std::vector<Value>::const_iterator const_iterator;
    const_iterator begin () const { return values_.begin(); }
    const_iterator end () const { return values_.end(); }

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
