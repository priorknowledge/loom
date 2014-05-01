#pragma once

#include <vector>
#include "common.hpp"

namespace loom
{

// This conforms to the Visitor interface implicit in
// distributions::protobuf::for_each_gridpoint(..., Visitor &)
// $DISTRIBUTIONS_PATH/include/distributions/io/protobuf.hpp

template<class Mixture>
class InferShared
{
public:

    typedef typename Mixture::Shared Shared;

    InferShared (
            Shared & shared,
            const Mixture & mixture,
            rng_t & rng) :
        shared_(shared),
        mixture_(mixture),
        rng_(rng)
    {
    }

    const Shared & shared () const
    {
        return shared_;
    }

    Shared & add ()
    {
        hypotheses_.push_back(shared_);
        return hypotheses_.back();
    }

    void done ()
    {
        const size_t size = hypotheses_.size();
        if (size == 1) {

            shared_ = hypotheses_[0];

        } else if (size > 1) {

            scores_.reserve(size);
            for (const auto & shared : hypotheses_) {
                scores_.push_back(mixture_.score_mixture(shared, rng_));
            }
            size_t i = sample_from_scores_overwrite(rng_, scores_);
            shared_ = hypotheses_[i];
        }
        hypotheses_.clear();
        scores_.clear();
    }

private:

    Shared & shared_;
    const Mixture & mixture_;
    std::vector<Shared> hypotheses_;
    VectorFloat scores_;
    rng_t & rng_;
};

} // namespace loom
