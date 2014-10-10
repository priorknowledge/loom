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
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <loom/timer.hpp>
#include <loom/cross_cat.hpp>

namespace loom
{

class QueryServer
{
public:

    typedef protobuf::Query Query;
    typedef google::protobuf::RepeatedPtrField<std::string> Errors;

    QueryServer (
            const std::vector<const CrossCat *> & cross_cats,
            const protobuf::Config & config,
            const char * rows_in) :
        config_(config),
        cross_cats_(cross_cats),
        rows_in_(rows_in)
    {
        LOOM_ASSERT(not cross_cats_.empty(), "no cross cats found");
    }

    void serve (
            rng_t & rng,
            const char * requests_in,
            const char * responses_out);

private:

    const ValueSchema schema () const { return cross_cats_[0]->schema; }
    const std::vector<ProductValue> tares () const
    {
        return cross_cats_[0]->tares;
    }

    bool validate (
            const Query::Sample::Request & request,
            Errors & errors) const;

    bool validate (
            const Query::Score::Request & request,
            Errors & errors) const;

    bool validate (
            const Query::Entropy::Request & request,
            Errors & errors) const;

    bool validate (
            const Query::ScoreDerivative::Request & request,
            Errors & errors) const;

    void call (
            rng_t & rng,
            const Query::Sample::Request & request,
            Query::Sample::Response & response) const;

    void call (
            rng_t & rng,
            const Query::Score::Request & request,
            Query::Score::Response & response) const;

    void call (
            rng_t & rng,
            const Query::Entropy::Request & request,
            Query::Entropy::Response & response) const;

    // not threadsafe
    void call (
            rng_t & rng,
            const Query::ScoreDerivative::Request & request,
            Query::ScoreDerivative::Response & response) const;

    const protobuf::Config config_;
    const std::vector<const CrossCat *> cross_cats_;
    const char * rows_in_;
    Timer timer_;
};

} // namespace loom
