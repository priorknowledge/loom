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

#include <loom/timer.hpp>
#include <loom/cross_cat.hpp>

namespace loom
{

class QueryServer
{
public:

    typedef protobuf::Query::Request Request;
    typedef protobuf::Query::Response Response;

    QueryServer (const std::vector<const CrossCat *> & cross_cats) :
        cross_cats_(cross_cats)
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

    void score_row (
            rng_t & rng,
            const Request & request,
            Response & response);

    void sample_rows (
            rng_t & rng,
            const Request & request,
            Response & response);

    const std::vector<const CrossCat *> cross_cats_;
    ProductValue::Diff temp_diff_;
    std::vector<ProductValue::Diff> partial_diffs_;
    std::vector<std::vector<ProductValue::Diff>> result_factors_;
    std::vector<ProductValue *> temp_values_;
    VectorFloat scores_;
    Timer timer_;
};

} // namespace loom
