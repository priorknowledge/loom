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

#include <loom/product_model.hpp>

namespace loom
{

void ProductModel::load (
        const protobuf::ProductModel_Shared & message,
        const std::vector<size_t> & featureids)
{
    clear();
    clustering.protobuf_load(message.clustering());

    size_t feature_count =
        message.bb_size() +
        message.dd_size() +
        message.dpd_size() +
        message.gp_size() +
        message.nich_size();
    LOOM_ASSERT(
        featureids.size() == feature_count,
        "kind has " << feature_count << " features, but featureids has "
        << featureids.size() << " entries");

    size_t absolute_pos = 0;

    for (size_t i = 0; i < message.bb_size(); ++i) {
        size_t featureid = featureids.at(absolute_pos++);
        features.bb.insert(featureid).protobuf_load(message.bb(i));
    }

    for (size_t i = 0; i < message.dd_size(); ++i) {
        size_t featureid = featureids.at(absolute_pos++);
        size_t dim = message.dd(i).alphas().size();
        if (dim <= 16) {
            features.dd16.insert(featureid).protobuf_load(message.dd(i));
        } else if (dim <= 256) {
            features.dd256.insert(featureid).protobuf_load(message.dd(i));
        } else {
            LOOM_ERROR("dim is too large: " << dim);
        }
    }

    for (size_t i = 0; i < message.dpd_size(); ++i) {
        size_t featureid = featureids.at(absolute_pos++);
        features.dpd.insert(featureid).protobuf_load(message.dpd(i));
    }

    for (size_t i = 0; i < message.gp_size(); ++i) {
        size_t featureid = featureids.at(absolute_pos++);
        features.gp.insert(featureid).protobuf_load(message.gp(i));
    }

    for (size_t i = 0; i < message.nich_size(); ++i) {
        size_t featureid = featureids.at(absolute_pos++);
        features.nich.insert(featureid).protobuf_load(message.nich(i));
    }

    LOOM_ASSERT_EQ(absolute_pos, featureids.size());

    update_schema();
}

struct ProductModel::dump_fun
{
    const Features & features;
    protobuf::ProductModel_Shared & message;

    template<class T>
    void operator() (T * t)
    {
        for (const auto & shared : features[t]) {
            shared.protobuf_dump(* protobuf::Fields<T>::get(message).Add());
        }
    }
};

void ProductModel::dump (protobuf::ProductModel_Shared & message) const
{
    clustering.protobuf_dump(* message.mutable_clustering());

    dump_fun fun = {features, message};
    for_each_feature_type(fun);
}


void ProductModel::update_schema ()
{
    schema.clear();
    schema.booleans_size += features.bb.size();
    schema.counts_size += features.dd16.size();
    schema.counts_size += features.dd256.size();
    schema.counts_size += features.dpd.size();
    schema.counts_size += features.gp.size();
    schema.reals_size += features.nich.size();
}

struct ProductModel::clear_fun
{
    Features & features;

    template<class T>
    void operator() (T * t)
    {
        features[t].clear();
    }
};

void ProductModel::clear ()
{
    schema.clear();

    clear_fun fun = {features};
    for_each_feature_type(fun);
}

struct ProductModel::extend_fun
{
    Features & destin;
    const Features & source;

    template<class T>
    void operator() (T * t)
    {
        destin[t].extend(source[t]);
    }
};

void ProductModel::extend (const ProductModel & other)
{
    schema += other.schema;

    extend_fun fun = {features, other.features};
    for_each_feature_type(fun);
}

} // namespace loom
