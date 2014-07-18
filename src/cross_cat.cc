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

#include <sstream>
#include <iomanip>
#include <distributions/io/protobuf.hpp>
#include <loom/protobuf_stream.hpp>
#include <loom/cross_cat.hpp>
#include <loom/infer_grid.hpp>

namespace loom
{

void CrossCat::model_load (const char * filename)
{
    protobuf::CrossCat message;
    protobuf::InFile(filename).read(message);

    schema.clear();
    featureid_to_kindid.clear();
    kinds.clear();

    const size_t kind_count = message.kinds_size();

    size_t feature_count = 0;
    for (size_t kindid = 0; kindid < kind_count; ++kindid) {
        size_t kind_feature_count = message.kinds(kindid).featureids_size();
        LOOM_ASSERT(
            kind_feature_count,
            "kind " << kindid << " has no features");
        feature_count += kind_feature_count;
    }
    featureid_to_kindid.clear();
    const uint32_t undefined = std::numeric_limits<uint32_t>::max();
    featureid_to_kindid.resize(feature_count, undefined);

    kinds.resize(kind_count);
    for (size_t kindid = 0; kindid < kind_count; ++kindid) {
        const auto & message_kind = message.kinds(kindid);
        auto & kind = kinds[kindid];

        kind.featureids.clear();
        std::vector<size_t> ordered_featureids;
        for (size_t i = 0; i < message_kind.featureids_size(); ++i) {
            size_t featureid = message_kind.featureids(i);
            LOOM_ASSERT(
                featureid < feature_count,
                "featureid out of bounds: " << featureid);
            kind.featureids.insert(featureid);
            ordered_featureids.push_back(featureid);
            LOOM_ASSERT(
                featureid_to_kindid[featureid] == undefined,
                "kind " << kindid << " has duplicate feature " << featureid);
            featureid_to_kindid[featureid] = kindid;
        }

        kind.model.load(message_kind.product_model(), ordered_featureids);
        schema += kind.model.schema;
    }

    for (size_t featureid = 0; featureid < feature_count; ++featureid) {
        LOOM_ASSERT(
            featureid_to_kindid[featureid] != undefined,
            "feature " << featureid << " appears in no kind");
    }

    topology.protobuf_load(message.topology());

    hyper_prior = message.hyper_prior();

    update_splitter();
}

void CrossCat::model_dump (const char * filename) const
{
    protobuf::CrossCat message;

    for (const auto & kind : kinds) {
        auto & message_kind = * message.add_kinds();

        std::vector<size_t> ordered_featureids(
            kind.featureids.begin(),
            kind.featureids.end());
        std::sort(ordered_featureids.begin(), ordered_featureids.end());

        for (size_t i : ordered_featureids) {
            message_kind.add_featureids(i);
        }

        kind.model.dump(* message_kind.mutable_product_model());
    }

    topology.protobuf_dump(* message.mutable_topology());

    * message.mutable_hyper_prior() = hyper_prior;

    protobuf::OutFile(filename).write(message);
}

std::string CrossCat::get_mixture_filename (
        const char * dirname,
        size_t kindid) const
{
    LOOM_ASSERT_LE(kindid, kinds.size());
    std::ostringstream filename;
    filename << dirname << "/mixture." <<
        std::setfill('0') << std::setw(3) << kindid << ".pbs.gz";
    return filename.str();
}

void CrossCat::mixture_load (
        const char * dirname,
        size_t empty_group_count,
        rng_t & rng)
{
    const size_t kind_count = kinds.size();
    const size_t feature_count = featureid_to_kindid.size();
    auto seed = rng();

    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t kindid = 0; kindid < kind_count; ++kindid) {
        rng_t rng(seed + kindid);
        Kind & kind = kinds[kindid];
        std::string filename = get_mixture_filename(dirname, kindid);
        kind.mixture.load_step_1_of_2(
            kind.model,
            filename.c_str(),
            empty_group_count);
    }
    seed += kind_count;

    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t featureid = 0; featureid < feature_count; ++featureid) {
        rng_t rng(seed + featureid);
        size_t kindid = featureid_to_kindid[featureid];
        auto & kind = kinds[kindid];
        kind.mixture.load_step_2_of_2(
            kind.model,
            featureid,
            empty_group_count,
            rng);
    }

    for (size_t kindid = 0; kindid < kind_count; ++kindid) {
        Kind & kind = kinds[kindid];
        kind.mixture.validate(kind.model);
    }
}

void CrossCat::mixture_dump (
        const char * dirname,
        const std::vector<std::vector<uint32_t>> & sorted_to_globals) const
{
    const size_t kind_count = kinds.size();
    LOOM_ASSERT(kind_count, "kind_count == 0, nothing to do");
    for (size_t kindid = 0; kindid < kind_count; ++kindid) {
        const Kind & kind = kinds[kindid];
        const auto & sorted_to_global = sorted_to_globals[kindid];
        std::string filename = get_mixture_filename(dirname, kindid);
        kind.mixture.dump(filename.c_str(), sorted_to_global);
    }
}

std::vector<std::vector<uint32_t>> CrossCat::get_sorted_groupids () const
{
    std::vector<std::vector<uint32_t>> sorted_to_globals(kinds.size());
    for (size_t k = 0; k < kinds.size(); ++k) {
        const auto & mixture = kinds[k].mixture;
        const auto & counts = mixture.clustering.counts();
        const auto & id_tracker = mixture.id_tracker;
        const size_t group_count = counts.size();
        std::vector<uint32_t> & sorted_to_global = sorted_to_globals[k];

        for (size_t packed = 0; packed < group_count; ++packed) {
            if (counts[packed]) {
                sorted_to_global.push_back(packed);
            }
        }
        std::sort(
            sorted_to_global.begin(),
            sorted_to_global.end(),
            [&](uint32_t x, uint32_t y) { return counts[x] > counts[y]; });
        for (uint32_t & packed : sorted_to_global) {
            packed = id_tracker.packed_to_global(packed);
        }
    }
    return sorted_to_globals;
}

float CrossCat::score_data (rng_t & rng) const
{
    float score = 0;
    std::vector<int> feature_counts;
    for (const auto & kind : kinds) {
        if (size_t feature_count = kind.featureids.size()) {
            feature_counts.push_back(feature_count);
            score += kind.mixture.score_data(kind.model, rng);
        }
    }
    score += topology.score_counts(feature_counts);
    return score;
}


} // namespace loom
