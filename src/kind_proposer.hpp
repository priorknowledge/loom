#pragma once

#include <utility>
#include <typeinfo>
#include <loom/common.hpp>
#include <loom/cross_cat.hpp>
#include <loom/timer.hpp>

namespace loom
{

struct KindProposer
{
    typedef CrossCat::Value Value;
    struct Kind
    {
        ProductModel::SimpleMixture mixture;
    };

    Clustering::Shared feature_clustering;
    ProductModel model;  // model.clustering is never used
    std::vector<Kind> kinds;

    void clear ();

    void model_load (const CrossCat & cross_cat);
    void model_update (const CrossCat & cross_cat);

    void mixture_init_empty (
            const CrossCat & cross_cat,
            rng_t & rng);

    std::pair<usec_t, usec_t> infer_assignments (
            std::vector<uint32_t> & featureid_to_kindid,
            size_t iterations,
            bool parallel,
            rng_t & rng) const;

    void validate (const CrossCat & cross_cat) const;

private:

    class BlockPitmanYorSampler;

    struct model_update_fun;
};

inline void KindProposer::validate (const CrossCat & cross_cat) const
{
    if (LOOM_DEBUG_LEVEL >= 1) {
        if (kinds.empty()) {
            LOOM_ASSERT_EQ(model.schema.total_size(), 0);
        } else {
            LOOM_ASSERT_EQ(model.schema, cross_cat.schema);
            LOOM_ASSERT_EQ(kinds.size(), cross_cat.kinds.size());
            for (const auto & kind : kinds) {
                kind.mixture.validate(model);
            }
            for (size_t i = 0; i < kinds.size(); ++i) {
                size_t proposer_group_count =
                    kinds[i].mixture.clustering.counts().size();
                size_t cross_cat_group_count =
                    cross_cat.kinds[i].mixture.clustering.counts().size();
                LOOM_ASSERT_EQ(proposer_group_count, cross_cat_group_count);
            }
        }
    }
}

} // namespace loom
