#pragma once

#include <typeinfo>
#include "common.hpp"
#include "cross_cat.hpp"

namespace loom
{

struct Algorithm8
{
    typedef CrossCat::Value Value;
    struct Kind
    {
        ProductModel::SimpleMixture mixture;
    };

    ProductModel model;
    std::vector<Kind> kinds;

    void clear ();

    void model_load (CrossCat & cross_cat);

    void mixture_init_empty (CrossCat & cross_cat, rng_t & rng);

    void infer_assignments (
            std::vector<uint32_t> & featureid_to_kindid,
            size_t iterations,
            rng_t & rng) const;

    void validate (const CrossCat & cross_cat) const;

private:

    class BlockPitmanYorSampler;
};

inline void Algorithm8::validate (const CrossCat & cross_cat) const
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
                size_t algorithm8_group_count =
                    kinds[i].mixture.clustering.counts().size();
                size_t cross_cat_group_count =
                    cross_cat.kinds[i].mixture.clustering.counts().size();
                LOOM_ASSERT_EQ(algorithm8_group_count, cross_cat_group_count);
            }
        }
    }
}

} // namespace loom
