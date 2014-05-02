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

    void mixture_init_empty (rng_t & rng, size_t kind_count);

    void infer_assignments (
            std::vector<uint32_t> & featureid_to_kindid,
            size_t iterations,
            rng_t & rng) const;

    void validate (const CrossCat & cross_cat) const;

private:

    //template<class ClusteringT>
    //static void sample_assignments (
    //        const ClusteringT & clustering,
    //        const std::vector<VectorFloat> & likelihoods,
    //        std::vector<uint32_t> & assignments,
    //        size_t iterations,
    //        rng_t & rng)
    //{
    //    LOOM_ERROR("not implemented for " << typeid(ClusteringT).name());
    //}

    static void sample_assignments (
            const distributions::Clustering<int>::PitmanYor & clustering,
            const std::vector<VectorFloat> & likelihoods,
            std::vector<uint32_t> & assignments,
            size_t iterations,
            rng_t & rng);
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
        }
    }
}

} // namespace loom
