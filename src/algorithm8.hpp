#pragma once

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

    protobuf::SparseValueSchema schema;
    ProductModel model;
    std::vector<Kind> kinds;

    void clear ();

    void model_load (CrossCat & cross_cat);
    void model_dump (CrossCat & cross_cat);

    void mixture_init_empty (rng_t & rng, size_t ephemeral_kind_count);
    void mixture_dump (CrossCat & cross_cat);
};

} // namespace loom
