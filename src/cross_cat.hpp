#pragma once

#include "common.hpp"
#include "protobuf.hpp"
#include "product_model.hpp"

namespace loom
{

struct CrossCat
{
    typedef protobuf::ProductModel::SparseValue Value;
    typedef distributions::Clustering<int>::PitmanYor Clustering;
    struct Kind
    {
        ProductModel model;
        ProductModel::Mixture mixture;
        std::vector<size_t> featureids;
    };

    std::vector<Kind> kinds;
    distributions::Clustering<int>::PitmanYor clustering;
    //Clustering clustering;
    std::vector<size_t> featureid_to_kindid;

    void load (const protobuf::CrossCatModel & message);

    void mixture_load (const char * dirname) { TODO("load mixtures"); }
    void mixture_dump (const char * dirname);
    void mixture_init (rng_t & rng)
    {
        for (auto & kind : kinds) {
            kind.model.mixture_init(kind.mixture, rng);
        }
    }
};

} // namespace loom
