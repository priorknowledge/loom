#include "algorithm8.hpp"

namespace loom
{

void Algorithm8::clear ()
{
    TODO("clear");
}

void Algorithm8::model_load (CrossCat & cross_cat)
{
    TODO("load model");
}

void Algorithm8::model_dump (CrossCat & cross_cat)
{
    TODO("dump model");
}

void Algorithm8::mixture_dump (CrossCat & cross_cat)
{
    TODO("dump mixtures");
}

void Algorithm8::mixture_init_empty (rng_t & rng, size_t ephemeral_kind_count)
{
    LOOM_ASSERT(ephemeral_kind_count > 0, "no ephemeral kinds");
    TODO("add ephemeral kinds");
    for (auto & kind : kinds) {
        kind.mixture.init_empty(model, rng);
    }
}

} // namespace loom
