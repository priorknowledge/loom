#include <sstream>
#include <distributions/protobuf.hpp>
#include "cross_cat.hpp"

namespace loom
{

void CrossCat::load (const protobuf::CrossCatModel & message)
{
    const size_t kind_count = message.kinds_size();
    kinds.resize(kind_count);
    for (size_t i = 0; i < kind_count; ++i) {
        Kind & kind = kinds[i];
        kind.model.load(message.kinds(i).product_model());
    }

    distributions::clustering_load(clustering, message.clustering());
}

void CrossCat::mixture_dump (const char * dirname)
{
    const size_t kind_count = kinds.size();
    for (size_t i = 0; i < kind_count; ++i) {
        Kind & kind = kinds[i];
        std::ostringstream filename;
        filename << dirname << "/" << i << ".";
        kind.model.mixture_dump(kind.mixture, filename.str().c_str());
    }
}

} // namespace loom
