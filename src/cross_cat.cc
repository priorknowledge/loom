#include <sstream>
#include <iomanip>
#include <distributions/io/protobuf.hpp>
#include "cross_cat.hpp"

namespace loom
{

void CrossCat::load (const protobuf::CrossCatModel & message)
{
    schema.clear();
    const size_t kind_count = message.kinds_size();
    kinds.resize(kind_count);
    for (size_t i = 0; i < kind_count; ++i) {
        ProductModel & model = kinds[i].model;
        model.load(message.kinds(i).product_model());
        schema += model.schema;
    }

    distributions::clustering_load(clustering, message.clustering());
}

void CrossCat::mixture_dump (const char * dirname)
{
    const size_t kind_count = kinds.size();
    for (size_t i = 0; i < kind_count; ++i) {
        Kind & kind = kinds[i];
        std::ostringstream filename;
        filename << dirname << "/mixture." <<
            std::setfill('0') << std::setw(3) << i << ".pbs";
        kind.model.mixture_dump(kind.mixture, filename.str().c_str());
    }
}

} // namespace loom
