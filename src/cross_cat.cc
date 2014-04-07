#include <sstream>
#include <iomanip>
#include <distributions/io/protobuf.hpp>
#include <distributions/io/protobuf_stream.hpp>
#include "cross_cat.hpp"

namespace loom
{

void CrossCat::load (const char * filename)
{
    protobuf::CrossCatModel message;
    protobuf::InFile(filename).read(message);

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

void CrossCat::mixture_load (const char * dirname, rng_t & rng)
{
    const size_t kind_count = kinds.size();
    for (size_t i = 0; i < kind_count; ++i) {
        Kind & kind = kinds[i];
        std::string filename = get_mixture_filename(dirname, i);
        kind.mixture.load(kind.model, filename.c_str(), rng);
    }
}

void CrossCat::mixture_dump (const char * dirname)
{
    const size_t kind_count = kinds.size();
    for (size_t i = 0; i < kind_count; ++i) {
        Kind & kind = kinds[i];
        std::string filename = get_mixture_filename(dirname, i);
        kind.mixture.dump(kind.model, filename.c_str());
    }
}

} // namespace loom
