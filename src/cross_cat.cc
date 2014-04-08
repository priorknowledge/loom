#include <sstream>
#include <iomanip>
#include <distributions/io/protobuf.hpp>
#include <distributions/io/protobuf_stream.hpp>
#include "cross_cat.hpp"

namespace loom
{

void CrossCat::model_load (const char * filename)
{
    protobuf::CrossCatModel message;
    protobuf::InFile(filename).read(message);

    schema.clear();
    const size_t kind_count = message.kinds_size();
    kinds.resize(kind_count);
    for (size_t kindid = 0; kindid < kind_count; ++kindid) {
        auto & kind = kinds[kindid];
        const auto & message_kind = message.kinds(kindid);
        kind.model.load(message_kind.product_model());
        schema += kind.model.schema;

        kind.featureids.clear();
        for (size_t i = 0; i < message_kind.featureids_size(); ++i) {
            kind.featureids.push_back(message_kind.featureids(i));
        }
    }

    distributions::clustering_load(clustering, message.clustering());

    featureid_to_kindid.clear();
    for (size_t i = 0; i < message.featureid_to_kindid_size(); ++i) {
        featureid_to_kindid.push_back(message.featureid_to_kindid(i));
    }
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
    for (size_t kindid = 0; kindid < kind_count; ++kindid) {
        Kind & kind = kinds[kindid];
        std::string filename = get_mixture_filename(dirname, kindid);
        kind.mixture.load(kind.model, filename.c_str(), rng);
    }
}

void CrossCat::mixture_dump (const char * dirname)
{
    const size_t kind_count = kinds.size();
    for (size_t kindid = 0; kindid < kind_count; ++kindid) {
        Kind & kind = kinds[kindid];
        std::string filename = get_mixture_filename(dirname, kindid);
        kind.mixture.dump(kind.model, filename.c_str());
    }
}

} // namespace loom
