#include <cstdlib>
#include <distributions/random.hpp>
#include "common.hpp"
#include "cross_cat.hpp"

namespace loom
{

void infer_kind (
        CrossCat::Kind & kind,
        protobuf::InFile & rows,
        rng_t & rng)
{
    const ProductModel & model = kind.model;
    ProductModel::Mixture & mixture = kind.mixture;
    VectorFloat scores;
    protobuf::SparseRow row;

    while (rows.try_read_stream(row)) {
        //const uint64_t id = row.id();
        //std::cout << id  << ' ' << std::flush; // DEBUG
        const ProductModel::Value & value = row.data();

        model.mixture_score(mixture, value, scores, rng);
        size_t groupid =
            distributions::sample_from_scores_overwrite(rng, scores);
        model.mixture_add_value(mixture, groupid, value, rng);
    }
}

void infer_kinds (
        CrossCat & cross_cat,
        protobuf::InFile & rows,
        rng_t & rng)
{
    const size_t kind_count = cross_cat.kinds.size();
    std::vector<ProductModel::Value> factors;
    VectorFloat scores;
    protobuf::SparseRow row;

    while (rows.try_read_stream(row)) {
        //const uint64_t id = row.id();
        //std::cout << id  << ' ' << std::flush; // DEBUG
        cross_cat.value_split(row.data(), factors);

        for (size_t i = 0; i < kind_count; ++i) {
            const auto & value = factors[i];
            auto & kind = cross_cat.kinds[i];
            const ProductModel & model = kind.model;
            ProductModel::Mixture & mixture = kind.mixture;

            model.mixture_score(mixture, value, scores, rng);
            size_t groupid =
                distributions::sample_from_scores_overwrite(rng, scores);
            model.mixture_add_value(mixture, groupid, value, rng);
        }
    }
}

} // namespace loom

const char * help_message =
"Usage: loom MODEL_IN GROUPS_IN ROWS_IN GROUPS_OUT"
"\nArguments:"
"\n  Any single input can be named '-' to indicate stdin."
"\n  Any single output can be named '-' to indicate stdout."
"\n  All filenames can end with .gz to indicate gzip compression."
"\n  GROUPS_IN can be named 'EMPTY' to indicate empty initialization."
;

int main (int argc, char ** argv)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    if (argc != 5) {
        std::cerr << help_message << std::endl;
        exit(1);
    }

    const char * model_in = argv[1];
    const char * groups_in = argv[2];
    const char * rows_in = argv[3];
    const char * groups_out = argv[4];

    distributions::rng_t rng;

    loom::CrossCat cross_cat;
    cross_cat.load(model_in);
    if (strcmp(groups_in, "EMPTY") == 0) {
        cross_cat.mixture_init_empty(rng);
    } else {
        cross_cat.mixture_load(groups_in, rng);
    }

    loom::protobuf::InFile rows(rows_in);

    if (cross_cat.kinds.size() == 1) {
        loom::infer_kind(cross_cat.kinds[0], rows, rng);
    } else {
        loom::infer_kinds(cross_cat, rows, rng);
    }

    cross_cat.mixture_dump(groups_out);

    return 0;
}
