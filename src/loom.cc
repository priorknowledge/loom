#include <cstdlib>
#include <distributions/random.hpp>
#include "common.hpp"
#include "cross_cat.hpp"

namespace loom
{

void infer_single_kind (
        CrossCat & cross_cat,
        protobuf::InFile & rows,
        rng_t & rng)
{
    auto & kind = cross_cat.kinds[0];
    const ProductModel & model = kind.model;
    ProductModel::Mixture & mixture = kind.mixture;

    VectorFloat scores;
    protobuf::SparseRow row;

    while (rows.try_read_stream(row)) {
        //const uint64_t id = row.id();
        //std::cout << row.id()  << ' ' << std::flush; // DEBUG
        const ProductModel::Value & value = row.data();
        LOOM_ASSERT2(
            value.observed_size() == model.feature_count,
            "bad row width: " << value.observed_size());

        model.mixture_score(mixture, value, scores, rng);
        size_t groupid =
            distributions::sample_from_scores_overwrite(rng, scores);
        model.mixture_add_value(mixture, groupid, value, rng);
    }
}

} // namespace loom

const char * help_message =
"Usage: loom MODEL_IN ROWS_IN GROUPS_OUT"
;

int main (int argc, char ** argv)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    if (argc != 4) {
        std::cerr << help_message << std::endl;
        exit(1);
    }

    const char * model_in = argv[1];
    const char * rows_in = argv[2];
    const char * groups_out = argv[3];

    distributions::rng_t rng;

    loom::CrossCat cross_cat;
    {
        loom::protobuf::CrossCatModel message;
        loom::protobuf::InFile(model_in).read(message);
        cross_cat.load(message);
    }
    cross_cat.mixture_init(rng);
    //cross_cat.mixture_load(groups_in);

    loom::protobuf::InFile rows(rows_in);

    if (cross_cat.kinds.size() == 1) {
        loom::infer_single_kind(cross_cat, rows, rng);
    } else {
        TODO("handle multiple kinds");
    }

    cross_cat.mixture_dump(groups_out);

    return 0;
}
