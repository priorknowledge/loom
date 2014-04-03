#include <cstdlib>
#include <distributions/random.hpp>
#include "common.hpp"
#include "product_model.hpp"
#include "protobuf.hpp"

const char * help_message =
"Usage: loom MODEL_IN VALUES_IN GROUPS_OUT"
;

int main (int argc, char ** argv)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    if (argc != 4) {
        std::cerr << help_message << std::endl;
        exit(1);
    }

    const char * model_in = argv[2];
    const char * values_in = argv[3];
    const char * groups_out = argv[4];

    distributions::rng_t rng;

    loom::ProductModel model;
    model.load(model_in);

    loom::ProductMixture mixture(model);
    mixture.init(rng);
    //mixture.load(groups_in);

    {
        distributions::VectorFloat scores;
        loom::protobuf::SparseRow row;
        loom::protobuf::InFileStream values_stream(values_in);

        while (values_stream.try_read(row)) {
            //const uint64_t id = row.id();
            const loom::ProductMixture::Value & value = row.value();

            mixture.score(value, scores, rng);
            size_t groupid = distributions::sample_discrete(
                rng,
                scores.size(),
                scores.data());
            mixture.add_value(groupid, value, rng);
        }
    }

    mixture.dump(groups_out);

    return 0;
}
