#include <loom/cat_kernel.hpp>
#include <loom/hyper_kernel.hpp>
#include <loom/kind_proposer.hpp>
#include <loom/infer_grid.hpp>

namespace loom
{

void generate_rows (
        const protobuf::Config::Generate & config,
        CrossCat & cross_cat,
        const char * rows_out,
        rng_t & rng)
{
    const size_t kind_count = cross_cat.kinds.size();
    const size_t row_count = config.row_count();
    const float density = config.density();
    LOOM_ASSERT_LE(0.0, density);
    LOOM_ASSERT_LE(density, 1.0);
    VectorFloat scores;
    std::vector<ProductModel::Value> partial_values(kind_count);
    CrossCat::ValueJoiner value_join(cross_cat);
    protobuf::SparseRow row;
    protobuf::OutFile rows(rows_out);

    for (size_t id = 0; id < row_count; ++id) {

        for (size_t k = 0; k < kind_count; ++k) {
            auto & kind = cross_cat.kinds[k];
            ProductModel & model = kind.model;
            auto & mixture = kind.mixture;
            ProductModel::Value & value = partial_values[k];

            scores.resize(mixture.clustering.counts().size());
            mixture.clustering.score_value(model.clustering, scores);
            distributions::scores_to_probs(scores);
            const VectorFloat & probs = scores;

            value.clear_observed();
            const size_t feature_count = kind.featureids.size();
            for (size_t f = 0; f < feature_count; ++f) {
                bool observed = distributions::sample_bernoulli(rng, density);
                value.add_observed(observed);
            }
            size_t groupid = mixture.sample_value(model, probs, value, rng);

            model.add_value(value, rng);
            mixture.add_value(model, groupid, value, rng);
        }

        row.set_id(id);
        value_join(* row.mutable_data(), partial_values);
        rows.write_stream(row);
    }
}

} // namespace loom
