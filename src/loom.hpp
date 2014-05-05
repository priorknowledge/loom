#include "common.hpp"
#include "cross_cat.hpp"
#include "algorithm8.hpp"
#include "assignments.hpp"

namespace loom
{

class Loom : noncopyable
{
public:

    Loom (
            rng_t & rng,
            const char * model_in,
            const char * groups_in = nullptr,
            const char * assign_in = nullptr,
            size_t empty_group_count = 1);

    void dump (
            const char * model_out = nullptr,
            const char * groups_out = nullptr,
            const char * assign_out = nullptr);

    void infer_single_pass (
            rng_t & rng,
            const char * rows_in,
            const char * assign_out = nullptr);

    void infer_multi_pass (
            rng_t & rng,
            const char * rows_in,
            double extra_passes);

    void infer_kind_structure (
            rng_t & rng,
            const char * rows_in,
            double extra_passes,
            size_t ephemeral_kind_count,
            size_t iterations);

    void posterior_enum (
            rng_t & rng,
            const char * rows_in,
            const char * samples_out,
            size_t sample_count);

    void posterior_enum (
            rng_t & rng,
            const char * rows_in,
            const char * samples_out,
            size_t sample_count,
            size_t ephemeral_kind_count,
            size_t iterations);

    void predict (
            rng_t & rng,
            const char * queries_in,
            const char * results_out);

    void validate_cross_cat () const;
    void validate_algorithm8 () const;
    void validate () const
    {
        validate_cross_cat();
        validate_algorithm8();
    }

    size_t count_untracked_rows () const;

private:

    void dump_posterior_enum (
            protobuf::PosteriorEnum::Sample & message,
            rng_t & rng);

    void prepare_algorithm8 (
            size_t ephemeral_kind_count,
            rng_t & rng);

    size_t run_algorithm8 (
            size_t ephemeral_kind_count,
            size_t iterations,
            rng_t & rng);

    void cleanup_algorithm8 (rng_t & rng);

    void run_hyper_inference (rng_t & rng);

    void add_featureless_kind (rng_t & rng);

    void remove_featureless_kind (size_t kindid);

    void init_featureless_kinds (
            size_t featureless_kind_count,
            rng_t & rng);

    void move_feature_to_kind (
            size_t featureid,
            size_t new_kindid,
            rng_t & rng);

    void add_row_noassign (
            rng_t & rng,
            const protobuf::SparseRow & row);

    void add_row (
            rng_t & rng,
            const protobuf::SparseRow & row,
            protobuf::Assignment & assignment_out);

    bool try_add_row (
            rng_t & rng,
            const protobuf::SparseRow & row);

    bool try_add_row_algorithm8 (
            rng_t & rng,
            const protobuf::SparseRow & row);

    void remove_row (
            rng_t & rng,
            const protobuf::SparseRow & row);

    void remove_row_algorithm8 (
            rng_t & rng,
            const protobuf::SparseRow & row);

    void predict_row (
            rng_t & rng,
            const protobuf::PreQL::Predict::Query & query,
            protobuf::PreQL::Predict::Result & result);

    const size_t empty_group_count_;
    CrossCat cross_cat_;
    Algorithm8 algorithm8_;
    Assignments assignments_;
    CrossCat::ValueJoiner value_join_;
    std::vector<ProductModel::Value> factors_;
    ProductModel::Value unobserved_;
    VectorFloat scores_;
};

inline void Loom::validate_cross_cat () const
{
    cross_cat_.validate();
    assignments_.validate();
    LOOM_ASSERT_EQ(cross_cat_.kinds.size(), factors_.size());
}

inline void Loom::validate_algorithm8 () const
{
    algorithm8_.validate(cross_cat_);
}

} // namespace loom
