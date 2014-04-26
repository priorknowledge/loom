#include "common.hpp"
#include "cross_cat.hpp"
#include "assignments.hpp"
#include "annealing_schedule.hpp"

namespace loom
{

class Loom : noncopyable
{
public:

    Loom (
            rng_t & rng,
            const char * model_in,
            const char * groups_in = nullptr,
            const char * assign_in = nullptr);

    void dump (
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

    void predict (
            rng_t & rng,
            const char * queries_in,
            const char * results_out);

private:

    void add_row_noassign (
            rng_t & rng,
            const protobuf::SparseRow & row);

    void add_row (
            rng_t & rng,
            const protobuf::SparseRow & row,
            protobuf::Assignment & assignment);

    bool try_add_row (
            rng_t & rng,
            const protobuf::SparseRow & row);

    void remove_row (
            rng_t & rng,
            const protobuf::SparseRow & row);

    void predict_row (
            rng_t & rng,
            const protobuf::PreQL::Predict::Query & query,
            protobuf::PreQL::Predict::Result & result);

    CrossCat cross_cat_;
    Assignments assignments_;
    CrossCat::ValueJoiner value_join_;
    std::vector<ProductModel::Value> factors_;
    VectorFloat scores_;
};

} // namespace loom
