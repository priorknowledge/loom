#include "loom.hpp"

namespace loom
{

//----------------------------------------------------------------------------
// StreamInterval

class StreamInterval : noncopyable
{
public:

    template<class RemoveRow>
    StreamInterval (
            const char * rows_in,
            Assignments & assignments,
            RemoveRow remove_row) :
        unassigned_(rows_in),
        assigned_(rows_in)
    {
        LOOM_ASSERT(assigned_.is_file(), "only files support StreamInterval");

        if (assignments.size()) {
            protobuf::SparseRow row;

            // point unassigned at first unassigned row
            const auto last_assigned_rowid = assignments.rowids().back();
            do {
                read_unassigned(row);
            } while (row.id() != last_assigned_rowid);

            // point rows_assigned at first assigned row
            const auto first_assigned_rowid = assignments.rowids().front();
            do {
                read_assigned(row);
            } while (row.id() != first_assigned_rowid);
            remove_row(row);
        }
    }

    void read_unassigned (protobuf::SparseRow & row)
    {
        unassigned_.cyclic_read_stream(row);
    }

    void read_assigned (protobuf::SparseRow & row)
    {
        assigned_.cyclic_read_stream(row);
    }

private:

    protobuf::InFile unassigned_;
    protobuf::InFile assigned_;
};

//----------------------------------------------------------------------------
// Loom

using ::distributions::sample_from_scores_overwrite;

Loom::Loom (
        rng_t & rng,
        const char * model_in,
        const char * groups_in,
        const char * assign_in) :
    cross_cat_(model_in),
    algorithm8_(),
    assignments_(cross_cat_.kinds.size()),
    value_join_(cross_cat_),
    factors_(cross_cat_.kinds.size()),
    scores_()
{
    LOOM_ASSERT(not cross_cat_.kinds.empty(), "no kinds, loom is empty");

    if (groups_in) {
        cross_cat_.mixture_load(groups_in, rng);
    } else {
        cross_cat_.mixture_init_empty(rng);
    }

    if (assign_in) {
        assignments_.load(assign_in);
        for (const auto & kind : cross_cat_.kinds) {
            LOOM_ASSERT_LE(
                assignments_.size(),
                kind.mixture.clustering.sample_size());
        }
    }
}

//----------------------------------------------------------------------------
// High level operations

void Loom::dump (
        const char * groups_out,
        const char * assign_out)
{
    if (groups_out) {
        cross_cat_.mixture_dump(groups_out);
    }

    if (assign_out) {
        assignments_.dump(assign_out);
    }
}

void Loom::infer_single_pass (
        rng_t & rng,
        const char * rows_in,
        const char * assign_out)
{
    protobuf::InFile rows(rows_in);
    protobuf::SparseRow row;

    if (assign_out) {

        protobuf::OutFile assignments(assign_out);
        protobuf::Assignment assignment;

        while (rows.try_read_stream(row)) {
            add_row(rng, row, assignment);
            assignments.write_stream(assignment);
        }

    } else {

        while (rows.try_read_stream(row)) {
            add_row_noassign(rng, row);
        }
    }
}

void Loom::infer_multi_pass (
        rng_t & rng,
        const char * rows_in,
        double extra_passes)
{
    auto _remove_row = [&](protobuf::SparseRow & row) { remove_row(rng, row); };
    StreamInterval rows(rows_in, assignments_, _remove_row);
    protobuf::SparseRow row;

    AnnealingSchedule schedule(extra_passes);
    while (true) {
        if (schedule.next_action_is_add()) {

            rows.read_unassigned(row);
            bool all_rows_assigned = not try_add_row(rng, row);
            if (LOOM_UNLIKELY(all_rows_assigned)) {
                break;
            }

        } else {

            rows.read_assigned(row);
            remove_row(rng, row);
        }
    }
}

class KindSchedule
{
public:

    KindSchedule (const Assignments & assignments) :
        assignments_(assignments)
    {
        reset();
    }

    bool run_after_removal ()
    {
        if (DIST_LIKELY(--timer_ == 0)) {
            return false;
        } else {
            reset();
            return true;
        }
    }

private:

    void reset () { timer_ = assignments_.size(); }

    const Assignments & assignments_;
    size_t timer_;
};

void Loom::infer_kind_structure (
        rng_t & rng,
        const char * rows_in,
        double extra_passes,
        size_t ephemeral_kind_count)
{
    algorithm8_.model_load(cross_cat_);
    algorithm8_.mixture_init_empty(rng, ephemeral_kind_count);

    auto _remove_row = [&](protobuf::SparseRow & row) { remove_row(rng, row); };
    StreamInterval rows(rows_in, assignments_, _remove_row);
    protobuf::SparseRow row;

    AnnealingSchedule annealing_schedule(extra_passes);
    KindSchedule kind_schedule(assignments_);

    while (true) {
        if (annealing_schedule.next_action_is_add()) {

            rows.read_unassigned(row);

            bool all_rows_assigned = not try_add_row_algorithm8(rng, row);
            if (LOOM_UNLIKELY(all_rows_assigned)) {
                break;
            }

        } else {

            rows.read_assigned(row);
            remove_row_algorithm8(rng, row);

            if (DIST_UNLIKELY(kind_schedule.run_after_removal())) {
                run_kind_inference(rng, ephemeral_kind_count);
            }
        }
    }

    algorithm8_.model_dump(cross_cat_);
    algorithm8_.mixture_dump(cross_cat_);
    algorithm8_.clear();
}

template<class Message>
inline std::vector<Message> load_stream (const char * filename)
{
    std::vector<Message> messages(1);
    protobuf::InFile stream(filename);
    while (stream.try_read_stream(messages.back())) {
        messages.resize(messages.size() + 1);
    }
    messages.pop_back();
    return messages;
}

void Loom::posterior_enum (
        rng_t & rng,
        const char * rows_in,
        const char * samples_out,
        size_t sample_count)
{
    infer_single_pass(rng, rows_in);

    const auto rows = load_stream<protobuf::SparseRow>(rows_in);
    protobuf::OutFile sample_stream(samples_out);
    protobuf::PosteriorEnum::Sample sample;
    for (size_t i = 0; i < sample_count; ++i) {
        for (const auto & row : rows) {
            remove_row(rng, row);
            try_add_row(rng, row);
        }
        dump(sample);
        sample_stream.write_stream(sample);
    }
}

void Loom::posterior_enum (
        rng_t & rng,
        const char * rows_in,
        const char * samples_out,
        size_t sample_count,
        size_t ephemeral_kind_count)
{
    infer_single_pass(rng, rows_in);

    TODO("sequentially infer kind structure");
}

void Loom::predict (
        rng_t & rng,
        const char * queries_in,
        const char * results_out)
{
    protobuf::InFile query_stream(queries_in);
    protobuf::OutFile result_stream(results_out);
    protobuf::PreQL::Predict::Query query;
    protobuf::PreQL::Predict::Result result;

    while (query_stream.try_read_stream(query)) {
        predict_row(rng, query, result);
        result_stream.write_stream(result);
        result_stream.flush();
    }
}

//----------------------------------------------------------------------------
// Low level operations

inline void Loom::dump (protobuf::PosteriorEnum::Sample & message)
{
    const size_t row_count = assignments_.size();
    const size_t kind_count = assignments_.dim();

    // assume assignments are sorted to simplify code
    LOOM_ASSERT_EQ(assignments_.rowids().front(), 0);
    LOOM_ASSERT_EQ(assignments_.rowids().back(), row_count - 1);

    message.Clear();
    for (auto kindid : cross_cat_.featureid_to_kindid) {
        message.add_featureid_to_kindid(kindid);
    }

    for (size_t i = 0; i < kind_count; ++i) {
        auto & kind = * message.add_kinds();
        for (auto groupid : assignments_.groupids(i)) {
            kind.add_groupids(groupid);
        }
    }
}

inline void Loom::add_row_noassign (
        rng_t & rng,
        const protobuf::SparseRow & row)
{
    cross_cat_.value_split(row.data(), factors_);

    const size_t kind_count = cross_cat_.kinds.size();
    for (size_t i = 0; i < kind_count; ++i) {
        const auto & value = factors_[i];
        auto & kind = cross_cat_.kinds[i];
        const ProductModel & model = kind.model;
        ProductModel::Mixture & mixture = kind.mixture;

        mixture.score(model, value, scores_, rng);
        size_t groupid = sample_from_scores_overwrite(rng, scores_);
        mixture.add_value(model, groupid, value, rng);
    }
}

inline void Loom::add_row (
        rng_t & rng,
        const protobuf::SparseRow & row,
        protobuf::Assignment & assignment)
{
    cross_cat_.value_split(row.data(), factors_);
    assignment.set_rowid(row.id());
    assignment.clear_groupids();

    const size_t kind_count = cross_cat_.kinds.size();
    for (size_t i = 0; i < kind_count; ++i) {
        const auto & value = factors_[i];
        auto & kind = cross_cat_.kinds[i];
        const ProductModel & model = kind.model;
        ProductModel::Mixture & mixture = kind.mixture;

        mixture.score(model, value, scores_, rng);
        size_t groupid = sample_from_scores_overwrite(rng, scores_);
        mixture.add_value(model, groupid, value, rng);
        assignment.add_groupids(groupid);
    }
}

inline bool Loom::try_add_row (
        rng_t & rng,
        const protobuf::SparseRow & row)
{
    bool already_added = not assignments_.rowids().try_push(row.id());
    if (LOOM_UNLIKELY(already_added)) {
        return false;
    }

    cross_cat_.value_split(row.data(), factors_);

    const size_t kind_count = cross_cat_.kinds.size();
    for (size_t i = 0; i < kind_count; ++i) {
        const auto & value = factors_[i];
        auto & kind = cross_cat_.kinds[i];
        const ProductModel & model = kind.model;
        ProductModel::Mixture & mixture = kind.mixture;

        mixture.score(model, value, scores_, rng);
        size_t groupid = sample_from_scores_overwrite(rng, scores_);
        mixture.add_value(model, groupid, value, rng);
        size_t global_groupid = mixture.id_tracker.packed_to_global(groupid);
        assignments_.groupids(i).push(global_groupid);
    }

    return true;
}

inline bool Loom::try_add_row_algorithm8 (
        rng_t & rng,
        const protobuf::SparseRow & row)
{
    bool already_added = not assignments_.rowids().try_push(row.id());
    if (LOOM_UNLIKELY(already_added)) {
        return false;
    }

    const ProductModel & model = algorithm8_.model;
    const auto & full_value = row.data();
    algorithm8_.value_split(full_value, factors_);

    const size_t kind_count = algorithm8_.kinds.size();
    for (size_t i = 0; i < kind_count; ++i) {
        const auto & partial_value = factors_[i];
        auto & kind = algorithm8_.kinds[i];
        ProductModel::Mixture & mixture = kind.mixture;

        mixture.score(model, partial_value, scores_, rng);
        size_t groupid = sample_from_scores_overwrite(rng, scores_);
        mixture.add_value(model, groupid, full_value, rng);
        size_t global_groupid = mixture.id_tracker.packed_to_global(groupid);
        assignments_.groupids(i).push(global_groupid);
    }

    return true;
}

inline void Loom::remove_row (
        rng_t & rng,
        const protobuf::SparseRow & row)
{
    assignments_.rowids().pop();
    cross_cat_.value_split(row.data(), factors_);

    const size_t kind_count = cross_cat_.kinds.size();
    for (size_t i = 0; i < kind_count; ++i) {
        const auto & value = factors_[i];
        auto & kind = cross_cat_.kinds[i];
        const ProductModel & model = kind.model;
        ProductModel::Mixture & mixture = kind.mixture;

        auto global_groupid = assignments_.groupids(i).pop();
        auto groupid = mixture.id_tracker.global_to_packed(global_groupid);
        mixture.remove_value(model, groupid, value, rng);
    }
}

inline void Loom::remove_row_algorithm8 (
        rng_t & rng,
        const protobuf::SparseRow & row)
{
    assignments_.rowids().pop();
    cross_cat_.value_split(row.data(), factors_);
    const ProductModel model = algorithm8_.model;

    const size_t kind_count = algorithm8_.kinds.size();
    for (size_t i = 0; i < kind_count; ++i) {
        const auto & value = factors_[i];
        auto & kind = algorithm8_.kinds[i];
        ProductModel::Mixture & mixture = kind.mixture;

        auto global_groupid = assignments_.groupids(i).pop();
        auto groupid = mixture.id_tracker.global_to_packed(global_groupid);
        mixture.remove_value(model, groupid, value, rng);
    }
}

void Loom::init_kind_inference (
        rng_t & rng,
        size_t ephemeral_kind_count)
{
    TODO("initialize sparse cross_cat");
}

void Loom::run_kind_inference (
        rng_t & rng,
        size_t ephemeral_kind_count)
{
    // Truncated approximation to Radford Neal's Algorithm 8

    TODO("score feature,kind pairs");
    TODO("run truncated algorithm 8 assignment");
    TODO("update assignments_");
    TODO("replace drifted groups with fresh groups (?)");
    TODO("resize cross_cat_.kinds to N + ephemeral_kind_count");
}

inline void Loom::predict_row (
        rng_t & rng,
        const protobuf::PreQL::Predict::Query & query,
        protobuf::PreQL::Predict::Result & result)
{
    result.Clear();
    result.set_id(query.id());
    if (not cross_cat_.schema.is_valid(query.data())) {
        result.set_error("invalid query data");
        return;
    }
    if (query.data().observed_size() != query.to_predict_size()) {
        result.set_error("observed size != to_predict size");
        return;
    }
    const size_t sample_count = query.sample_count();
    if (sample_count == 0) {
        return;
    }

    cross_cat_.value_split(query.data(), factors_);
    std::vector<std::vector<ProductModel::Value>> result_factors(1);
    {
        ProductModel::Value sample;
        * sample.mutable_observed() = query.to_predict();
        cross_cat_.value_resize(sample);
        cross_cat_.value_split(sample, result_factors[0]);
        result_factors.resize(sample_count, result_factors[0]);
    }

    const size_t kind_count = cross_cat_.kinds.size();
    for (size_t i = 0; i < kind_count; ++i) {
        if (protobuf::SparseValueSchema::total_size(result_factors[0][i])) {
            const auto & value = factors_[i];
            auto & kind = cross_cat_.kinds[i];
            const ProductModel & model = kind.model;
            ProductModel::Mixture & mixture = kind.mixture;

            mixture.score(model, value, scores_, rng);
            float total = distributions::scores_to_likelihoods(scores_);
            distributions::vector_scale(
                scores_.size(),
                scores_.data(),
                1.f / total);
            const VectorFloat & probs = scores_;

            for (auto & result_values : result_factors) {
                mixture.sample_value(model, probs, result_values[i], rng);
            }
        }
    }

    for (const auto & result_values : result_factors) {
        value_join_(* result.add_samples(), result_values);
    }
}

} // namespace loom
