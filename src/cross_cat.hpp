#pragma once

#include <unordered_set>
#include <distributions/vector.hpp>
#include <loom/common.hpp>
#include <loom/protobuf.hpp>
#include <loom/product_model.hpp>

namespace loom
{

struct CrossCat : noncopyable
{
    typedef protobuf::ProductModel::SparseValue Value;
    typedef ProductModel::CachedMixture ProductMixture;
    struct Kind
    {
        ProductModel model;
        ProductMixture mixture;
        std::unordered_set<size_t> featureids;
    };

    protobuf::SparseValueSchema schema;
    protobuf::CrossCat_HyperPrior hyper_prior;
    Clustering::Shared feature_clustering;
    distributions::Packed_<Kind> kinds;
    std::vector<uint32_t> featureid_to_kindid;

    void model_load (const char * filename);
    void model_dump (const char * filename) const;

    void mixture_init_empty (
            size_t empty_group_count,
            rng_t & rng);
    void mixture_load (
            const char * dirname,
            size_t empty_group_count,
            rng_t & rng);
    void mixture_dump (
            const char * dirname,
            const std::vector<std::vector<uint32_t>> & sorted_to_globals) const;

    std::vector<std::vector<uint32_t>> get_sorted_groupids () const;

    void value_split (
            const Value & full_value,
            std::vector<Value> & partial_values) const;

    void value_split_observed (
            const Value & full_value,
            std::vector<Value> & partial_values) const;

    struct ValueJoiner;
    void value_join (
            Value & full_value,
            const std::vector<Value> & partial_values) const;

    float score_data (rng_t & rng) const;

    void validate () const;

private:

    void validate (const Value & full_value) const;
    void validate (const std::vector<Value> & partial_values) const;

    std::string get_mixture_filename (
            const char * dirname,
            size_t kindid) const;

    struct value_split_fun;
    struct value_split_observed_fun;
    struct value_join_fun;
};

inline void CrossCat::mixture_init_empty (size_t empty_group_count, rng_t & rng)
{
    const std::vector<int> counts(empty_group_count, 0);
    for (auto & kind : kinds) {
        kind.mixture.init_unobserved(kind.model, counts, rng);
    }
}

struct CrossCat::value_split_fun
{
    const CrossCat & cross_cat;
    const Value & full_value;
    std::vector<Value> & partial_values;
    size_t absolute_pos;

    template<class FieldType>
    inline void operator() (FieldType *, size_t size)
    {
        typedef protobuf::Fields<FieldType> Fields;
        const auto & full_fields = Fields::get(full_value);
        for (size_t i = 0, packed_pos = 0; i < size; ++i, ++absolute_pos) {
            auto kindid = cross_cat.featureid_to_kindid[absolute_pos];
            auto & partial_value = partial_values[kindid];
            bool observed = full_value.observed(absolute_pos);
            partial_value.add_observed(observed);
            if (observed) {
                Fields::get(partial_value).Add(full_fields.Get(packed_pos++));
            }
        }
    }
};

inline void CrossCat::value_split (
        const Value & full_value,
        std::vector<Value> & partial_values) const
{
    validate(full_value);

    partial_values.resize(kinds.size());
    for (auto & partial_value : partial_values) {
        partial_value.Clear();
    }
    value_split_fun fun = {* this, full_value, partial_values, 0};
    schema.for_each_datatype(fun);

    if (LOOM_DEBUG_LEVEL >= 1) {
        size_t feature_count = featureid_to_kindid.size();
        LOOM_ASSERT_EQ(fun.absolute_pos, feature_count);
    }

    validate(partial_values);
}

struct CrossCat::value_split_observed_fun
{
    const CrossCat & cross_cat;
    const Value & full_value;
    std::vector<Value> & partial_values;
    size_t absolute_pos;

    template<class FieldType>
    inline void operator() (FieldType *, size_t size)
    {
        for (size_t i = 0; i < size; ++i, ++absolute_pos) {
            auto kindid = cross_cat.featureid_to_kindid[absolute_pos];
            auto & partial_value = partial_values[kindid];
            bool observed = full_value.observed(absolute_pos);
            partial_value.add_observed(observed);
        }
    }
};

inline void CrossCat::value_split_observed (
        const Value & full_value,
        std::vector<Value> & partial_values) const
{
    partial_values.resize(kinds.size());
    for (auto & partial_value : partial_values) {
        partial_value.Clear();
    }
    value_split_observed_fun fun = {* this, full_value, partial_values, 0};
    schema.for_each_datatype(fun);

    if (LOOM_DEBUG_LEVEL >= 1) {
        size_t feature_count = featureid_to_kindid.size();
        LOOM_ASSERT_EQ(fun.absolute_pos, feature_count);
    }
}

struct CrossCat::value_join_fun
{
    const CrossCat & cross_cat;
    std::vector<size_t> & absolute_pos_list;
    std::vector<size_t> & packed_pos_list;
    Value & full_value;
    const std::vector<Value> & partial_values;
    size_t featureid;

    template<class FieldType>
    void operator() (FieldType *, size_t size)
    {
        if (size) {
            typedef protobuf::Fields<FieldType> Fields;
            auto & full_fields = Fields::get(full_value);
            std::fill(packed_pos_list.begin(), packed_pos_list.end(), 0);
            for (size_t end = featureid + size; featureid < end; ++featureid) {
                auto kindid = cross_cat.featureid_to_kindid[featureid];
                auto & partial_value = partial_values[kindid];
                auto & absolute_pos = absolute_pos_list[kindid];
                bool observed = partial_value.observed(absolute_pos++);
                full_value.add_observed(observed);
                if (observed) {
                    auto & packed_pos = packed_pos_list[kindid];
                    auto & partial_fields = Fields::get(partial_value);
                    full_fields.Add(partial_fields.Get(packed_pos++));
                }
            }
        }
    }
};

struct CrossCat::ValueJoiner
{
    ValueJoiner (const CrossCat & cross_cat) : cross_cat_(cross_cat) {}

    void operator() (
            Value & full_value,
            const std::vector<Value> & partial_values)
    {
        //LOOM_DEBUG(partial_values);
        cross_cat_.validate(partial_values);

        full_value.Clear();
        absolute_pos_list_.clear();
        absolute_pos_list_.resize(cross_cat_.kinds.size(), 0);
        packed_pos_list_.resize(cross_cat_.kinds.size());
        CrossCat::value_join_fun fun = {
            cross_cat_,
            absolute_pos_list_,
            packed_pos_list_,
            full_value,
            partial_values,
            0};
        cross_cat_.schema.for_each_datatype(fun);

        if (LOOM_DEBUG_LEVEL >= 1) {
            size_t feature_count = cross_cat_.featureid_to_kindid.size();
            LOOM_ASSERT_EQ(fun.featureid, feature_count);
        }

        cross_cat_.validate(full_value);
    }

private:

    const CrossCat & cross_cat_;
    std::vector<size_t> absolute_pos_list_;
    std::vector<size_t> packed_pos_list_;
};

inline void CrossCat::value_join (
        Value & full_value,
        const std::vector<Value> & partial_values) const
{
    ValueJoiner(* this)(full_value, partial_values);
}

inline void CrossCat::validate () const
{
    if (LOOM_DEBUG_LEVEL >= 1) {
        LOOM_ASSERT_LT(0, schema.total_size());
        protobuf::SparseValueSchema expected_schema;
        for (const auto & kind : kinds) {
            kind.mixture.validate(kind.model);
            expected_schema += kind.model.schema;
        }
        LOOM_ASSERT_EQ(schema, expected_schema);
    }
    if (LOOM_DEBUG_LEVEL >= 2) {
        for (size_t f = 0; f < featureid_to_kindid.size(); ++f) {
            size_t k = featureid_to_kindid[f];
            const auto & featureids = kinds[k].featureids;
            LOOM_ASSERT(
                featureids.find(f) != featureids.end(),
                "kind.featureids is missing " << f);
        }
        for (size_t k = 0; k < kinds.size(); ++k) {
            for (size_t f : kinds[k].featureids) {
                LOOM_ASSERT_EQ(featureid_to_kindid[f], k);
            }
        }
    }
    if (LOOM_DEBUG_LEVEL >= 3) {
        std::vector<size_t> row_counts;
        for (const auto & kind : kinds) {
            row_counts.push_back(kind.mixture.count_rows());
        }
        for (size_t k = 1; k < kinds.size(); ++k) {
            LOOM_ASSERT_EQ(row_counts[k], row_counts[0]);
        }
    }
}

inline void CrossCat::validate (const Value & full_value) const
{
    if (LOOM_DEBUG_LEVEL >= 2) {
        schema.validate(full_value);
    }
}

inline void CrossCat::validate (const std::vector<Value> & partial_values) const
{
    if (LOOM_DEBUG_LEVEL >= 2) {
        const size_t kind_count = kinds.size();
        LOOM_ASSERT_EQ(partial_values.size(), kind_count);
        for (size_t k = 0; k < kind_count; ++k) {
            kinds[k].model.schema.validate(partial_values[k]);
        }
    }
}

} // namespace loom
