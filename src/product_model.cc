#include "product_model.hpp"

namespace loom
{

template<class MixtureT>
void ProductModel::Mixture::init_empty_factors (
        size_t empty_group_count,
        const std::vector<typename MixtureT::Model> & models,
        std::vector<MixtureT> & mixtures,
        rng_t & rng)
{
    const size_t model_count = models.size();
    mixtures.clear();
    mixtures.resize(model_count);
    for (size_t i = 0; i < model_count; ++i) {
        const auto & model = models[i];
        auto & mixture = mixtures[i];
        mixture.groups.resize(empty_group_count);
        for (auto & group : mixture.groups) {
            group.init(model, rng);
        }
        mixture.init(model, rng);
    }
}

void ProductModel::Mixture::init_empty (
        const ProductModel & model,
        rng_t & rng,
        size_t empty_group_count)
{
    std::vector<int> counts(empty_group_count, 0);
    clustering.init(model.clustering, counts);

    init_empty_factors(empty_group_count, model.dd, dd, rng);
    init_empty_factors(empty_group_count, model.dpd, dpd, rng);
    init_empty_factors(empty_group_count, model.gp, gp, rng);
    init_empty_factors(empty_group_count, model.nich, nich, rng);

    id_tracker.init(empty_group_count);

    _validate(model);
}

void ProductModel::load (const protobuf::ProductModel & message)
{
    schema.clear();
    distributions::clustering_load(clustering, message.clustering());

    schema.booleans_size += message.bb_size();
    for (size_t i = 0; i < message.bb_size(); ++i) {
        TODO("load bb models");
    }

    schema.counts_size += message.dd_size();
    dd.resize(message.dd_size());
    for (size_t i = 0; i < message.dd_size(); ++i) {
        distributions::model_load(dd[i], message.dd(i));
        LOOM_ASSERT1(dd[i].dim > 1, "invalid dim: " << dd[i].dim);
    }

    schema.counts_size += message.dpd_size();
    dpd.resize(message.dpd_size());
    for (size_t i = 0; i < message.dpd_size(); ++i) {
        distributions::model_load(dpd[i], message.dpd(i));
        LOOM_ASSERT1(
            dpd[i].betas.size() > 1,
            "invalid dim: " << dpd[i].betas.size());
    }

    schema.counts_size += message.gp_size();
    gp.resize(message.gp_size());
    for (size_t i = 0; i < message.gp_size(); ++i) {
        distributions::model_load(gp[i], message.gp(i));
    }

    schema.reals_size += message.nich_size();
    nich.resize(message.nich_size());
    for (size_t i = 0; i < message.nich_size(); ++i) {
        distributions::model_load(nich[i], message.nich(i));
    }
}

struct ProductModel::Mixture::load_group_fun
{
    size_t groupid;
    const protobuf::ProductModel::Group & message;

    template<class Mixture>
    void operator() (
            size_t index,
            const typename Mixture::Model & model,
            Mixture & mixture)
    {
        mixture.groups.resize(mixture.groups.size() + 1);
        distributions::group_load(
                model,
                mixture.groups[groupid],
                protobuf::Groups<typename Mixture::Group>::get(message).Get(index));
    }
};

struct ProductModel::Mixture::init_fun
{
    rng_t & rng;

    template<class Mixture>
    void operator() (
            size_t,
            const typename Mixture::Model & model,
            Mixture & mixture)
    {
        mixture.init(model, rng);
    }
};

void ProductModel::Mixture::load (
        const ProductModel & model,
        const char * filename,
        rng_t & rng,
        size_t empty_group_count)
{
    init_empty(model, rng, empty_group_count);
    protobuf::InFile groups(filename);
    protobuf::ProductModel::Group message;
    for (size_t groupid = 0; groups.try_read_stream(message); ++groupid) {
        clustering.add_value(model.clustering, groupid, message.count());
        load_group_fun fun = {groupid, message};
        apply_dense(model, fun);
    }
    init_fun fun = {rng};
    apply_dense(model, fun);
    id_tracker.init(clustering.counts().size());
    _validate(model);
}

struct ProductModel::Mixture::dump_group_fun
{
    size_t groupid;
    protobuf::ProductModel::Group & message;

    template<class Mixture>
    void operator() (
            size_t,
            const typename Mixture::Model & model,
            const Mixture & mixture)
    {
        distributions::group_dump(
                model,
                mixture.groups[groupid],
                * protobuf::Groups<typename Mixture::Group>::get(message).Add());
    }
};

void ProductModel::Mixture::dump (
        const ProductModel & model,
        const char * filename)
{
    protobuf::OutFile groups_stream(filename);
    protobuf::ProductModel::Group message;
    const size_t group_count = clustering.counts().size();
    for (size_t i = 0; i < group_count; ++i) {
        bool group_is_not_empty = clustering.counts(i);
        if (group_is_not_empty) {
            message.set_count(clustering.counts(i));
            dump_group_fun fun = {i, message};
            apply_dense(model, fun);
            groups_stream.write_stream(message);
            message.Clear();
        }
    }
}


} // namespace loom
