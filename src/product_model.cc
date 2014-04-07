#include "product_model.hpp"

namespace loom
{

template<class Model>
void ProductModel::mixture_init_empty_factors (
        const std::vector<Model> & models,
        std::vector<typename Model::Classifier> & mixtures,
        rng_t & rng) const
{
    const size_t model_count = models.size();
    mixtures.clear();
    mixtures.resize(model_count);
    for (size_t i = 0; i < model_count; ++i) {
        const auto & model = models[i];
        auto & mixture = mixtures[i];
        mixture.groups.resize(1);
        model.group_init(mixture.groups[0], rng);
        model.classifier_init(mixture, rng);
    }
}

void ProductModel::mixture_init_empty (
        Mixture & mixture,
        rng_t & rng) const
{
    mixture.clustering.counts.resize(1);
    mixture.clustering.counts[0] = 0;
    clustering.mixture_init(mixture.clustering);

    mixture_init_empty_factors(dd, mixture.dd, rng);
    mixture_init_empty_factors(dpd, mixture.dpd, rng);
    mixture_init_empty_factors(gp, mixture.gp, rng);
    mixture_init_empty_factors(nich, mixture.nich, rng);
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

struct ProductModel::load_group_fun
{
    size_t groupid;
    const protobuf::ProductModel::Group & message;

    template<class Model>
    void operator() (
            const Model & model,
            typename Model::Classifier & mixture)
    {
        mixture.groups.resize(mixture.groups.size() + 1);
        distributions::group_load(
                model,
                mixture.groups[groupid],
                protobuf::Groups<Model>::get(message).Get(groupid));
    }
};

struct ProductModel::mixture_init_fun
{
    rng_t & rng;

    template<class Model>
    void operator() (
            const Model & model,
            typename Model::Classifier & mixture)
    {
        model.classifier_init(mixture, rng);
    }
};

void ProductModel::mixture_load (
        Mixture & mixture,
        const char * filename,
        rng_t & rng) const
{
    mixture_init_empty(mixture, rng);
    protobuf::InFile groups_stream(filename);
    protobuf::ProductModel::Group message;
    for (size_t i = 0; groups_stream.try_read_stream(message); ++i) {
        clustering.mixture_add_value(mixture.clustering, i, message.count());
        load_group_fun fun = {i, message};
        apply_dense(fun, mixture);
    }
    mixture_init_fun fun = {rng};
    apply_dense(fun, mixture);
}

struct ProductModel::dump_group_fun
{
    size_t groupid;
    protobuf::ProductModel::Group & message;

    template<class Model>
    void operator() (
            const Model & model,
            const typename Model::Classifier & mixture)
    {
        distributions::group_dump(
                model,
                mixture.groups[groupid],
                * protobuf::Groups<Model>::get(message).Add());
    }
};

void ProductModel::mixture_dump (
        Mixture & mixture,
        const char * filename) const
{
    protobuf::OutFile groups_stream(filename);
    protobuf::ProductModel::Group message;
    const size_t group_count = mixture.clustering.counts.size();
    for (size_t i = 0; i < group_count; ++i) {
        bool group_is_not_empty = mixture.clustering.counts[i];
        if (group_is_not_empty) {
            message.set_count(mixture.clustering.counts[i]);
            dump_group_fun fun = {i, message};
            apply_dense(fun, mixture);
            groups_stream.write_stream(message);
            message.Clear();
        }
    }
}


} // namespace loom
