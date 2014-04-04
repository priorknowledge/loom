#include "product_model.hpp"

namespace loom
{

void ProductModel::load (const protobuf::ProductModel & message)
{
    distributions::clustering_load(clustering, message.clustering());

    for (size_t i = 0; i < message.bb_size(); ++i) {
        TODO("load bb models");
    }

    dd.resize(message.dd_size());
    for (size_t i = 0; i < message.dd_size(); ++i) {
        distributions::model_load(dd[i], message.dd(i));
    }

    dpd.resize(message.dpd_size());
    for (size_t i = 0; i < message.dpd_size(); ++i) {
        distributions::model_load(dpd[i], message.dpd(i));
    }

    gp.resize(message.gp_size());
    for (size_t i = 0; i < message.gp_size(); ++i) {
        distributions::model_load(gp[i], message.gp(i));
    }

    nich.resize(message.nich_size());
    for (size_t i = 0; i < message.nich_size(); ++i) {
        distributions::model_load(nich[i], message.nich(i));
    }
}

void ProductModel::mixture_load (
        Mixture & mixture,
        const char * filename) const
{
    TODO("load");
}

struct ProductModel::dump_group_fun
{
    size_t groupid;
    protobuf::ProductModel::Group & message;

    void operator() (
            const distributions::DirichletDiscrete<16> & model,
            const distributions::DirichletDiscrete<16>::Classifier & mixture)
    {
        distributions::group_dump(
                model,
                mixture.groups[groupid],
                * message.add_dd());
    }

    void operator() (
            const distributions::DirichletProcessDiscrete & model,
            const distributions::DirichletProcessDiscrete::Classifier &
                mixture)
    {
        distributions::group_dump(
                model,
                mixture.groups[groupid],
                * message.add_dpd());
    }

    void operator() (
            const distributions::GammaPoisson & model,
            const distributions::GammaPoisson::Classifier & mixture)
    {
        distributions::group_dump(
                model,
                mixture.groups[groupid],
                * message.add_gp());
    }

    void operator() (
            const distributions::NormalInverseChiSq & model,
            const distributions::NormalInverseChiSq::Classifier & mixture)
    {
        distributions::group_dump(
                model,
                mixture.groups[groupid],
                * message.add_nich());
    }
};

void ProductModel::mixture_dump (
        Mixture & mixture,
        const char * filename) const
{
    loom::protobuf::OutFile groups_stream(filename);
    protobuf::ProductModel::Group message;
    const size_t group_count = mixture.clustering.counts.size();
    for (size_t i = 0; i < group_count; ++i) {
        dump_group_fun fun = {i, message};
        apply_dense(fun, mixture);
        groups_stream.write_stream(message);
        message.Clear();
    }
}


} // namespace loom
