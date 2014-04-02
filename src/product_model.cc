#include <fstream>
#include "product_model.hpp"

namespace loom
{

void ProductModel::load (const char * filename)
{
    std::fstream file(filename, std::ios::in | std::ios::binary);
    LOOM_ASSERT(file, "models file not found");
    protobuf::ProductModel product_model;
    bool info = product_model.ParseFromIstream(&file);
    LOOM_ASSERT(info, "failed to parse model from file");

    clustering.alpha = product_model.clustering().pitman_yor().alpha();
    clustering.d = product_model.clustering().pitman_yor().d();

    for (size_t i = 0; i < product_model.bb_size(); ++i) {
        TODO("load bb models");
    }

    dd.resize(product_model.dd_size());
    for (size_t i = 0; i < product_model.dd_size(); ++i) {
        distributions::model_load(dd[i], product_model.dd(i));
    }

    dpd.resize(product_model.dpd_size());
    for (size_t i = 0; i < product_model.dpd_size(); ++i) {
        distributions::model_load(dpd[i], product_model.dpd(i));
    }

    gp.resize(product_model.gp_size());
    for (size_t i = 0; i < product_model.gp_size(); ++i) {
        distributions::model_load(gp[i], product_model.gp(i));
    }

    nich.resize(product_model.nich_size());
    for (size_t i = 0; i < product_model.nich_size(); ++i) {
        distributions::model_load(nich[i], product_model.nich(i));
    }
}

struct ProductMixture::dump_group_fun
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

void ProductMixture::dump (const char * filename)
{
    loom::protobuf::OutFileStream groups_stream(filename);
    protobuf::ProductModel::Group message;
    const size_t group_count = clustering.counts.size();
    for (size_t i = 0; i < group_count; ++i) {
        dump_group_fun fun = {i, message};
        apply_dense(fun);
        groups_stream.write(message);
        message.Clear();
    }
}


} // namespace loom
