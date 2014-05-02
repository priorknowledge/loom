#include "product_model.hpp"

namespace loom
{

void ProductModel::load (
        const protobuf::ProductModel_Shared & message,
        const std::vector<size_t> & featureids)
{
    clear();
    distributions::clustering_load(clustering, message.clustering());

    // HACK --------------------
    auto & dd = features.dd256;
    auto & dpd = features.dpd;
    auto & gp = features.gp;
    auto & nich = features.nich;
    //--------------------------

    size_t absolute_pos = 0;

    for (size_t i = 0; i < message.bb_size(); ++i) {
        TODO("load bb models");
    }

    for (size_t i = 0; i < message.dd_size(); ++i) {
        auto & shared = dd.insert(featureids.at(absolute_pos++));
        distributions::shared_load(shared, message.dd(i));
        LOOM_ASSERT1(shared.dim > 1, "invalid dim: " << shared.dim);
    }

    for (size_t i = 0; i < message.dpd_size(); ++i) {
        auto & shared = dpd.insert(featureids.at(absolute_pos++));
        distributions::shared_load(shared, message.dpd(i));
        size_t dim = shared.betas.size();
        LOOM_ASSERT1(dim > 1, "invalid dim: " << dim);
    }

    for (size_t i = 0; i < message.gp_size(); ++i) {
        auto & shared = gp.insert(featureids.at(absolute_pos++));
        distributions::shared_load(shared, message.gp(i));
    }

    for (size_t i = 0; i < message.nich_size(); ++i) {
        auto & shared = nich.insert(featureids.at(absolute_pos++));
        distributions::shared_load(shared, message.nich(i));
    }

    update_schema();
}


void ProductModel::update_schema ()
{
    // HACK --------------------
    auto & dd = features.dd256;
    auto & dpd = features.dpd;
    auto & gp = features.gp;
    auto & nich = features.nich;
    //--------------------------

    schema.booleans_size = 0;
    schema.counts_size = dd.size() + dpd.size() + gp.size();
    schema.reals_size = nich.size();
}

struct ProductModel::clear_fun
{
    Features & features;

    template<class T>
    void operator() (T * t)
    {
        features[t].clear();
    }
};

void ProductModel::clear ()
{
    schema.clear();

    clear_fun fun = {features};
    for_each_feature_type(fun);
}

struct ProductModel::extend_fun
{
    Features & destin;
    const Features & source;

    template<class T>
    void operator() (T * t)
    {
        destin[t].extend(source[t]);
    }
};

void ProductModel::extend (const ProductModel & other)
{
    schema += other.schema;

    extend_fun fun = {features, other.features};
    for_each_feature_type(fun);
}

} // namespace loom
