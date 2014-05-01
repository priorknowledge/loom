#include "product_model.hpp"

namespace loom
{

void ProductModel::load (
        const protobuf::ProductModel_Shared & message,
        const std::vector<size_t> & featureids)
{
    clear();
    distributions::clustering_load(clustering, message.clustering());

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
}


void ProductModel::update_schema ()
{
    schema.booleans_size = 0;  // TODO("implement bb");
    schema.counts_size = dd.size() + dpd.size() + gp.size();
    schema.reals_size = nich.size();
}

void ProductModel::clear ()
{
    schema.clear();
    dd.clear();
    dpd.clear();
    gp.clear();
    nich.clear();
}

} // namespace loom
