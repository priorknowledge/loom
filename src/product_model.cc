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

    schema.booleans_size += message.bb_size();
    for (size_t i = 0; i < message.bb_size(); ++i) {
        TODO("load bb models");
    }

    schema.counts_size += message.dd_size();
    for (size_t i = 0; i < message.dd_size(); ++i) {
        auto & shared = dd.insert(featureids.at(absolute_pos++));
        distributions::shared_load(shared, message.dd(i));
        LOOM_ASSERT1(shared.dim > 1, "invalid dim: " << shared.dim);
    }

    schema.counts_size += message.dpd_size();
    for (size_t i = 0; i < message.dpd_size(); ++i) {
        auto & shared = dpd.insert(featureids.at(absolute_pos++));
        distributions::shared_load(shared, message.dpd(i));
        LOOM_ASSERT1(
            shared.betas.size() > 1,
            "invalid dim: " << shared.betas.size());
    }

    schema.counts_size += message.gp_size();
    for (size_t i = 0; i < message.gp_size(); ++i) {
        auto & shared = gp.insert(featureids.at(absolute_pos++));
        distributions::shared_load(shared, message.gp(i));
    }

    schema.reals_size += message.nich_size();
    for (size_t i = 0; i < message.nich_size(); ++i) {
        auto & shared = nich.insert(featureids.at(absolute_pos++));
        distributions::shared_load(shared, message.nich(i));
    }
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
