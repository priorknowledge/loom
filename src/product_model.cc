#include "product_model.hpp"

namespace loom
{

void ProductModel::load (const protobuf::ProductModel_Shared & message)
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
        distributions::shared_load(dd[i], message.dd(i));
        LOOM_ASSERT1(dd[i].dim > 1, "invalid dim: " << dd[i].dim);
    }

    schema.counts_size += message.dpd_size();
    dpd.resize(message.dpd_size());
    for (size_t i = 0; i < message.dpd_size(); ++i) {
        distributions::shared_load(dpd[i], message.dpd(i));
        LOOM_ASSERT1(
            dpd[i].betas.size() > 1,
            "invalid dim: " << dpd[i].betas.size());
    }

    schema.counts_size += message.gp_size();
    gp.resize(message.gp_size());
    for (size_t i = 0; i < message.gp_size(); ++i) {
        distributions::shared_load(gp[i], message.gp(i));
    }

    schema.reals_size += message.nich_size();
    nich.resize(message.nich_size());
    for (size_t i = 0; i < message.nich_size(); ++i) {
        distributions::shared_load(nich[i], message.nich(i));
    }
}

} // namespace loom
