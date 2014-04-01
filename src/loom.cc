#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <distributions/models/dd.hpp>
#include <distributions/models/dpd.hpp>
#include <distributions/models/nich.hpp>
#include <distributions/models/gp.hpp>
#include <distributions/protobuf.hpp>
#include "common.hpp"

#define TODO(message) LOOM_ERROR("TODO " << message)

namespace loom
{

distributions::rng_t rng;


struct ProductModel
{
    std::vector<distributions::DirichletDiscrete<16>> dd;
    std::vector<distributions::DirichletProcessDiscrete> dpd;
    std::vector<distributions::GammaPoisson> gp;
    std::vector<distributions::NormalInverseChiSq> nich;

    void load (const char * filename);
};

void ProductModel::load (const char * filename)
{
    std::fstream file(filename, std::ios::in | std::ios::binary);
    LOOM_ASSERT(file, "models file not found");
    distributions::protobuf::ProductModel product_model;
    bool info = product_model.ParseFromIstream(&file);
    LOOM_ASSERT(info, "failed to parse model from file");

    for (size_t i = 0; i < product_model.bb_size(); ++i) {
        TODO("load bb models");
    }

    for (size_t i = 0; i < product_model.dd_size(); ++i) {
        distributions::model_load(dd[i], product_model.dd(i));
    }

    for (size_t i = 0; i < product_model.dpd_size(); ++i) {
        distributions::model_load(dpd[i], product_model.dpd(i));
    }

    for (size_t i = 0; i < product_model.gp_size(); ++i) {
        distributions::model_load(gp[i], product_model.gp(i));
    }

    for (size_t i = 0; i < product_model.nich_size(); ++i) {
        distributions::model_load(nich[i], product_model.nich(i));
    }
}


struct ProductClassifier
{
    const ProductModel & model;
    std::vector<distributions::DirichletDiscrete<16>::Classifier> dd;
    std::vector<distributions::DirichletProcessDiscrete::Classifier> dpd;
    std::vector<distributions::GammaPoisson::Classifier> gp;
    std::vector<distributions::NormalInverseChiSq::Classifier> nich;

    ProductClassifier (const ProductModel & m) : model(m) {}

    void init ();
    void load (const char * filename) { TODO("load"); }
    void dump (const char * filename) const { TODO("dump"); }

    void score (
            const distributions::protobuf::ProductValue & value,
            distributions::VectorFloat & scores);

private:

    template<class Model>
    void init_classifiers (
            const std::vector<Model> & models,
            std::vector<typename Model::Classifier> & classifiers);
};

template<class Model>
void ProductClassifier::init_classifiers (
        const std::vector<Model> & models,
        std::vector<typename Model::Classifier> & classifiers)
{
    const size_t count = models.size();
    classifiers.clear();
    classifiers.resize(count);
    for (size_t i = 0; i < count; ++i) {
        const auto & model = models[i];
        auto & classifier = classifiers[i];
        classifier.groups.resize(1);
        model.group_init(classifier.groups[0], rng);
        model.classifier_init(classifier, rng);
    }
}

void ProductClassifier::init ()
{
    init_classifiers(model.dd, dd);
    init_classifiers(model.dpd, dpd);
    init_classifiers(model.gp, gp);
    init_classifiers(model.nich, nich);
}

inline void ProductClassifier::score (
        const distributions::protobuf::ProductValue & row,
        distributions::VectorFloat & scores)
{
    size_t observed_pos = 0;

    if (row.booleans_size()) {
        TODO("implement bb");
    }

    if (row.counts_size()) {
        size_t data_pos = 0;

        for (size_t i = 0; i < dd.size(); ++i) {
            if (row.observed(observed_pos++)) {
                const auto value = row.counts(data_pos++);
                model.dd[i].classifier_score(dd[i], value, scores, rng);
            }
        }

        for (size_t i = 0; i < dpd.size(); ++i) {
            if (row.observed(observed_pos++)) {
                const auto value = row.counts(data_pos++);
                model.dpd[i].classifier_score(dpd[i], value, scores, rng);
            }
        }

        for (size_t i = 0; i < gp.size(); ++i) {
            if (row.observed(observed_pos++)) {
                const auto value = row.counts(data_pos++);
                model.gp[i].classifier_score(gp[i], value, scores, rng);
            }
        }
    }

    if (row.reals_size()) {
        size_t data_pos = 0;

        for (size_t i = 0; i < gp.size(); ++i) {
            if (row.observed(observed_pos++)) {
                const auto value = row.reals(data_pos++);
                model.nich[i].classifier_score(nich[i], value, scores, rng);
            }
        }
    }
}


} // namespace loom


const char * help_message =
"Usage: loom MODEL_IN GROUPS_IN GROUPS_OUT < OBSERVATIONS > ASSIGNMENTS"
;


int main (int argc, char ** argv)
{
    if (argc != 4) {
        std::cerr << help_message << std::endl;
        exit(1);
    }

    const char * model_in = argv[2];
    const char * classifier_in = argv[3];
    const char * classifier_out = argv[4];

    loom::ProductModel model;
    model.load(model_in);

    loom::ProductClassifier classifier(model);
    classifier.load(classifier_in);

    TODO("stream in data");

    classifier.dump(classifier_out);

    return 0;
}
