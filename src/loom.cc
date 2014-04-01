#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <google/protobuf/io/coded_stream.h>
#include <distributions/random.hpp>
#include <distributions/clustering.hpp>
#include <distributions/models/dd.hpp>
#include <distributions/models/dpd.hpp>
#include <distributions/models/nich.hpp>
#include <distributions/models/gp.hpp>
#include <distributions/protobuf.hpp>
#include "common.hpp"

#define TODO(message) LOOM_ERROR("TODO " << message)

namespace loom
{

using distributions::rng_t;
using distributions::VectorFloat;


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


struct ProductClassifier
{
    const ProductModel & model;
    std::vector<distributions::DirichletDiscrete<16>::Classifier> dd;
    std::vector<distributions::DirichletProcessDiscrete::Classifier> dpd;
    std::vector<distributions::GammaPoisson::Classifier> gp;
    std::vector<distributions::NormalInverseChiSq::Classifier> nich;

    ProductClassifier (const ProductModel & m) : model(m) {}

    void init (rng_t & rng);
    void load (const char * filename) { TODO("load"); }
    void dump (const char * filename) const { TODO("dump"); }

    void score (
            const distributions::protobuf::ProductValue & value,
            VectorFloat & scores,
            rng_t & rng);

    void add_group (rng_t & rng);

    void add_value (
            size_t groupid,
            const distributions::protobuf::ProductValue & value,
            rng_t & rng);

private:

    template<class Model>
    void init_factors (
            const std::vector<Model> & models,
            std::vector<typename Model::Classifier> & classifiers,
            rng_t & rng);

    template<class Fun>
    void apply_dense (Fun & fun);

    template<class Fun>
    void apply_sparse (
            Fun & fun,
            const distributions::protobuf::ProductValue & value);

    struct score_fun;
    struct add_group_fun;
    struct add_value_fun;
};

template<class Model>
void ProductClassifier::init_factors (
        const std::vector<Model> & models,
        std::vector<typename Model::Classifier> & classifiers,
        rng_t & rng)
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

void ProductClassifier::init (rng_t & rng)
{
    init_factors(model.dd, dd, rng);
    init_factors(model.dpd, dpd, rng);
    init_factors(model.gp, gp, rng);
    init_factors(model.nich, nich, rng);
}

template<class Fun>
inline void ProductClassifier::apply_dense (Fun & fun)
{
    //TODO("implement bb");
    for (size_t i = 0; i < dd.size(); ++i) {
        fun(model.dd[i], dd[i]);
    }
    for (size_t i = 0; i < dpd.size(); ++i) {
        fun(model.dpd[i], dpd[i]);
    }
    for (size_t i = 0; i < gp.size(); ++i) {
        fun(model.gp[i], gp[i]);
    }
    for (size_t i = 0; i < nich.size(); ++i) {
        fun(model.nich[i], nich[i]);
    }
}

template<class Fun>
inline void ProductClassifier::apply_sparse (
        Fun & fun,
        const distributions::protobuf::ProductValue & value)
{
    size_t observed_pos = 0;

    if (value.booleans_size()) {
        TODO("implement bb");
    }

    if (value.counts_size()) {
        size_t data_pos = 0;
        for (size_t i = 0; i < dd.size(); ++i) {
            if (value.observed(observed_pos++)) {
                fun(model.dd[i], dd[i], value.counts(data_pos++));
            }
        }
        for (size_t i = 0; i < dpd.size(); ++i) {
            if (value.observed(observed_pos++)) {
                fun(model.dpd[i], dpd[i], value.counts(data_pos++));
            }
        }
        for (size_t i = 0; i < gp.size(); ++i) {
            if (value.observed(observed_pos++)) {
                fun(model.gp[i], gp[i], value.counts(data_pos++));
            }
        }
    }

    if (value.reals_size()) {
        size_t data_pos = 0;
        for (size_t i = 0; i < gp.size(); ++i) {
            if (value.observed(observed_pos++)) {
                fun(model.nich[i], nich[i], value.reals(data_pos++));
            }
        }
    }
}

struct ProductClassifier::score_fun
{
    VectorFloat & scores;
    rng_t & rng;

    template<class Model>
    void operator() (
        const Model & model,
        const typename Model::Classifier & classifier,
        const typename Model::Value & value)
    {
        model.classifier_score(classifier, value, scores, rng);
    }
};

inline void ProductClassifier::score (
        const distributions::protobuf::ProductValue & value,
        VectorFloat & scores,
        rng_t & rng)
{
    TODO("score clustering");
    score_fun fun = {scores, rng};
    apply_sparse(fun, value);
}

struct ProductClassifier::add_group_fun
{
    rng_t & rng;

    template<class Model>
    void operator() (
            const Model & model,
            typename Model::Classifier & classifier)
    {
        model.classifier_add_group(classifier, rng);
    }
};

inline void ProductClassifier::add_group (rng_t & rng)
{
    TODO("update clustering");
    add_group_fun fun = {rng};
    apply_dense(fun);
}

struct ProductClassifier::add_value_fun
{
    const size_t groupid;
    rng_t & rng;

    template<class Model>
    void operator() (
        const Model & model,
        typename Model::Classifier & classifier,
        const typename Model::Value & value)
    {
        model.classifier_add_value(classifier, groupid, value, rng);
    }
};

inline void ProductClassifier::add_value (
        size_t groupid,
        const distributions::protobuf::ProductValue & value,
        rng_t & rng)
{
    TODO("update clustering");
    add_value_fun fun = {groupid, rng};
    apply_sparse(fun, value);
}


} // namespace loom


const char * help_message =
"Usage: loom MODEL_IN GROUPS_OUT < OBSERVATIONS > ASSIGNMENTS"
;


int main (int argc, char ** argv)
{
    if (argc != 3) {
        std::cerr << help_message << std::endl;
        exit(1);
    }

    const char * model_in = argv[2];
    const char * classifier_out = argv[3];

    distributions::rng_t rng;

    loom::ProductModel model;
    model.load(model_in);

    loom::ProductClassifier classifier(model);
    classifier.init(rng);
    //classifier.load(classifier_in);

    {
        distributions::protobuf::ProductValue value;
        distributions::VectorFloat scores;
        while (std::cin) {
            TODO("value.ParseFromIstream(std::cin);");
            classifier.score(value, scores, rng);
            size_t groupid = distributions::sample_discrete(
                rng,
                scores.size(),
                scores.data());
            classifier.add_value(groupid, value, rng);
        }
    }

    classifier.dump(classifier_out);

    return 0;
}
