#if __clang__ || __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include <gtest/gtest.h>
#if __clang__ || __GNUC__
#pragma GCC diagnostic pop
#endif
#include <filesystem>
#include <fstream>
#include <sstream>
#include <svmegn.h>
#include <unordered_set>

namespace fs = std::filesystem;

namespace
{

const fs::path source_dir = SVMEGN_SOURCE_DIR;
std::unordered_set<int> not_a_linear_type{8, 9, 10, 14, 15, 16, 17, 18, 19, 20};

std::pair<svmegn::MatrixD, svmegn::VectorD>
get_test_data()
{
    const auto X = (svmegn::MatrixD{20, 2} << 1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0)
                       .finished();
    const auto y = (svmegn::VectorD{20} << -1,
                    1,
                    -1,
                    1,
                    -1,
                    1,
                    -1,
                    1,
                    -1,
                    1,
                    -1,
                    1,
                    -1,
                    1,
                    -1,
                    1,
                    -1,
                    1,
                    -1,
                    1)
                       .finished();
    return std::make_pair(X, y);
}

std::pair<svmegn::SpaMatrixD, svmegn::VectorD>
get_sparse_test_data(const bool full)
{
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.push_back({0, 0, 1});
    triplets.push_back({0, 1, 1});
    triplets.push_back({1, 0, 0});
    triplets.push_back({1, 1, 0});
    triplets.push_back({2, 0, 1});
    triplets.push_back({2, 1, 1});
    triplets.push_back({3, 0, 0});
    triplets.push_back({3, 1, 0});
    triplets.push_back({4, 0, 1});
    triplets.push_back({4, 1, 1});
    if (full)
    {
        // to test some missing
        triplets.push_back({5, 0, 0});
    }
    triplets.push_back({5, 1, 0});
    triplets.push_back({6, 0, 1});
    triplets.push_back({6, 1, 1});
    triplets.push_back({7, 0, 0});
    triplets.push_back({7, 1, 0});
    triplets.push_back({8, 0, 1});
    if (full)
    {
        // to test some missing
        triplets.push_back({8, 1, 1});
    }
    triplets.push_back({9, 0, 0});
    triplets.push_back({9, 1, 0});
    svmegn::SpaMatrixD X{10, 2};
    X.setFromTriplets(triplets.begin(), triplets.end());

    const auto y = (svmegn::VectorD{20} << -1,
                    1,
                    -1,
                    1,
                    -1,
                    1,
                    -1,
                    1,
                    -1,
                    1,
                    -1,
                    1,
                    -1,
                    1,
                    -1,
                    1,
                    -1,
                    1,
                    -1,
                    1)
                       .finished();
    return std::make_pair(X, y);
}

template <typename Mat>
void
generic_train_predict(const svmegn::Params& params,
                      const Mat& X,
                      const svmegn::VectorD& y)
{
    auto assert_model_info = [&params](const svmegn::Model& m) {
        ASSERT_EQ(2, m.nr_features());
        ASSERT_EQ(2, m.nr_class());
        if (params.svm_type != svmegn::SvmType::EPSILON_SVR &&
            params.svm_type != svmegn::SvmType::NU_SVR &&
            params.svm_type != svmegn::SvmType::ONE_CLASS &&
            params.linear_type != svmegn::LinearType::ONECLASS_SVM &&
            params.linear_type != svmegn::LinearType::L2R_L1LOSS_SVR_DUAL &&
            params.linear_type != svmegn::LinearType::L2R_L2LOSS_SVR_DUAL &&
            params.linear_type != svmegn::LinearType::L2R_L2LOSS_SVR)
        {
            const auto labels = (svmegn::VectorI{2} << 1, -1).finished();
            ASSERT_TRUE(m.labels());
            ASSERT_EQ(labels, *m.labels());
        }
        else
        {
            ASSERT_FALSE(m.labels());
        }
    };

    auto assert_prediction =
        [&params](const auto& p0, const auto& X, const auto& model) {
            const auto p1 = model.predict(X, params.probability);
            ASSERT_EQ(p0.y, p1.y);
            if (params.probability)
            {
                if (params.svm_type != svmegn::SvmType::EPSILON_SVR &&
                    params.svm_type != svmegn::SvmType::NU_SVR)
                {
                    ASSERT_EQ(*p0.prob, *p1.prob);
                }
            }
        };

    const auto svm0 = svmegn::Model::train(params, X, y);
    assert_model_info(svm0);
    const auto p0 = svm0.predict(X, params.probability);
    ASSERT_EQ(X.rows(), p0.y.rows());
    if (params.probability)
    {
        ASSERT_EQ(X.rows(), p0.prob->rows());
    }
    // predict again
    assert_prediction(p0, X, svm0);
    // test copy constructor
    const svmegn::Model svm1{svm0};
    assert_model_info(svm1);
    assert_prediction(p0, X, svm1);
    // test copy assignment
    svmegn::Model svm2{svm0};
    svm2 = svm1;
    assert_model_info(svm2);
    assert_prediction(p0, X, svm2);
    // test move constructor
    const svmegn::Model svm3{std::move(svm0)};
    assert_model_info(svm3);
    assert_prediction(p0, X, svm3);
    // test move assignment
    svmegn::Model svm4{svm1};
    svm4 = std::move(svm2);
    assert_model_info(svm4);
    assert_prediction(p0, X, svm4);
    // test save & load
    std::stringstream ss;
    svm4.save(ss);
    ss.seekg(0);
    const auto svm5 = svmegn::Model::load(ss);
    assert_model_info(svm5);
    assert_prediction(p0, X, svm5);
}

svmegn::MatrixD
load_csv(const fs::path filename)
{
    std::ifstream is{source_dir / "test" / filename};
    std::string line;
    std::vector<double> values;
    Eigen::Index rows = 0;
    while (std::getline(is, line))
    {
        std::stringstream ls(line);
        std::string cell;
        while (std::getline(ls, cell, ' '))
        {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Eigen::Map<svmegn::MatrixD>{
        values.data(), rows, static_cast<Eigen::Index>(values.size()) / rows};
}

std::pair<svmegn::MatrixD, svmegn::VectorD>
load_regression_data()
{
    const auto csv = load_csv("boston.csv");
    const svmegn::VectorD y = csv.col(13);
    const svmegn::MatrixD X = csv(Eigen::all, Eigen::seq(0, 12));
    return std::make_pair(std::move(X), std::move(y));
}

std::pair<svmegn::SpaMatrixD, svmegn::VectorD>
load_sparse_regression_data()
{
    auto data = load_regression_data();
    return std::make_pair(data.first.sparseView(), std::move(data.second));
}

std::pair<svmegn::MatrixD, svmegn::VectorD>
load_two_class_data()
{
    auto csv = load_regression_data();
    const auto mean = csv.second.mean();
    svmegn::VectorD labels{csv.second.rows()};
    for (Eigen::Index i = 0; i < labels.rows(); ++i)
    {
        labels(i) = csv.second(i) > mean ? 1 : -1;
    }
    return std::make_pair(std::move(csv.first), std::move(labels));
}

std::pair<svmegn::SpaMatrixD, svmegn::VectorD>
load_sparse_two_class_data()
{
    auto data = load_two_class_data();
    return std::make_pair(data.first.sparseView(), std::move(data.second));
}

std::pair<svmegn::MatrixD, svmegn::VectorD>
load_four_class_data()
{
    auto csv = load_regression_data();
    double min = 0;
    csv.second.cwiseMin(min);
    double max = 0;
    csv.second.cwiseMax(max);
    const double two = csv.second.mean();
    const auto one = (two - min) / 2;
    const auto three = (max - two) / 2;
    svmegn::VectorD labels{csv.second.rows()};
    for (Eigen::Index i = 0; i < labels.rows(); ++i)
    {
        if (csv.second(i) <= one)
        {
            labels(i) = 0;
        }
        else if (csv.second(i) <= two)
        {
            labels(i) = 1;
        }
        else if (csv.second(i) <= three)
        {
            labels(i) = 2;
        }
        else
        {
            labels(i) = 3;
        }
    }
    return std::make_pair(std::move(csv.first), std::move(labels));
}

std::pair<svmegn::SpaMatrixD, svmegn::VectorD>
load_sparse_four_class_data()
{
    auto data = load_four_class_data();
    return std::make_pair(data.first.sparseView(), std::move(data.second));
}

} // namespace

TEST(svmegn, svm_generic_train_predict)
{
    for (int svm = 0; svm < 5; ++svm)
    {
        for (int kern = 0; kern < 5; ++kern)
        {
            for (int shrink = 0; shrink < 2; ++shrink)
            {
                for (int prob = 0; prob < 2; ++prob)
                {
                    for (int ww = 0; ww < 2; ++ww)
                    {
                        svmegn::Params params;
                        if (prob == 1)
                        {
                            // Need to adjust params to avoid this error:
                            // "WARNING: number of positive or negative decision
                            // values <5; too few to do a probability
                            // estimation."
                            params.nu = 1e-6;
                            params.coef0 = 0.5;
                        }
                        params.model_type = svmegn::ModelType::SVM;
                        params.svm_type = static_cast<svmegn::SvmType>(svm);
                        params.kernel_type =
                            static_cast<svmegn::KernelType>(kern);
                        params.shrinking = static_cast<bool>(shrink);
                        params.probability = static_cast<bool>(prob);
                        if (ww == 1)
                        {
                            params.weight_label =
                                (Eigen::VectorXi{2} << -1, 1).finished();
                            params.weight =
                                (Eigen::VectorXd{2} << 0.4, 0.6).finished();
                        }
                        const auto data = get_test_data();
                        generic_train_predict(params, data.first, data.second);
                        const auto spadata_full = get_sparse_test_data(true);
                        generic_train_predict(
                            params, spadata_full.first, spadata_full.second);
                        if (prob == 0)
                        {
                            const auto spadata = get_sparse_test_data(false);
                            generic_train_predict(
                                params, spadata.first, spadata.second);
                        }
                    }
                }
            }
        }
    }
}

TEST(svmegn, linear_generic_train_predict)
{
    for (int lin = 0; lin < 22; ++lin)
    {
        if (not_a_linear_type.count(lin) > 0)
        {
            continue;
        }
        for (int regb = 0; regb < 2; ++regb)
        {
            if (regb == 0)
            {
                if (lin != 0 && lin != 2 && lin != 5 && lin != 6 && lin != 11)
                {
                    continue;
                }
            }
            for (int bias = -1; bias < 2; ++bias)
            {
                if (regb == 0 && bias != 1.0)
                {
                    continue;
                }
                if (bias >= 0 && lin == 21)
                {
                    continue;
                }
                for (int prob = 0; prob < 2; ++prob)
                {
                    if (prob == 1)
                    {
                        if (lin != 0 && lin != 6 && lin != 7)
                        {
                            continue;
                        }
                    }
                    for (int ww = 0; ww < 2; ++ww)
                    {
                        for (int init = 0; init < 2; ++init)
                        {
                            if (init > 0)
                            {
                                if (lin != 0 && lin != 2 && lin != 11)
                                {
                                    continue;
                                }
                            }
                            svmegn::Params params;
                            params.model_type = svmegn::LINEAR;
                            params.linear_type =
                                static_cast<svmegn::LinearType>(lin);
                            params.regularize_bias = static_cast<bool>(regb);
                            params.bias = bias;
                            params.probability = static_cast<bool>(prob);
                            if (ww == 1)
                            {
                                params.weight_label =
                                    (Eigen::VectorXi{2} << -1, 1).finished();
                                params.weight =
                                    (Eigen::VectorXd{2} << 0.4, 0.6).finished();
                            }
                            if (init == 1)
                            {
                                params.init_sol =
                                    (Eigen::VectorXd{2} << 0.1, 0.9).finished();
                            }
                            const auto data = get_test_data();
                            generic_train_predict(
                                params, data.first, data.second);
                            const auto spadata_full =
                                get_sparse_test_data(true);
                            generic_train_predict(params,
                                                  spadata_full.first,
                                                  spadata_full.second);
                            if (prob == 0)
                            {
                                const auto spadata =
                                    get_sparse_test_data(false);
                                generic_train_predict(
                                    params, spadata.first, spadata.second);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(svmegn, svm_cross_validation)
{
    for (int svm = 0; svm < 5; ++svm)
    {
        for (int kern = 0; kern < 5; ++kern)
        {
            svmegn::Params params;
            params.model_type = svmegn::ModelType::SVM;
            params.svm_type = static_cast<svmegn::SvmType>(svm);
            params.kernel_type = static_cast<svmegn::KernelType>(kern);
            const auto data = get_test_data();
            const auto y0 =
                svmegn::Model::cross_validate(params, data.first, data.second);
            ASSERT_EQ(data.second.size(), y0.size());
            for (const auto full : {true, false})
            {
                const auto spadata = get_sparse_test_data(full);
                const auto y1 = svmegn::Model::cross_validate(
                    params, spadata.first, spadata.second);
                ASSERT_EQ(spadata.second.size(), y1.size());
            }
        }
    }
}

TEST(svmegn, linear_cross_validation)
{
    for (int lin = 0; lin < 22; ++lin)
    {
        if (not_a_linear_type.count(lin) > 0)
        {
            continue;
        }
        svmegn::Params params;
        params.model_type = svmegn::ModelType::LINEAR;
        params.linear_type = static_cast<svmegn::LinearType>(lin);
        const auto data = get_test_data();
        const auto y0 =
            svmegn::Model::cross_validate(params, data.first, data.second);
        ASSERT_EQ(data.second.size(), y0.size());
        for (const auto full : {true, false})
        {
            const auto spadata = get_sparse_test_data(full);
            const auto y1 = svmegn::Model::cross_validate(
                params, spadata.first, spadata.second);
            ASSERT_EQ(spadata.second.size(), y1.size());
        }
    }
}

TEST(svmegn, impl_library_version)
{
    ASSERT_EQ(331, svmegn::impl_library_version(svmegn::ModelType::SVM));
    ASSERT_EQ(245, svmegn::impl_library_version(svmegn::ModelType::LINEAR));
}

namespace
{

template <typename Data>
void
test_svm_regression_epsilon_svr(const Data& data)
{
    svmegn::Params params;
    params.probability = true;
    params.svm_type = svmegn::SvmType::EPSILON_SVR;
    params.C = 100;
    params.p = 0.01;
    const auto model = svmegn::Model::train(params, data.first, data.second);
    ASSERT_EQ(data.first.cols(), model.nr_features());
    ASSERT_EQ(2, model.nr_class());
    const auto pred = model.predict(data.first);
    ASSERT_EQ(data.first.rows(), pred.y.rows());
    ASSERT_LT((data.second - pred.y).norm(), 1);
    const auto pred2 = model.predict(data.first, true);
    ASSERT_EQ(pred.y.rows(),
              pred2.y.rows()); // actual predicted values may be different
    ASSERT_EQ(pred.y.rows(), pred2.prob->rows());
    ASSERT_EQ(model.nr_class(), pred2.prob->cols());
}

} // namespace

TEST(svmegn, svm_regression_epsilon_svr)
{
    test_svm_regression_epsilon_svr(load_regression_data());
    test_svm_regression_epsilon_svr(load_sparse_regression_data());
}

namespace
{

template <typename Data>
void
test_svm_regression_nu_svr(const Data& data)
{
    svmegn::Params params;
    params.probability = true;
    params.svm_type = svmegn::SvmType::NU_SVR;
    params.C = 100;
    params.nu = 0.5;
    const auto model = svmegn::Model::train(params, data.first, data.second);
    ASSERT_EQ(data.first.cols(), model.nr_features());
    ASSERT_EQ(2, model.nr_class());
    const auto pred = model.predict(data.first);
    ASSERT_EQ(data.first.rows(), pred.y.rows());
    ASSERT_LT((data.second - pred.y).norm(), 1);
    const auto pred2 = model.predict(data.first, true);
    ASSERT_EQ(pred.y.rows(),
              pred2.y.rows()); // actual predicted values may be different
    ASSERT_EQ(pred.y.rows(), pred2.prob->rows());
    ASSERT_EQ(model.nr_class(), pred2.prob->cols());
}

} // namespace

TEST(svmegn, svm_regression_nu_svr)
{
    test_svm_regression_nu_svr(load_regression_data());
    test_svm_regression_nu_svr(load_sparse_regression_data());
}

namespace
{

template <typename Data>
void
test_linear_regression_l2r_l2loss_svr(const Data& data)
{
    svmegn::Params params;
    params.model_type = svmegn::ModelType::LINEAR;
    params.linear_type = svmegn::LinearType::L2R_L2LOSS_SVR;
    params.C = 10;
    const auto model = svmegn::Model::train(params, data.first, data.second);
    ASSERT_EQ(data.first.cols(), model.nr_features());
    ASSERT_EQ(2, model.nr_class());
    const auto pred = model.predict(data.first);
    ASSERT_EQ(data.first.rows(), pred.y.rows());
    ASSERT_LT((data.second - pred.y).norm(), 550);
}

} // namespace

TEST(svmegn, linear_regression_l2r_l2loss_svr)
{
    test_linear_regression_l2r_l2loss_svr(load_regression_data());
    test_linear_regression_l2r_l2loss_svr(load_sparse_regression_data());
}

namespace
{

template <typename Data>
void
test_svm_two_class_c_svc(const Data& data)
{
    svmegn::Params params;
    params.probability = true;
    params.C = 1;
    const auto model = svmegn::Model::train(params, data.first, data.second);
    ASSERT_EQ(data.first.cols(), model.nr_features());
    ASSERT_EQ(2, model.nr_class());
    const auto pred = model.predict(data.first);
    ASSERT_EQ(data.first.rows(), pred.y.rows());
    ASSERT_LT((data.second - pred.y).norm(), 17);
    const auto pred2 = model.predict(data.first, true);
    ASSERT_EQ(pred.y.rows(),
              pred2.y.rows()); // actual predicted values may be different
    ASSERT_EQ(pred.y.rows(), pred2.prob->rows());
    ASSERT_EQ(model.nr_class(), pred2.prob->cols());
}

} // namespace

TEST(svmegn, svm_two_class_c_svc)
{
    test_svm_two_class_c_svc(load_two_class_data());
    test_svm_two_class_c_svc(load_sparse_two_class_data());
}

namespace
{

template <typename Data>
void
test_svm_two_class_nu_svc(const Data& data)
{
    svmegn::Params params;
    params.probability = true;
    params.svm_type = svmegn::SvmType::NU_SVC;
    params.C = 1;
    const auto model = svmegn::Model::train(params, data.first, data.second);
    ASSERT_EQ(data.first.cols(), model.nr_features());
    ASSERT_EQ(2, model.nr_class());
    const auto pred = model.predict(data.first);
    ASSERT_EQ(data.first.rows(), pred.y.rows());
    ASSERT_LT((data.second - pred.y).norm(), 17);
    const auto pred2 = model.predict(data.first, true);
    ASSERT_EQ(pred.y.rows(),
              pred2.y.rows()); // actual predicted values may be different
    ASSERT_EQ(pred.y.rows(), pred2.prob->rows());
    ASSERT_EQ(model.nr_class(), pred2.prob->cols());
}

} // namespace

TEST(svmegn, svm_two_class_nu_svc)
{
    test_svm_two_class_nu_svc(load_two_class_data());
    test_svm_two_class_nu_svc(load_sparse_two_class_data());
}

namespace
{

template <typename Data>
void
test_svm_two_class_one_class(const Data& data)
{
    svmegn::Params params;
    params.probability = true;
    params.svm_type = svmegn::SvmType::ONE_CLASS;
    params.C = 10;
    const auto model = svmegn::Model::train(params, data.first, data.second);
    ASSERT_EQ(data.first.cols(), model.nr_features());
    ASSERT_EQ(2, model.nr_class());
    const auto pred = model.predict(data.first);
    ASSERT_EQ(data.first.rows(), pred.y.rows());
    ASSERT_LT((data.second - pred.y).norm(), 33);
    const auto pred2 = model.predict(data.first, true);
    ASSERT_EQ(pred.y.rows(),
              pred2.y.rows()); // actual predicted values may be different
    ASSERT_EQ(pred.y.rows(), pred2.prob->rows());
    ASSERT_EQ(model.nr_class(), pred2.prob->cols());
}

} // namespace

TEST(svmegn, svm_two_class_one_class)
{
    test_svm_two_class_one_class(load_two_class_data());
    test_svm_two_class_one_class(load_sparse_two_class_data());
}

namespace
{

template <typename Data>
void
test_linear_two_class_l2r_l2loss_svc_dual(const Data& data)
{
    svmegn::Params params;
    params.model_type = svmegn::ModelType::LINEAR;
    params.C = 100;
    const auto model = svmegn::Model::train(params, data.first, data.second);
    ASSERT_EQ(data.first.cols(), model.nr_features());
    ASSERT_EQ(2, model.nr_class());
    const auto pred = model.predict(data.first);
    ASSERT_EQ(data.first.rows(), pred.y.rows());
    ASSERT_LT((data.second - pred.y).norm(), 17);
}

} // namespace

TEST(svmegn, linear_two_class_l2r_l2loss_svc_dual)
{
    test_linear_two_class_l2r_l2loss_svc_dual(load_two_class_data());
    test_linear_two_class_l2r_l2loss_svc_dual(load_sparse_two_class_data());
}

namespace
{

template <typename Data>
void
test_linear_two_class_l2r_lr(const Data& data)
{
    svmegn::Params params;
    params.model_type = svmegn::ModelType::LINEAR;
    params.linear_type = svmegn::LinearType::L2R_LR;
    params.C = 100;
    const auto model = svmegn::Model::train(params, data.first, data.second);
    ASSERT_EQ(data.first.cols(), model.nr_features());
    ASSERT_EQ(2, model.nr_class());
    const auto pred = model.predict(data.first);
    ASSERT_EQ(data.first.rows(), pred.y.rows());
    ASSERT_LT((data.second - pred.y).norm(), 17);
    const auto pred2 = model.predict(data.first, true);
    ASSERT_EQ(pred.y.rows(),
              pred2.y.rows()); // actual predicted values may be different
    ASSERT_EQ(pred.y.rows(), pred2.prob->rows());
    ASSERT_EQ(model.nr_class(), pred2.prob->cols());
}

} // namespace

TEST(svmegn, linear_two_class_l2r_lr)
{
    test_linear_two_class_l2r_lr(load_two_class_data());
    test_linear_two_class_l2r_lr(load_sparse_two_class_data());
}

namespace
{

template <typename Data>
void
test_svm_four_class_c_svc(const Data& data)
{
    svmegn::Params params;
    params.probability = true;
    params.C = 1;
    const auto model = svmegn::Model::train(params, data.first, data.second);
    ASSERT_EQ(data.first.cols(), model.nr_features());
    ASSERT_EQ(3, model.nr_class());
    const auto pred = model.predict(data.first);
    ASSERT_EQ(data.first.rows(), pred.y.rows());
    ASSERT_LT((data.second - pred.y).norm(), 18);
    const auto pred2 = model.predict(data.first, true);
    ASSERT_EQ(pred.y.rows(),
              pred2.y.rows()); // actual predicted values may be different
    ASSERT_EQ(pred.y.rows(), pred2.prob->rows());
    ASSERT_EQ(model.nr_class(), pred2.prob->cols());
}

} // namespace

TEST(svmegn, svm_four_class_c_svc)
{
    test_svm_four_class_c_svc(load_four_class_data());
    test_svm_four_class_c_svc(load_sparse_four_class_data());
}

namespace
{

template <typename Data>
void
test_svm_four_class_nu_svc(const Data& data)
{
    svmegn::Params params;
    params.probability = true;
    params.svm_type = svmegn::SvmType::NU_SVC;
    params.C = 10;
    params.nu = 0.01;
    const auto model = svmegn::Model::train(params, data.first, data.second);
    ASSERT_EQ(data.first.cols(), model.nr_features());
    ASSERT_EQ(3, model.nr_class());
    const auto pred = model.predict(data.first);
    ASSERT_EQ(data.first.rows(), pred.y.rows());
    ASSERT_LT((data.second - pred.y).norm(), 18);
    const auto pred2 = model.predict(data.first, true);
    ASSERT_EQ(pred.y.rows(),
              pred2.y.rows()); // actual predicted values may be different
    ASSERT_EQ(pred.y.rows(), pred2.prob->rows());
    ASSERT_EQ(model.nr_class(), pred2.prob->cols());
}

} // namespace

TEST(svmegn, svm_four_class_nu_svc)
{
    test_svm_four_class_nu_svc(load_four_class_data());
    test_svm_four_class_nu_svc(load_sparse_four_class_data());
}

namespace
{

template <typename Data>
void
test_linear_four_class_l2r_l2loss_svc_dual(const Data& data)
{
    svmegn::Params params;
    params.model_type = svmegn::ModelType::LINEAR;
    params.C = 100;
    const auto model = svmegn::Model::train(params, data.first, data.second);
    ASSERT_EQ(data.first.cols(), model.nr_features());
    ASSERT_EQ(3, model.nr_class());
    const auto pred = model.predict(data.first);
    ASSERT_EQ(data.first.rows(), pred.y.rows());
    ASSERT_LT((data.second - pred.y).norm(), 18);
}

} // namespace

TEST(svmegn, linear_four_class_l2r_l2loss_svc_dual)
{
    test_linear_four_class_l2r_l2loss_svc_dual(load_four_class_data());
    test_linear_four_class_l2r_l2loss_svc_dual(load_sparse_four_class_data());
}

namespace
{

template <typename Data>
void
test_linear_four_class_l2r_lr(const Data& data)
{
    svmegn::Params params;
    params.model_type = svmegn::ModelType::LINEAR;
    params.linear_type = svmegn::LinearType::L2R_LR;
    params.C = 100;
    const auto model = svmegn::Model::train(params, data.first, data.second);
    ASSERT_EQ(data.first.cols(), model.nr_features());
    ASSERT_EQ(3, model.nr_class());
    const auto pred = model.predict(data.first);
    ASSERT_EQ(data.first.rows(), pred.y.rows());
    ASSERT_LT((data.second - pred.y).norm(), 18);
    const auto pred2 = model.predict(data.first, true);
    ASSERT_EQ(pred.y.rows(),
              pred2.y.rows()); // actual predicted values may be different
    ASSERT_EQ(pred.y.rows(), pred2.prob->rows());
    ASSERT_EQ(model.nr_class(), pred2.prob->cols());
}

} // namespace

TEST(svmegn, linear_four_class_l2r_lr)
{
    test_linear_four_class_l2r_lr(load_four_class_data());
    test_linear_four_class_l2r_lr(load_sparse_four_class_data());
}

int
main(int argc, char** argv)
{
    svmegn::remove_print_string_function(svmegn::ModelType::SVM);
    svmegn::remove_print_string_function(svmegn::ModelType::LINEAR);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
