#if __clang__ || __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include <gtest/gtest.h>
#if __clang__ || __GNUC__
#pragma GCC diagnostic pop
#endif
#include <sstream>
#include <svmegn.h>
#include <unordered_set>

namespace
{

template <typename Mat>
void
generic_train_predict_impl(const svmegn::Params& params,
                           const Mat& X,
                           const svmegn::VectorD& y)
{
    auto svm0 = svmegn::Model::train(params, X, y);
    const auto p0 = svm0.predict(X, params.probability);
    ASSERT_EQ(X.rows(), p0.y.rows());
    if (params.probability)
    {
        ASSERT_EQ(X.rows(), p0.prob->rows());
    }
    // test copy constructor
    const svmegn::Model svm1{svm0};
    const auto p1 = svm1.predict(X, params.probability);
    ASSERT_EQ(p0.y, p1.y);
    if (params.probability)
    {
        ASSERT_EQ(*p0.prob, *p1.prob);
    }
    // test copy assignment
    svmegn::Model svm2{svm0};
    svm2 = svm1;
    const auto p2 = svm2.predict(X, params.probability);
    ASSERT_EQ(p0.y, p2.y);
    if (params.probability)
    {
        ASSERT_EQ(*p0.prob, *p2.prob);
    }
    // test move constructor
    const svmegn::Model svm3{std::move(svm0)};
    const auto p3 = svm3.predict(X, params.probability);
    ASSERT_EQ(p0.y, p3.y);
    if (params.probability)
    {
        ASSERT_EQ(*p0.prob, *p3.prob);
    }
    // test move assignment
    svmegn::Model svm4{svm1};
    svm4 = std::move(svm2);
    const auto p4 = svm4.predict(X, params.probability);
    ASSERT_EQ(p0.y, p4.y);
    if (params.probability)
    {
        ASSERT_EQ(*p0.prob, *p4.prob);
    }
    // test save & load
    std::stringstream ss;
    svm4.save(ss);
    ss.seekg(0);
    const auto svm5 = svmegn::Model::load(ss);
    const auto p5 = svm5.predict(X, params.probability);
    ASSERT_EQ(p0.y, p5.y);
    if (params.probability)
    {
        ASSERT_EQ(*p0.prob, *p5.prob);
    }
}

void
generic_train_predict(svmegn::MatrixD, const svmegn::Params& params)
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
    generic_train_predict_impl(params, X, y);
}

void
generic_train_predict(svmegn::SpaMatrixD,
                      const svmegn::Params& params,
                      const bool full)
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
    generic_train_predict_impl(params, X, y);
}

} // namespace

TEST(svmegn, svm_generic_combinations)
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
                        params.svm_type = static_cast<svmegn::SVMType>(svm);
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
                        generic_train_predict(svmegn::MatrixD{}, params);
                        generic_train_predict(
                            svmegn::SpaMatrixD{}, params, true);
                        if (prob == 0)
                        {
                            generic_train_predict(
                                svmegn::SpaMatrixD{}, params, false);
                        }
                    }
                }
            }
        }
    }
}

TEST(svmegn, linear_generic_combinations)
{
    std::unordered_set<int> lin_skipped{8, 9, 10, 14, 15, 16, 17, 18, 19, 20};
    for (int lin = 0; lin < 22; ++lin)
    {
        if (lin_skipped.count(lin) > 0)
        {
            continue;
        }
        for (int regb = 0; regb < 2; ++regb)
        {
            for (int bias = -1; bias < 2; ++bias)
            {
                for (int prob = 0; prob < 2; ++prob)
                {
                    for (int ww = 0; ww < 2; ++ww)
                    {
                        for (int init = 0; init < 2; ++init)
                        {
                            svmegn::Params params;
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
                            generic_train_predict(svmegn::MatrixD{}, params);
                            generic_train_predict(
                                svmegn::SpaMatrixD{}, params, true);
                            generic_train_predict(
                                svmegn::SpaMatrixD{}, params, false);
                        }
                    }
                }
            }
        }
    }
}

int
main(int argc, char** argv)
{
    svmegn::remove_print_string_function(svmegn::ModelType::SVM);
    svmegn::remove_print_string_function(svmegn::ModelType::LINEAR);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
