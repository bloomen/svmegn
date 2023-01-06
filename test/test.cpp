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

void
generic_train_predict(const svmegn::Params& params)
{
    const auto X = (Eigen::MatrixXd{20, 2} << 1,
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
    const auto y = (Eigen::VectorXd{20} << -1,
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
    auto svm0 = svmegn::Model::train(params, X, y);
    const auto p0 = svm0.predict(X);
    ASSERT_EQ(X.rows(), p0.rows());
    // test copy constructor
    const svmegn::Model svm1{svm0};
    const auto p1 = svm1.predict(X);
    ASSERT_EQ(p0, p1);
    // test copy assignment
    svmegn::Model svm2{svm0};
    svm2 = svm1;
    const auto p2 = svm2.predict(X);
    ASSERT_EQ(p0, p2);
    // test move constructor
    const svmegn::Model svm3{std::move(svm0)};
    const auto p3 = svm3.predict(X);
    ASSERT_EQ(p0, p3);
    // test move assignment
    svmegn::Model svm4{svm1};
    svm4 = std::move(svm2);
    const auto p4 = svm4.predict(X);
    ASSERT_EQ(p0, p4);
    // test save & load
    std::stringstream ss;
    svm4.save(ss);
    ss.seekg(0);
    const auto svm5 = svmegn::Model::load(ss);
    const auto p5 = svm5.predict(X);
    ASSERT_EQ(p0, p5);
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
                        generic_train_predict(params);
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
                for (int ww = 0; ww < 2; ++ww)
                {
                    for (int init = 0; init < 2; ++init)
                    {
                        svmegn::Params params;
                        params.linear_type =
                            static_cast<svmegn::LinearType>(lin);
                        params.regularize_bias = static_cast<bool>(regb);
                        params.bias = bias;
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
                        generic_train_predict(params);
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
