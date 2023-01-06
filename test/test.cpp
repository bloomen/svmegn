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

namespace
{

void
generic_train_predict(const svmegn::ModelType model_type,
                      const svmegn::SVMType svm_type,
                      const svmegn::KernelType kernel_type,
                      const svmegn::LinearType linear_type =
                          svmegn::LinearType::L2R_L2LOSS_SVC_DUAL)
{
    svmegn::Parameters params;
    params.model_type = model_type;
    params.svm_type = svm_type;
    params.kernel_type = kernel_type;
    params.linear_type = linear_type;
    const auto X = (Eigen::MatrixXd{10, 2} << 1,
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
    const auto y =
        (Eigen::VectorXd{10} << 0, 1, 0, 1, 0, 1, 0, 1, 0, 1).finished();
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
            generic_train_predict(svmegn::ModelType::SVM,
                                  static_cast<svmegn::SVMType>(svm),
                                  static_cast<svmegn::KernelType>(kern));
        }
    }
}

TEST(svmegn, linear_generic_combinations)
{
    for (int lin = 0; lin < 8; ++lin)
    {
        if (lin == 4)
        {
            // TODO figure out why this solver crashes
            continue;
        }
        generic_train_predict(svmegn::ModelType::LINEAR,
                              svmegn::SVMType::C_SVC,
                              svmegn::KernelType::LINEAR,
                              static_cast<svmegn::LinearType>(lin));
    }
    for (int lin = 11; lin < 14; ++lin)
    {
        generic_train_predict(svmegn::ModelType::LINEAR,
                              svmegn::SVMType::C_SVC,
                              svmegn::KernelType::LINEAR,
                              static_cast<svmegn::LinearType>(lin));
    }
    generic_train_predict(svmegn::ModelType::LINEAR,
                          svmegn::SVMType::C_SVC,
                          svmegn::KernelType::LINEAR,
                          svmegn::LinearType::ONECLASS_SVM);
}

int
main(int argc, char** argv)
{
    svmegn::SVM::remove_print_string_function();
    svmegn::Linear::remove_print_string_function();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
