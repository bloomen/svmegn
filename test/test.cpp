#include <catch2/catch_all.hpp>
#include <sstream>
#include <svmegn.h>

namespace
{

void
generic_train_predict(const svmegn::SVMType svm_type,
                      const svmegn::KernelType kernel_type)
{
    svmegn::Parameters params;
    params.svm_type = svm_type;
    params.kernel_type = kernel_type;
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
    auto svm0 = svmegn::SVM::train(params, X, y);
    const auto p0 = svm0.predict(X);
    REQUIRE(X.rows() == p0.rows());
    // test copy constructor
    const svmegn::SVM svm1{svm0};
    const auto p1 = svm1.predict(X);
    REQUIRE(p0 == p1);
    // test copy assignment
    svmegn::SVM svm2{svm0};
    svm2 = svm1;
    const auto p2 = svm2.predict(X);
    REQUIRE(p0 == p2);
    // test move constructor
    const svmegn::SVM svm3{std::move(svm0)};
    const auto p3 = svm3.predict(X);
    REQUIRE(p0 == p3);
    // test move assignment
    svmegn::SVM svm4{svm1};
    svm4 = std::move(svm2);
    const auto p4 = svm4.predict(X);
    REQUIRE(p0 == p4);
    // test save & load
    //    std::stringstream ss;
    //    svm4.save(ss);
    //    ss.seekg(0);
    //    const auto svm5 = svmegn::SVM::load(ss);
    //    const auto p5 = svm5.predict(X);
    //    REQUIRE(p0 == p5);
}

} // namespace

TEST_CASE("generic_combinations")
{
    for (int svm = 0; svm < 5; ++svm)
    {
        for (int kern = 0; kern < 5; ++kern)
        {
            generic_train_predict(static_cast<svmegn::SVMType>(svm),
                                  static_cast<svmegn::KernelType>(kern));
        }
    }
}

int
main(int argc, char** argv)
{
    return Catch::Session().run(argc, argv);
}
