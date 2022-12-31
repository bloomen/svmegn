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
    const auto svm0 = svmegn::SVM::train(params, X, y);
    const auto p0 = svm0.predict(X);
    REQUIRE(X.rows() == p0.rows());
    // test save & load
    //    std::stringstream ss;
    //    svm0.save(ss);
    //    ss.seekg(0);
    //    const auto svm1 = svmegn::SVM::load(ss);
    //    const auto p1 = svm1.predict(X);
    //    REQUIRE(p0 == p1);
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
