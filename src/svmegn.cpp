// svmegn is C++ wrapper library around libsvm using Eigen
// Repo: https://github.com/bloomen/svmegn
// Author: Christian Blume
// License: MIT http://www.opensource.org/licenses/mit-license.php

#include <svmegn.h>

#include <svm/svm.h>

namespace svmegn
{

struct SVM::Impl
{
};

SVM::~SVM()
{
}

SVM
SVM::train(Parameters params,
           const Eigen::MatrixXd& X,
           const Eigen::MatrixXd& y)
{
    return {};
}

Eigen::MatrixXd
SVM::predict(const Eigen::MatrixXd& X) const
{
    return {};
}

void
SVM::save(std::ostream& os) const
{
}

SVM
SVM::load(std::istream& os)
{
    return {};
}

} // namespace svmegn
