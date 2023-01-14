# svmegn

[![Actions](https://github.com/bloomen/svmegn/actions/workflows/svmegn-tests.yml/badge.svg?branch=main)](https://github.com/bloomen/svmegn/actions/workflows/svmegn-tests.yml?query=branch%3Amain)

This is a C++ wrapper library around [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) and [liblinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) using the [Eigen](https://eigen.tuxfamily.org) linear algebra library.

Sample usage:
```
// Let X be the Eigen matrix of features (dense or sparse)
// Let y be the Eigen vector of targets (class labels in this case)
svmegn::Params params;
params.model_type = svmegn::ModelType::SVM; // = libsvm. Use LINEAR for liblinear
params.svm_type = svmegn::SvmType::C_SVC;
params.C = 10;
params.gamma = 0.1;
const auto model = svmegn::Model::train(params, X, y);
const auto prediction = model.predict(X);
// prediction.y is now the Eigen vector of responses
```
