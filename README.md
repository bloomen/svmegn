# svmegn

[![Actions](https://github.com/bloomen/svmegn/actions/workflows/svmegn-tests.yml/badge.svg?branch=main)](https://github.com/bloomen/svmegn/actions/workflows/svmegn-tests.yml?query=branch%3Amain)

svmegn is a C++ library for supervised learning using established methods of support vector machines.
It is wrapping [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) and [liblinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/)
while using the [Eigen](https://eigen.tuxfamily.org) linear algebra library for interfacing. Requires a C++17 compliant compiler. Tested with Clang, GCC, and Visual Studio.


### Sample usage
```cpp
// Let X be the matrix of features (dense or sparse)
// Let y be the vector of targets (class labels in this case)
svmegn::Params params;
params.model_type = svmegn::ModelType::SVM; // = libsvm. Use LINEAR for liblinear
params.svm_type = svmegn::SvmType::C_SVC;
params.C = 10;
params.gamma = 0.1;
const auto model = svmegn::Model::train(params, X, y);
const auto prediction = model.predict(X);
// prediction.y is now the vector of responses
```

### Dependencies

svmegn only depends on the [Eigen](https://eigen.tuxfamily.org) header-only library.

### Running the tests

Requires: cmake, python

```
python3 bootstrap.py  # uses conan to install Eigen and gtest
mkdir build && cd build
cmake -Dsvmegn_build_tests=ON ..
cmake --build .
ctest --verbose
```
