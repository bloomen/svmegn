# svmegn

[![Actions](https://github.com/bloomen/svmegn/actions/workflows/svmegn-tests.yml/badge.svg?branch=main)](https://github.com/bloomen/svmegn/actions/workflows/svmegn-tests.yml?query=branch%3Amain)

This a C++ wrapper library around [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) & [liblinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) using the [Eigen](https://eigen.tuxfamily.org) linear algebra library.

Sample code:
```
svmegn::Params params;
params.C = 10;
params.gamma = 0.1;
const auto model = svmegn::Model::train(params, X, y);
const auto pred = model.predict(X);
```
