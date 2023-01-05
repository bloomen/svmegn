// svmegn is C++ wrapper library around libsvm/liblinear using Eigen
// Repo: https://github.com/bloomen/svmegn
// Author: Christian Blume
// License: MIT http://www.opensource.org/licenses/mit-license.php

#pragma once

#include <memory>
#include <optional>

#if __clang__ || __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#endif
#include <Eigen/Core>
#if __clang__ || __GNUC__
#pragma GCC diagnostic pop
#endif

// TODO Use fixed size types only

namespace svmegn
{

void
set_print_string_function(void (*)(const char*));

void
remove_print_string_function();

enum ModelType
{
    SVM = 0,
    LINEAR = 1
};

enum class SVMType
{
    C_SVC = 0,
    NU_SVC = 1,
    ONE_CLASS = 2,
    EPSILON_SVR = 3,
    NU_SVR = 4
};

enum class KernelType
{
    LINEAR = 0,
    POLY = 1,
    RBF = 2,
    SIGMOID = 3,
    PRECOMPUTED = 4
};

struct Parameters
{
    ModelType model_type = ModelType::SVM;
    SVMType svm_type = SVMType::C_SVC;
    KernelType kernel_type = KernelType::RBF;
    int degree = 3; // for poly
    double gamma = 1.0; // for poly/rbf/sigmoid
    double coef0 = 0.0; // for poly/sigmoid
    double cache_size = 200; // in MB
    double eps = 0.001; // stopping criteria
    double C = 1.0; // for C_SVC, EPSILON_SVR and NU_SVR
    int nr_weight = 0; // for C_SVC
    Eigen::VectorXi weight_label; // for C_SVC
    Eigen::VectorXd weight; // for C_SVC
    double nu = 0.5; // for NU_SVC, ONE_CLASS, and NU_SVR
    double p = 0.1; // for EPSILON_SVR
    bool shrinking = true; // use the shrinking heuristics
    bool probability = false; // do probability estimates
};

class ModelError : public std::runtime_error
{
public:
    explicit ModelError(const std::string& message)
        : std::runtime_error{message}
    {
    }
};

class Model
{
public:
    ~Model();

    Model(const Model&);
    Model&
    operator=(const Model&);

    Model(Model&&);
    Model&
    operator=(Model&&);

    static Model
    train(Parameters params,
          const Eigen::MatrixXd& X,
          const Eigen::VectorXd& y);

    const Parameters&
    parameters() const;

    Eigen::VectorXd
    predict(const Eigen::MatrixXd& X) const;

    void
    save(std::ostream& os) const;

    static Model
    load(std::istream& is);

private:
    Model() = default;
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

class SVM : public Model
{
public:
    // add extra functions here
};

// class Linear : public Model<ModelType::Linear>
//{
// public:
//    // add extra functions here
//};

} // namespace svmegn
