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

enum class LinearType
{
    L2R_LR = 0,
    L2R_L2LOSS_SVC_DUAL = 1,
    L2R_L2LOSS_SVC = 2,
    L2R_L1LOSS_SVC_DUAL = 3,
    MCSVM_CS = 4,
    L1R_L2LOSS_SVC = 5,
    L1R_LR = 6,
    L2R_LR_DUAL = 7,
    L2R_L2LOSS_SVR = 11,
    L2R_L2LOSS_SVR_DUAL = 12,
    L2R_L1LOSS_SVR_DUAL = 13,
    ONECLASS_SVM = 21
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
    LinearType linear_type = LinearType::L2R_L2LOSS_SVC_DUAL;
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
    Eigen::VectorXd init_sol;
    bool regularize_bias = true;
    double bias = -1;
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
    struct SvmImpl;
    friend struct SvmImpl;
    static std::unique_ptr<Impl>
    make_impl(const ModelType);
    static std::unique_ptr<Impl>
    make_impl(std::istream&);
    static std::unique_ptr<Impl>
    make_impl(const Impl&);
};

class SVM : public Model
{
public:
    static void
    set_print_string_function(void (*)(const char*));

    static void
    remove_print_string_function();
};

class Linear : public Model
{
public:
    static void
    set_print_string_function(void (*)(const char*));

    static void
    remove_print_string_function();
};

} // namespace svmegn
