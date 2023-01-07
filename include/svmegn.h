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

namespace svmegn
{

using SmallInt = std::int8_t;
using VectorInt = Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic>;

enum ModelType : SmallInt
{
    SVM = 0, // libsvm
    LINEAR = 1 // liblinear
};

void
set_print_string_function(ModelType model_type, void (*func)(const char*));

void
remove_print_string_function(ModelType model_type);

enum class SVMType : SmallInt // Used for ModelType::SVM
{
    C_SVC = 0,
    NU_SVC = 1,
    ONE_CLASS = 2,
    EPSILON_SVR = 3,
    NU_SVR = 4
};

enum class KernelType : SmallInt // Used for ModelType::SVM
{
    LINEAR = 0,
    POLY = 1,
    RBF = 2,
    SIGMOID = 3,
    PRECOMPUTED = 4
};

enum class LinearType : SmallInt // Used for ModelType::Linear
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

struct Params
{
    ModelType model_type = ModelType::SVM;

    // For ModelType::SVM
    SVMType svm_type = SVMType::C_SVC;

    // For ModelType::SVM
    KernelType kernel_type = KernelType::RBF;

    // For ModelType::LINEAR
    LinearType linear_type = LinearType::L2R_L2LOSS_SVC_DUAL;

    // For ModelType::SVM. For poly
    SmallInt degree = 3;

    // For ModelType::SVM. For poly/rbf/sigmoid
    double gamma = 1.0;

    // For ModelType::SVM. For poly/sigmoid
    double coef0 = 0.0;

    // For ModelType::SVM. In MB
    double cache_size = 200;

    // For ModelType::SVM/LINEAR. stopping criteria
    double eps = 0.001;

    // For ModelType::SVM/LINEAR. For C_SVC, EPSILON_SVR and NU_SVR
    double C = 1.0;

    // For ModelType::SVM/LINEAR. For C_SVC
    VectorInt weight_label;

    // For ModelType::SVM/LINEAR. For C_SVC
    Eigen::VectorXd weight;

    // For ModelType::SVM/LINEAR. For NU_SVC, ONE_CLASS, and NU_SVR
    double nu = 0.5;

    // For ModelType::SVM/LINEAR. For EPSILON_SVR
    double p = 0.1;

    // For ModelType::SVM. Use the shrinking heuristics
    bool shrinking = true;

    // For ModelType::SVM. Do probability estimates
    bool probability = false;

    // For ModelType::LINEAR
    Eigen::VectorXd init_sol;

    // For ModelType::LINEAR
    bool regularize_bias = true;

    // For ModelType::LINEAR
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
    train(Params params, const Eigen::MatrixXd& X, const Eigen::VectorXd& y);

    const Params&
    params() const;

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
    struct LinearImpl;
    friend struct LinearImpl;
    static std::unique_ptr<Impl>
    make_impl(const ModelType);
};

} // namespace svmegn
