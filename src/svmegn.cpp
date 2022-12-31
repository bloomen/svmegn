// svmegn is C++ wrapper library around libsvm using Eigen
// Repo: https://github.com/bloomen/svmegn
// Author: Christian Blume
// License: MIT http://www.opensource.org/licenses/mit-license.php

#include <svmegn.h>

#include <svm/svm.h>

#define SVM_ASSERT(condition)                                                  \
    if (!(condition))                                                          \
    ErrorThrower{} << __FILE__ << ":" << __LINE__ << ":" << __func__           \
                   << ": Assert failed: '" << #condition << "' "

namespace svmegn
{

namespace
{

constexpr int prob_density_mark_count = 10;

class ErrorThrower
{
public:
    explicit ErrorThrower() = default;

    ErrorThrower(const ErrorThrower&) = default;
    ErrorThrower&
    operator=(const ErrorThrower&) = default;
    ErrorThrower(ErrorThrower&&) = default;
    ErrorThrower&
    operator=(ErrorThrower&&) = default;

    ~ErrorThrower() noexcept(false)
    {
        throw SVMError{m_msg};
    }

    template <typename T>
    ErrorThrower&
    operator<<(T&& data) &
    {
        std::ostringstream os;
        os << std::forward<T>(data);
        m_msg += os.str();
        return *this;
    }

    template <typename T>
    ErrorThrower&&
    operator<<(T&& data) &&
    {
        return std::move(operator<<(std::forward<T>(data)));
    }

private:
    std::string m_msg;
};

void
free_content(svm_model& model)
{
    //    if (model.SV)
    //    {
    //        for (int i = 0; i < model.l; ++i)
    //        {
    //                        delete[] model.SV[i];
    //        }
    //    }
    svm_free_model_content(&model);
}

void
copy_content(const svm_model& from, svm_model& to)
{
    to.param = from.param;
    to.nr_class = from.nr_class;
    to.l = from.l;

    if (from.SV)
    {
        to.SV = (svm_node**)malloc(to.l);
        for (int i = 0; i < to.l; ++i)
        {
            int j = 0;
            while (from.SV[i][j++].index >= 0)
                ;
            to.SV[i] = (svm_node*)malloc(j);
            std::copy(from.SV[i], from.SV[i] + j, to.SV[i]);
        }
    }

    if (from.sv_coef)
    {
        const auto size = to.nr_class - 1;
        to.sv_coef = (double**)malloc(size);
        for (int i = 0; i < size; ++i)
        {
            to.sv_coef[i] = (double*)malloc(to.l);
            std::copy(from.sv_coef[i], from.sv_coef[i] + to.l, to.sv_coef[i]);
        }
    }

    if (from.rho)
    {
        const auto size = to.nr_class * (to.nr_class - 1) / 2;
        to.rho = (double*)malloc(size);
        std::copy(from.rho, from.rho + size, to.rho);
    }

    if (from.probA)
    {
        const auto size = to.nr_class * (to.nr_class - 1) / 2;
        to.probA = (double*)malloc(size);
        std::copy(from.probA, from.probA + size, to.probA);
    }

    if (from.probB)
    {
        const auto size = to.nr_class * (to.nr_class - 1) / 2;
        to.probB = (double*)malloc(size);
        std::copy(from.probB, from.probB + size, to.probB);
    }

    if (from.prob_density_marks)
    {
        constexpr auto size = prob_density_mark_count;
        to.prob_density_marks = (double*)malloc(size);
        std::copy(from.prob_density_marks,
                  from.prob_density_marks + size,
                  to.prob_density_marks);
    }

    if (from.sv_indices)
    {
        const auto size = to.l;
        to.sv_indices = (int*)malloc(size);
        std::copy(from.sv_indices, from.sv_indices + size, to.sv_indices);
    }

    if (from.label)
    {
        const auto size = to.nr_class;
        to.label = (int*)malloc(size);
        std::copy(from.label, from.label + size, to.label);
    }

    if (from.nSV)
    {
        const auto size = to.nr_class;
        to.nSV = (int*)malloc(size);
        std::copy(from.nSV, from.nSV + size, to.nSV);
    }

    to.free_sv = from.free_sv;
}

svm_parameter
convert(const Parameters& ip)
{
    svm_parameter op;
    op.svm_type = static_cast<int>(ip.svm_type);
    op.kernel_type = static_cast<int>(ip.kernel_type);
    op.degree = ip.degree;
    op.gamma = ip.gamma;
    op.coef0 = ip.coef0;
    if (ip.training)
    {
        op.cache_size = ip.training->cache_size;
        op.eps = ip.training->eps;
        op.C = ip.training->C;
        op.nr_weight = ip.training->nr_weight;
        op.nu = ip.training->nu;
        op.p = ip.training->p;
        op.shrinking = ip.training->shrinking ? 1 : 0;
        op.probability = ip.training->probability ? 1 : 0;

        if (op.nr_weight > 0)
        {
            SVM_ASSERT(op.nr_weight == ip.training->weight_label.size());
            SVM_ASSERT(op.nr_weight == ip.training->weight.size());
            op.weight_label =
                const_cast<int*>(ip.training->weight_label.data());
            op.weight = const_cast<double*>(ip.training->weight.data());
        }
    }
    return op;
}

struct ProblemDeleter
{
    void
    operator()(svm_problem* prob) const
    {
        if (prob)
        {
            if (prob->x)
            {
                for (int i = 0; i < prob->l; ++i)
                {
                    delete[] prob->x[i];
                }
                delete[] prob->x;
            }
            delete prob;
        }
    }
};

svm_node*
make_record(const Eigen::RowVectorXd& row)
{
    auto record = new svm_node[row.cols() + 1];
    for (int j = 0; j < row.cols(); ++j)
    {
        record[j] = svm_node{j, row(j)};
    }
    record[row.cols()] = svm_node{-1, 0};
    return record;
}

std::unique_ptr<svm_problem, ProblemDeleter>
make_problem(const Eigen::MatrixXd& X, const Eigen::VectorXd& y)
{
    SVM_ASSERT(X.rows() == y.rows());
    std::unique_ptr<svm_problem, ProblemDeleter> prob{new svm_problem{},
                                                      ProblemDeleter{}};
    prob->l = static_cast<int>(X.rows());
    prob->y = const_cast<double*>(y.data());
    prob->x = new svm_node*[prob->l];
    for (int i = 0; i < prob->l; ++i)
    {
        prob->x[i] = make_record(X.row(i));
    }
    return prob;
}

struct ModelDeleter
{
    void
    operator()(svm_model* model) const
    {
        if (model)
        {
            free_content(*model);
            free(model);
        }
    }
};

} // namespace

struct SVM::Impl
{
    void
    train(const Parameters& params,
          const Eigen::MatrixXd& X,
          const Eigen::MatrixXd& y)
    {
        const auto svm_params = convert(params);
        const auto prob = make_problem(X, y);
        std::unique_ptr<const char[]> error{
            svm_check_parameter(prob.get(), &svm_params)};
        SVM_ASSERT(error == nullptr) << error.get();
        const auto model = svm_train(prob.get(), &svm_params);
        m_model =
            std::unique_ptr<svm_model, ModelDeleter>{model, ModelDeleter{}};
    }

    std::unique_ptr<svm_model, ModelDeleter> m_model;
};

SVM::SVM(const SVM& other)
{
    *this = other;
}

SVM&
SVM::operator=(const SVM& other)
{
    if (this != &other)
    {
        m_impl.reset();
        m_impl = std::make_unique<Impl>();
        m_impl->m_model = std::unique_ptr<svm_model, ModelDeleter>{
            new svm_model{}, ModelDeleter{}};
        copy_content(*other.m_impl->m_model, *m_impl->m_model);
    }
    return *this;
}

SVM::~SVM()
{
}

SVM
SVM::train(const Parameters& params,
           const Eigen::MatrixXd& X,
           const Eigen::VectorXd& y)
{
    SVM svm;
    svm.m_impl = std::make_unique<Impl>();
    svm.m_impl->train(params, X, y);
    return svm;
}

Eigen::VectorXd
SVM::predict(const Eigen::MatrixXd& X) const
{
    Eigen::VectorXd y{X.rows()};
    for (int i = 0; i < X.rows(); ++i)
    {
        std::unique_ptr<svm_node[]> record{make_record(X.row(i))};
        y(i) = svm_predict(m_impl->m_model.get(), record.get());
    }
    return y;
}

void
SVM::save(std::ostream& os) const
{
}

SVM
SVM::load(std::istream& is)
{
    return {};
}

} // namespace svmegn
