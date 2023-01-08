// svmegn is C++ wrapper library around libsvm/liblinear using Eigen
// Repo: https://github.com/bloomen/svmegn
// Author: Christian Blume
// License: MIT http://www.opensource.org/licenses/mit-license.php

#include <type_traits>
#include <unordered_set>

#include "liblinear/linear.h"
#include "libsvm/svm.h"

#include <svmegn.h>

#define SVMEGN_ASSERT(condition)                                               \
    if (!(condition))                                                          \
    ErrorThrower{} << __FILE__ << ":" << __LINE__ << ":" << __func__           \
                   << ": Assert failed: '" << #condition << "' "

namespace svmegn
{

namespace
{

constexpr int serialize_version = 0;
constexpr int prob_density_mark_count = 10;

template <typename T, typename U>
T*
allocate(const U size, const bool zero = false)
{
    auto ptr = (T*)malloc(static_cast<std::size_t>(size) * sizeof(T));
    if (zero)
    {
        std::memset(ptr, 0, sizeof(T));
    }
    return ptr;
}

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

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4722)
#endif
    ~ErrorThrower() noexcept(false)
    {
        throw ModelError{m_msg};
    }
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

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

template <typename T>
struct is_eigen : std::false_type
{
};

template <typename T, int... Is>
struct is_eigen<Eigen::Matrix<T, Is...>> : std::true_type
{
};

template <bool is_enum, bool is_eigen>
struct Serializer;

template <>
struct Serializer<false, false>
{
    template <typename T>
    static void
    write(std::ostream& os, const T& value)
    {
        os.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }
    template <typename T>
    static void
    read(std::istream& is, T& value)
    {
        is.read(reinterpret_cast<char*>(&value), sizeof(T));
    }
};

template <>
struct Serializer<true, false>
{
    template <typename T>
    static void
    write(std::ostream& os, const T& value)
    {
        using type = std::underlying_type_t<T>;
        const auto enum_value = static_cast<type>(value);
        os.write(reinterpret_cast<const char*>(&enum_value), sizeof(type));
    }
    template <typename T>
    static void
    read(std::istream& is, T& value)
    {
        using type = std::underlying_type_t<T>;
        type enum_value;
        is.read(reinterpret_cast<char*>(&enum_value), sizeof(type));
        value = static_cast<T>(enum_value);
    }
};

template <>
struct Serializer<false, true>
{
    template <typename T>
    static void
    write(std::ostream& os, const T& value)
    {
        const auto rows = static_cast<SizeType>(value.rows());
        os.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        const auto cols = static_cast<SizeType>(value.cols());
        os.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        os.write(reinterpret_cast<const char*>(value.data()),
                 static_cast<std::streamsize>(sizeof(typename T::value_type) *
                                              rows * cols));
    }
    template <typename T>
    static void
    read(std::istream& is, T& value)
    {
        SizeType rows;
        is.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        SizeType cols;
        is.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        T matrix{rows, cols};
        is.read(reinterpret_cast<char*>(matrix.data()),
                static_cast<std::streamsize>(sizeof(typename T::value_type) *
                                             rows * cols));
        value = std::move(matrix);
    }
};

template <typename T>
void
write(std::ostream& os, const T& value)
{
    Serializer<std::is_enum<T>::value, is_eigen<T>::value>::write(os, value);
}

template <typename T>
void
write_array(std::ostream& os, const T* data, const SizeType size)
{
    os.write(reinterpret_cast<const char*>(data),
             static_cast<std::streamsize>(sizeof(T) * size));
}

template <typename T>
void
read(std::istream& is, T& value)
{
    Serializer<std::is_enum<T>::value, is_eigen<T>::value>::read(is, value);
}

template <typename T>
void
read_array(std::istream& is, T* data, const SizeType size)
{
    is.read(reinterpret_cast<char*>(data),
            static_cast<std::streamsize>(sizeof(T) * size));
}

void
write_parameters(std::ostream& os, const Params& params)
{
    write(os, params.model_type);
    write(os, params.svm_type);
    write(os, params.kernel_type);
    write(os, params.linear_type);
    write(os, params.degree);
    write(os, params.gamma);
    write(os, params.coef0);
    write(os, params.cache_size);
    write(os, params.eps);
    write(os, params.C);
    write(os, params.weight_label);
    write(os, params.weight);
    write(os, params.nu);
    write(os, params.p);
    write(os, params.shrinking);
    write(os, params.probability);
    write(os, params.init_sol);
    write(os, params.regularize_bias);
    write(os, params.bias);
}

void
read_parameters(std::istream& is, Params& params)
{
    // Note: model_type already read at this point
    read(is, params.svm_type);
    read(is, params.kernel_type);
    read(is, params.linear_type);
    read(is, params.degree);
    read(is, params.gamma);
    read(is, params.coef0);
    read(is, params.cache_size);
    read(is, params.eps);
    read(is, params.C);
    read(is, params.weight_label);
    read(is, params.weight);
    read(is, params.nu);
    read(is, params.p);
    read(is, params.shrinking);
    read(is, params.probability);
    read(is, params.init_sol);
    read(is, params.regularize_bias);
    read(is, params.bias);
}

libsvm::svm_parameter
to_svm_params(const Params& ip)
{
    libsvm::svm_parameter op;
    op.svm_type = static_cast<int>(ip.svm_type);
    op.kernel_type = static_cast<int>(ip.kernel_type);
    op.degree = ip.degree;
    op.gamma = ip.gamma;
    op.coef0 = ip.coef0;
    op.cache_size = ip.cache_size;
    op.eps = ip.eps;
    op.C = ip.C;
    op.nr_weight = static_cast<int>(ip.weight.size());
    op.nu = ip.nu;
    op.p = ip.p;
    op.shrinking = ip.shrinking ? 1 : 0;
    op.probability = ip.probability ? 1 : 0;
    if (op.nr_weight > 0)
    {
        SVMEGN_ASSERT(ip.weight.size() == ip.weight_label.size());
        op.weight_label = allocate<int>(op.nr_weight);
        op.weight = allocate<double>(op.nr_weight);
        std::copy(ip.weight_label.data(),
                  ip.weight_label.data() + op.nr_weight,
                  op.weight_label);
        std::copy(ip.weight.data(), ip.weight.data() + op.nr_weight, op.weight);
    }
    else
    {
        op.weight_label = nullptr;
        op.weight = nullptr;
    }
    return op;
}

liblinear::parameter
to_linear_params(const Params& ip)
{
    liblinear::parameter op;
    op.solver_type = static_cast<int>(ip.linear_type);
    op.eps = ip.eps;
    op.C = ip.C;
    op.nr_weight = static_cast<int>(ip.weight.size());
    op.p = ip.p;
    op.nu = ip.nu;
    if (op.nr_weight > 0)
    {
        SVMEGN_ASSERT(ip.weight.size() == ip.weight_label.size());
        op.weight_label = allocate<int>(op.nr_weight);
        op.weight = allocate<double>(op.nr_weight);
        std::copy(ip.weight_label.data(),
                  ip.weight_label.data() + op.nr_weight,
                  op.weight_label);
        std::copy(ip.weight.data(), ip.weight.data() + op.nr_weight, op.weight);
    }
    else
    {
        op.weight_label = nullptr;
        op.weight = nullptr;
    }
    if (ip.init_sol.size() > 0)
    {
        op.init_sol = allocate<double>(ip.init_sol.size());
        std::copy(ip.init_sol.data(),
                  ip.init_sol.data() + ip.init_sol.size(),
                  op.init_sol);
    }
    else
    {
        op.init_sol = nullptr;
    }
    op.regularize_bias = ip.regularize_bias;
    return op;
}

std::unique_ptr<libsvm::svm_node, decltype(std::free)*>
make_svm_record(const Eigen::RowVectorXd& row)
{
    auto record = allocate<libsvm::svm_node>(row.cols() + 1);
    for (int j = 0; j < row.cols(); ++j)
    {
        record[j] = libsvm::svm_node{j + 1, row(j)};
    }
    record[row.cols()] = libsvm::svm_node{-1, 0};
    return std::unique_ptr<libsvm::svm_node, decltype(std::free)*>{record,
                                                                   std::free};
}

std::unique_ptr<libsvm::svm_node, decltype(std::free)*>
make_svm_record(SpaMatrixD::InnerIterator it)
{
    std::vector<libsvm::svm_node> row;
    for (; it; ++it)
    {
        row.push_back({static_cast<int>(it.col()) + 1, it.value()});
    }
    row.push_back({-1, 0});
    auto record = allocate<libsvm::svm_node>(row.size());
    std::copy(row.begin(), row.end(), record);
    return std::unique_ptr<libsvm::svm_node, decltype(std::free)*>{record,
                                                                   std::free};
}

std::unique_ptr<liblinear::feature_node, decltype(std::free)*>
make_linear_record(const Eigen::RowVectorXd& row)
{
    auto record = allocate<liblinear::feature_node>(row.cols() + 1);
    for (int j = 0; j < row.cols(); ++j)
    {
        record[j] = liblinear::feature_node{j + 1, row(j)};
    }
    record[row.cols()] = liblinear::feature_node{-1, 0};
    return std::unique_ptr<liblinear::feature_node, decltype(std::free)*>{
        record, std::free};
}

std::unique_ptr<liblinear::feature_node, decltype(std::free)*>
make_linear_record(SpaMatrixD::InnerIterator it)
{
    std::vector<liblinear::feature_node> row;
    for (; it; ++it)
    {
        row.push_back({static_cast<int>(it.col()) + 1, it.value()});
    }
    row.push_back({-1, 0});
    auto record = allocate<liblinear::feature_node>(row.size());
    std::copy(row.begin(), row.end(), record);
    return std::unique_ptr<liblinear::feature_node, decltype(std::free)*>{
        record, std::free};
}

class SvmProblem
{
public:
    SvmProblem(const MatrixD& X, const VectorD& y)
    {
        m_prob->l = static_cast<int>(X.rows());
        m_prob->y = allocate<double>(m_prob->l);
        std::copy(y.data(), y.data() + m_prob->l, m_prob->y);
        m_prob->x = allocate<libsvm::svm_node*>(m_prob->l);
        for (int i = 0; i < m_prob->l; ++i)
        {
            m_prob->x[i] = make_svm_record(X.row(i)).release();
        }
    }

    SvmProblem(const SpaMatrixD& X, const VectorD& y)
    {
        m_prob->l = static_cast<int>(X.outerSize());
        m_prob->y = allocate<double>(m_prob->l);
        std::copy(y.data(), y.data() + m_prob->l, m_prob->y);
        m_prob->x = allocate<libsvm::svm_node*>(m_prob->l);
        for (int i = 0; i < m_prob->l; ++i)
        {
            SpaMatrixD::InnerIterator it{X, i};
            m_prob->x[i] = make_svm_record(it).release();
        }
    }

    ~SvmProblem()
    {
        std::free(m_prob->y);
        for (int i = 0; i < m_prob->l; ++i)
        {
            if (m_sv_indices.find(i + 1) != m_sv_indices.end())
            {
                continue;
            }
            std::free(m_prob->x[i]);
        }
        std::free(m_prob->x);
        std::free(m_prob);
    }

    SvmProblem(const SvmProblem&) = delete;
    SvmProblem&
    operator=(const SvmProblem&) = delete;
    SvmProblem(SvmProblem&&) = delete;
    SvmProblem&
    operator=(SvmProblem&&) = delete;

    libsvm::svm_problem&
    get() const
    {
        return *m_prob;
    }

    void
    set_sv_indices(const int* idx, const int l)
    {
        m_sv_indices = std::unordered_set<int>{idx, idx + l};
    }

private:
    std::unordered_set<int> m_sv_indices;
    libsvm::svm_problem* m_prob = allocate<libsvm::svm_problem>(1);
};

class LinearProblem
{
public:
    LinearProblem(const MatrixD& X, const VectorD& y, const double bias)
    {
        m_prob->l = static_cast<int>(X.rows());
        m_prob->n = static_cast<int>(X.cols());
        m_prob->y = allocate<double>(m_prob->l);
        std::copy(y.data(), y.data() + m_prob->l, m_prob->y);
        m_prob->x = allocate<liblinear::feature_node*>(m_prob->l);
        for (int i = 0; i < m_prob->l; ++i)
        {
            m_prob->x[i] = make_linear_record(X.row(i)).release();
        }
        m_prob->bias = bias;
    }

    LinearProblem(const SpaMatrixD& X, const VectorD& y, const double bias)
    {
        m_prob->l = static_cast<int>(X.outerSize());
        m_prob->n = static_cast<int>(X.cols());
        m_prob->y = allocate<double>(m_prob->l);
        std::copy(y.data(), y.data() + m_prob->l, m_prob->y);
        m_prob->x = allocate<liblinear::feature_node*>(m_prob->l);
        for (int i = 0; i < m_prob->l; ++i)
        {
            SpaMatrixD::InnerIterator it{X, i};
            m_prob->x[i] = make_linear_record(it).release();
        }
        m_prob->bias = bias;
    }

    ~LinearProblem()
    {
        std::free(m_prob->y);
        for (int i = 0; i < m_prob->l; ++i)
        {
            std::free(m_prob->x[i]);
        }
        std::free(m_prob->x);
        std::free(m_prob);
    }

    LinearProblem(const LinearProblem&) = delete;
    LinearProblem&
    operator=(const LinearProblem&) = delete;
    LinearProblem(LinearProblem&&) = delete;
    LinearProblem&
    operator=(LinearProblem&&) = delete;

    liblinear::problem&
    get() const
    {
        return *m_prob;
    }

private:
    liblinear::problem* m_prob = allocate<liblinear::problem>(1);
};

} // namespace

void
set_print_string_function(const ModelType model_type, void (*func)(const char*))
{
    switch (model_type)
    {
    case ModelType::SVM:
        libsvm::svm_set_print_string_function(func);
        return;
    case ModelType::LINEAR:
        liblinear::set_print_string_function(func);
        return;
    }
    SVMEGN_ASSERT(false) << "No such model type: " << model_type;
    return;
}

void
remove_print_string_function(const ModelType model_type)
{
    set_print_string_function(model_type, [](const char*) {});
}

struct Model::Impl
{
    Impl() = default;
    virtual ~Impl() = default;

    Impl(const Impl&) = delete;
    Impl&
    operator=(const Impl&) = delete;
    Impl(Impl&&) = delete;
    Impl&
    operator=(Impl&&) = delete;

    virtual void
    copy_from(const Impl& i) = 0;
    virtual const Params&
    params() const = 0;
    virtual void
    train(Params params, const MatrixD& X, const VectorD& y) = 0;
    virtual void
    train(Params params, const SpaMatrixD& X, const VectorD& y) = 0;
    virtual VectorD
    cross_validate(const Params& params,
                   const MatrixD& X,
                   const VectorD& y,
                   int nr_fold) = 0;
    virtual VectorD
    cross_validate(const Params& params,
                   const SpaMatrixD& X,
                   const VectorD& y,
                   int nr_fold) = 0;
    virtual int
    nr_class() const = 0;
    virtual std::optional<VectorI>
    labels() const = 0;
    virtual double
    predict(const Eigen::RowVectorXd& row) const = 0;
    virtual double
    predict(SpaMatrixD::InnerIterator it) const = 0;
    virtual std::pair<double, Eigen::RowVectorXd>
    predict_proba(const Eigen::RowVectorXd& row) const = 0;
    virtual std::pair<double, Eigen::RowVectorXd>
    predict_proba(SpaMatrixD::InnerIterator it) const = 0;
    virtual void
    save(std::ostream& os) const = 0;
    virtual void
    load(std::istream& is, int version) = 0;
};

struct Model::SvmImpl : public Model::Impl
{
    ~SvmImpl()
    {
        destroy_model(m_model);
    }

    void
    copy_from(const Model::Impl& i) override
    {
        const auto& svmi = static_cast<const Model::SvmImpl&>(i);
        destroy_model(m_model);
        m_model = allocate<libsvm::svm_model>(1, true);
        copy_model(*svmi.m_model, *m_model);
        m_params = svmi.m_params;
        m_model->param = to_svm_params(m_params);
    }

    const Params&
    params() const override
    {
        return m_params;
    }

    void
    train(Params params, const MatrixD& X, const VectorD& y) override
    {
        train_impl(std::move(params), X, y);
    }

    void
    train(Params params, const SpaMatrixD& X, const VectorD& y) override
    {
        train_impl(std::move(params), X, y);
    }

    VectorD
    cross_validate(const Params& params,
                   const MatrixD& X,
                   const VectorD& y,
                   const int nr_fold) override
    {
        return cross_validate_impl(params, X, y, nr_fold);
    }

    VectorD
    cross_validate(const Params& params,
                   const SpaMatrixD& X,
                   const VectorD& y,
                   const int nr_fold) override
    {
        return cross_validate_impl(params, X, y, nr_fold);
    }

    int
    nr_class() const override
    {
        return libsvm::svm_get_nr_class(m_model);
    }

    std::optional<VectorI>
    labels() const override
    {
        if (m_model->label)
        {
            VectorI vec{nr_class()};
            std::copy(m_model->label, m_model->label + nr_class(), vec.data());
            return vec;
        }
        return std::nullopt;
    }

    double
    predict(const Eigen::RowVectorXd& row) const override
    {
        auto record = make_svm_record(row);
        return libsvm::svm_predict(m_model, record.get());
    }

    double
    predict(SpaMatrixD::InnerIterator it) const override
    {
        auto record = make_svm_record(it);
        return libsvm::svm_predict(m_model, record.get());
    }

    std::pair<double, Eigen::RowVectorXd>
    predict_proba(const Eigen::RowVectorXd& row) const override
    {
        SVMEGN_ASSERT(libsvm::svm_check_probability_model(m_model) != 0)
            << "model cannot estimate probas";
        auto record = make_svm_record(row);
        Eigen::RowVectorXd prob{nr_class()};
        const auto y =
            libsvm::svm_predict_probability(m_model, record.get(), prob.data());
        return std::make_pair(std::move(y), std::move(prob));
    }

    std::pair<double, Eigen::RowVectorXd>
    predict_proba(SpaMatrixD::InnerIterator it) const override
    {
        SVMEGN_ASSERT(libsvm::svm_check_probability_model(m_model) != 0)
            << "model cannot estimate probas";
        auto record = make_svm_record(it);
        Eigen::RowVectorXd prob{nr_class()};
        const auto y =
            libsvm::svm_predict_probability(m_model, record.get(), prob.data());
        return std::make_pair(std::move(y), std::move(prob));
    }

    void
    save(std::ostream& os) const override
    {
        SVMEGN_ASSERT(m_model != nullptr);
        write(os, serialize_version);
        write_parameters(os, m_params);

        const bool have_model = m_model != nullptr;
        write(os, have_model);
        if (have_model)
        {
            write(os, m_model->nr_class);
            write(os, m_model->l);

            const bool have_SV = m_model->SV != nullptr;
            write(os, have_SV);
            if (have_SV)
            {
                for (int i = 0; i < m_model->l; ++i)
                {
                    SizeType j = 0;
                    while (m_model->SV[i][j++].index >= 0)
                        ;
                    write(os, j);
                    write_array(os, m_model->SV[i], j);
                }
            }

            const bool have_sv_coeff = m_model->sv_coef != nullptr;
            write(os, have_sv_coeff);
            if (have_sv_coeff)
            {
                const auto size = m_model->nr_class - 1;
                for (int i = 0; i < size; ++i)
                {
                    write_array(os,
                                m_model->sv_coef[i],
                                static_cast<SizeType>(m_model->l));
                }
            }

            const bool have_rho = m_model->rho != nullptr;
            write(os, have_rho);
            if (have_rho)
            {
                const auto size =
                    m_model->nr_class * (m_model->nr_class - 1) / 2;
                write_array(os, m_model->rho, static_cast<SizeType>(size));
            }

            const bool have_probA = m_model->probA != nullptr;
            write(os, have_probA);
            if (have_probA)
            {
                const auto size =
                    m_model->nr_class * (m_model->nr_class - 1) / 2;
                write_array(os, m_model->probA, static_cast<SizeType>(size));
            }

            const bool have_probB = m_model->probB != nullptr;
            write(os, have_probB);
            if (have_probB)
            {
                const auto size =
                    m_model->nr_class * (m_model->nr_class - 1) / 2;
                write_array(os, m_model->probB, static_cast<SizeType>(size));
            }

            const bool have_prob_density_marks =
                m_model->prob_density_marks != nullptr;
            write(os, have_prob_density_marks);
            if (have_prob_density_marks)
            {
                constexpr auto size = prob_density_mark_count;
                write_array(os,
                            m_model->prob_density_marks,
                            static_cast<SizeType>(size));
            }

            const bool have_sv_indices = m_model->sv_indices != nullptr;
            write(os, have_sv_indices);
            if (have_sv_indices)
            {
                const auto size = m_model->l;
                write_array(
                    os, m_model->sv_indices, static_cast<SizeType>(size));
            }

            const bool have_label = m_model->label != nullptr;
            write(os, have_label);
            if (have_label)
            {
                const auto size = m_model->nr_class;
                write_array(os, m_model->label, static_cast<SizeType>(size));
            }

            const bool have_nSV = m_model->nSV != nullptr;
            write(os, have_nSV);
            if (have_nSV)
            {
                const auto size = m_model->nr_class;
                write_array(os, m_model->nSV, static_cast<SizeType>(size));
            }

            write(os, m_model->free_sv);
        }
    }

    void
    load(std::istream& is, const int version) override
    {
        (void)version;
        read_parameters(is, m_params);

        bool have_model;
        read(is, have_model);
        if (have_model)
        {
            destroy_model(m_model);
            m_model = allocate<libsvm::svm_model>(1, true);
            m_model->param = to_svm_params(m_params);
            read(is, m_model->nr_class);
            read(is, m_model->l);

            bool have_SV;
            read(is, have_SV);
            if (have_SV)
            {
                m_model->SV = allocate<libsvm::svm_node*>(m_model->l);
                for (int i = 0; i < m_model->l; ++i)
                {
                    SizeType j;
                    read(is, j);
                    m_model->SV[i] = allocate<libsvm::svm_node>(j);
                    read_array(is, m_model->SV[i], j);
                }
            }

            bool have_sv_coeff;
            read(is, have_sv_coeff);
            if (have_sv_coeff)
            {
                const auto size = m_model->nr_class - 1;
                m_model->sv_coef = allocate<double*>(size);
                for (int i = 0; i < size; ++i)
                {
                    m_model->sv_coef[i] = allocate<double>(m_model->l);
                    read_array(is,
                               m_model->sv_coef[i],
                               static_cast<SizeType>(m_model->l));
                }
            }

            bool have_rho;
            read(is, have_rho);
            if (have_rho)
            {
                const auto size =
                    m_model->nr_class * (m_model->nr_class - 1) / 2;
                m_model->rho = allocate<double>(size);
                read_array(is, m_model->rho, static_cast<SizeType>(size));
            }

            bool have_probA;
            read(is, have_probA);
            if (have_probA)
            {
                const auto size =
                    m_model->nr_class * (m_model->nr_class - 1) / 2;
                m_model->probA = allocate<double>(size);
                read_array(is, m_model->probA, static_cast<SizeType>(size));
            }

            bool have_probB;
            read(is, have_probB);
            if (have_probB)
            {
                const auto size =
                    m_model->nr_class * (m_model->nr_class - 1) / 2;
                m_model->probB = allocate<double>(size);
                read_array(is, m_model->probB, static_cast<SizeType>(size));
            }

            bool have_prob_density_marks;
            read(is, have_prob_density_marks);
            if (have_prob_density_marks)
            {
                constexpr auto size = prob_density_mark_count;
                m_model->prob_density_marks = allocate<double>(size);
                read_array(is,
                           m_model->prob_density_marks,
                           static_cast<SizeType>(size));
            }

            bool have_sv_indices;
            read(is, have_sv_indices);
            if (have_sv_indices)
            {
                const auto size = m_model->l;
                m_model->sv_indices = allocate<int>(size);
                read_array(
                    is, m_model->sv_indices, static_cast<SizeType>(size));
            }

            bool have_label;
            read(is, have_label);
            if (have_label)
            {
                const auto size = m_model->nr_class;
                m_model->label = allocate<int>(size);
                read_array(is, m_model->label, static_cast<SizeType>(size));
            }

            bool have_nSV;
            read(is, have_nSV);
            if (have_nSV)
            {
                const auto size = m_model->nr_class;
                m_model->nSV = allocate<int>(size);
                read_array(is, m_model->nSV, static_cast<SizeType>(size));
            }

            read(is, m_model->free_sv);
        }
    }

private:
    template <typename Mat>
    void
    train_impl(Params params, const Mat& X, const VectorD& y)
    {
        m_params = std::move(params);
        const auto svm_params = to_svm_params(m_params);
        SvmProblem prob{X, y};
        const auto error =
            libsvm::svm_check_parameter(&prob.get(), &svm_params);
        SVMEGN_ASSERT(error == nullptr) << error;
        m_model = libsvm::svm_train(&prob.get(), &svm_params);
        SVMEGN_ASSERT(m_model != nullptr) << "libsvm::svm_train() failed";
        prob.set_sv_indices(m_model->sv_indices, m_model->l);
    }

    template <typename Mat>
    VectorD
    cross_validate_impl(const Params& params,
                        const Mat& X,
                        const VectorD& y,
                        const int nr_fold)
    {
        const auto svm_params = to_svm_params(params);
        SvmProblem prob{X, y};
        const auto error =
            libsvm::svm_check_parameter(&prob.get(), &svm_params);
        SVMEGN_ASSERT(error == nullptr) << error;
        VectorD resp{y.size()};
        libsvm::svm_cross_validation(
            &prob.get(), &svm_params, nr_fold, resp.data());
        return resp;
    }

    static void
    destroy_model(libsvm::svm_model*& model)
    {
        if (!model)
        {
            return;
        }
        if (model->SV)
        {
            for (int i = 0; i < model->l; ++i)
            {
                std::free(model->SV[i]);
            }
        }
        libsvm::svm_destroy_param(&model->param);
        libsvm::svm_free_and_destroy_model(&model);
        model = nullptr;
    }

    static void
    copy_model(const libsvm::svm_model& from, libsvm::svm_model& to)
    {
        // Note: param is copied separately
        to.nr_class = from.nr_class;
        to.l = from.l;

        if (from.SV)
        {
            to.SV = allocate<libsvm::svm_node*>(to.l);
            for (int i = 0; i < to.l; ++i)
            {
                int j = 0;
                while (from.SV[i][j++].index >= 0)
                    ;
                to.SV[i] = allocate<libsvm::svm_node>(j);
                std::copy(from.SV[i], from.SV[i] + j, to.SV[i]);
            }
        }

        if (from.sv_coef)
        {
            const auto size = to.nr_class - 1;
            to.sv_coef = allocate<double*>(size);
            for (int i = 0; i < size; ++i)
            {
                to.sv_coef[i] = allocate<double>(to.l);
                std::copy(
                    from.sv_coef[i], from.sv_coef[i] + to.l, to.sv_coef[i]);
            }
        }

        if (from.rho)
        {
            const auto size = to.nr_class * (to.nr_class - 1) / 2;
            to.rho = allocate<double>(size);
            std::copy(from.rho, from.rho + size, to.rho);
        }

        if (from.probA)
        {
            const auto size = to.nr_class * (to.nr_class - 1) / 2;
            to.probA = allocate<double>(size);
            std::copy(from.probA, from.probA + size, to.probA);
        }

        if (from.probB)
        {
            const auto size = to.nr_class * (to.nr_class - 1) / 2;
            to.probB = allocate<double>(size);
            std::copy(from.probB, from.probB + size, to.probB);
        }

        if (from.prob_density_marks)
        {
            constexpr auto size = prob_density_mark_count;
            to.prob_density_marks = allocate<double>(size);
            std::copy(from.prob_density_marks,
                      from.prob_density_marks + size,
                      to.prob_density_marks);
        }

        if (from.sv_indices)
        {
            const auto size = to.l;
            to.sv_indices = allocate<int>(size);
            std::copy(from.sv_indices, from.sv_indices + size, to.sv_indices);
        }

        if (from.label)
        {
            const auto size = to.nr_class;
            to.label = allocate<int>(size);
            std::copy(from.label, from.label + size, to.label);
        }

        if (from.nSV)
        {
            const auto size = to.nr_class;
            to.nSV = allocate<int>(size);
            std::copy(from.nSV, from.nSV + size, to.nSV);
        }

        to.free_sv = from.free_sv;
    }

    libsvm::svm_model* m_model = nullptr;
    Params m_params;
};

struct Model::LinearImpl : public Model::Impl
{
    ~LinearImpl()
    {
        destroy_model(m_model);
    }

    void
    copy_from(const Model::Impl& i) override
    {
        const auto& svmi = static_cast<const Model::LinearImpl&>(i);
        destroy_model(m_model);
        m_model = allocate<liblinear::model>(1, true);
        copy_model(*svmi.m_model, *m_model);
        m_params = svmi.m_params;
        m_model->param = to_linear_params(m_params);
    }

    const Params&
    params() const override
    {
        return m_params;
    }

    void
    train(Params params, const MatrixD& X, const VectorD& y) override
    {
        train_impl(std::move(params), X, y);
    }

    void
    train(Params params, const SpaMatrixD& X, const VectorD& y) override
    {
        train_impl(std::move(params), X, y);
    }

    VectorD
    cross_validate(const Params& params,
                   const MatrixD& X,
                   const VectorD& y,
                   const int nr_fold) override
    {
        return cross_validate_impl(params, X, y, nr_fold);
    }

    VectorD
    cross_validate(const Params& params,
                   const SpaMatrixD& X,
                   const VectorD& y,
                   const int nr_fold) override
    {
        return cross_validate_impl(params, X, y, nr_fold);
    }

    int
    nr_class() const override
    {
        return liblinear::get_nr_class(m_model);
    }

    std::optional<VectorI>
    labels() const override
    {
        if (m_model->label)
        {
            VectorI vec{nr_class()};
            std::copy(m_model->label, m_model->label + nr_class(), vec.data());
            return vec;
        }
        return std::nullopt;
    }

    double
    predict(const Eigen::RowVectorXd& row) const override
    {
        auto record = make_linear_record(row);
        return liblinear::predict(m_model, record.get());
    }

    double
    predict(SpaMatrixD::InnerIterator it) const override
    {
        auto record = make_linear_record(it);
        return liblinear::predict(m_model, record.get());
    }

    std::pair<double, Eigen::RowVectorXd>
    predict_proba(const Eigen::RowVectorXd& row) const override
    {
        SVMEGN_ASSERT(liblinear::check_probability_model(m_model) != 0)
            << "model cannot estimate probas";
        auto record = make_linear_record(row);
        Eigen::RowVectorXd prob{nr_class()};
        const auto y =
            liblinear::predict_probability(m_model, record.get(), prob.data());
        return std::make_pair(std::move(y), std::move(prob));
    }

    std::pair<double, Eigen::RowVectorXd>
    predict_proba(SpaMatrixD::InnerIterator it) const override
    {
        SVMEGN_ASSERT(liblinear::check_probability_model(m_model) != 0)
            << "model cannot estimate probas";
        auto record = make_linear_record(it);
        Eigen::RowVectorXd prob{nr_class()};
        const auto y =
            liblinear::predict_probability(m_model, record.get(), prob.data());
        return std::make_pair(std::move(y), std::move(prob));
    }

    void
    save(std::ostream& os) const override
    {
        SVMEGN_ASSERT(m_model != nullptr);
        write(os, serialize_version);
        write_parameters(os, m_params);

        const bool have_model = m_model != nullptr;
        write(os, have_model);
        if (have_model)
        {
            write(os, m_model->nr_class);
            write(os, m_model->nr_feature);

            const bool have_w = m_model->w != nullptr;
            write(os, have_w);
            if (have_w)
            {
                write_array(
                    os, m_model->w, static_cast<SizeType>(m_model->nr_feature));
            }

            const bool have_label = m_model->label != nullptr;
            write(os, have_label);
            if (have_label)
            {
                write_array(os,
                            m_model->label,
                            static_cast<SizeType>(m_model->nr_class));
            }

            write(os, m_model->bias);
            write(os, m_model->rho);
        }
    }

    void
    load(std::istream& is, const int version) override
    {
        (void)version;
        read_parameters(is, m_params);

        bool have_model;
        read(is, have_model);
        if (have_model)
        {
            destroy_model(m_model);
            m_model = allocate<liblinear::model>(1, true);
            m_model->param = to_linear_params(m_params);
            read(is, m_model->nr_class);
            read(is, m_model->nr_feature);

            bool have_w;
            read(is, have_w);
            if (have_w)
            {
                const auto size = m_model->nr_feature;
                m_model->w = allocate<double>(size);
                read_array(is, m_model->w, static_cast<SizeType>(size));
            }

            bool have_label;
            read(is, have_label);
            if (have_label)
            {
                const auto size = m_model->nr_class;
                m_model->label = allocate<int>(size);
                read_array(is, m_model->label, static_cast<SizeType>(size));
            }

            read(is, m_model->bias);
            read(is, m_model->rho);
        }
    }

private:
    template <typename Mat>
    void
    train_impl(Params params, const Mat& X, const VectorD& y)
    {
        m_params = std::move(params);
        const auto linear_params = to_linear_params(m_params);
        LinearProblem prob{X, y, m_params.bias};
        const auto error =
            liblinear::check_parameter(&prob.get(), &linear_params);
        SVMEGN_ASSERT(error == nullptr) << error;
        m_model = liblinear::train(&prob.get(), &linear_params);
        SVMEGN_ASSERT(m_model != nullptr) << "liblinear::train() failed";
    }

    template <typename Mat>
    VectorD
    cross_validate_impl(const Params& params,
                        const Mat& X,
                        const VectorD& y,
                        const int nr_fold)
    {
        const auto linear_params = to_linear_params(params);
        LinearProblem prob{X, y, m_params.bias};
        const auto error =
            liblinear::check_parameter(&prob.get(), &linear_params);
        SVMEGN_ASSERT(error == nullptr) << error;
        VectorD resp{y.size()};
        liblinear::cross_validation(
            &prob.get(), &linear_params, nr_fold, resp.data());
        return resp;
    }

    static void
    destroy_model(liblinear::model*& model)
    {
        if (!model)
        {
            return;
        }
        liblinear::destroy_param(&model->param);
        liblinear::free_and_destroy_model(&model);
        model = nullptr;
    }

    static void
    copy_model(const liblinear::model& from, liblinear::model& to)
    {
        // Note: param is copied separately
        to.nr_class = from.nr_class;
        to.nr_feature = from.nr_feature;

        if (from.w)
        {
            to.w = allocate<double>(to.nr_feature);
            std::copy(from.w, from.w + from.nr_feature, to.w);
        }

        if (from.label)
        {
            to.label = allocate<int>(to.nr_class);
            std::copy(from.label, from.label + from.nr_class, to.label);
        }

        to.bias = from.bias;
        to.rho = from.rho;
    }

    liblinear::model* m_model = nullptr;
    Params m_params;
};

std::unique_ptr<Model::Impl>
Model::make_impl(const ModelType model_type)
{
    switch (model_type)
    {
    case ModelType::SVM:
        return std::make_unique<Model::SvmImpl>();
    case ModelType::LINEAR:
        return std::make_unique<Model::LinearImpl>();
    }
    SVMEGN_ASSERT(false) << "No such model type: " << model_type;
    return nullptr;
}

Model::~Model()
{
}

Model::Model(const Model& other)
    : m_impl{make_impl(other.params().model_type)}
{
    m_impl->copy_from(*other.m_impl);
}

Model&
Model::operator=(const Model& other)
{
    if (this != &other)
    {
        m_impl.reset();
        m_impl = make_impl(other.params().model_type);
        m_impl->copy_from(*other.m_impl);
    }
    return *this;
}

Model::Model(Model&&) = default;

Model&
Model::operator=(Model&&) = default;

Model
Model::train(Params params, const MatrixD& X, const VectorD& y)
{
    Model model;
    model.m_impl = make_impl(params.model_type);
    model.m_impl->train(std::move(params), X, y);
    return model;
}

Model
Model::train(Params params, const SpaMatrixD& X, const VectorD& y)
{
    Model model;
    model.m_impl = make_impl(params.model_type);
    model.m_impl->train(std::move(params), X, y);
    return model;
}

VectorD
Model::cross_validate(const Params& params,
                      const MatrixD& X,
                      const VectorD& y,
                      const int nr_fold)
{
    return make_impl(params.model_type)->cross_validate(params, X, y, nr_fold);
}

VectorD
Model::cross_validate(const Params& params,
                      const SpaMatrixD& X,
                      const VectorD& y,
                      const int nr_fold)
{
    return make_impl(params.model_type)->cross_validate(params, X, y, nr_fold);
}

const Params&
Model::params() const
{
    return m_impl->params();
}

int
Model::nr_class() const
{
    return m_impl->nr_class();
}

std::optional<VectorI>
Model::labels() const
{
    return m_impl->labels();
}

Prediction
Model::predict(const MatrixD& X, const bool prob) const
{
    Prediction pred;
    pred.y = VectorD{X.rows()};
    if (prob)
    {
        pred.prob = MatrixD{X.rows(), m_impl->nr_class()};
        for (int i = 0; i < X.rows(); ++i)
        {
            auto row = m_impl->predict_proba(X.row(i));
            pred.y(i) = std::move(row.first);
            pred.prob->row(i) = std::move(row.second);
        }
    }
    else
    {
        for (int i = 0; i < X.rows(); ++i)
        {
            pred.y(i) = m_impl->predict(X.row(i));
        }
    }
    return pred;
}

Prediction
Model::predict(const SpaMatrixD& X, const bool prob) const
{
    Prediction pred;
    pred.y = VectorD{X.outerSize()};
    if (prob)
    {
        pred.prob = MatrixD{X.outerSize(), m_impl->nr_class()};
        for (int i = 0; i < X.outerSize(); ++i)
        {
            SpaMatrixD::InnerIterator it{X, i};
            auto row = m_impl->predict_proba(it);
            pred.y(i) = std::move(row.first);
            pred.prob->row(i) = std::move(row.second);
        }
    }
    else
    {
        for (int i = 0; i < X.outerSize(); ++i)
        {
            SpaMatrixD::InnerIterator it{X, i};
            pred.y(i) = m_impl->predict(it);
        }
    }
    return pred;
}

void
Model::save(std::ostream& os) const
{
    m_impl->save(os);
}

Model
Model::load(std::istream& is)
{
    int version;
    read(is, version);
    ModelType model_type;
    read(is, model_type);
    Model model;
    model.m_impl = make_impl(model_type);
    model.m_impl->load(is, version);
    return model;
}

} // namespace svmegn
