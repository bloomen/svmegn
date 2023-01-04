// svmegn is C++ wrapper library around libsvm using Eigen
// Repo: https://github.com/bloomen/svmegn
// Author: Christian Blume
// License: MIT http://www.opensource.org/licenses/mit-license.php

#include <type_traits>
#include <unordered_set>

#include <svm/svm.h>

#include <svmegn.h>

#define SVMEGN_ASSERT(condition)                                               \
    if (!(condition))                                                          \
    ErrorThrower{} << __FILE__ << ":" << __LINE__ << ":" << __func__           \
                   << ": Assert failed: '" << #condition << "' "

namespace svmegn
{

void
set_print_string_function(void (*print_func)(const char*))
{
    svm_set_print_string_function(print_func);
}

void
remove_print_string_function()
{
    set_print_string_function([](const char*) {});
}

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
        const auto enum_value = static_cast<int>(value);
        os.write(reinterpret_cast<const char*>(&enum_value), sizeof(int));
    }
    template <typename T>
    static void
    read(std::istream& is, T& value)
    {
        int enum_value;
        is.read(reinterpret_cast<char*>(&enum_value), sizeof(int));
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
        const auto rows = value.rows();
        os.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        const auto cols = value.cols();
        os.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        os.write(reinterpret_cast<const char*>(value.data()),
                 sizeof(typename T::value_type) * rows * cols);
    }
    template <typename T>
    static void
    read(std::istream& is, T& value)
    {
        using Index = decltype(std::declval<Eigen::MatrixXd>().rows());
        Index rows;
        is.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        Index cols;
        is.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        T matrix{rows, cols};
        is.read(reinterpret_cast<char*>(matrix.data()),
                sizeof(typename T::value_type) * rows * cols);
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
write_array(std::ostream& os, const T* data, const std::size_t size)
{
    os.write(reinterpret_cast<const char*>(data), sizeof(T) * size);
}

template <typename T>
void
read(std::istream& is, T& value)
{
    Serializer<std::is_enum<T>::value, is_eigen<T>::value>::read(is, value);
}

template <typename T>
void
read_array(std::istream& is, T* data, const std::size_t size)
{
    is.read(reinterpret_cast<char*>(data), sizeof(T) * size);
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
    op.cache_size = ip.cache_size;
    op.eps = ip.eps;
    op.C = ip.C;
    op.nr_weight = ip.nr_weight;
    op.nu = ip.nu;
    op.p = ip.p;
    op.shrinking = ip.shrinking ? 1 : 0;
    op.probability = ip.probability ? 1 : 0;
    if (op.nr_weight > 0)
    {
        SVMEGN_ASSERT(op.nr_weight == ip.weight_label.size());
        SVMEGN_ASSERT(op.nr_weight == ip.weight.size());
        op.weight_label = const_cast<int*>(ip.weight_label.data());
        op.weight = const_cast<double*>(ip.weight.data());
    }
    return op;
}

std::unique_ptr<svm_node, decltype(std::free)*>
make_record(const Eigen::RowVectorXd& row)
{
    auto record = allocate<svm_node>(row.cols() + 1);
    for (int j = 0; j < row.cols(); ++j)
    {
        record[j] = svm_node{j, row(j)};
    }
    record[row.cols()] = svm_node{-1, 0};
    return std::unique_ptr<svm_node, decltype(std::free)*>{record, std::free};
}

class Problem
{
public:
    Problem(const Eigen::MatrixXd& X, const Eigen::VectorXd& y)
    {
        m_prob->l = static_cast<int>(X.rows());
        m_prob->y = const_cast<double*>(y.data());
        m_prob->x = new svm_node*[m_prob->l];
        for (int i = 0; i < m_prob->l; ++i)
        {
            m_prob->x[i] = make_record(X.row(i)).release();
        }
    }

    ~Problem()
    {
        if (m_prob->x)
        {
            for (int i = 0; i < m_prob->l; ++i)
            {
                if (m_sv_indices.find(i + 1) != m_sv_indices.end())
                {
                    continue;
                }
                std::free(m_prob->x[i]);
            }
            delete[] m_prob->x;
        }
        delete m_prob;
    }

    Problem(const Problem&) = delete;
    Problem&
    operator=(const Problem&) = delete;
    Problem(Problem&&) = delete;
    Problem&
    operator=(Problem&&) = delete;

    svm_problem&
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
    svm_problem* m_prob = new svm_problem;
};

class Model
{
public:
    Model() = default;
    explicit Model(svm_model* model, Parameters params)
        : m_model{model}
        , m_params(std::move(params))
    {
    }

    ~Model()
    {
        destroy(m_model);
    }

    Model(const Model& other)
        : m_model{allocate<svm_model>(1, true)}
        , m_params{other.m_params}
    {
        copy(*other.m_model, *m_model);
    }

    Model&
    operator=(const Model& other)
    {
        if (this != &other)
        {
            destroy(m_model);
            m_model = allocate<svm_model>(1, true);
            copy(*other.m_model, *m_model);
            m_params = other.m_params;
        }
        return *this;
    }

    Model(Model&& other)
        : m_model{other.m_model}
        , m_params{std::move(other.m_params)}
    {
        other.m_model = nullptr;
    }

    Model&
    operator=(Model&& other)
    {
        if (this != &other)
        {
            m_model = other.m_model;
            other.m_model = nullptr;
            m_params = std::move(other.m_params);
        }
        return *this;
    }

    svm_model&
    get() const
    {
        return *m_model;
    }

    const Parameters&
    params() const
    {
        return m_params;
    }

    void
    save(std::ostream& os) const
    {
        SVMEGN_ASSERT(m_model != nullptr);
        write(os, serialize_version);
        write(os, m_params.svm_type);
        write(os, m_params.kernel_type);
        write(os, m_params.degree);
        write(os, m_params.gamma);
        write(os, m_params.coef0);
        write(os, m_params.cache_size);
        write(os, m_params.eps);
        write(os, m_params.C);
        write(os, m_params.nr_weight);
        write(os, m_params.weight_label);
        write(os, m_params.weight);
        write(os, m_params.nu);
        write(os, m_params.p);
        write(os, m_params.shrinking);
        write(os, m_params.probability);

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
                    int j = 0;
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
                    write_array(os, m_model->sv_coef[i], m_model->l);
                }
            }

            const bool have_rho = m_model->rho != nullptr;
            write(os, have_rho);
            if (have_rho)
            {
                const auto size =
                    m_model->nr_class * (m_model->nr_class - 1) / 2;
                write_array(os, m_model->rho, size);
            }

            const bool have_probA = m_model->probA != nullptr;
            write(os, have_probA);
            if (have_probA)
            {
                const auto size =
                    m_model->nr_class * (m_model->nr_class - 1) / 2;
                write_array(os, m_model->probA, size);
            }

            const bool have_probB = m_model->probB != nullptr;
            write(os, have_probB);
            if (have_probB)
            {
                const auto size =
                    m_model->nr_class * (m_model->nr_class - 1) / 2;
                write_array(os, m_model->probB, size);
            }

            const bool have_prob_density_marks =
                m_model->prob_density_marks != nullptr;
            write(os, have_prob_density_marks);
            if (have_prob_density_marks)
            {
                constexpr auto size = prob_density_mark_count;
                write_array(os, m_model->prob_density_marks, size);
            }

            const bool have_sv_indices = m_model->sv_indices != nullptr;
            write(os, have_sv_indices);
            if (have_sv_indices)
            {
                const auto size = m_model->l;
                write_array(os, m_model->sv_indices, size);
            }

            const bool have_label = m_model->label != nullptr;
            write(os, have_label);
            if (have_label)
            {
                const auto size = m_model->nr_class;
                write_array(os, m_model->label, size);
            }

            const bool have_nSV = m_model->nSV != nullptr;
            write(os, have_nSV);
            if (have_nSV)
            {
                const auto size = m_model->nr_class;
                write_array(os, m_model->nSV, size);
            }

            write(os, m_model->free_sv);
        }
    }

    void
    load(std::istream& is)
    {
        int version;
        read(is, version);
        read(is, m_params.svm_type);
        read(is, m_params.kernel_type);
        read(is, m_params.degree);
        read(is, m_params.gamma);
        read(is, m_params.coef0);
        read(is, m_params.cache_size);
        read(is, m_params.eps);
        read(is, m_params.C);
        read(is, m_params.nr_weight);
        read(is, m_params.weight_label);
        read(is, m_params.weight);
        read(is, m_params.nu);
        read(is, m_params.p);
        read(is, m_params.shrinking);
        read(is, m_params.probability);
        bool have_model;
        read(is, have_model);
        if (have_model)
        {
            destroy(m_model);
            m_model = allocate<svm_model>(1, true);
            m_model->param = convert(m_params);
            read(is, m_model->nr_class);
            read(is, m_model->l);

            bool have_SV;
            read(is, have_SV);
            if (have_SV)
            {
                m_model->SV = allocate<svm_node*>(m_model->l);
                for (int i = 0; i < m_model->l; ++i)
                {
                    int j;
                    read(is, j);
                    m_model->SV[i] = allocate<svm_node>(j);
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
                    read_array(is, m_model->sv_coef[i], m_model->l);
                }
            }

            bool have_rho;
            read(is, have_rho);
            if (have_rho)
            {
                const auto size =
                    m_model->nr_class * (m_model->nr_class - 1) / 2;
                m_model->rho = allocate<double>(size);
                read_array(is, m_model->rho, size);
            }

            bool have_probA;
            read(is, have_probA);
            if (have_probA)
            {
                const auto size =
                    m_model->nr_class * (m_model->nr_class - 1) / 2;
                m_model->probA = allocate<double>(size);
                read_array(is, m_model->probA, size);
            }

            bool have_probB;
            read(is, have_probB);
            if (have_probB)
            {
                const auto size =
                    m_model->nr_class * (m_model->nr_class - 1) / 2;
                m_model->probB = allocate<double>(size);
                read_array(is, m_model->probB, size);
            }

            bool have_prob_density_marks;
            read(is, have_prob_density_marks);
            if (have_prob_density_marks)
            {
                constexpr auto size = prob_density_mark_count;
                m_model->prob_density_marks = allocate<double>(size);
                read_array(is, m_model->prob_density_marks, size);
            }

            bool have_sv_indices;
            read(is, have_sv_indices);
            if (have_sv_indices)
            {
                const auto size = m_model->l;
                m_model->sv_indices = allocate<int>(size);
                read_array(is, m_model->sv_indices, size);
            }

            bool have_label;
            read(is, have_label);
            if (have_label)
            {
                const auto size = m_model->nr_class;
                m_model->label = allocate<int>(size);
                read_array(is, m_model->label, size);
            }

            bool have_nSV;
            read(is, have_nSV);
            if (have_nSV)
            {
                const auto size = m_model->nr_class;
                m_model->nSV = allocate<int>(size);
                read_array(is, m_model->nSV, size);
            }

            read(is, m_model->free_sv);
        }
    }

private:
    static void
    destroy(svm_model* model)
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
        svm_free_and_destroy_model(&model);
    }

    static void
    copy(const svm_model& from, svm_model& to)
    {
        to.param = from.param; // TODO make deep copy
        to.nr_class = from.nr_class;
        to.l = from.l;

        if (from.SV)
        {
            to.SV = allocate<svm_node*>(to.l);
            for (int i = 0; i < to.l; ++i)
            {
                int j = 0;
                while (from.SV[i][j++].index >= 0)
                    ;
                to.SV[i] = allocate<svm_node>(j);
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

    svm_model* m_model = nullptr;
    Parameters m_params;
};

} // namespace

struct SVM::Impl
{
    Impl(const Impl& other)
        : m_model{other.m_model}
    {
    }

    Impl(Parameters params, const Eigen::MatrixXd& X, const Eigen::MatrixXd& y)
    {
        const auto svm_params = convert(params);
        Problem prob{X, y};
        std::unique_ptr<const char[]> error{
            svm_check_parameter(&prob.get(), &svm_params)};
        SVMEGN_ASSERT(error == nullptr) << error.get();
        m_model = Model{svm_train(&prob.get(), &svm_params), std::move(params)};
        prob.set_sv_indices(m_model.get().sv_indices, m_model.get().l);
    }

    Impl(std::istream& is)
    {
        m_model.load(is);
    }

    double
    predict(const Eigen::RowVectorXd& row) const
    {
        auto record = make_record(row);
        return svm_predict(&m_model.get(), record.get());
    }

    void
    save(std::ostream& os) const
    {
        m_model.save(os);
    }

    Model m_model;
};

SVM::~SVM()
{
}

SVM::SVM(const SVM& other)
    : m_impl{std::make_unique<Impl>(*other.m_impl)}
{
}

SVM&
SVM::operator=(const SVM& other)
{
    if (this != &other)
    {
        m_impl.reset();
        m_impl = std::make_unique<Impl>(*other.m_impl);
    }
    return *this;
}

SVM::SVM(SVM&&) = default;

SVM&
SVM::operator=(SVM&&) = default;

SVM
SVM::train(Parameters params,
           const Eigen::MatrixXd& X,
           const Eigen::VectorXd& y)
{
    SVM svm;
    svm.m_impl = std::make_unique<Impl>(std::move(params), X, y);
    return svm;
}

const Parameters&
SVM::parameters() const
{
    return m_impl->m_model.params();
}

Eigen::VectorXd
SVM::predict(const Eigen::MatrixXd& X) const
{
    Eigen::VectorXd y{X.rows()};
    for (int i = 0; i < X.rows(); ++i)
    {
        y(i) = m_impl->predict(X.row(i));
    }
    return y;
}

void
SVM::save(std::ostream& os) const
{
    m_impl->save(os);
}

SVM
SVM::load(std::istream& is)
{
    SVM svm;
    svm.m_impl = std::make_unique<Impl>(is);
    return svm;
}

} // namespace svmegn
