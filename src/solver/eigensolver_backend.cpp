// src/solver/eigensolver_backend.cpp
// CPU eigensolver using Spectra shift-and-invert Lanczos.
//
// Solves: K φ = λ M φ
// Transform: (K - σM)^{-1} M φ = ν φ,  ν = 1/(λ - σ)
// Spectra finds ν → returns original λ = σ + 1/ν.
//
// compute() takes two SortRule arguments:
//   selection: used during Lanczos restarts; must match Lanczos convergence.
//              Lanczos naturally converges to LARGEST |ν| (modes closest to σ),
//              so use LargestMagn.
//   sorting:   applied to the final λ values (after ν→λ conversion);
//              SmallestAlge gives ascending eigenvalue order (lowest freq first).

#include "solver/eigensolver_backend.hpp"
#include "core/exceptions.hpp"

#include <Spectra/SymGEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <Eigen/SparseCholesky>
#include <format>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string_view>

#ifdef HAVE_ACCELERATE
#include <Eigen/AccelerateSupport>
#endif
#ifdef EIGEN_CHOLMOD_SUPPORT
#include <Eigen/CholmodSupport>
#endif

namespace vibestran {

namespace {

#if defined(HAVE_ACCELERATE)

struct AccelerateOrderConfig {
    SparseOrder_t order;
    const char* label;
};

AccelerateOrderConfig accelerate_order_config() {
    const char* env = std::getenv("VIBESTRAN_ACCELERATE_ORDER");
    if (env == nullptr || env[0] == '\0')
        return {SparseOrderMetis, "metis"};
    if (std::strcmp(env, "default") == 0)
        return {SparseOrderDefault, "default"};
    if (std::strcmp(env, "amd") == 0)
        return {SparseOrderAMD, "amd"};
    if (std::strcmp(env, "metis") == 0)
        return {SparseOrderMetis, "metis"};

    spdlog::warn("Ignoring invalid VIBESTRAN_ACCELERATE_ORDER='{}' "
                 "(valid: default, amd, metis)",
                 env);
    return {SparseOrderMetis, "metis"};
}

#endif

template <typename Solver>
void apply_solve(const Solver& solver, std::string_view backend_label,
                 const double* x_in, double* y_out, Eigen::Index n) {
    using Vector = Eigen::VectorXd;
    using MapConstVec = Eigen::Map<const Vector>;
    using MapVec = Eigen::Map<Vector>;

    MapConstVec x(x_in, n);
    MapVec y(y_out, n);
    y.noalias() = solver.solve(x);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(std::format(
            "shift-invert back-substitution failed using {}", backend_label));
    }
}

class SpectraDirectShiftInvert {
public:
    using Scalar = double;
    using SparseMatrix = Eigen::SparseMatrix<Scalar>;
    using Index = Eigen::Index;

    SpectraDirectShiftInvert(const SparseMatrix& K, const SparseMatrix& M)
        : K_(K), M_(M), n_(K.rows()) {
        if (n_ != K.cols() || n_ != M.rows() || n_ != M.cols()) {
            throw std::invalid_argument(
                "SpectraDirectShiftInvert: K and M must be square matrices of the same size");
        }
    }

    [[nodiscard]] Index rows() const { return n_; }
    [[nodiscard]] Index cols() const { return n_; }
    [[nodiscard]] std::string_view backend_label() const noexcept {
        return backend_label_;
    }

    void set_shift(const Scalar& sigma) {
        shift_ = SparseMatrix(K_.template triangularView<Eigen::Lower>());
        SparseMatrix M_lower(M_.template triangularView<Eigen::Lower>());
        shift_ -= sigma * M_lower;
        shift_.makeCompressed();

#if defined(HAVE_ACCELERATE)
        const auto order_cfg = accelerate_order_config();

        accelerate_llt_.setOrder(order_cfg.order);
        accelerate_llt_.compute(shift_);
        if (accelerate_llt_.info() == Eigen::Success) {
            active_backend_ = ActiveBackend::AccelerateLLT;
            backend_label_ = std::format("accelerate-llt(order={})", order_cfg.label);
            return;
        }

        accelerate_ldlt_.setOrder(order_cfg.order);
        accelerate_ldlt_.compute(shift_);
        if (accelerate_ldlt_.info() == Eigen::Success) {
            active_backend_ = ActiveBackend::AccelerateLDLT;
            backend_label_ = std::format("accelerate-ldlt(order={})", order_cfg.label);
            return;
        }

        active_backend_ = ActiveBackend::None;
        backend_label_ = std::format("accelerate(order={})", order_cfg.label);
        throw std::runtime_error(std::format(
            "Accelerate factorization failed for K - sigma*M with sigma={:.16g}",
            sigma));
#elif defined(EIGEN_CHOLMOD_SUPPORT)
        cholmod_.compute(shift_);
        if (cholmod_.info() == Eigen::Success) {
            active_backend_ = ActiveBackend::Cholmod;
            backend_label_ = "cholmod";
            return;
        }

        simplicial_ldlt_.compute(shift_);
        if (simplicial_ldlt_.info() == Eigen::Success) {
            active_backend_ = ActiveBackend::SimplicialLDLT;
            backend_label_ = "simplicial-ldlt";
            return;
        }

        active_backend_ = ActiveBackend::None;
        backend_label_ = "cholmod+ldlt";
        throw std::runtime_error(std::format(
            "CHOLMOD factorization failed for K - sigma*M with sigma={:.16g}; "
            "SimplicialLDLT fallback also failed",
            sigma));
#else
        simplicial_llt_.compute(shift_);
        if (simplicial_llt_.info() == Eigen::Success) {
            active_backend_ = ActiveBackend::SimplicialLLT;
            backend_label_ = "simplicial-llt";
            return;
        }

        simplicial_ldlt_.compute(shift_);
        if (simplicial_ldlt_.info() == Eigen::Success) {
            active_backend_ = ActiveBackend::SimplicialLDLT;
            backend_label_ = "simplicial-ldlt";
            return;
        }

        active_backend_ = ActiveBackend::None;
        backend_label_ = "eigen-llt+ldlt";
        throw std::runtime_error(std::format(
            "Eigen sparse factorization failed for K - sigma*M with sigma={:.16g}; "
            "both SimplicialLLT and SimplicialLDLT failed",
            sigma));
#endif
    }

    void perform_op(const Scalar* x_in, Scalar* y_out) const {
        switch (active_backend_) {
#if defined(HAVE_ACCELERATE)
        case ActiveBackend::AccelerateLLT:
            apply_solve(accelerate_llt_, backend_label_, x_in, y_out, n_);
            return;
        case ActiveBackend::AccelerateLDLT:
            apply_solve(accelerate_ldlt_, backend_label_, x_in, y_out, n_);
            return;
#endif
#if defined(EIGEN_CHOLMOD_SUPPORT)
        case ActiveBackend::Cholmod:
            apply_solve(cholmod_, backend_label_, x_in, y_out, n_);
            return;
#endif
        case ActiveBackend::SimplicialLLT:
            apply_solve(simplicial_llt_, backend_label_, x_in, y_out, n_);
            return;
        case ActiveBackend::SimplicialLDLT:
            apply_solve(simplicial_ldlt_, backend_label_, x_in, y_out, n_);
            return;
        case ActiveBackend::None:
            break;
        }

        throw std::logic_error(
            "SpectraDirectShiftInvert::perform_op called before successful factorization");
    }

private:
    enum class ActiveBackend {
        None,
#if defined(HAVE_ACCELERATE)
        AccelerateLLT,
        AccelerateLDLT,
#endif
#if defined(EIGEN_CHOLMOD_SUPPORT)
        Cholmod,
#endif
        SimplicialLLT,
        SimplicialLDLT,
    };

    const SparseMatrix& K_;
    const SparseMatrix& M_;
    const Index n_;
    SparseMatrix shift_;
    ActiveBackend active_backend_{ActiveBackend::None};
    std::string backend_label_{"uninitialized"};

#if defined(HAVE_ACCELERATE)
    Eigen::AccelerateLLT<SparseMatrix> accelerate_llt_;
    Eigen::AccelerateLDLT<SparseMatrix> accelerate_ldlt_;
#endif
#if defined(EIGEN_CHOLMOD_SUPPORT)
    Eigen::CholmodDecomposition<SparseMatrix> cholmod_;
#endif
    Eigen::SimplicialLLT<SparseMatrix> simplicial_llt_;
    Eigen::SimplicialLDLT<SparseMatrix> simplicial_ldlt_;
};

} // namespace

std::vector<EigenPair> SpectraEigensolverBackend::solve(
    const Eigen::SparseMatrix<double>& K,
    const Eigen::SparseMatrix<double>& M,
    int nd, double sigma)
{
    const int n = static_cast<int>(K.rows());
    if (n < 1)
        throw SolverError("Eigensolver: system has no free DOFs");
    if (nd < 1)
        throw SolverError("Eigensolver: nd must be >= 1");
    if (nd >= n)
        nd = n - 1; // Spectra requires nd < n

    // Lanczos basis size: must satisfy nd < ncv <= n
    int ncv = std::min(n, std::max(2 * nd + 10, 3 * nd));

    // The Lanczos iteration itself is mostly serial, but the shift-invert
    // factorization and triangular solves dominate runtime for large models.
    // Use the same sparse direct backend priority as the static CPU solver:
    // Apple Accelerate on macOS, CHOLMOD when available, otherwise Eigen's
    // simplicial factorizations. This avoids Spectra's default SparseLU path.
    using OpType  = SpectraDirectShiftInvert;
    using BOpType = Spectra::SparseSymMatProd<double>;

    OpType op(K, M);
    BOpType Bop(M);
    std::optional<Spectra::SymGEigsShiftSolver<
        OpType, BOpType, Spectra::GEigsMode::ShiftInvert>> solver;
    try {
        solver.emplace(op, Bop, nd, ncv, sigma);
    } catch (const std::exception& e) {
        throw SolverError(std::format(
            "Spectra eigensolver setup failed for sigma={:.16g}: {}",
            sigma, e.what()));
    }

    spdlog::info("Spectra shift-invert factorization backend: {}",
                 op.backend_label());

    solver->init();
    // LargestMagn selection: keep Ritz pairs with largest |ν| during restarts,
    // matching the natural convergence of the shift-inverted Lanczos.
    // SmallestAlge sorting: after ν→λ conversion, sort final results ascending by λ.
    int nconv = solver->compute(Spectra::SortRule::LargestMagn, 1000, 1e-10,
                                Spectra::SortRule::SmallestAlge);

    if (solver->info() != Spectra::CompInfo::Successful || nconv < 1)
        throw SolverError(std::format(
            "Spectra eigensolver: converged only {}/{} eigenvalues (info={})",
            nconv, nd, static_cast<int>(solver->info())));

    Eigen::VectorXd eigenvalues  = solver->eigenvalues();
    Eigen::MatrixXd eigenvectors = solver->eigenvectors();

    std::vector<EigenPair> results;
    results.reserve(nconv);
    for (int i = 0; i < nconv; ++i) {
        EigenPair ep;
        ep.eigenvalue  = eigenvalues(i);
        ep.eigenvector = eigenvectors.col(i);
        results.push_back(std::move(ep));
    }
    return results;
}

} // namespace vibestran
