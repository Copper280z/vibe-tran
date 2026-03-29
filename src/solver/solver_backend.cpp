// src/solver/solver_backend.cpp
#include "solver/solver_backend.hpp"
#include "core/factor_ratio_check.hpp"
#include "core/types.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <chrono>
#include <cstring>
#include <cstdlib>
#ifdef HAVE_ACCELERATE
#include <Eigen/AccelerateSupport>
#endif
#ifdef EIGEN_CHOLMOD_SUPPORT
#include <Eigen/CholmodSupport>
#endif
#include <format>
#include <spdlog/spdlog.h>

namespace vibestran {

namespace {

#if !defined(HAVE_ACCELERATE)
std::vector<double>
extract_csr_diagonal(const SparseMatrixBuilder::CsrData &csr) {
  std::vector<double> diag(static_cast<std::size_t>(csr.n), 0.0);
  for (int row = 0; row < csr.n; ++row) {
    for (int j = csr.row_ptr[row]; j < csr.row_ptr[row + 1]; ++j) {
      if (csr.col_ind[j] == row) {
        diag[static_cast<std::size_t>(row)] = csr.values[j];
        break;
      }
    }
  }
  return diag;
}

template <typename DiagonalExpr>
std::vector<double> copy_diagonal(const DiagonalExpr &diag_expr) {
  Eigen::VectorXd dense = diag_expr;
  return std::vector<double>(dense.data(), dense.data() + dense.size());
}

[[maybe_unused]] void
report_factor_ratio_result(const FactorRatioCheckPolicy &policy,
                           const FactorRatioCheckResult &result,
                           std::string_view backend_label) {
  if (result.status == FactorRatioCheckStatus::Ok)
    return;

  const std::string matrix_name =
      policy.matrix_name.empty() ? "matrix" : policy.matrix_name;
  if (result.status == FactorRatioCheckStatus::NonPositiveFactorDiagonal) {
    throw SolverError(std::format(
        "{} {} has non-positive or near-zero factor diagonal at row {} "
        "(matrix diag {:.3e}, factor diag {:.3e}, backend {})",
        policy.context, matrix_name, result.row_1based, result.matrix_diag,
        result.factor_diag, backend_label));
  }

  const std::string message = std::format(
      "{} {} diagonal / factor diagonal ratio {:.3e} exceeds "
      "MAXRATIO={:.3e} at row {} (matrix diag {:.3e}, factor diag {:.3e}, "
      "backend {})",
      policy.context, matrix_name, result.max_ratio, policy.maxratio,
      result.row_1based, result.matrix_diag, result.factor_diag, backend_label);
  if (policy.fatal)
    throw SolverError(message);
  spdlog::warn("{}", message);
}
#endif

} // namespace

#if defined(HAVE_ACCELERATE)
namespace {

struct AccelerateOrderConfig {
  SparseOrder_t order;
  const char *label;
};

AccelerateOrderConfig accelerate_order_config() {
  const char *env = std::getenv("VIBESTRAN_ACCELERATE_ORDER");
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

} // namespace
#endif

std::vector<double>
EigenSolverBackend::solve(const SparseMatrixBuilder::CsrData &K_csr,
                          const std::vector<double> &F,
                          const FactorRatioCheckPolicy *factor_ratio_policy) {
  using Clock = std::chrono::steady_clock;
  using Ms = std::chrono::duration<double, std::milli>;

  const int n = K_csr.n;
  if (n == 0)
    throw SolverError("Stiffness matrix is empty — no free DOFs");
  if (static_cast<int>(F.size()) != n)
    throw SolverError(
        std::format("Force vector size {} != matrix size {}", F.size(), n));

  // Map CsrData arrays directly into an Eigen RowMajor sparse matrix.
  // No copies: Eigen::Map references the existing vectors in-place.
  using ESMR = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  Eigen::Map<const ESMR> K_row(
      n, n, K_csr.nnz,
      K_csr.row_ptr.data(),
      K_csr.col_ind.data(),
      K_csr.values.data());

  const bool lower_only = K_csr.stores_lower_triangle_only();

  // Convert to a compressed column-major matrix, then factorize only the lower
  // triangle. This keeps the direct path aligned with symmetric storage even
  // when a caller passed a full mirrored matrix.
  using ESM = Eigen::SparseMatrix<double>;
  const auto t_convert_begin = Clock::now();
  ESM K_input(K_row);
  K_input.makeCompressed();
  const auto t_convert_end = Clock::now();
  const int input_nnz = static_cast<int>(K_input.nonZeros());

  const auto t_lower_begin = Clock::now();
  ESM K;
  if (lower_only) {
    K = std::move(K_input);
  } else {
    K = ESM(K_input.template triangularView<Eigen::Lower>());
    K.makeCompressed();
  }
  const auto t_lower_end = Clock::now();
#if !defined(HAVE_ACCELERATE)
  const std::vector<double> matrix_diag = extract_csr_diagonal(K_csr);
#endif

  // Map force vector
  Eigen::Map<const Eigen::VectorXd> F_eigen(F.data(), n);

  // Factorize and solve.
  // Priority: Apple Accelerate (macOS) > SuiteSparse CHOLMOD (Linux) > SimplicialLLT.
  // Each tier falls back to LDLT for near-singular / mildly indefinite systems.
#if defined(HAVE_ACCELERATE)
  const auto order_cfg = accelerate_order_config();
  const char *backend_label = "accelerate-llt";
  Eigen::AccelerateLLT<ESM> solver;
  solver.setOrder(order_cfg.order);
  const auto t_compute_begin = Clock::now();
  solver.compute(K);
  const auto t_compute_end = Clock::now();
  if (solver.info() != Eigen::Success) {
    backend_label = "accelerate-ldlt";
    Eigen::AccelerateLDLT<ESM> ldlt;
    ldlt.setOrder(order_cfg.order);
    const auto t_fallback_compute_begin = Clock::now();
    ldlt.compute(K);
    const auto t_fallback_compute_end = Clock::now();
    if (ldlt.info() != Eigen::Success)
      throw SolverError(
          "Stiffness matrix factorization failed (Accelerate + LDLT fallback) — "
          "check boundary conditions (SPCs)");
    if (factor_ratio_policy != nullptr) {
      spdlog::debug(
          "[cpu-direct] skipping MAXRATIO check for {} because factor diagonal "
          "is unavailable",
          backend_label);
    }
    const auto t_fallback_solve_begin = Clock::now();
    Eigen::VectorXd u = ldlt.solve(F_eigen);
    const auto t_fallback_solve_end = Clock::now();
    if (ldlt.info() != Eigen::Success)
      throw SolverError("Back-substitution failed (Accelerate LDLT fallback)");
    spdlog::debug(
        "[cpu-direct] backend={}, order={}, storage={}, n={}, nnz(input)={}, nnz(factor)={}, "
        "convert CSR->Eigen: {:.3f} ms, build lower: {:.3f} ms, failed LLT attempt: {:.3f} ms, "
        "fallback factorization: {:.3f} ms, fallback solve: {:.3f} ms",
        backend_label, order_cfg.label, lower_only ? "lower" : "full", n, input_nnz, K.nonZeros(),
        Ms(t_convert_end - t_convert_begin).count(),
        Ms(t_lower_end - t_lower_begin).count(),
        Ms(t_compute_end - t_compute_begin).count(),
        Ms(t_fallback_compute_end - t_fallback_compute_begin).count(),
        Ms(t_fallback_solve_end - t_fallback_solve_begin).count());
    return std::vector<double>(u.data(), u.data() + n);
  }
  if (factor_ratio_policy != nullptr) {
    spdlog::debug(
        "[cpu-direct] skipping MAXRATIO check for {} because factor diagonal "
        "is unavailable",
        backend_label);
  }
  const auto t_solve_begin = Clock::now();
  Eigen::VectorXd u = solver.solve(F_eigen);
  const auto t_solve_end = Clock::now();
  if (solver.info() != Eigen::Success)
    throw SolverError("Back-substitution failed (Accelerate)");
  spdlog::debug(
      "[cpu-direct] backend={}, order={}, storage={}, n={}, nnz(input)={}, nnz(factor)={}, "
      "convert CSR->Eigen: {:.3f} ms, build lower: {:.3f} ms, factorization: {:.3f} ms, solve: {:.3f} ms",
      backend_label, order_cfg.label, lower_only ? "lower" : "full", n, input_nnz, K.nonZeros(),
      Ms(t_convert_end - t_convert_begin).count(),
      Ms(t_lower_end - t_lower_begin).count(),
      Ms(t_compute_end - t_compute_begin).count(),
      Ms(t_solve_end - t_solve_begin).count());
#elif defined(EIGEN_CHOLMOD_SUPPORT)
  const char *backend_label = "cholmod";
  Eigen::CholmodDecomposition<ESM> solver;
  const auto t_compute_begin = Clock::now();
  solver.compute(K);
  const auto t_compute_end = Clock::now();
  if (solver.info() != Eigen::Success) {
    // Cholesky failed (matrix not numerically PD) — fall back to LDLT which
    // handles near-singular and mildly indefinite systems.
    backend_label = "simplicial-ldlt";
    Eigen::SimplicialLDLT<ESM> ldlt;
    const auto t_fallback_compute_begin = Clock::now();
    ldlt.compute(K);
    const auto t_fallback_compute_end = Clock::now();
    if (ldlt.info() != Eigen::Success)
      throw SolverError(
          "Stiffness matrix factorization failed (CHOLMOD + LDLT fallback) — "
          "check boundary conditions (SPCs)");
    if (factor_ratio_policy != nullptr) {
      report_factor_ratio_result(
          *factor_ratio_policy,
          evaluate_factor_ratio_check(matrix_diag, copy_diagonal(ldlt.vectorD()),
                                      factor_ratio_policy->maxratio),
          backend_label);
    }
    const auto t_fallback_solve_begin = Clock::now();
    Eigen::VectorXd u = ldlt.solve(F_eigen);
    const auto t_fallback_solve_end = Clock::now();
    if (ldlt.info() != Eigen::Success)
      throw SolverError("Back-substitution failed (LDLT fallback)");
    spdlog::debug(
        "[cpu-direct] backend={}, storage={}, n={}, nnz(input)={}, nnz(factor)={}, "
        "convert CSR->Eigen: {:.3f} ms, build lower: {:.3f} ms, failed CHOLMOD attempt: {:.3f} ms, "
        "fallback factorization: {:.3f} ms, fallback solve: {:.3f} ms",
        backend_label, lower_only ? "lower" : "full", n, input_nnz, K.nonZeros(),
        Ms(t_convert_end - t_convert_begin).count(),
        Ms(t_lower_end - t_lower_begin).count(),
        Ms(t_compute_end - t_compute_begin).count(),
        Ms(t_fallback_compute_end - t_fallback_compute_begin).count(),
        Ms(t_fallback_solve_end - t_fallback_solve_begin).count());
    return std::vector<double>(u.data(), u.data() + n);
  }
  if (factor_ratio_policy != nullptr) {
    spdlog::debug(
        "[cpu-direct] skipping MAXRATIO check for {} because factor diagonal "
        "is unavailable",
        backend_label);
  }
  const auto t_solve_begin = Clock::now();
  Eigen::VectorXd u = solver.solve(F_eigen);
  const auto t_solve_end = Clock::now();
  if (solver.info() != Eigen::Success)
    throw SolverError("Back-substitution failed (CHOLMOD)");
  spdlog::debug(
      "[cpu-direct] backend={}, storage={}, n={}, nnz(input)={}, nnz(factor)={}, "
      "convert CSR->Eigen: {:.3f} ms, build lower: {:.3f} ms, factorization: {:.3f} ms, solve: {:.3f} ms",
      backend_label, lower_only ? "lower" : "full", n, input_nnz, K.nonZeros(),
      Ms(t_convert_end - t_convert_begin).count(),
      Ms(t_lower_end - t_lower_begin).count(),
      Ms(t_compute_end - t_compute_begin).count(),
      Ms(t_solve_end - t_solve_begin).count());
#else
  const char *backend_label = "simplicial-llt";
  Eigen::SimplicialLLT<ESM> solver;
  const auto t_compute_begin = Clock::now();
  solver.compute(K);
  const auto t_compute_end = Clock::now();
  if (solver.info() != Eigen::Success) {
    // Fall back to LDLT which is more robust for near-singular systems
    backend_label = "simplicial-ldlt";
    Eigen::SimplicialLDLT<ESM> ldlt;
    const auto t_fallback_compute_begin = Clock::now();
    ldlt.compute(K);
    const auto t_fallback_compute_end = Clock::now();
    if (ldlt.info() != Eigen::Success)
      throw SolverError(
          "Stiffness matrix factorization failed — check boundary conditions");
    if (factor_ratio_policy != nullptr) {
      report_factor_ratio_result(
          *factor_ratio_policy,
          evaluate_factor_ratio_check(matrix_diag, copy_diagonal(ldlt.vectorD()),
                                      factor_ratio_policy->maxratio),
          backend_label);
    }
    const auto t_fallback_solve_begin = Clock::now();
    Eigen::VectorXd u = ldlt.solve(F_eigen);
    const auto t_fallback_solve_end = Clock::now();
    if (ldlt.info() != Eigen::Success)
      throw SolverError("Back-substitution failed");
    spdlog::debug(
        "[cpu-direct] backend={}, storage={}, n={}, nnz(input)={}, nnz(factor)={}, "
        "convert CSR->Eigen: {:.3f} ms, build lower: {:.3f} ms, failed LLT attempt: {:.3f} ms, "
        "fallback factorization: {:.3f} ms, fallback solve: {:.3f} ms",
        backend_label, lower_only ? "lower" : "full", n, input_nnz, K.nonZeros(),
        Ms(t_convert_end - t_convert_begin).count(),
        Ms(t_lower_end - t_lower_begin).count(),
        Ms(t_compute_end - t_compute_begin).count(),
        Ms(t_fallback_compute_end - t_fallback_compute_begin).count(),
        Ms(t_fallback_solve_end - t_fallback_solve_begin).count());
    return std::vector<double>(u.data(), u.data() + n);
  }
  if (factor_ratio_policy != nullptr) {
    report_factor_ratio_result(
        *factor_ratio_policy,
        evaluate_factor_ratio_check(matrix_diag,
                                    copy_diagonal(solver.matrixL().diagonal()),
                                    factor_ratio_policy->maxratio),
        backend_label);
  }
  const auto t_solve_begin = Clock::now();
  Eigen::VectorXd u = solver.solve(F_eigen);
  const auto t_solve_end = Clock::now();
  if (solver.info() != Eigen::Success)
    throw SolverError("Back-substitution failed");
  spdlog::debug(
      "[cpu-direct] backend={}, storage={}, n={}, nnz(input)={}, nnz(factor)={}, "
      "convert CSR->Eigen: {:.3f} ms, build lower: {:.3f} ms, factorization: {:.3f} ms, solve: {:.3f} ms",
      backend_label, lower_only ? "lower" : "full", n, input_nnz, K.nonZeros(),
      Ms(t_convert_end - t_convert_begin).count(),
      Ms(t_lower_end - t_lower_begin).count(),
      Ms(t_compute_end - t_compute_begin).count(),
      Ms(t_solve_end - t_solve_begin).count());
#endif

  return std::vector<double>(u.data(), u.data() + n);
}


std::vector<double>
EigenPCGSolverBackend::solve(const SparseMatrixBuilder::CsrData& K_csr,
                              const std::vector<double>& F,
                              const FactorRatioCheckPolicy*) {
  using Clock = std::chrono::steady_clock;
  using Ms = std::chrono::duration<double, std::milli>;

  const int n = K_csr.n;
  if (n == 0)
    throw SolverError("Stiffness matrix is empty — no free DOFs");
  if (static_cast<int>(F.size()) != n)
    throw SolverError(
        std::format("Force vector size {} != matrix size {}", F.size(), n));

  const bool lower_only = K_csr.stores_lower_triangle_only();

  using ESMR = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  using ESM = Eigen::SparseMatrix<double>;
  Eigen::Map<const ESMR> K_row(
      n, n, K_csr.nnz,
      K_csr.row_ptr.data(),
      K_csr.col_ind.data(),
      K_csr.values.data());

  const auto t_convert_begin = Clock::now();
  ESM K(K_row);
  K.makeCompressed();
  const auto t_convert_end = Clock::now();

  Eigen::Map<const Eigen::VectorXd> F_eigen(F.data(), n);

  using Precond = Eigen::IncompleteCholesky<double, Eigen::Lower>;

  auto configure = [&](auto &cg) {
    if (max_iters_ > 0)
      cg.setMaxIterations(max_iters_);
    cg.setTolerance(tolerance_);
  };

  if (lower_only) {
    Eigen::ConjugateGradient<ESM, Eigen::Lower, Precond> cg;
    configure(cg);

    const auto t_compute_begin = Clock::now();
    cg.compute(K);
    const auto t_compute_end = Clock::now();
    if (cg.info() != Eigen::Success)
      throw SolverError(
          "PCG preconditioner (IncompleteCholesky) setup failed — "
          "matrix may not be positive definite. Check boundary conditions.");

    const auto t_solve_begin = Clock::now();
    Eigen::VectorXd u = cg.solve(F_eigen);
    const auto t_solve_end = Clock::now();

    last_iters_ = static_cast<int>(cg.iterations());
    last_error_ = cg.error();

    spdlog::debug(
        "[cpu-pcg] storage=lower, n={}, nnz={}, Eigen threads={}, convert CSR->Eigen: {:.3f} ms, "
        "preconditioner: {:.3f} ms, iterative solve: {:.3f} ms, iterations: {}, estimated error: {:.3e}",
        n, K.nonZeros(), Eigen::nbThreads(),
        Ms(t_convert_end - t_convert_begin).count(),
        Ms(t_compute_end - t_compute_begin).count(),
        Ms(t_solve_end - t_solve_begin).count(),
        last_iters_, last_error_);

    if (cg.info() == Eigen::NoConvergence)
      throw SolverError(
          std::format("PCG did not converge after {} iterations "
                      "(estimated error {:.3e} > tolerance {:.3e}). "
                      "Consider increasing max iterations or checking "
                      "boundary conditions.",
                      last_iters_, last_error_, tolerance_));

    return std::vector<double>(u.data(), u.data() + n);
  }

  Eigen::ConjugateGradient<ESM, Eigen::Lower | Eigen::Upper, Precond> cg;
  configure(cg);

  const auto t_compute_begin = Clock::now();
  cg.compute(K);
  const auto t_compute_end = Clock::now();
  if (cg.info() != Eigen::Success)
    throw SolverError(
        "PCG preconditioner (IncompleteCholesky) setup failed — "
        "matrix may not be positive definite. Check boundary conditions.");

  const auto t_solve_begin = Clock::now();
  Eigen::VectorXd u = cg.solve(F_eigen);
  const auto t_solve_end = Clock::now();

  last_iters_ = static_cast<int>(cg.iterations());
  last_error_ = cg.error();

  spdlog::debug(
      "[cpu-pcg] storage=full, n={}, nnz={}, Eigen threads={}, convert CSR->Eigen: {:.3f} ms, "
      "preconditioner: {:.3f} ms, iterative solve: {:.3f} ms, iterations: {}, estimated error: {:.3e}",
      n, K.nonZeros(), Eigen::nbThreads(),
      Ms(t_convert_end - t_convert_begin).count(),
      Ms(t_compute_end - t_compute_begin).count(),
      Ms(t_solve_end - t_solve_begin).count(),
      last_iters_, last_error_);

  if (cg.info() == Eigen::NoConvergence)
    throw SolverError(
        std::format("PCG did not converge after {} iterations "
                    "(estimated error {:.3e} > tolerance {:.3e}). "
                    "Consider increasing max iterations or checking "
                    "boundary conditions.",
                    last_iters_, last_error_, tolerance_));

  return std::vector<double>(u.data(), u.data() + n);
}

} // namespace vibestran
