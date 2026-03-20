// src/solver/solver_backend.cpp
#include "solver/solver_backend.hpp"
#include "core/types.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#ifdef HAVE_ACCELERATE
#include <Eigen/AccelerateSupport>
#endif
#ifdef EIGEN_CHOLMOD_SUPPORT
#include <Eigen/CholmodSupport>
#endif
#include <format>

namespace nastran {

std::vector<double>
EigenSolverBackend::solve(const SparseMatrixBuilder::CsrData &K_csr,
                          const std::vector<double> &F) {
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

  // SimplicialLLT/LDLT expect ColMajor; convert with a single structural copy.
  using ESM = Eigen::SparseMatrix<double>;
  ESM K(K_row);

  // Map force vector
  Eigen::Map<const Eigen::VectorXd> F_eigen(F.data(), n);

  // Factorize and solve.
  // Priority: Apple Accelerate (macOS) > SuiteSparse CHOLMOD (Linux) > SimplicialLLT.
  // Each tier falls back to LDLT for near-singular / mildly indefinite systems.
#if defined(HAVE_ACCELERATE)
  Eigen::AccelerateLLT<ESM> solver;
  solver.compute(K);
  if (solver.info() != Eigen::Success) {
    Eigen::AccelerateLDLT<ESM> ldlt;
    ldlt.compute(K);
    if (ldlt.info() != Eigen::Success)
      throw SolverError(
          "Stiffness matrix factorization failed (Accelerate + LDLT fallback) — "
          "check boundary conditions (SPCs)");
    Eigen::VectorXd u = ldlt.solve(F_eigen);
    if (ldlt.info() != Eigen::Success)
      throw SolverError("Back-substitution failed (Accelerate LDLT fallback)");
    return std::vector<double>(u.data(), u.data() + n);
  }
  Eigen::VectorXd u = solver.solve(F_eigen);
  if (solver.info() != Eigen::Success)
    throw SolverError("Back-substitution failed (Accelerate)");
#elif defined(EIGEN_CHOLMOD_SUPPORT)
  Eigen::CholmodDecomposition<ESM> solver;
  solver.compute(K);
  if (solver.info() != Eigen::Success) {
    // Cholesky failed (matrix not numerically PD) — fall back to LDLT which
    // handles near-singular and mildly indefinite systems.
    Eigen::SimplicialLDLT<ESM> ldlt;
    ldlt.compute(K);
    if (ldlt.info() != Eigen::Success)
      throw SolverError(
          "Stiffness matrix factorization failed (CHOLMOD + LDLT fallback) — "
          "check boundary conditions (SPCs)");
    Eigen::VectorXd u = ldlt.solve(F_eigen);
    if (ldlt.info() != Eigen::Success)
      throw SolverError("Back-substitution failed (LDLT fallback)");
    return std::vector<double>(u.data(), u.data() + n);
  }
  Eigen::VectorXd u = solver.solve(F_eigen);
  if (solver.info() != Eigen::Success)
    throw SolverError("Back-substitution failed (CHOLMOD)");
#else
  Eigen::SimplicialLLT<ESM> solver;
  solver.compute(K);
  if (solver.info() != Eigen::Success) {
    // Fall back to LDLT which is more robust for near-singular systems
    Eigen::SimplicialLDLT<ESM> ldlt;
    ldlt.compute(K);
    if (ldlt.info() != Eigen::Success)
      throw SolverError(
          "Stiffness matrix factorization failed — check boundary conditions");
    Eigen::VectorXd u = ldlt.solve(F_eigen);
    if (ldlt.info() != Eigen::Success)
      throw SolverError("Back-substitution failed");
    return std::vector<double>(u.data(), u.data() + n);
  }
  Eigen::VectorXd u = solver.solve(F_eigen);
  if (solver.info() != Eigen::Success)
    throw SolverError("Back-substitution failed");
#endif

  return std::vector<double>(u.data(), u.data() + n);
}


std::vector<double>
EigenPCGSolverBackend::solve(const SparseMatrixBuilder::CsrData& K_csr,
                              const std::vector<double>& F) {
    const int n = K_csr.n;
    if (n == 0)
        throw SolverError("Stiffness matrix is empty — no free DOFs");
    if (static_cast<int>(F.size()) != n)
        throw SolverError(
            std::format("Force vector size {} != matrix size {}", F.size(), n));

    // Map CsrData into Eigen ColMajor sparse matrix (required by ConjugateGradient).
    using ESMR = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using ESM  = Eigen::SparseMatrix<double>;
    Eigen::Map<const ESMR> K_row(n, n, K_csr.nnz,
                                  K_csr.row_ptr.data(),
                                  K_csr.col_ind.data(),
                                  K_csr.values.data());
    ESM K(K_row);

    Eigen::Map<const Eigen::VectorXd> F_eigen(F.data(), n);

    // ConjugateGradient with Incomplete Cholesky (zero fill-in) preconditioner.
    // IncompleteCholesky typically reduces iteration counts by 5-10× vs Jacobi
    // for FEM stiffness matrices, at the cost of O(nnz) factorization memory
    // (same asymptotic order as the matrix itself — no fill-in by default).
    using Precond = Eigen::IncompleteCholesky<double, Eigen::Lower | Eigen::Upper>;
    Eigen::ConjugateGradient<ESM, Eigen::Lower | Eigen::Upper, Precond> cg;

    if (max_iters_ > 0)
        cg.setMaxIterations(max_iters_);
    cg.setTolerance(tolerance_);

    cg.compute(K);
    if (cg.info() != Eigen::Success)
        throw SolverError(
            "PCG preconditioner (IncompleteCholesky) setup failed — "
            "matrix may not be positive definite. Check boundary conditions.");

    Eigen::VectorXd u = cg.solve(F_eigen);

    last_iters_ = static_cast<int>(cg.iterations());
    last_error_ = cg.error();

    if (cg.info() == Eigen::NoConvergence)
        throw SolverError(
            std::format("PCG did not converge after {} iterations "
                        "(estimated error {:.3e} > tolerance {:.3e}). "
                        "Consider increasing max iterations or checking "
                        "boundary conditions.",
                        last_iters_, last_error_, tolerance_));

    return std::vector<double>(u.data(), u.data() + n);
}

} // namespace nastran
