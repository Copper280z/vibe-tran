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
#include <Spectra/MatOp/SymShiftInvert.h>

#include <algorithm>
#include <format>

namespace vibetran {

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

    // SymShiftInvert<double> defaults: TypeA=Eigen::Sparse, TypeB=Eigen::Sparse,
    // UploA=Eigen::Lower, UploB=Eigen::Lower — factorises (K - sigma*M) via SparseLU.
    using OpType  = Spectra::SymShiftInvert<double>;
    using BOpType = Spectra::SparseSymMatProd<double>;

    // SymShiftInvert factorises (K - sigma*M) once, solves at each Lanczos step.
    OpType  op(K, M);
    BOpType Bop(M);

    Spectra::SymGEigsShiftSolver<OpType, BOpType, Spectra::GEigsMode::ShiftInvert>
        solver(op, Bop, nd, ncv, sigma);

    solver.init();
    // LargestMagn selection: keep Ritz pairs with largest |ν| during restarts,
    // matching the natural convergence of the shift-inverted Lanczos.
    // SmallestAlge sorting: after ν→λ conversion, sort final results ascending by λ.
    int nconv = solver.compute(Spectra::SortRule::LargestMagn, 1000, 1e-10,
                               Spectra::SortRule::SmallestAlge);

    if (solver.info() != Spectra::CompInfo::Successful || nconv < 1)
        throw SolverError(std::format(
            "Spectra eigensolver: converged only {}/{} eigenvalues (info={})",
            nconv, nd, static_cast<int>(solver.info())));

    Eigen::VectorXd eigenvalues  = solver.eigenvalues();
    Eigen::MatrixXd eigenvectors = solver.eigenvectors();

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

} // namespace vibetran
