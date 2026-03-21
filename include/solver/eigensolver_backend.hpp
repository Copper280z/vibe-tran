#pragma once
// include/solver/eigensolver_backend.hpp
// Abstract interface for generalized eigenvalue solvers: K φ = λ M φ
//
// Mirrors SolverBackend to allow future GPU eigensolver backends.

#include <Eigen/SparseCore>
#include <string>
#include <vector>

namespace vibetran {

/// One mode shape result (single eigenvalue/eigenvector pair)
struct EigenPair {
    double eigenvalue;          ///< λ = ω² (rad²/s²), unshifted
    Eigen::VectorXd eigenvector; ///< φ, mass-normalised (φᵀMφ = 1)
};

/// Abstract interface for generalized eigensolver backends.
/// Solves: K φ = λ M φ  for the nd eigenvalues nearest to sigma.
class EigensolverBackend {
public:
    virtual ~EigensolverBackend() = default;

    /// Solve the generalised eigenvalue problem K φ = λ M φ.
    /// @param K     Stiffness matrix (symmetric positive semi-definite)
    /// @param M     Mass matrix (symmetric positive definite)
    /// @param nd    Number of desired eigenpairs
    /// @param sigma Shift: find eigenvalues closest to this value
    /// @return      Up to nd EigenPairs sorted by eigenvalue (ascending)
    [[nodiscard]] virtual std::vector<EigenPair> solve(
        const Eigen::SparseMatrix<double>& K,
        const Eigen::SparseMatrix<double>& M,
        int nd, double sigma) = 0;

    [[nodiscard]] virtual std::string name() const = 0;
};

/// CPU implementation using Spectra (shift-and-invert Lanczos).
class SpectraEigensolverBackend : public EigensolverBackend {
public:
    [[nodiscard]] std::vector<EigenPair> solve(
        const Eigen::SparseMatrix<double>& K,
        const Eigen::SparseMatrix<double>& M,
        int nd, double sigma) override;

    [[nodiscard]] std::string name() const override {
        return "Spectra CPU (shift-invert Lanczos)";
    }
};

} // namespace vibetran
