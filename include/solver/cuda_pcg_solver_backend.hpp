#pragma once
// include/solver/cuda_pcg_solver_backend.hpp
// CUDA Preconditioned Conjugate Gradient solver backend.
//
// Algorithm: PCG with IC0 → ILU0 → Jacobi fallback preconditioning.
//   - cuSPARSE generic API for sparse matrix-vector products (K*p)
//   - cuSPARSE SpSV for triangular forward/backward solves (IC0/ILU0)
//   - cuBLAS for dot products and axpy operations
//   - Custom CUDA kernels for Jacobi preconditioner apply and axpby
//
// Memory footprint: O(nnz + n) device memory — the matrix is stored once on
// the device and no fill-in factorisation is performed.  This makes the PCG
// backend suitable for very large systems (millions of DOFs) that would exhaust
// device memory with cuDSS sparse Cholesky.
//
// Convergence: relative residual ||r||_2 / ||b||_2 < tolerance.
// Default tolerance: 1e-8.
//
// Use try_create() to construct — returns nullopt when no CUDA device is
// present so the caller can fall back without exception handling.

#ifdef HAVE_CUDA

#include "solver/solver_backend.hpp"
#include <memory>
#include <optional>

namespace vibestran {

// Opaque RAII context (defined in cuda_pcg_solver_backend.cu).
struct CudaPCGContext;

class CudaPCGSolverBackend final : public SolverBackend {
public:
    ~CudaPCGSolverBackend() override;
    CudaPCGSolverBackend(CudaPCGSolverBackend&&) noexcept;
    CudaPCGSolverBackend& operator=(CudaPCGSolverBackend&&) noexcept;
    CudaPCGSolverBackend(const CudaPCGSolverBackend&) = delete;
    CudaPCGSolverBackend& operator=(const CudaPCGSolverBackend&) = delete;

    /// Factory — returns nullopt when no CUDA device is available.
    /// @param tolerance             Relative residual convergence threshold
    ///                               (<=0 selects the default 1e-8).
    /// @param max_iters             Maximum PCG iterations (0 = default: 10000).
    [[nodiscard]] static std::optional<CudaPCGSolverBackend>
    try_create(double tolerance = 0.0,
               int max_iters = 0) noexcept;

    /// Solve K*u = F using PCG with IC0 → ILU0 → Jacobi preconditioning.
    /// Throws SolverError if the system does not converge or on GPU errors.
    [[nodiscard]] std::vector<double> solve(
        const SparseMatrixBuilder::CsrData& K,
        const std::vector<double>& F,
        const FactorRatioCheckPolicy* factor_ratio_policy = nullptr) override;

    [[nodiscard]] std::string_view name() const noexcept override;

    // ── Diagnostics (valid after each solve()) ─────────────────────────────

    /// Number of PCG iterations used in the most recent solve().
    [[nodiscard]] int last_iteration_count() const noexcept override;

    /// Relative residual ||r||_2 / ||b||_2 achieved in the most recent solve().
    [[nodiscard]] double last_relative_residual() const noexcept;

    /// Alias for last_relative_residual() to satisfy SolverBackend diagnostics.
    [[nodiscard]] double last_estimated_error() const noexcept override;

    /// GPU device name reported by the CUDA runtime.
    [[nodiscard]] std::string_view device_name() const noexcept;

private:
    explicit CudaPCGSolverBackend(std::unique_ptr<CudaPCGContext> ctx) noexcept;
    std::unique_ptr<CudaPCGContext> ctx_;
};

} // namespace vibestran

#endif // HAVE_CUDA
