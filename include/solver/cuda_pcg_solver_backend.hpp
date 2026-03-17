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
// Single-precision mode: pass use_single_precision=true to try_create() to
// perform the entire solve in float32.  Halves device memory usage vs float64,
// which is useful for very large problems (hundreds of millions of nnz).
//
// Memory footprint: O(nnz + n) device memory — the matrix is stored once on
// the device and no fill-in factorisation is performed.  This makes the PCG
// backend suitable for very large systems (millions of DOFs) that would exhaust
// device memory with cuDSS sparse Cholesky.
//
// Convergence: relative preconditioned residual ||r||/||b|| < tolerance.
// Default tolerance 1e-8 is stricter than typical engineering requirements to
// leave headroom for mildly ill-conditioned systems.
//
// Use try_create() to construct — returns nullopt when no CUDA device is
// present so the caller can fall back without exception handling.

#ifdef HAVE_CUDA

#include "solver/solver_backend.hpp"
#include <memory>
#include <optional>

namespace nastran {

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
    /// @param use_single_precision  Perform the solve in float32 (halves VRAM usage).
    /// @param tolerance             Relative residual convergence threshold.
    /// @param max_iters             Maximum PCG iterations (0 = default: 10000).
    [[nodiscard]] static std::optional<CudaPCGSolverBackend>
    try_create(bool use_single_precision = false,
               double tolerance = 1e-8,
               int max_iters = 0) noexcept;

    /// Solve K*u = F using PCG with Jacobi preconditioning on the GPU.
    /// Throws SolverError if the system does not converge or on GPU errors.
    [[nodiscard]] std::vector<double> solve(
        const SparseMatrixBuilder::CsrData& K,
        const std::vector<double>& F) override;

    [[nodiscard]] std::string_view name() const noexcept override;

    // ── Diagnostics (valid after each solve()) ─────────────────────────────

    /// Number of PCG iterations used in the most recent solve().
    [[nodiscard]] int last_iteration_count() const noexcept;

    /// Relative residual achieved in the most recent solve().
    [[nodiscard]] double last_relative_residual() const noexcept;

    /// GPU device name reported by the CUDA runtime.
    [[nodiscard]] std::string_view device_name() const noexcept;

    /// True when the backend was created with use_single_precision=true.
    [[nodiscard]] bool uses_single_precision() const noexcept;

private:
    explicit CudaPCGSolverBackend(std::unique_ptr<CudaPCGContext> ctx) noexcept;
    std::unique_ptr<CudaPCGContext> ctx_;
};

} // namespace nastran

#endif // HAVE_CUDA
