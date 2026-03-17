#pragma once
// include/solver/cuda_solver_backend.hpp
// CUDA sparse solver backend using NVIDIA cuSOLVER.
//
// Algorithm: Sparse Cholesky factorization (cusolverSpDcsrlsvchol) for SPD
// FEM stiffness matrices, with automatic fallback to sparse LU
// (cusolverSpDcsrlsvlu) if the matrix is detected as non-SPD or ill-conditioned.
// Both solvers use AMD reordering to minimize fill-in.
//
// Both paths are host-side cuSOLVER APIs: input/output data remains on the CPU
// and cuSOLVER manages device memory and kernel dispatch internally.  This keeps
// the backend free of explicit CUDA memory management while still leveraging
// GPU acceleration via NVIDIA libraries.
//
// Use try_create() to construct — returns nullopt when no CUDA device is
// present so the caller can fall back without exception handling.

#ifdef HAVE_CUDA

#include "solver/solver_backend.hpp"
#include <memory>
#include <optional>

namespace nastran {

// Opaque RAII wrapper around cuSOLVER/cuSPARSE handles.
// Defined in cuda_solver_backend.cu to keep CUDA headers out of this file.
struct CudaContext;

class CudaSolverBackend final : public SolverBackend {
public:
    ~CudaSolverBackend() override;
    CudaSolverBackend(CudaSolverBackend&&) noexcept;
    CudaSolverBackend& operator=(CudaSolverBackend&&) noexcept;
    CudaSolverBackend(const CudaSolverBackend&) = delete;
    CudaSolverBackend& operator=(const CudaSolverBackend&) = delete;

    /// Factory — returns nullopt when no CUDA device is available.
    [[nodiscard]] static std::optional<CudaSolverBackend> try_create() noexcept;

    /// Solve K*u = F using cuSOLVER sparse Cholesky (with LU fallback).
    /// Throws SolverError on failure.
    [[nodiscard]] std::vector<double> solve(
        const SparseMatrixBuilder::CsrData& K,
        const std::vector<double>& F) override;

    [[nodiscard]] std::string_view name() const noexcept override;

    // ── Diagnostics (valid after each solve()) ─────────────────────────────

    /// Returns true if the most recent solve used sparse Cholesky (false = LU fallback).
    [[nodiscard]] bool last_solve_used_cholesky() const noexcept;

    /// GPU device name reported by CUDA runtime.
    [[nodiscard]] std::string_view device_name() const noexcept;

private:
    explicit CudaSolverBackend(std::unique_ptr<CudaContext> ctx) noexcept;

    std::unique_ptr<CudaContext> ctx_;
    bool last_cholesky_{true};
};

} // namespace nastran

#endif // HAVE_CUDA
