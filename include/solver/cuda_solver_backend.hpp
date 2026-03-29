#pragma once
// include/solver/cuda_solver_backend.hpp
// CUDA sparse solver backend using NVIDIA cuDSS (direct sparse solver library).
//
// Algorithm: Sparse Cholesky (CUDSS_MTYPE_SPD) for SPD FEM stiffness matrices,
// with automatic fallback to sparse LU (CUDSS_MTYPE_GENERAL) if the matrix is
// not positive definite or the Cholesky residual is too large.
// Both phases run fully on-device via cudssExecute.
//
// For large problems where cuDSS's internal workspace exceeds device memory,
// fp32 mode halves the device memory footprint by downcasting inputs to float
// before the solve and upcasting the result back to double.
//
// Use try_create() to construct — returns nullopt when no CUDA device is
// present so the caller can fall back without exception handling.

#ifdef HAVE_CUDA

#include "solver/solver_backend.hpp"
#include <memory>
#include <optional>

namespace vibestran {

enum class CudaSolverPrecision {
    Float64,
    Float32,
};

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
    /// precision: downcast inputs to float before the GPU solve and upcast the
    /// result back to double when Float32 is selected.
    [[nodiscard]] static std::optional<CudaSolverBackend>
    try_create(CudaSolverPrecision precision = CudaSolverPrecision::Float64) noexcept;

    /// Solve K*u = F using cuDSS sparse Cholesky (with LU fallback).
    /// Throws SolverError on failure.
    [[nodiscard]] std::vector<double> solve(
        const SparseMatrixBuilder::CsrData& K,
        const std::vector<double>& F,
        const FactorRatioCheckPolicy* factor_ratio_policy = nullptr) override;

    [[nodiscard]] std::string_view name() const noexcept override;

    // ── Diagnostics (valid after each solve()) ─────────────────────────────

    /// Returns true if the most recent solve used sparse Cholesky (false = LU fallback).
    [[nodiscard]] bool last_solve_used_cholesky() const noexcept;

    /// Precision mode selected for this backend instance.
    [[nodiscard]] CudaSolverPrecision precision() const noexcept;

    /// GPU device name reported by CUDA runtime.
    [[nodiscard]] std::string_view device_name() const noexcept;

private:
    explicit CudaSolverBackend(std::unique_ptr<CudaContext> ctx) noexcept;

    std::unique_ptr<CudaContext> ctx_;
    bool last_cholesky_{true};
};

} // namespace vibestran

#endif // HAVE_CUDA
