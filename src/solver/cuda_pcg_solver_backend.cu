#define HAVE_CUDA 1

#include "solver/cuda_pcg_solver_backend.hpp"
#include "cuda_pcg_backend_impl.hpp"

namespace vibestran {

CudaPCGSolverBackend::CudaPCGSolverBackend(std::unique_ptr<CudaPCGContext> ctx) noexcept
    : ctx_(std::move(ctx)) {}

CudaPCGSolverBackend::~CudaPCGSolverBackend() {
    destroy_cuda_pcg_context(ctx_);
}

CudaPCGSolverBackend::CudaPCGSolverBackend(CudaPCGSolverBackend&&) noexcept = default;
CudaPCGSolverBackend& CudaPCGSolverBackend::operator=(CudaPCGSolverBackend&&) noexcept = default;

std::optional<CudaPCGSolverBackend>
CudaPCGSolverBackend::try_create(double tolerance, int max_iters) noexcept {
    auto ctx = try_create_cuda_pcg_context(
        tolerance > 0.0 ? tolerance : 1e-8, max_iters);
    if (!ctx)
        return std::nullopt;
    return CudaPCGSolverBackend(std::move(ctx));
}

std::string_view CudaPCGSolverBackend::name() const noexcept {
    return "CUDA PCG + IC0/ILU0 float64 (GPU)";
}

int CudaPCGSolverBackend::last_iteration_count() const noexcept {
    return ctx_->last_iters;
}

double CudaPCGSolverBackend::last_relative_residual() const noexcept {
    return ctx_->last_rel_res;
}

double CudaPCGSolverBackend::last_estimated_error() const noexcept {
    return ctx_->last_rel_res;
}

std::string_view CudaPCGSolverBackend::device_name() const noexcept {
    return ctx_->device_name;
}

std::vector<double>
CudaPCGSolverBackend::solve(const SparseMatrixBuilder::CsrData& K,
                            const std::vector<double>& F,
                            const FactorRatioCheckPolicy*) {
    return solve_cuda_pcg_stable(*ctx_, K, F);
}

} // namespace vibestran
