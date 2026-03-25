#define HAVE_CUDA 1

#include "solver/cuda_mixed_pcg_solver_backend.hpp"
#include "cuda_pcg_backend_impl.hpp"

namespace vibestran {

CudaMixedPCGSolverBackend::CudaMixedPCGSolverBackend(
    std::unique_ptr<CudaPCGContext> ctx) noexcept
    : ctx_(std::move(ctx)) {}

CudaMixedPCGSolverBackend::~CudaMixedPCGSolverBackend() {
    destroy_cuda_pcg_context(ctx_);
}

CudaMixedPCGSolverBackend::CudaMixedPCGSolverBackend(
    CudaMixedPCGSolverBackend&&) noexcept = default;
CudaMixedPCGSolverBackend& CudaMixedPCGSolverBackend::operator=(
    CudaMixedPCGSolverBackend&&) noexcept = default;

std::optional<CudaMixedPCGSolverBackend>
CudaMixedPCGSolverBackend::try_create(double tolerance, int max_iters) noexcept {
    auto ctx = try_create_cuda_pcg_context(
        tolerance > 0.0 ? tolerance : 1e-6, max_iters);
    if (!ctx)
        return std::nullopt;
    return CudaMixedPCGSolverBackend(std::move(ctx));
}

std::vector<double>
CudaMixedPCGSolverBackend::solve(const SparseMatrixBuilder::CsrData& K,
                                 const std::vector<double>& F) {
    return solve_cuda_pcg_mixed(*ctx_, K, F);
}

std::string_view CudaMixedPCGSolverBackend::name() const noexcept {
    return "CUDA mixed-precision PCG + iterative refinement (GPU)";
}

int CudaMixedPCGSolverBackend::last_iteration_count() const noexcept {
    return ctx_->last_iters;
}

double CudaMixedPCGSolverBackend::last_estimated_error() const noexcept {
    return ctx_->last_rel_res;
}

std::string_view CudaMixedPCGSolverBackend::device_name() const noexcept {
    return ctx_->device_name;
}

} // namespace vibestran
