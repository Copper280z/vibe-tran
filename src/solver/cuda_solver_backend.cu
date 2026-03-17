// src/solver/cuda_solver_backend.cu
// CUDA sparse solver backend: cuSOLVER sparse Cholesky + LU fallback.
//
// Solver selection logic:
//   1. Try cusolverSpDcsrlsvchol (direct sparse Cholesky, AMD reordering).
//      Optimal for SPD FEM stiffness matrices; fills in far less than LU.
//   2. If Cholesky reports singularity (singularity != -1), retry with
//      cusolverSpDcsrlsvlu (sparse LU, AMD reordering) which handles
//      non-SPD and mildly ill-conditioned matrices.
//
// Both APIs are host-side (data lives on the CPU); cuSOLVER handles all
// device-side memory management and kernel dispatch internally.
//
// Note: this file is only compiled when the CUDA backend is enabled in
// meson (have_cuda_backend=true), so HAVE_CUDA is guaranteed to be defined.
// The #ifdef guard is in the header only, to prevent the class from being
// declared in non-CUDA builds.
//
// <format> / std::format is intentionally avoided here: nvcc uses its
// bundled g++-12 as the host compiler, and g++-12 does not provide <format>.
// Use std::string concatenation and std::to_string() for error messages.
#define HAVE_CUDA 1

#include "solver/cuda_solver_backend.hpp"
// Include exceptions.hpp directly (not types.hpp) to avoid pulling in <format>.
#include "core/exceptions.hpp"

#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace nastran {

// ── RAII helpers ──────────────────────────────────────────────────────────────

struct CudaContext {
    cusolverSpHandle_t cusolver = nullptr;
    cusparseHandle_t   cusparse = nullptr;
    std::string        device_name;
};

// ── Constructor / destructor ──────────────────────────────────────────────────

CudaSolverBackend::CudaSolverBackend(std::unique_ptr<CudaContext> ctx) noexcept
    : ctx_(std::move(ctx)) {}

CudaSolverBackend::~CudaSolverBackend() {
    if (!ctx_) return;
    if (ctx_->cusparse) cusparseDestroy(ctx_->cusparse);
    if (ctx_->cusolver) cusolverSpDestroy(ctx_->cusolver);
}

CudaSolverBackend::CudaSolverBackend(CudaSolverBackend&&) noexcept = default;
CudaSolverBackend& CudaSolverBackend::operator=(CudaSolverBackend&&) noexcept = default;

// ── Factory ───────────────────────────────────────────────────────────────────

std::optional<CudaSolverBackend> CudaSolverBackend::try_create() noexcept {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        return std::nullopt;

    // Select device 0; a future extension could let the user pick.
    if (cudaSetDevice(0) != cudaSuccess)
        return std::nullopt;

    auto ctx = std::make_unique<CudaContext>();

    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, 0) == cudaSuccess)
        ctx->device_name = props.name;

    if (cusolverSpCreate(&ctx->cusolver) != CUSOLVER_STATUS_SUCCESS)
        return std::nullopt;

    if (cusparseCreate(&ctx->cusparse) != CUSPARSE_STATUS_SUCCESS) {
        cusolverSpDestroy(ctx->cusolver);
        return std::nullopt;
    }

    return CudaSolverBackend(std::move(ctx));
}

// ── Accessors ─────────────────────────────────────────────────────────────────

std::string_view CudaSolverBackend::name() const noexcept {
    return "CUDA cuSOLVER sparse Cholesky";
}

bool CudaSolverBackend::last_solve_used_cholesky() const noexcept {
    return last_cholesky_;
}

std::string_view CudaSolverBackend::device_name() const noexcept {
    return ctx_->device_name;
}

// ── Residual validation ───────────────────────────────────────────────────────

// Compute relative residual ||K*u - F||_2 / ||F||_2 on the CPU.
// Used to detect garbage solutions from near-singular matrices that cuSOLVER's
// Cholesky factorises without reporting singularity (e.g. SPSD matrices where
// the near-zero pivot exceeds the factorisation tolerance).
static double relative_residual(
        const SparseMatrixBuilder::CsrData& K,
        const std::vector<double>& u,
        const std::vector<double>& F) {
    const int n = K.n;
    double res_sq = 0.0;
    double rhs_sq = 0.0;
    for (int row = 0; row < n; ++row) {
        double row_val = 0.0;
        for (int idx = K.row_ptr[row]; idx < K.row_ptr[row + 1]; ++idx)
            row_val += K.values[idx] * u[K.col_ind[idx]];
        double diff = row_val - F[row];
        res_sq += diff * diff;
        rhs_sq += F[row] * F[row];
    }
    if (rhs_sq == 0.0) return (res_sq == 0.0) ? 0.0 : 1.0;
    return std::sqrt(res_sq / rhs_sq);
}

// ── solve ─────────────────────────────────────────────────────────────────────

std::vector<double>
CudaSolverBackend::solve(const SparseMatrixBuilder::CsrData& K,
                          const std::vector<double>& F) {
    const int n   = K.n;
    const int nnz = K.nnz;

    if (n == 0)
        throw SolverError("CUDA solver: stiffness matrix is empty -- no free DOFs");
    if (static_cast<int>(F.size()) != n)
        throw SolverError(
            "CUDA solver: force vector size " + std::to_string(F.size()) +
            " != matrix size " + std::to_string(n));

    // Build cusparse matrix descriptor (general, zero-based indexing).
    cusparseMatDescr_t descr = nullptr;
    if (cusparseCreateMatDescr(&descr) != CUSPARSE_STATUS_SUCCESS)
        throw SolverError("CUDA solver: cusparseCreateMatDescr failed");

    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);

    std::vector<double> u(n, 0.0);
    int singularity = -1;

    // ── Path 1: sparse Cholesky (host-side) ──────────────────────────────────
    // Host variant: all pointers are CPU memory; cuSOLVER manages device
    // allocations internally.
    // reorder=1: AMD (Approximate Minimum Degree) reordering minimises fill-in.
    // tol=1e-10: singularity threshold; singularity is set to first zero pivot
    // row if the matrix is (nearly) singular.
    cusolverStatus_t status = cusolverSpDcsrlsvcholHost(
        ctx_->cusolver,
        n, nnz, descr,
        K.values.data(),
        K.row_ptr.data(),
        K.col_ind.data(),
        F.data(),
        /*tol=*/1e-10,
        /*reorder=*/1,
        u.data(),
        &singularity
    );

    if (status != CUSOLVER_STATUS_SUCCESS) {
        cusparseDestroyMatDescr(descr);
        throw SolverError(
            "CUDA solver: cusolverSpDcsrlsvcholHost failed with status " +
            std::to_string(static_cast<int>(status)));
    }

    // ── Cholesky result evaluation ────────────────────────────────────────────
    // cuSOLVER's Cholesky can silently "succeed" for near-singular SPSD matrices
    // if the near-zero pivot exceeds the factorization tolerance.  Validate the
    // residual to catch these cases before returning a garbage solution.
    bool chol_ok = (singularity == -1);
    if (chol_ok) {
        double rel_res = relative_residual(K, u, F);
        if (rel_res > 1e-6) {
            std::clog << "[cuda] Cholesky residual " << rel_res
                      << " > 1e-6 -- retrying with sparse LU\n";
            chol_ok = false;
        }
    }

    if (chol_ok) {
        last_cholesky_ = true;
        cusparseDestroyMatDescr(descr);
        std::clog << "[cuda] Cholesky solve: n=" << n << ", nnz=" << nnz
                  << ", device='" << ctx_->device_name << "'\n";
        return u;
    }

    // ── Path 2: LU fallback ───────────────────────────────────────────────────
    if (singularity != -1)
        std::clog << "[cuda] Cholesky reported singularity at row " << singularity
                  << " -- retrying with sparse LU\n";

    std::fill(u.begin(), u.end(), 0.0);
    singularity = -1;

    // Host variant: all pointers are CPU memory.
    // reorder=3: COLAMD, better suited for unsymmetric/non-SPD systems than AMD.
    status = cusolverSpDcsrlsvluHost(
        ctx_->cusolver,
        n, nnz, descr,
        K.values.data(),
        K.row_ptr.data(),
        K.col_ind.data(),
        F.data(),
        /*tol=*/1e-10,
        /*reorder=*/3,
        u.data(),
        &singularity
    );

    cusparseDestroyMatDescr(descr);

    if (status != CUSOLVER_STATUS_SUCCESS)
        throw SolverError(
            "CUDA solver: cusolverSpDcsrlsvluHost failed with status " +
            std::to_string(static_cast<int>(status)));

    if (singularity != -1)
        throw SolverError(
            "CUDA solver: stiffness matrix is singular at row " +
            std::to_string(singularity) +
            " -- check boundary conditions (SPCs)");

    // LU succeeded; validate residual as a final sanity check.
    double rel_res_lu = relative_residual(K, u, F);
    if (rel_res_lu > 1e-6)
        throw SolverError(
            "CUDA solver: LU produced large residual " +
            std::to_string(rel_res_lu) +
            " -- stiffness matrix is singular or very ill-conditioned. "
            "Check boundary conditions (SPCs)");

    last_cholesky_ = false;
    std::clog << "[cuda] LU solve: n=" << n << ", nnz=" << nnz
              << ", device='" << ctx_->device_name << "'\n";
    return u;
}

} // namespace nastran
