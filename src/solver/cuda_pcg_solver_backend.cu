// src/solver/cuda_pcg_solver_backend.cu
// CUDA Preconditioned Conjugate Gradient solver backend.
//
// Preconditioner selection (tried in order):
//   1. IC0 (Incomplete Cholesky, zero fill-in) via cusparseDcsric02.
//      Optimal for SPD FEM stiffness matrices; reduces iteration count 10-100×
//      vs Jacobi.  Factorization is in-place on a copy of K; apply = two
//      triangular solves with cusparseSpSV (forward L, backward L^T).
//   2. ILU0 (Incomplete LU, zero fill-in) via cusparseDcsrilu02.
//      Used when IC0 setup fails (zero pivot, non-SPD matrix).  Apply = forward
//      L solve + backward U solve.
//   3. Jacobi (diagonal scaling).  Always succeeds; weakest preconditioner.
//
// PCG loop:
//   r = F, z = M^{-1}r, p = z, rz = r·z
//   while not converged:
//     Ap = K*p                (cusparseSpMV)
//     alpha = rz / (p·Ap)
//     u += alpha*p, r -= alpha*Ap
//     z = M^{-1}r             (two triangular solves or Jacobi kernel)
//     rz_new = r·z
//     if sqrt(rz_new/rz0) < tol: done
//     p = z + (rz_new/rz)*p  (axpby kernel)
//     rz = rz_new
//
// Dense vector descriptors used in analysis must be the same handles reused in
// solve; they are stored inside TriangPrecond and point to fixed device buffers.
//
// Note: <format> / std::format is intentionally avoided — nvcc uses its bundled
// g++-12 as host compiler, which does not provide <format>.
#define HAVE_CUDA 1

#include "solver/cuda_pcg_solver_backend.hpp"
#include "core/exceptions.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

// Log free/total device memory alongside a label.
static void log_mem(const char* label, std::size_t extra_bytes = 0) {
    std::size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    constexpr double kMiB = 1024.0 * 1024.0;
    std::clog << "[cuda-pcg] " << label
              << ": free=" << free_bytes / kMiB << " MiB"
              << ", total=" << total_bytes / kMiB << " MiB";
    if (extra_bytes > 0)
        std::clog << ", allocating=" << extra_bytes / kMiB << " MiB";
    std::clog << "\n";
}

} // anonymous namespace

namespace nastran {

// ── RAII device buffer ────────────────────────────────────────────────────────

template <typename T>
struct PCGDeviceBuffer {
    T* ptr = nullptr;

    PCGDeviceBuffer() = default;
    explicit PCGDeviceBuffer(std::size_t count) {
        if (count == 0) return;
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr),
                                     count * sizeof(T));
        if (err != cudaSuccess)
            throw SolverError(
                "CUDA PCG: cudaMalloc failed for " + std::to_string(count) +
                " elements: " + cudaGetErrorString(err));
    }
    ~PCGDeviceBuffer() { if (ptr) cudaFree(ptr); }

    PCGDeviceBuffer(const PCGDeviceBuffer&)            = delete;
    PCGDeviceBuffer& operator=(const PCGDeviceBuffer&) = delete;
    PCGDeviceBuffer(PCGDeviceBuffer&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
    PCGDeviceBuffer& operator=(PCGDeviceBuffer&& o) noexcept {
        if (this != &o) { if (ptr) cudaFree(ptr); ptr = o.ptr; o.ptr = nullptr; }
        return *this;
    }

    void upload(const T* host, std::size_t count) {
        cudaError_t err = cudaMemcpy(ptr, host, count * sizeof(T),
                                     cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            throw SolverError(
                std::string("CUDA PCG: cudaMemcpy H->D failed: ") +
                cudaGetErrorString(err));
    }

    void download(T* host, std::size_t count) const {
        cudaError_t err = cudaMemcpy(host, ptr, count * sizeof(T),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            throw SolverError(
                std::string("CUDA PCG: cudaMemcpy D->H failed: ") +
                cudaGetErrorString(err));
    }

    void zero(std::size_t count) { cudaMemset(ptr, 0, count * sizeof(T)); }
};

// ── Custom CUDA kernels ───────────────────────────────────────────────────────

// Jacobi preconditioner: z[i] = r[i] * d_inv[i]
__global__ static void jacobi_kernel(
    const double* __restrict__ r,
    const double* __restrict__ d_inv,
    double* __restrict__ z,
    int n)
{
    int i = static_cast<int>(blockIdx.x) * blockDim.x +
            static_cast<int>(threadIdx.x);
    if (i < n) z[i] = r[i] * d_inv[i];
}

// axpby: y[i] = alpha*x[i] + beta*y[i]
__global__ static void axpby_kernel(
    const double* __restrict__ x,
    double alpha,
    double beta,
    double* __restrict__ y,
    int n)
{
    int i = static_cast<int>(blockIdx.x) * blockDim.x +
            static_cast<int>(threadIdx.x);
    if (i < n) y[i] = alpha * x[i] + beta * y[i];
}

static constexpr int kBlock = 256;

static void launch_jacobi(const double* r, const double* d_inv, double* z, int n) {
    jacobi_kernel<<<(n + kBlock - 1) / kBlock, kBlock>>>(r, d_inv, z, n);
}

static void launch_axpby(const double* x, double a, double b, double* y, int n) {
    axpby_kernel<<<(n + kBlock - 1) / kBlock, kBlock>>>(x, a, b, y, n);
}

// ── Preconditioner type ───────────────────────────────────────────────────────

enum class PrecondKind { IC0, ILU0, Jacobi };

// Stores all GPU resources for a triangular (IC0 or ILU0) preconditioner.
// Dense vector descriptors (vec_r, vec_tmp, vec_z) are created during setup
// and reused in each apply call — cusparseSpSV requires the same descriptor
// handles in solve() as were used in analysis().
struct TriangPrecond {
    PCGDeviceBuffer<double> d_M_vals;    // in-place IC0 or ILU0 factored values
    PCGDeviceBuffer<char>   d_factor_buf; // scratch for csric02 / csrilu02

    // Sparse matrix descriptors for triangular factors.
    // IC0:  mat_L only (FILL_MODE_LOWER); backward solve uses TRANSPOSE.
    // ILU0: mat_L (FILL_MODE_LOWER, unit-diag) + mat_U (FILL_MODE_UPPER).
    cusparseSpMatDescr_t mat_L = nullptr;
    cusparseSpMatDescr_t mat_U = nullptr;  // ILU0 only; null for IC0

    // Triangular-solve descriptors.  sv_L = forward solve, sv_UT = backward.
    cusparseSpSVDescr_t sv_L  = nullptr;
    cusparseSpSVDescr_t sv_UT = nullptr;
    PCGDeviceBuffer<char> d_sv_L_buf;
    PCGDeviceBuffer<char> d_sv_UT_buf;

    // Intermediate buffer: tmp = L^{-1}r, then z = U^{-1}tmp.
    PCGDeviceBuffer<double> d_tmp;

    // Dense vector descriptors.  MUST be the same handles passed to both
    // cusparseSpSV_analysis() and cusparseSpSV_solve().
    cusparseDnVecDescr_t vec_r   = nullptr;  // points to PCG d_r
    cusparseDnVecDescr_t vec_tmp = nullptr;  // points to d_tmp
    cusparseDnVecDescr_t vec_z   = nullptr;  // points to PCG d_z

    ~TriangPrecond() {
        if (vec_r)   cusparseDestroyDnVec(vec_r);
        if (vec_tmp) cusparseDestroyDnVec(vec_tmp);
        if (vec_z)   cusparseDestroyDnVec(vec_z);
        if (mat_L)   cusparseDestroySpMat(mat_L);
        if (mat_U)   cusparseDestroySpMat(mat_U);
        if (sv_L)    cusparseSpSV_destroyDescr(sv_L);
        if (sv_UT)   cusparseSpSV_destroyDescr(sv_UT);
    }
};

// ── IC0 factorization and SpSV setup ─────────────────────────────────────────
// Returns false on structural/numerical zero pivots so the caller can retry.
static bool setup_ic0(
    cusparseHandle_t cusparse,
    int n, int nnz,
    const int* d_row_ptr, const int* d_col_ind, const double* d_values,
    double* d_r, double* d_z,
    TriangPrecond& tp)
{
    // Working copy of K values (IC0 factorizes lower triangle in-place).
    tp.d_M_vals = PCGDeviceBuffer<double>(nnz);
    tp.d_tmp    = PCGDeviceBuffer<double>(n);

    cudaMemcpy(tp.d_M_vals.ptr, d_values,
               static_cast<std::size_t>(nnz) * sizeof(double),
               cudaMemcpyDeviceToDevice);

    // GENERAL + FILL_MODE_LOWER: csric02 reads only the lower triangle.
    // SYMMETRIC type is not supported for csric02 in CUDA 12.x.
    cusparseMatDescr_t descr = nullptr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);

    csric02Info_t info = nullptr;
    cusparseCreateCsric02Info(&info);

    int buf_size = 0;
    cusparseStatus_t cs = cusparseDcsric02_bufferSize(
        cusparse, n, nnz, descr,
        tp.d_M_vals.ptr, d_row_ptr, d_col_ind, info, &buf_size);
    if (cs != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroyCsric02Info(info);
        cusparseDestroyMatDescr(descr);
        throw SolverError("CUDA PCG: IC0 bufferSize failed, status=" +
                          std::to_string(static_cast<int>(cs)));
    }

    // factored copy of K values + intermediate tmp vector + factor scratch
    const std::size_t precond_bytes = static_cast<std::size_t>(nnz) * sizeof(double)
                                    + static_cast<std::size_t>(n) * sizeof(double)
                                    + static_cast<std::size_t>(buf_size > 0 ? buf_size : 1);
    log_mem("IC0 factor alloc", precond_bytes);
    std::clog << "[cuda-pcg] IC0 factor scratch=" << buf_size / (1024.0 * 1024.0)
              << " MiB\n";

    tp.d_factor_buf = PCGDeviceBuffer<char>(buf_size > 0 ? buf_size : 1);

    cusparseDcsric02_analysis(cusparse, n, nnz, descr,
        tp.d_M_vals.ptr, d_row_ptr, d_col_ind, info,
        CUSPARSE_SOLVE_POLICY_NO_LEVEL, tp.d_factor_buf.ptr);

    int structural_zero = -1;
    cusparseXcsric02_zeroPivot(cusparse, info, &structural_zero);
    if (structural_zero >= 0 || cs != CUSPARSE_STATUS_SUCCESS) {
        std::clog << "[cuda-pcg] IC0: structural zero at row " << structural_zero
                  << " -- retrying with ILU0\n";
        cusparseDestroyCsric02Info(info);
        cusparseDestroyMatDescr(descr);
        return false;
    }

    cs = cusparseDcsric02(cusparse, n, nnz, descr,
        tp.d_M_vals.ptr, d_row_ptr, d_col_ind, info,
        CUSPARSE_SOLVE_POLICY_NO_LEVEL, tp.d_factor_buf.ptr);

    int numerical_zero = -1;
    cusparseXcsric02_zeroPivot(cusparse, info, &numerical_zero);
    cusparseDestroyCsric02Info(info);
    cusparseDestroyMatDescr(descr);

    if (numerical_zero >= 0 || cs != CUSPARSE_STATUS_SUCCESS) {
        std::clog << "[cuda-pcg] IC0: numerical zero at row " << numerical_zero
                  << " -- retrying with ILU0\n";
        return false;
    }

    // ── SpSV setup ────────────────────────────────────────────────────────────
    // mat_L = lower triangle of d_M_vals (contains L after IC0).
    // Forward solve: L * tmp = r  (NON_TRANSPOSE).
    // Backward solve: L^T * z = tmp  (TRANSPOSE on mat_L).
    cusparseCreateCsr(&tp.mat_L,
        static_cast<int64_t>(n), static_cast<int64_t>(n),
        static_cast<int64_t>(nnz),
        const_cast<int*>(d_row_ptr), const_cast<int*>(d_col_ind),
        tp.d_M_vals.ptr,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    {
        cusparseFillMode_t fill_L  = CUSPARSE_FILL_MODE_LOWER;
        cusparseDiagType_t diag_nu = CUSPARSE_DIAG_TYPE_NON_UNIT;
        cusparseSpMatSetAttribute(tp.mat_L, CUSPARSE_SPMAT_FILL_MODE,
                                  &fill_L,  sizeof(fill_L));
        cusparseSpMatSetAttribute(tp.mat_L, CUSPARSE_SPMAT_DIAG_TYPE,
                                  &diag_nu, sizeof(diag_nu));
    }
    tp.mat_U = nullptr; // IC0 uses transpose of mat_L, not a separate descriptor

    // Create dense vector descriptors that persist for the lifetime of TriangPrecond.
    cusparseCreateDnVec(&tp.vec_r,   n, d_r,          CUDA_R_64F);
    cusparseCreateDnVec(&tp.vec_tmp, n, tp.d_tmp.ptr, CUDA_R_64F);
    cusparseCreateDnVec(&tp.vec_z,   n, d_z,          CUDA_R_64F);

    cusparseSpSV_createDescr(&tp.sv_L);
    cusparseSpSV_createDescr(&tp.sv_UT);

    const double one = 1.0;

    // Forward: L * tmp = r
    std::size_t sz = 0;
    cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
        tp.mat_L, tp.vec_r, tp.vec_tmp,
        CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_L, &sz);
    tp.d_sv_L_buf = PCGDeviceBuffer<char>(sz > 0 ? sz : 1);
    std::clog << "[cuda-pcg] IC0 SpSV forward scratch=" << sz / (1024.0 * 1024.0)
              << " MiB\n";
    cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
        tp.mat_L, tp.vec_r, tp.vec_tmp,
        CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_L, tp.d_sv_L_buf.ptr);

    // Backward: L^T * z = tmp
    sz = 0;
    cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_TRANSPOSE, &one,
        tp.mat_L, tp.vec_tmp, tp.vec_z,
        CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_UT, &sz);
    tp.d_sv_UT_buf = PCGDeviceBuffer<char>(sz > 0 ? sz : 1);
    std::clog << "[cuda-pcg] IC0 SpSV backward scratch=" << sz / (1024.0 * 1024.0)
              << " MiB\n";
    cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_TRANSPOSE, &one,
        tp.mat_L, tp.vec_tmp, tp.vec_z,
        CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_UT, tp.d_sv_UT_buf.ptr);

    log_mem("after IC0 setup");
    std::clog << "[cuda-pcg] IC0 preconditioner setup successful\n";
    return true;
}

// ── ILU0 factorization and SpSV setup ────────────────────────────────────────
// Returns false on zero pivot so the caller can fall back to Jacobi.
static bool setup_ilu0(
    cusparseHandle_t cusparse,
    int n, int nnz,
    const int* d_row_ptr, const int* d_col_ind, const double* d_values,
    double* d_r, double* d_z,
    TriangPrecond& tp)
{
    tp.d_M_vals = PCGDeviceBuffer<double>(nnz);
    tp.d_tmp    = PCGDeviceBuffer<double>(n);

    cudaMemcpy(tp.d_M_vals.ptr, d_values,
               static_cast<std::size_t>(nnz) * sizeof(double),
               cudaMemcpyDeviceToDevice);

    cusparseMatDescr_t descr = nullptr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    csrilu02Info_t info = nullptr;
    cusparseCreateCsrilu02Info(&info);

    int buf_size = 0;
    cusparseStatus_t cs = cusparseDcsrilu02_bufferSize(
        cusparse, n, nnz, descr,
        tp.d_M_vals.ptr, d_row_ptr, d_col_ind, info, &buf_size);
    if (cs != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroyCsrilu02Info(info);
        cusparseDestroyMatDescr(descr);
        throw SolverError("CUDA PCG: ILU0 bufferSize failed, status=" +
                          std::to_string(static_cast<int>(cs)));
    }

    const std::size_t precond_bytes_ilu = static_cast<std::size_t>(nnz) * sizeof(double)
                                        + static_cast<std::size_t>(n) * sizeof(double)
                                        + static_cast<std::size_t>(buf_size > 0 ? buf_size : 1);
    log_mem("ILU0 factor alloc", precond_bytes_ilu);
    std::clog << "[cuda-pcg] ILU0 factor scratch=" << buf_size / (1024.0 * 1024.0)
              << " MiB\n";

    tp.d_factor_buf = PCGDeviceBuffer<char>(buf_size > 0 ? buf_size : 1);

    cusparseDcsrilu02_analysis(cusparse, n, nnz, descr,
        tp.d_M_vals.ptr, d_row_ptr, d_col_ind, info,
        CUSPARSE_SOLVE_POLICY_NO_LEVEL, tp.d_factor_buf.ptr);

    int structural_zero = -1;
    cusparseXcsrilu02_zeroPivot(cusparse, info, &structural_zero);
    if (structural_zero >= 0) {
        std::clog << "[cuda-pcg] ILU0: structural zero at row " << structural_zero
                  << " -- falling back to Jacobi\n";
        cusparseDestroyCsrilu02Info(info);
        cusparseDestroyMatDescr(descr);
        return false;
    }

    cs = cusparseDcsrilu02(cusparse, n, nnz, descr,
        tp.d_M_vals.ptr, d_row_ptr, d_col_ind, info,
        CUSPARSE_SOLVE_POLICY_NO_LEVEL, tp.d_factor_buf.ptr);

    int numerical_zero = -1;
    cusparseXcsrilu02_zeroPivot(cusparse, info, &numerical_zero);
    cusparseDestroyCsrilu02Info(info);
    cusparseDestroyMatDescr(descr);

    if (numerical_zero >= 0 || cs != CUSPARSE_STATUS_SUCCESS) {
        std::clog << "[cuda-pcg] ILU0: numerical zero at row " << numerical_zero
                  << " -- falling back to Jacobi\n";
        return false;
    }

    // ── SpSV setup ────────────────────────────────────────────────────────────
    // After ILU0: lower triangle = L (unit-diagonal), upper triangle = U.
    // Forward: L * tmp = r  (NON_TRANSPOSE, LOWER, unit-diagonal).
    // Backward: U * z = tmp (NON_TRANSPOSE, UPPER, non-unit-diagonal).
    cusparseCreateCsr(&tp.mat_L,
        static_cast<int64_t>(n), static_cast<int64_t>(n),
        static_cast<int64_t>(nnz),
        const_cast<int*>(d_row_ptr), const_cast<int*>(d_col_ind),
        tp.d_M_vals.ptr,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    {
        cusparseFillMode_t fill_L = CUSPARSE_FILL_MODE_LOWER;
        cusparseDiagType_t diag_u = CUSPARSE_DIAG_TYPE_UNIT;
        cusparseSpMatSetAttribute(tp.mat_L, CUSPARSE_SPMAT_FILL_MODE,
                                  &fill_L,  sizeof(fill_L));
        cusparseSpMatSetAttribute(tp.mat_L, CUSPARSE_SPMAT_DIAG_TYPE,
                                  &diag_u,  sizeof(diag_u));
    }

    cusparseCreateCsr(&tp.mat_U,
        static_cast<int64_t>(n), static_cast<int64_t>(n),
        static_cast<int64_t>(nnz),
        const_cast<int*>(d_row_ptr), const_cast<int*>(d_col_ind),
        tp.d_M_vals.ptr,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    {
        cusparseFillMode_t fill_U  = CUSPARSE_FILL_MODE_UPPER;
        cusparseDiagType_t diag_nu = CUSPARSE_DIAG_TYPE_NON_UNIT;
        cusparseSpMatSetAttribute(tp.mat_U, CUSPARSE_SPMAT_FILL_MODE,
                                  &fill_U,  sizeof(fill_U));
        cusparseSpMatSetAttribute(tp.mat_U, CUSPARSE_SPMAT_DIAG_TYPE,
                                  &diag_nu, sizeof(diag_nu));
    }

    cusparseCreateDnVec(&tp.vec_r,   n, d_r,          CUDA_R_64F);
    cusparseCreateDnVec(&tp.vec_tmp, n, tp.d_tmp.ptr, CUDA_R_64F);
    cusparseCreateDnVec(&tp.vec_z,   n, d_z,          CUDA_R_64F);

    cusparseSpSV_createDescr(&tp.sv_L);
    cusparseSpSV_createDescr(&tp.sv_UT);

    const double one = 1.0;

    // Forward: L * tmp = r
    std::size_t sz = 0;
    cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
        tp.mat_L, tp.vec_r, tp.vec_tmp,
        CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_L, &sz);
    tp.d_sv_L_buf = PCGDeviceBuffer<char>(sz > 0 ? sz : 1);
    std::clog << "[cuda-pcg] ILU0 SpSV forward scratch=" << sz / (1024.0 * 1024.0)
              << " MiB\n";
    cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
        tp.mat_L, tp.vec_r, tp.vec_tmp,
        CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_L, tp.d_sv_L_buf.ptr);

    // Backward: U * z = tmp
    sz = 0;
    cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
        tp.mat_U, tp.vec_tmp, tp.vec_z,
        CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_UT, &sz);
    tp.d_sv_UT_buf = PCGDeviceBuffer<char>(sz > 0 ? sz : 1);
    std::clog << "[cuda-pcg] ILU0 SpSV backward scratch=" << sz / (1024.0 * 1024.0)
              << " MiB\n";
    cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
        tp.mat_U, tp.vec_tmp, tp.vec_z,
        CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_UT, tp.d_sv_UT_buf.ptr);

    log_mem("after ILU0 setup");
    std::clog << "[cuda-pcg] ILU0 preconditioner setup successful\n";
    return true;
}

// ── Context ───────────────────────────────────────────────────────────────────

struct CudaPCGContext {
    cublasHandle_t   cublas   = nullptr;
    cusparseHandle_t cusparse = nullptr;
    std::string      device_name;
    double           tolerance = 1e-8;
    int              max_iters = 10000;

    int    last_iters   = 0;
    double last_rel_res = 0.0;
};

// ── Constructor / destructor ──────────────────────────────────────────────────

CudaPCGSolverBackend::CudaPCGSolverBackend(std::unique_ptr<CudaPCGContext> ctx) noexcept
    : ctx_(std::move(ctx)) {}

CudaPCGSolverBackend::~CudaPCGSolverBackend() {
    if (!ctx_) return;
    if (ctx_->cusparse) cusparseDestroy(ctx_->cusparse);
    if (ctx_->cublas)   cublasDestroy(ctx_->cublas);
}

CudaPCGSolverBackend::CudaPCGSolverBackend(CudaPCGSolverBackend&&) noexcept = default;
CudaPCGSolverBackend& CudaPCGSolverBackend::operator=(CudaPCGSolverBackend&&) noexcept = default;

// ── Factory ───────────────────────────────────────────────────────────────────

std::optional<CudaPCGSolverBackend>
CudaPCGSolverBackend::try_create(double tolerance, int max_iters) noexcept {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        return std::nullopt;
    if (cudaSetDevice(0) != cudaSuccess)
        return std::nullopt;

    auto ctx = std::make_unique<CudaPCGContext>();

    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, 0) == cudaSuccess)
        ctx->device_name = props.name;

    ctx->tolerance = tolerance;
    ctx->max_iters = (max_iters > 0) ? max_iters : 10000;

    if (cublasCreate(&ctx->cublas) != CUBLAS_STATUS_SUCCESS)
        return std::nullopt;
    if (cusparseCreate(&ctx->cusparse) != CUSPARSE_STATUS_SUCCESS) {
        cublasDestroy(ctx->cublas);
        return std::nullopt;
    }

    return CudaPCGSolverBackend(std::move(ctx));
}

// ── Accessors ─────────────────────────────────────────────────────────────────

std::string_view CudaPCGSolverBackend::name() const noexcept {
    return "CUDA PCG + IC0/ILU0 (GPU)";
}

int CudaPCGSolverBackend::last_iteration_count() const noexcept {
    return ctx_->last_iters;
}

double CudaPCGSolverBackend::last_relative_residual() const noexcept {
    return ctx_->last_rel_res;
}

std::string_view CudaPCGSolverBackend::device_name() const noexcept {
    return ctx_->device_name;
}

// ── solve ─────────────────────────────────────────────────────────────────────

std::vector<double>
CudaPCGSolverBackend::solve(const SparseMatrixBuilder::CsrData& K,
                             const std::vector<double>& F) {
    const int n   = K.n;
    const int nnz = K.nnz;

    if (n == 0)
        throw SolverError("CUDA PCG: stiffness matrix is empty -- no free DOFs");
    if (static_cast<int>(F.size()) != n)
        throw SolverError("CUDA PCG: force vector size " +
                          std::to_string(F.size()) + " != matrix size " +
                          std::to_string(n));

    // ── VRAM estimate ─────────────────────────────────────────────────────────
    // Compute a conservative lower-bound estimate of total device memory before
    // allocating anything.  The SpSV scratch buffers are runtime-determined (we
    // can't query them without allocating the matrix first), so they are omitted
    // from the estimate; in practice they are small (<< preconditioner copy).
    //
    // Components:
    //   Matrix CSR:      nnz*8 (values) + (n+1)*4 (row_ptr) + nnz*4 (col_ind)
    //   PCG vectors:     7 * n*8  (F, u, r, z, p, Ap, diag_inv)
    //   Preconditioner:  nnz*8 (factored copy) + n*8 (tmp vector)
    //                    [Jacobi path has no extra cost beyond diag_inv above]
    const std::size_t bytes_matrix  = static_cast<std::size_t>(nnz) * sizeof(double)
                                    + static_cast<std::size_t>(n + 1) * sizeof(int)
                                    + static_cast<std::size_t>(nnz) * sizeof(int);
    const std::size_t bytes_vectors = 7UL * static_cast<std::size_t>(n) * sizeof(double);
    const std::size_t bytes_precond = static_cast<std::size_t>(nnz) * sizeof(double)
                                    + static_cast<std::size_t>(n) * sizeof(double);
    const std::size_t bytes_estimate = bytes_matrix + bytes_vectors + bytes_precond;

    {
        std::size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        constexpr double kMiB = 1024.0 * 1024.0;
        std::clog << "[cuda-pcg] VRAM estimate (lower bound): "
                  << bytes_estimate / kMiB << " MiB"
                  << "  (matrix=" << bytes_matrix / kMiB << " MiB"
                  << ", vectors=" << bytes_vectors / kMiB << " MiB"
                  << ", precond=" << bytes_precond / kMiB << " MiB)\n"
                  << "[cuda-pcg] device: free=" << free_bytes / kMiB << " MiB"
                  << ", total=" << total_bytes / kMiB << " MiB"
                  << ", n=" << n << ", nnz=" << nnz << "\n";
        if (bytes_estimate > free_bytes)
            std::clog << "[cuda-pcg] WARNING: estimate exceeds available VRAM ("
                      << free_bytes / kMiB << " MiB free) -- solve may fail\n";
    }

    log_mem("before PCG alloc", bytes_matrix + bytes_vectors);
    std::clog << "[cuda-pcg] problem: n=" << n << ", nnz=" << nnz
              << ", matrix=" << bytes_matrix / (1024.0 * 1024.0) << " MiB"
              << ", vectors=" << bytes_vectors / (1024.0 * 1024.0) << " MiB\n";

    PCGDeviceBuffer<double> d_values(nnz);
    PCGDeviceBuffer<int>    d_row_ptr(n + 1);
    PCGDeviceBuffer<int>    d_col_ind(nnz);
    PCGDeviceBuffer<double> d_F(n);
    PCGDeviceBuffer<double> d_u(n);
    PCGDeviceBuffer<double> d_r(n);
    PCGDeviceBuffer<double> d_z(n);
    PCGDeviceBuffer<double> d_p(n);
    PCGDeviceBuffer<double> d_Ap(n);

    d_values.upload(K.values.data(),   nnz);
    d_row_ptr.upload(K.row_ptr.data(), n + 1);
    d_col_ind.upload(K.col_ind.data(), nnz);
    d_F.upload(F.data(), n);
    d_u.zero(n);

    // ── Jacobi diagonal (fallback preconditioner and zero-pivot detection) ────
    std::vector<double> diag_inv(n, 1.0);
    for (int i = 0; i < n; ++i) {
        for (int j = K.row_ptr[i]; j < K.row_ptr[i + 1]; ++j) {
            if (K.col_ind[j] == i) {
                double kii = K.values[j];
                if (kii == 0.0)
                    throw SolverError(
                        "CUDA PCG: zero diagonal at row " + std::to_string(i) +
                        " -- matrix is singular. Check boundary conditions.");
                diag_inv[i] = 1.0 / kii;
                break;
            }
        }
    }
    PCGDeviceBuffer<double> d_diag_inv(n);
    d_diag_inv.upload(diag_inv.data(), n);

    // ── Preconditioner setup (IC0 → ILU0 → Jacobi) ───────────────────────────
    PrecondKind precond_kind = PrecondKind::Jacobi;
    std::unique_ptr<TriangPrecond> tp;

    {
        auto try_tp = std::make_unique<TriangPrecond>();
        if (setup_ic0(ctx_->cusparse, n, nnz,
                      d_row_ptr.ptr, d_col_ind.ptr, d_values.ptr,
                      d_r.ptr, d_z.ptr, *try_tp)) {
            precond_kind = PrecondKind::IC0;
            tp = std::move(try_tp);
        } else {
            auto try_ilu = std::make_unique<TriangPrecond>();
            if (setup_ilu0(ctx_->cusparse, n, nnz,
                           d_row_ptr.ptr, d_col_ind.ptr, d_values.ptr,
                           d_r.ptr, d_z.ptr, *try_ilu)) {
                precond_kind = PrecondKind::ILU0;
                tp = std::move(try_ilu);
            } else {
                std::clog << "[cuda-pcg] Using Jacobi preconditioner\n";
            }
        }
    }

    const char* precond_name =
        precond_kind == PrecondKind::IC0   ? "IC0"   :
        precond_kind == PrecondKind::ILU0  ? "ILU0"  : "Jacobi";

    // ── cuSPARSE SpMV setup ───────────────────────────────────────────────────
    cusparseSpMatDescr_t mat_K = nullptr;
    cusparseCreateCsr(&mat_K,
        static_cast<int64_t>(n), static_cast<int64_t>(n),
        static_cast<int64_t>(nnz),
        d_row_ptr.ptr, d_col_ind.ptr, d_values.ptr,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    struct MatKGuard {
        cusparseSpMatDescr_t h;
        ~MatKGuard() { if (h) cusparseDestroySpMat(h); }
    } mat_K_guard{mat_K};

    cusparseDnVecDescr_t vec_p  = nullptr;
    cusparseDnVecDescr_t vec_Ap = nullptr;
    cusparseCreateDnVec(&vec_p,  n, d_p.ptr,  CUDA_R_64F);
    cusparseCreateDnVec(&vec_Ap, n, d_Ap.ptr, CUDA_R_64F);
    struct DnVecGuard {
        cusparseDnVecDescr_t h;
        ~DnVecGuard() { if (h) cusparseDestroyDnVec(h); }
    } vp_guard{vec_p}, vap_guard{vec_Ap};

    const double one  = 1.0;
    const double zero = 0.0;

    std::size_t spmv_sz = 0;
    cusparseSpMV_bufferSize(ctx_->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one, mat_K, vec_p, &zero, vec_Ap,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_sz);
    std::clog << "[cuda-pcg] SpMV scratch=" << spmv_sz / (1024.0 * 1024.0) << " MiB\n";
    log_mem("after all PCG allocs");
    PCGDeviceBuffer<char> d_spmv_buf(spmv_sz > 0 ? spmv_sz : 1);

    // ── Apply-preconditioner helper ───────────────────────────────────────────
    // Computes z = M^{-1} r using the stored dense-vector descriptors.
    // The descriptors tp->vec_r and tp->vec_z point to d_r.ptr and d_z.ptr
    // respectively, so they always reflect the current residual and output.
    auto apply_precond = [&]() {
        if (precond_kind != PrecondKind::Jacobi) {
            // Forward: L * tmp = r
            cusparseSpSV_solve(ctx_->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one, tp->mat_L, tp->vec_r, tp->vec_tmp,
                CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, tp->sv_L);

            // Backward: L^T * z = tmp (IC0) or U * z = tmp (ILU0)
            if (precond_kind == PrecondKind::IC0) {
                cusparseSpSV_solve(ctx_->cusparse, CUSPARSE_OPERATION_TRANSPOSE,
                    &one, tp->mat_L, tp->vec_tmp, tp->vec_z,
                    CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, tp->sv_UT);
            } else {
                cusparseSpSV_solve(ctx_->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one, tp->mat_U, tp->vec_tmp, tp->vec_z,
                    CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, tp->sv_UT);
            }
        } else {
            launch_jacobi(d_r.ptr, d_diag_inv.ptr, d_z.ptr, n);
        }
    };

    // ── PCG initialisation ────────────────────────────────────────────────────
    // r = F  (u0 = 0)
    cudaMemcpy(d_r.ptr, d_F.ptr,
               static_cast<std::size_t>(n) * sizeof(double),
               cudaMemcpyDeviceToDevice);

    apply_precond(); // z = M^{-1} r

    // p = z
    cudaMemcpy(d_p.ptr, d_z.ptr,
               static_cast<std::size_t>(n) * sizeof(double),
               cudaMemcpyDeviceToDevice);

    double rz = 0.0;
    cublasDdot(ctx_->cublas, n, d_r.ptr, 1, d_z.ptr, 1, &rz);

    const double rz0 = rz;
    if (rz0 == 0.0) {
        ctx_->last_iters   = 0;
        ctx_->last_rel_res = 0.0;
        return std::vector<double>(n, 0.0);
    }

    // ── PCG iteration ─────────────────────────────────────────────────────────
    const double tol       = ctx_->tolerance;
    const int    max_iters = ctx_->max_iters;

    int iter = 0;
    for (; iter < max_iters; ++iter) {
        // Update vec_p to point to current d_p (handles are fixed).
        cusparseDnVecSetValues(vec_p,  d_p.ptr);
        cusparseDnVecSetValues(vec_Ap, d_Ap.ptr);

        // Ap = K * p
        cusparseStatus_t cs = cusparseSpMV(
            ctx_->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, mat_K, vec_p, &zero, vec_Ap,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buf.ptr);
        if (cs != CUSPARSE_STATUS_SUCCESS)
            throw SolverError("CUDA PCG: cusparseSpMV failed at iteration " +
                              std::to_string(iter));

        // pAp = p · Ap
        double pAp = 0.0;
        cublasDdot(ctx_->cublas, n, d_p.ptr, 1, d_Ap.ptr, 1, &pAp);

        if (pAp <= 0.0)
            throw SolverError(
                "CUDA PCG: non-positive p·Ap=" + std::to_string(pAp) +
                " at iteration " + std::to_string(iter) +
                " -- matrix may not be positive definite. "
                "Check boundary conditions (SPCs).");

        const double alpha     =  rz / pAp;
        const double neg_alpha = -alpha;

        cublasDaxpy(ctx_->cublas, n, &alpha,     d_p.ptr,  1, d_u.ptr, 1);
        cublasDaxpy(ctx_->cublas, n, &neg_alpha, d_Ap.ptr, 1, d_r.ptr, 1);

        apply_precond(); // z = M^{-1} r

        double rz_new = 0.0;
        cublasDdot(ctx_->cublas, n, d_r.ptr, 1, d_z.ptr, 1, &rz_new);

        const double rel = std::sqrt(std::abs(rz_new) / rz0);
        if (rel < tol) {
            rz = rz_new;
            ++iter;
            break;
        }

        const double beta = rz_new / rz;
        rz = rz_new;

        // p = z + beta * p
        launch_axpby(d_z.ptr, 1.0, beta, d_p.ptr, n);
    }

    cudaDeviceSynchronize();

    ctx_->last_iters   = iter;
    ctx_->last_rel_res = std::sqrt(std::abs(rz) / rz0);

    if (iter >= max_iters)
        throw SolverError(
            "CUDA PCG: did not converge after " + std::to_string(max_iters) +
            " iterations (relative residual " +
            std::to_string(ctx_->last_rel_res) + " > tolerance " +
            std::to_string(tol) +
            "). Consider increasing max iterations or using a direct solver.");

    std::clog << "[cuda-pcg] " << precond_name
              << " converged in " << ctx_->last_iters
              << " iterations, rel_res=" << ctx_->last_rel_res
              << ", n=" << n << ", nnz=" << nnz
              << ", device='" << ctx_->device_name << "'\n";

    std::vector<double> u(n);
    d_u.download(u.data(), n);
    return u;
}

} // namespace nastran
