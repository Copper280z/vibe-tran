// src/solver/cuda_pcg_solver_backend.cu
// CUDA Preconditioned Conjugate Gradient solver backend.
//
// Scalar precision:
//   Double precision (default): all device buffers and arithmetic use float64.
//   Single precision (--cuda-single-precision): all device buffers and
//   arithmetic use float32, halving VRAM for every allocation (matrix, IC0/ILU0
//   factor copy, SpSV scratch, PCG vectors). Input K and F are downcast to
//   float32 before the solve; the result is upcast back to double64. Achievable
//   accuracy is limited by float32 roundoff (~1e-7).
//
// Preconditioner selection (tried in order):
//   1. IC0 (Incomplete Cholesky, zero fill-in) via cusparseTcsric02.
//      Optimal for SPD FEM stiffness matrices; reduces iteration count 10-100×
//      vs Jacobi.  Factorization is in-place on a copy of K; apply = two
//      triangular solves with cusparseSpSV (forward L, backward L^T).
//   2. ILU0 (Incomplete LU, zero fill-in) via cusparseTcsrilu02.
//      Used when IC0 setup fails (zero pivot, non-SPD matrix).
//   3. Jacobi (diagonal scaling).  Always succeeds; weakest preconditioner.
//
// PCG loop (executed entirely in type T):
//   r = F, z = M^{-1}r, p = z, rz = r·z
//   while not converged:
//     Ap = K*p  (cusparseSpMV)
//     alpha = rz / (p·Ap)
//     u += alpha*p, r -= alpha*Ap
//     z = M^{-1}r
//     rz_new = r·z
//     if sqrt(rz_new/rz0) < tol: done
//     p = z + (rz_new/rz)*p
//     rz = rz_new
//
// Dense vector descriptors (vec_r, vec_tmp, vec_z) are created during
// preconditioner setup and reused in every apply call — cusparseSpSV requires
// the same descriptor handles in solve() as were used in analysis().
//
// Note: <format> / std::format intentionally avoided — nvcc uses its bundled
// g++-12 as host compiler, which does not provide <format>.
#define HAVE_CUDA 1

#include "solver/cuda_pcg_solver_backend.hpp"
#include "core/exceptions.hpp"
#include "core/logger.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <cmath>
#include <string>
#include <vector>

// ── Memory logging helper ─────────────────────────────────────────────────────

namespace {

static void log_mem(const char* label, std::size_t extra_bytes = 0) {
    std::size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    constexpr std::size_t kMiB = 1024UL * 1024UL;
    std::string msg = std::string("[cuda-pcg] ") + label +
                      ": free=" + std::to_string(free_bytes / kMiB) + " MiB"
                      ", total=" + std::to_string(total_bytes / kMiB) + " MiB";
    if (extra_bytes > 0)
        msg += ", allocating=" + std::to_string(extra_bytes / kMiB) + " MiB";
    vibestran::log_debug(msg);
}

} // anonymous namespace

namespace vibestran {

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
                " elements (" + std::to_string(count * sizeof(T)) +
                " bytes): " + cudaGetErrorString(err));
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

// ── Scalar type traits ────────────────────────────────────────────────────────
// Dispatch to the correct cuBLAS and cuSPARSE type-specific functions.

template<typename T> struct ScalarTraits;

template<> struct ScalarTraits<double> {
    static constexpr cudaDataType_t cuda_dtype = CUDA_R_64F;
    static const char* name() { return "float64"; }

    static cublasStatus_t dot(cublasHandle_t h, int n,
        const double* x, const double* y, double* r) {
        return cublasDdot(h, n, x, 1, y, 1, r);
    }
    static cublasStatus_t axpy(cublasHandle_t h, int n,
        const double* alpha, const double* x, double* y) {
        return cublasDaxpy(h, n, alpha, x, 1, y, 1);
    }

    static cusparseStatus_t csric02_buffer_size(
        cusparseHandle_t h, int m, int nnz, const cusparseMatDescr_t d,
        double* vals, const int* row, const int* col,
        csric02Info_t info, int* buf_sz) {
        return cusparseDcsric02_bufferSize(h, m, nnz, d, vals, row, col, info, buf_sz);
    }
    static cusparseStatus_t csric02_analysis(
        cusparseHandle_t h, int m, int nnz, const cusparseMatDescr_t d,
        double* vals, const int* row, const int* col,
        csric02Info_t info, cusparseSolvePolicy_t pol, void* buf) {
        return cusparseDcsric02_analysis(h, m, nnz, d, vals, row, col, info, pol, buf);
    }
    static cusparseStatus_t csric02(
        cusparseHandle_t h, int m, int nnz, const cusparseMatDescr_t d,
        double* vals, const int* row, const int* col,
        csric02Info_t info, cusparseSolvePolicy_t pol, void* buf) {
        return cusparseDcsric02(h, m, nnz, d, vals, row, col, info, pol, buf);
    }

    static cusparseStatus_t csrilu02_buffer_size(
        cusparseHandle_t h, int m, int nnz, const cusparseMatDescr_t d,
        double* vals, const int* row, const int* col,
        csrilu02Info_t info, int* buf_sz) {
        return cusparseDcsrilu02_bufferSize(h, m, nnz, d, vals, row, col, info, buf_sz);
    }
    static cusparseStatus_t csrilu02_analysis(
        cusparseHandle_t h, int m, int nnz, const cusparseMatDescr_t d,
        double* vals, const int* row, const int* col,
        csrilu02Info_t info, cusparseSolvePolicy_t pol, void* buf) {
        return cusparseDcsrilu02_analysis(h, m, nnz, d, vals, row, col, info, pol, buf);
    }
    static cusparseStatus_t csrilu02(
        cusparseHandle_t h, int m, int nnz, const cusparseMatDescr_t d,
        double* vals, const int* row, const int* col,
        csrilu02Info_t info, cusparseSolvePolicy_t pol, void* buf) {
        return cusparseDcsrilu02(h, m, nnz, d, vals, row, col, info, pol, buf);
    }
};

template<> struct ScalarTraits<float> {
    static constexpr cudaDataType_t cuda_dtype = CUDA_R_32F;
    static const char* name() { return "float32"; }

    static cublasStatus_t dot(cublasHandle_t h, int n,
        const float* x, const float* y, float* r) {
        return cublasSdot(h, n, x, 1, y, 1, r);
    }
    static cublasStatus_t axpy(cublasHandle_t h, int n,
        const float* alpha, const float* x, float* y) {
        return cublasSaxpy(h, n, alpha, x, 1, y, 1);
    }

    static cusparseStatus_t csric02_buffer_size(
        cusparseHandle_t h, int m, int nnz, const cusparseMatDescr_t d,
        float* vals, const int* row, const int* col,
        csric02Info_t info, int* buf_sz) {
        return cusparseScsric02_bufferSize(h, m, nnz, d, vals, row, col, info, buf_sz);
    }
    static cusparseStatus_t csric02_analysis(
        cusparseHandle_t h, int m, int nnz, const cusparseMatDescr_t d,
        float* vals, const int* row, const int* col,
        csric02Info_t info, cusparseSolvePolicy_t pol, void* buf) {
        return cusparseScsric02_analysis(h, m, nnz, d, vals, row, col, info, pol, buf);
    }
    static cusparseStatus_t csric02(
        cusparseHandle_t h, int m, int nnz, const cusparseMatDescr_t d,
        float* vals, const int* row, const int* col,
        csric02Info_t info, cusparseSolvePolicy_t pol, void* buf) {
        return cusparseScsric02(h, m, nnz, d, vals, row, col, info, pol, buf);
    }

    static cusparseStatus_t csrilu02_buffer_size(
        cusparseHandle_t h, int m, int nnz, const cusparseMatDescr_t d,
        float* vals, const int* row, const int* col,
        csrilu02Info_t info, int* buf_sz) {
        return cusparseScsrilu02_bufferSize(h, m, nnz, d, vals, row, col, info, buf_sz);
    }
    static cusparseStatus_t csrilu02_analysis(
        cusparseHandle_t h, int m, int nnz, const cusparseMatDescr_t d,
        float* vals, const int* row, const int* col,
        csrilu02Info_t info, cusparseSolvePolicy_t pol, void* buf) {
        return cusparseScsrilu02_analysis(h, m, nnz, d, vals, row, col, info, pol, buf);
    }
    static cusparseStatus_t csrilu02(
        cusparseHandle_t h, int m, int nnz, const cusparseMatDescr_t d,
        float* vals, const int* row, const int* col,
        csrilu02Info_t info, cusparseSolvePolicy_t pol, void* buf) {
        return cusparseScsrilu02(h, m, nnz, d, vals, row, col, info, pol, buf);
    }
};

// ── Custom CUDA kernels ───────────────────────────────────────────────────────

template<typename T>
__global__ static void jacobi_kernel(
    const T* __restrict__ r,
    const T* __restrict__ d_inv,
    T* __restrict__ z,
    int n)
{
    int i = static_cast<int>(blockIdx.x) * blockDim.x +
            static_cast<int>(threadIdx.x);
    if (i < n) z[i] = r[i] * d_inv[i];
}

template<typename T>
__global__ static void axpby_kernel(
    const T* __restrict__ x,
    T alpha,
    T beta,
    T* __restrict__ y,
    int n)
{
    int i = static_cast<int>(blockIdx.x) * blockDim.x +
            static_cast<int>(threadIdx.x);
    if (i < n) y[i] = alpha * x[i] + beta * y[i];
}

static constexpr int kBlock = 256;

template<typename T>
static void launch_jacobi(const T* r, const T* d_inv, T* z, int n) {
    jacobi_kernel<T><<<(n + kBlock - 1) / kBlock, kBlock>>>(r, d_inv, z, n);
}

template<typename T>
static void launch_axpby(const T* x, T a, T b, T* y, int n) {
    axpby_kernel<T><<<(n + kBlock - 1) / kBlock, kBlock>>>(x, a, b, y, n);
}

// ── Preconditioner type ───────────────────────────────────────────────────────

enum class PrecondKind { IC0, ILU0, Jacobi };

// Stores all GPU resources for a triangular (IC0 or ILU0) preconditioner.
// Dense vector descriptors (vec_r, vec_tmp, vec_z) are created during setup
// and reused in every apply call — cusparseSpSV requires the same descriptor
// handles in solve() as were used in analysis().
template<typename T>
struct TriangPrecond {
    PCGDeviceBuffer<T>    d_M_vals;     // in-place IC0 or ILU0 factored values
    PCGDeviceBuffer<char> d_factor_buf; // scratch for csric02 / csrilu02

    cusparseSpMatDescr_t mat_L = nullptr;
    cusparseSpMatDescr_t mat_U = nullptr; // ILU0 only; null for IC0

    cusparseSpSVDescr_t sv_L  = nullptr;
    cusparseSpSVDescr_t sv_UT = nullptr;
    PCGDeviceBuffer<char> d_sv_L_buf;
    PCGDeviceBuffer<char> d_sv_UT_buf;

    PCGDeviceBuffer<T> d_tmp; // intermediate: tmp = L^{-1}r

    // Dense vector descriptors held for the lifetime of this object.
    cusparseDnVecDescr_t vec_r   = nullptr; // points to PCG d_r
    cusparseDnVecDescr_t vec_tmp = nullptr; // points to d_tmp
    cusparseDnVecDescr_t vec_z   = nullptr; // points to PCG d_z

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
// Returns false on zero pivots so the caller can retry with ILU0.
template<typename T>
static bool setup_ic0(
    cusparseHandle_t cusparse,
    int n, int nnz,
    const int* d_row_ptr, const int* d_col_ind, const T* d_values,
    T* d_r, T* d_z,
    TriangPrecond<T>& tp)
{
    using Tr = ScalarTraits<T>;
    constexpr double kMiB = 1024.0 * 1024.0;

    tp.d_M_vals = PCGDeviceBuffer<T>(nnz);
    tp.d_tmp    = PCGDeviceBuffer<T>(n);

    cudaMemcpy(tp.d_M_vals.ptr, d_values,
               static_cast<std::size_t>(nnz) * sizeof(T),
               cudaMemcpyDeviceToDevice);

    // GENERAL + FILL_MODE_LOWER: csric02 reads only the lower triangle.
    cusparseMatDescr_t descr = nullptr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);

    csric02Info_t info = nullptr;
    cusparseCreateCsric02Info(&info);

    int buf_size = 0;
    cusparseStatus_t cs = Tr::csric02_buffer_size(
        cusparse, n, nnz, descr,
        tp.d_M_vals.ptr, d_row_ptr, d_col_ind, info, &buf_size);
    if (cs != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroyCsric02Info(info);
        cusparseDestroyMatDescr(descr);
        throw SolverError("CUDA PCG: IC0 bufferSize failed, status=" +
                          std::to_string(static_cast<int>(cs)));
    }

    const std::size_t precond_bytes =
        static_cast<std::size_t>(nnz) * sizeof(T)   // d_M_vals
      + static_cast<std::size_t>(n)   * sizeof(T)   // d_tmp
      + static_cast<std::size_t>(buf_size > 0 ? buf_size : 1);
    log_mem("IC0 factor alloc", precond_bytes);
    vibestran::log_debug("[cuda-pcg] IC0 factor scratch=" +
                        std::to_string(static_cast<std::size_t>(buf_size) / kMiB) + " MiB");

    tp.d_factor_buf = PCGDeviceBuffer<char>(buf_size > 0 ? buf_size : 1);

    Tr::csric02_analysis(cusparse, n, nnz, descr,
        tp.d_M_vals.ptr, d_row_ptr, d_col_ind, info,
        CUSPARSE_SOLVE_POLICY_NO_LEVEL, tp.d_factor_buf.ptr);

    int structural_zero = -1;
    cusparseXcsric02_zeroPivot(cusparse, info, &structural_zero);
    if (structural_zero >= 0 || cs != CUSPARSE_STATUS_SUCCESS) {
        vibestran::log_warn("[cuda-pcg] IC0: structural zero at row " +
                           std::to_string(structural_zero) + " -- retrying with ILU0");
        cusparseDestroyCsric02Info(info);
        cusparseDestroyMatDescr(descr);
        return false;
    }

    cs = Tr::csric02(cusparse, n, nnz, descr,
        tp.d_M_vals.ptr, d_row_ptr, d_col_ind, info,
        CUSPARSE_SOLVE_POLICY_NO_LEVEL, tp.d_factor_buf.ptr);

    int numerical_zero = -1;
    cusparseXcsric02_zeroPivot(cusparse, info, &numerical_zero);
    cusparseDestroyCsric02Info(info);
    cusparseDestroyMatDescr(descr);

    if (numerical_zero >= 0 || cs != CUSPARSE_STATUS_SUCCESS) {
        vibestran::log_warn("[cuda-pcg] IC0: numerical zero at row " +
                           std::to_string(numerical_zero) + " -- retrying with ILU0");
        return false;
    }

    // ── SpSV setup ────────────────────────────────────────────────────────────
    cusparseCreateCsr(&tp.mat_L,
        static_cast<int64_t>(n), static_cast<int64_t>(n),
        static_cast<int64_t>(nnz),
        const_cast<int*>(d_row_ptr), const_cast<int*>(d_col_ind),
        tp.d_M_vals.ptr,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, Tr::cuda_dtype);
    {
        cusparseFillMode_t fill_L  = CUSPARSE_FILL_MODE_LOWER;
        cusparseDiagType_t diag_nu = CUSPARSE_DIAG_TYPE_NON_UNIT;
        cusparseSpMatSetAttribute(tp.mat_L, CUSPARSE_SPMAT_FILL_MODE,
                                  &fill_L,  sizeof(fill_L));
        cusparseSpMatSetAttribute(tp.mat_L, CUSPARSE_SPMAT_DIAG_TYPE,
                                  &diag_nu, sizeof(diag_nu));
    }
    tp.mat_U = nullptr;

    cusparseCreateDnVec(&tp.vec_r,   n, d_r,          Tr::cuda_dtype);
    cusparseCreateDnVec(&tp.vec_tmp, n, tp.d_tmp.ptr, Tr::cuda_dtype);
    cusparseCreateDnVec(&tp.vec_z,   n, d_z,          Tr::cuda_dtype);

    cusparseSpSV_createDescr(&tp.sv_L);
    cusparseSpSV_createDescr(&tp.sv_UT);

    const T one{1};

    std::size_t sz = 0;
    cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
        tp.mat_L, tp.vec_r, tp.vec_tmp,
        Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_L, &sz);
    vibestran::log_debug("[cuda-pcg] IC0 SpSV forward scratch=" +
                        std::to_string(sz / kMiB) + " MiB");
    tp.d_sv_L_buf = PCGDeviceBuffer<char>(sz > 0 ? sz : 1);
    cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
        tp.mat_L, tp.vec_r, tp.vec_tmp,
        Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_L, tp.d_sv_L_buf.ptr);

    sz = 0;
    cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_TRANSPOSE, &one,
        tp.mat_L, tp.vec_tmp, tp.vec_z,
        Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_UT, &sz);
    vibestran::log_debug("[cuda-pcg] IC0 SpSV backward scratch=" +
                        std::to_string(sz / kMiB) + " MiB");
    tp.d_sv_UT_buf = PCGDeviceBuffer<char>(sz > 0 ? sz : 1);
    cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_TRANSPOSE, &one,
        tp.mat_L, tp.vec_tmp, tp.vec_z,
        Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_UT, tp.d_sv_UT_buf.ptr);

    log_mem("after IC0 setup");
    vibestran::log_debug("[cuda-pcg] IC0 preconditioner setup successful");
    return true;
}

// ── ILU0 factorization and SpSV setup ────────────────────────────────────────
template<typename T>
static bool setup_ilu0(
    cusparseHandle_t cusparse,
    int n, int nnz,
    const int* d_row_ptr, const int* d_col_ind, const T* d_values,
    T* d_r, T* d_z,
    TriangPrecond<T>& tp)
{
    using Tr = ScalarTraits<T>;
    constexpr double kMiB = 1024.0 * 1024.0;

    tp.d_M_vals = PCGDeviceBuffer<T>(nnz);
    tp.d_tmp    = PCGDeviceBuffer<T>(n);

    cudaMemcpy(tp.d_M_vals.ptr, d_values,
               static_cast<std::size_t>(nnz) * sizeof(T),
               cudaMemcpyDeviceToDevice);

    cusparseMatDescr_t descr = nullptr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    csrilu02Info_t info = nullptr;
    cusparseCreateCsrilu02Info(&info);

    int buf_size = 0;
    cusparseStatus_t cs = Tr::csrilu02_buffer_size(
        cusparse, n, nnz, descr,
        tp.d_M_vals.ptr, d_row_ptr, d_col_ind, info, &buf_size);
    if (cs != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroyCsrilu02Info(info);
        cusparseDestroyMatDescr(descr);
        throw SolverError("CUDA PCG: ILU0 bufferSize failed, status=" +
                          std::to_string(static_cast<int>(cs)));
    }

    const std::size_t precond_bytes =
        static_cast<std::size_t>(nnz) * sizeof(T)
      + static_cast<std::size_t>(n)   * sizeof(T)
      + static_cast<std::size_t>(buf_size > 0 ? buf_size : 1);
    log_mem("ILU0 factor alloc", precond_bytes);
    vibestran::log_debug("[cuda-pcg] ILU0 factor scratch=" +
                        std::to_string(static_cast<std::size_t>(buf_size) / kMiB) + " MiB");

    tp.d_factor_buf = PCGDeviceBuffer<char>(buf_size > 0 ? buf_size : 1);

    Tr::csrilu02_analysis(cusparse, n, nnz, descr,
        tp.d_M_vals.ptr, d_row_ptr, d_col_ind, info,
        CUSPARSE_SOLVE_POLICY_NO_LEVEL, tp.d_factor_buf.ptr);

    int structural_zero = -1;
    cusparseXcsrilu02_zeroPivot(cusparse, info, &structural_zero);
    if (structural_zero >= 0) {
        vibestran::log_warn("[cuda-pcg] ILU0: structural zero at row " +
                           std::to_string(structural_zero) + " -- falling back to Jacobi");
        cusparseDestroyCsrilu02Info(info);
        cusparseDestroyMatDescr(descr);
        return false;
    }

    cs = Tr::csrilu02(cusparse, n, nnz, descr,
        tp.d_M_vals.ptr, d_row_ptr, d_col_ind, info,
        CUSPARSE_SOLVE_POLICY_NO_LEVEL, tp.d_factor_buf.ptr);

    int numerical_zero = -1;
    cusparseXcsrilu02_zeroPivot(cusparse, info, &numerical_zero);
    cusparseDestroyCsrilu02Info(info);
    cusparseDestroyMatDescr(descr);

    if (numerical_zero >= 0 || cs != CUSPARSE_STATUS_SUCCESS) {
        vibestran::log_warn("[cuda-pcg] ILU0: numerical zero at row " +
                           std::to_string(numerical_zero) + " -- falling back to Jacobi");
        return false;
    }

    // ── SpSV setup ────────────────────────────────────────────────────────────
    cusparseCreateCsr(&tp.mat_L,
        static_cast<int64_t>(n), static_cast<int64_t>(n),
        static_cast<int64_t>(nnz),
        const_cast<int*>(d_row_ptr), const_cast<int*>(d_col_ind),
        tp.d_M_vals.ptr,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, Tr::cuda_dtype);
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
        CUSPARSE_INDEX_BASE_ZERO, Tr::cuda_dtype);
    {
        cusparseFillMode_t fill_U  = CUSPARSE_FILL_MODE_UPPER;
        cusparseDiagType_t diag_nu = CUSPARSE_DIAG_TYPE_NON_UNIT;
        cusparseSpMatSetAttribute(tp.mat_U, CUSPARSE_SPMAT_FILL_MODE,
                                  &fill_U,  sizeof(fill_U));
        cusparseSpMatSetAttribute(tp.mat_U, CUSPARSE_SPMAT_DIAG_TYPE,
                                  &diag_nu, sizeof(diag_nu));
    }

    cusparseCreateDnVec(&tp.vec_r,   n, d_r,          Tr::cuda_dtype);
    cusparseCreateDnVec(&tp.vec_tmp, n, tp.d_tmp.ptr, Tr::cuda_dtype);
    cusparseCreateDnVec(&tp.vec_z,   n, d_z,          Tr::cuda_dtype);

    cusparseSpSV_createDescr(&tp.sv_L);
    cusparseSpSV_createDescr(&tp.sv_UT);

    const T one{1};

    std::size_t sz = 0;
    cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
        tp.mat_L, tp.vec_r, tp.vec_tmp,
        Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_L, &sz);
    vibestran::log_debug("[cuda-pcg] ILU0 SpSV forward scratch=" +
                        std::to_string(sz / kMiB) + " MiB");
    tp.d_sv_L_buf = PCGDeviceBuffer<char>(sz > 0 ? sz : 1);
    cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
        tp.mat_L, tp.vec_r, tp.vec_tmp,
        Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_L, tp.d_sv_L_buf.ptr);

    sz = 0;
    cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
        tp.mat_U, tp.vec_tmp, tp.vec_z,
        Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_UT, &sz);
    vibestran::log_debug("[cuda-pcg] ILU0 SpSV backward scratch=" +
                        std::to_string(sz / kMiB) + " MiB");
    tp.d_sv_UT_buf = PCGDeviceBuffer<char>(sz > 0 ? sz : 1);
    cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
        tp.mat_U, tp.vec_tmp, tp.vec_z,
        Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_UT, tp.d_sv_UT_buf.ptr);

    log_mem("after ILU0 setup");
    vibestran::log_debug("[cuda-pcg] ILU0 preconditioner setup successful");
    return true;
}

// ── Typed PCG solve ───────────────────────────────────────────────────────────
// Performs the full PCG iteration in scalar type T.  K values and F must
// already be on the host in type T (caller handles downcasting for float32).
// Returns the solution as std::vector<double> (upcasting from T if needed).
template<typename T>
static std::vector<double> solve_pcg(
    cublasHandle_t   cublas,
    cusparseHandle_t cusparse,
    const std::string& device_name,
    int n, int nnz,
    const std::vector<T>& h_values,
    const std::vector<int>& h_row_ptr,
    const std::vector<int>& h_col_ind,
    const std::vector<T>& h_F,
    const std::vector<T>& h_diag_inv,
    double tolerance,
    int max_iters,
    int& out_iters,
    double& out_rel_res)
{
    using Tr = ScalarTraits<T>;
    constexpr double kMiB = 1024.0 * 1024.0;

    // ── Device allocation and upload ─────────────────────────────────────────
    const std::size_t bytes_matrix  = static_cast<std::size_t>(nnz) * sizeof(T)
                                    + static_cast<std::size_t>(n + 1) * sizeof(int)
                                    + static_cast<std::size_t>(nnz) * sizeof(int);
    const std::size_t bytes_vectors = 7UL * static_cast<std::size_t>(n) * sizeof(T);
    const std::size_t bytes_precond = static_cast<std::size_t>(nnz) * sizeof(T)
                                    + static_cast<std::size_t>(n) * sizeof(T);
    const std::size_t bytes_estimate = bytes_matrix + bytes_vectors + bytes_precond;

    {
        std::size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        vibestran::log_debug(
            std::string("[cuda-pcg] VRAM estimate (lower bound, ") + Tr::name() + "): " +
            std::to_string(bytes_estimate / kMiB) + " MiB"
            "  (matrix=" + std::to_string(bytes_matrix / kMiB) + " MiB"
            ", vectors=" + std::to_string(bytes_vectors / kMiB) + " MiB"
            ", precond=" + std::to_string(bytes_precond / kMiB) + " MiB)");
        vibestran::log_debug(
            "[cuda-pcg] device: free=" + std::to_string(free_bytes / kMiB) + " MiB"
            ", total=" + std::to_string(total_bytes / kMiB) + " MiB"
            ", n=" + std::to_string(n) + ", nnz=" + std::to_string(nnz));
        if (bytes_estimate > free_bytes)
            vibestran::log_warn(
                "[cuda-pcg] WARNING: estimate exceeds available VRAM (" +
                std::to_string(free_bytes / kMiB) + " MiB free) -- solve may fail");
    }

    log_mem("before PCG alloc", bytes_matrix + bytes_vectors);
    vibestran::log_debug("[cuda-pcg] matrix=" + std::to_string(bytes_matrix / kMiB) + " MiB"
                        ", vectors=" + std::to_string(bytes_vectors / kMiB) + " MiB");

    PCGDeviceBuffer<T>   d_values(nnz);
    PCGDeviceBuffer<int> d_row_ptr(n + 1);
    PCGDeviceBuffer<int> d_col_ind(nnz);
    PCGDeviceBuffer<T>   d_F(n);
    PCGDeviceBuffer<T>   d_u(n);
    PCGDeviceBuffer<T>   d_r(n);
    PCGDeviceBuffer<T>   d_z(n);
    PCGDeviceBuffer<T>   d_p(n);
    PCGDeviceBuffer<T>   d_Ap(n);

    d_values.upload(h_values.data(),   nnz);
    d_row_ptr.upload(h_row_ptr.data(), n + 1);
    d_col_ind.upload(h_col_ind.data(), nnz);
    d_F.upload(h_F.data(), n);
    d_u.zero(n);

    PCGDeviceBuffer<T> d_diag_inv(n);
    d_diag_inv.upload(h_diag_inv.data(), n);

    // ── Preconditioner setup (IC0 → ILU0 → Jacobi) ───────────────────────────
    PrecondKind precond_kind = PrecondKind::Jacobi;
    std::unique_ptr<TriangPrecond<T>> tp;

    {
        auto try_tp = std::make_unique<TriangPrecond<T>>();
        if (setup_ic0<T>(cusparse, n, nnz,
                         d_row_ptr.ptr, d_col_ind.ptr, d_values.ptr,
                         d_r.ptr, d_z.ptr, *try_tp)) {
            precond_kind = PrecondKind::IC0;
            tp = std::move(try_tp);
        } else {
            auto try_ilu = std::make_unique<TriangPrecond<T>>();
            if (setup_ilu0<T>(cusparse, n, nnz,
                              d_row_ptr.ptr, d_col_ind.ptr, d_values.ptr,
                              d_r.ptr, d_z.ptr, *try_ilu)) {
                precond_kind = PrecondKind::ILU0;
                tp = std::move(try_ilu);
            } else {
                vibestran::log_debug("[cuda-pcg] Using Jacobi preconditioner");
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
        CUSPARSE_INDEX_BASE_ZERO, Tr::cuda_dtype);
    struct MatKGuard {
        cusparseSpMatDescr_t h;
        ~MatKGuard() { if (h) cusparseDestroySpMat(h); }
    } mat_K_guard{mat_K};

    cusparseDnVecDescr_t vec_p  = nullptr;
    cusparseDnVecDescr_t vec_Ap = nullptr;
    cusparseCreateDnVec(&vec_p,  n, d_p.ptr,  Tr::cuda_dtype);
    cusparseCreateDnVec(&vec_Ap, n, d_Ap.ptr, Tr::cuda_dtype);
    struct DnVecGuard {
        cusparseDnVecDescr_t h;
        ~DnVecGuard() { if (h) cusparseDestroyDnVec(h); }
    } vp_guard{vec_p}, vap_guard{vec_Ap};

    const T one{1};
    const T zero{0};

    std::size_t spmv_sz = 0;
    cusparseSpMV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one, mat_K, vec_p, &zero, vec_Ap,
        Tr::cuda_dtype, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_sz);
    vibestran::log_debug("[cuda-pcg] SpMV scratch=" + std::to_string(spmv_sz / kMiB) + " MiB");
    log_mem("after all PCG allocs");
    PCGDeviceBuffer<char> d_spmv_buf(spmv_sz > 0 ? spmv_sz : 1);

    // ── Apply-preconditioner helper ───────────────────────────────────────────
    auto apply_precond = [&]() {
        if (precond_kind != PrecondKind::Jacobi) {
            cusparseSpSV_solve(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one, tp->mat_L, tp->vec_r, tp->vec_tmp,
                Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, tp->sv_L);
            if (precond_kind == PrecondKind::IC0) {
                cusparseSpSV_solve(cusparse, CUSPARSE_OPERATION_TRANSPOSE,
                    &one, tp->mat_L, tp->vec_tmp, tp->vec_z,
                    Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, tp->sv_UT);
            } else {
                cusparseSpSV_solve(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one, tp->mat_U, tp->vec_tmp, tp->vec_z,
                    Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, tp->sv_UT);
            }
        } else {
            launch_jacobi<T>(d_r.ptr, d_diag_inv.ptr, d_z.ptr, n);
        }
    };

    // ── PCG initialisation ────────────────────────────────────────────────────
    cudaMemcpy(d_r.ptr, d_F.ptr,
               static_cast<std::size_t>(n) * sizeof(T),
               cudaMemcpyDeviceToDevice);

    apply_precond();

    cudaMemcpy(d_p.ptr, d_z.ptr,
               static_cast<std::size_t>(n) * sizeof(T),
               cudaMemcpyDeviceToDevice);

    T rz_T{0};
    Tr::dot(cublas, n, d_r.ptr, d_z.ptr, &rz_T);
    double rz  = static_cast<double>(rz_T);
    const double rz0 = rz;

    if (rz0 == 0.0) {
        out_iters   = 0;
        out_rel_res = 0.0;
        return std::vector<double>(n, 0.0);
    }

    // ── PCG iteration ─────────────────────────────────────────────────────────
    int iter = 0;
    for (; iter < max_iters; ++iter) {
        cusparseDnVecSetValues(vec_p,  d_p.ptr);
        cusparseDnVecSetValues(vec_Ap, d_Ap.ptr);

        cusparseStatus_t cs = cusparseSpMV(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, mat_K, vec_p, &zero, vec_Ap,
            Tr::cuda_dtype, CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buf.ptr);
        if (cs != CUSPARSE_STATUS_SUCCESS)
            throw SolverError("CUDA PCG: cusparseSpMV failed at iteration " +
                              std::to_string(iter));

        T pAp_T{0};
        Tr::dot(cublas, n, d_p.ptr, d_Ap.ptr, &pAp_T);
        const double pAp = static_cast<double>(pAp_T);

        if (pAp <= 0.0) {
            vibestran::log_warn(
                "[cuda-pcg] non-positive p*Ap=" + std::to_string(pAp) +
                " at iteration " + std::to_string(iter) +
                " -- matrix may not be positive definite or"
                " preconditioner is ill-conditioned; stopping.");
            iter = max_iters; // trigger the non-convergence path
            break;
        }

        const T alpha     = static_cast<T>(rz / pAp);
        const T neg_alpha = -alpha;

        Tr::axpy(cublas, n, &alpha,     d_p.ptr,  d_u.ptr);
        Tr::axpy(cublas, n, &neg_alpha, d_Ap.ptr, d_r.ptr);

        apply_precond();

        T rz_new_T{0};
        Tr::dot(cublas, n, d_r.ptr, d_z.ptr, &rz_new_T);
        const double rz_new = static_cast<double>(rz_new_T);

        const double rel = std::sqrt(std::abs(rz_new) / rz0);
        if (rel < tolerance) {
            rz = rz_new;
            ++iter;
            break;
        }

        const T beta = static_cast<T>(rz_new / rz);
        rz = rz_new;

        launch_axpby<T>(d_z.ptr, T{1}, beta, d_p.ptr, n);
    }

    cudaDeviceSynchronize();

    out_iters   = iter;
    out_rel_res = std::sqrt(std::abs(rz) / rz0);

    if (iter >= max_iters)
        throw SolverError(
            "CUDA PCG: did not converge after " + std::to_string(max_iters) +
            " iterations (relative residual " +
            std::to_string(out_rel_res) + " > tolerance " +
            std::to_string(tolerance) +
            "). Consider increasing max iterations or using a direct solver.");

    vibestran::log_info(
        std::string("[cuda-pcg] ") + precond_name +
        " (" + Tr::name() + ") converged in " + std::to_string(out_iters) +
        " iterations, rel_res=" + std::to_string(out_rel_res) +
        ", n=" + std::to_string(n) + ", nnz=" + std::to_string(nnz) +
        ", device='" + device_name + "'");

    std::vector<double> u(n);
    if constexpr (std::is_same_v<T, double>) {
        d_u.download(u.data(), n);
    } else {
        std::vector<T> u_T(n);
        d_u.download(u_T.data(), n);
        for (int i = 0; i < n; ++i)
            u[i] = static_cast<double>(u_T[i]);
    }
    return u;
}

// ── Context ───────────────────────────────────────────────────────────────────

struct CudaPCGContext {
    cublasHandle_t   cublas   = nullptr;
    cusparseHandle_t cusparse = nullptr;
    std::string      device_name;
    double           tolerance          = 1e-8;
    int              max_iters          = 10000;
    bool             use_single_precision = false;

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
CudaPCGSolverBackend::try_create(bool use_single_precision,
                                  double tolerance,
                                  int max_iters) noexcept {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        return std::nullopt;
    if (cudaSetDevice(0) != cudaSuccess)
        return std::nullopt;

    auto ctx = std::make_unique<CudaPCGContext>();

    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, 0) == cudaSuccess)
        ctx->device_name = props.name;

    ctx->use_single_precision = use_single_precision;
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
    return ctx_->use_single_precision
        ? "CUDA PCG + IC0/ILU0 float32 (GPU)"
        : "CUDA PCG + IC0/ILU0 float64 (GPU)";
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

bool CudaPCGSolverBackend::uses_single_precision() const noexcept {
    return ctx_->use_single_precision;
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

    // Build Jacobi diagonal (host, for fallback preconditioner and zero detection).
    std::vector<double> diag_inv_d(n, 1.0);
    for (int i = 0; i < n; ++i) {
        for (int j = K.row_ptr[i]; j < K.row_ptr[i + 1]; ++j) {
            if (K.col_ind[j] == i) {
                double kii = K.values[j];
                if (kii == 0.0)
                    throw SolverError(
                        "CUDA PCG: zero diagonal at row " + std::to_string(i) +
                        " -- matrix is singular. Check boundary conditions.");
                diag_inv_d[i] = 1.0 / kii;
                break;
            }
        }
    }

    if (ctx_->use_single_precision) {
        std::vector<float> vals_f(nnz), F_f(n), diag_inv_f(n);
        for (int i = 0; i < nnz; ++i) vals_f[i]     = static_cast<float>(K.values[i]);
        for (int i = 0; i < n;   ++i) F_f[i]         = static_cast<float>(F[i]);
        for (int i = 0; i < n;   ++i) diag_inv_f[i]  = static_cast<float>(diag_inv_d[i]);

        return solve_pcg<float>(
            ctx_->cublas, ctx_->cusparse, ctx_->device_name,
            n, nnz, vals_f, K.row_ptr, K.col_ind, F_f, diag_inv_f,
            ctx_->tolerance, ctx_->max_iters,
            ctx_->last_iters, ctx_->last_rel_res);
    }

    return solve_pcg<double>(
        ctx_->cublas, ctx_->cusparse, ctx_->device_name,
        n, nnz, K.values, K.row_ptr, K.col_ind, F, diag_inv_d,
        ctx_->tolerance, ctx_->max_iters,
        ctx_->last_iters, ctx_->last_rel_res);
}

} // namespace vibestran
