// src/solver/cuda_eigensolver_backend.cu
// CUDA shift-and-invert Lanczos eigensolver for K φ = λ M φ.
//
// Algorithm overview (M-inner-product Lanczos):
//   Shift: C = K - sigma*M.
//   Operator: A = C^{-1} M,  A is M-symmetric.
//   Ritz values nu_i of A satisfy:  nu_i = 1 / (lambda_i - sigma).
//   The nd Ritz values with largest |nu| correspond to lambda nearest sigma.
//
//   1. Factorize C once on GPU via cuDSS (SPD Cholesky first, LU fallback).
//   2. Run ncv Lanczos steps with M-inner-product and full double
//      re-orthogonalization:
//        w_k   = M * v_k               (cuSPARSE SpMV)
//        z_k   = C^{-1} * w_k          (cuDSS SOLVE, reuses factorization)
//        alpha_k = w_k · z_k            (cuBLAS dot == <v_k, A v_k>_M)
//        r     = z_k - alpha_k v_k - beta_{k-1} v_{k-1}
//        [double re-orthogonalization via cuBLAS dgemv against all prior v_j]
//        w_{k+1} = M * r                (cuSPARSE SpMV)
//        beta_k  = sqrt(r · w_{k+1})    (cuBLAS dot, = ||r||_M)
//        v_{k+1} = r / beta_k,  w_{k+1} /= beta_k
//   3. Tridiagonal T (nstep × nstep) eigendecomposition on CPU via Eigen.
//   4. Select nd Ritz pairs with largest |nu|; Ritz vectors = V * Y (CPU).
//   5. Convert nu -> lambda = sigma + 1/nu, sort ascending, return.
//
// Note: <format> / std::format intentionally absent — nvcc uses its bundled
// g++-12 as host compiler, which does not ship <format>.  Use std::string
// concatenation and std::to_string() for all error / log messages.
#define HAVE_CUDA_EIGENSOLVER 1

#include "solver/cuda_eigensolver_backend.hpp"
#include "core/exceptions.hpp"
#include "core/logger.hpp"

#include <cuda_runtime.h>
#include <cudss.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace vibetran {

// ── RAII helpers (anonymous namespace) ───────────────────────────────────────

namespace {

template <typename T>
struct EigDevBuf {
    T* ptr = nullptr;

    EigDevBuf() = default;
    explicit EigDevBuf(std::size_t count) {
        if (count == 0) return;
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr),
                                     count * sizeof(T));
        if (err != cudaSuccess)
            throw SolverError(
                std::string("CUDA eigensolver: cudaMalloc(") +
                std::to_string(count) + "): " + cudaGetErrorString(err));
    }
    ~EigDevBuf() { if (ptr) cudaFree(ptr); }
    EigDevBuf(const EigDevBuf&)            = delete;
    EigDevBuf& operator=(const EigDevBuf&) = delete;
    EigDevBuf(EigDevBuf&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
    EigDevBuf& operator=(EigDevBuf&& o) noexcept {
        if (this != &o) { if (ptr) cudaFree(ptr); ptr = o.ptr; o.ptr = nullptr; }
        return *this;
    }

    void upload(const T* h, std::size_t cnt) {
        if (cudaMemcpy(ptr, h, cnt * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess)
            throw SolverError("CUDA eigensolver: H->D copy failed");
    }
    void download(T* h, std::size_t cnt) const {
        if (cudaMemcpy(h, ptr, cnt * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess)
            throw SolverError("CUDA eigensolver: D->H copy failed");
    }
};

struct EigCuDSSCfg {
    cudssConfig_t cfg = nullptr;
    EigCuDSSCfg() {
        if (cudssConfigCreate(&cfg) != CUDSS_STATUS_SUCCESS)
            throw SolverError("CUDA eigensolver: cudssConfigCreate failed");
    }
    ~EigCuDSSCfg() { if (cfg) cudssConfigDestroy(cfg); }
    EigCuDSSCfg(const EigCuDSSCfg&)            = delete;
    EigCuDSSCfg& operator=(const EigCuDSSCfg&) = delete;
};

struct EigCuDSSData {
    cudssHandle_t handle;
    cudssData_t   data = nullptr;
    explicit EigCuDSSData(cudssHandle_t h) : handle(h) {
        if (cudssDataCreate(h, &data) != CUDSS_STATUS_SUCCESS)
            throw SolverError("CUDA eigensolver: cudssDataCreate failed");
    }
    ~EigCuDSSData() { if (data) cudssDataDestroy(handle, data); }
    EigCuDSSData(const EigCuDSSData&)            = delete;
    EigCuDSSData& operator=(const EigCuDSSData&) = delete;
};

struct EigCuDSSMat {
    cudssMatrix_t mat = nullptr;
    EigCuDSSMat() = default;
    ~EigCuDSSMat() { if (mat) cudssMatrixDestroy(mat); }
    EigCuDSSMat(const EigCuDSSMat&)            = delete;
    EigCuDSSMat& operator=(const EigCuDSSMat&) = delete;
    EigCuDSSMat(EigCuDSSMat&& o) noexcept : mat(o.mat) { o.mat = nullptr; }
    EigCuDSSMat& operator=(EigCuDSSMat&& o) noexcept {
        if (this != &o) {
            if (mat) cudssMatrixDestroy(mat);
            mat = o.mat; o.mat = nullptr;
        }
        return *this;
    }
};

struct SpMatD {
    cusparseSpMatDescr_t d = nullptr;
    SpMatD() = default;
    ~SpMatD() { if (d) cusparseDestroySpMat(d); }
    SpMatD(const SpMatD&)            = delete;
    SpMatD& operator=(const SpMatD&) = delete;
};

struct DnVecD {
    cusparseDnVecDescr_t d = nullptr;
    DnVecD() = default;
    ~DnVecD() { if (d) cusparseDestroyDnVec(d); }
    DnVecD(const DnVecD&)            = delete;
    DnVecD& operator=(const DnVecD&) = delete;
};

static void ck(cudaError_t e, const char* s) {
    if (e != cudaSuccess)
        throw SolverError(std::string("CUDA eigensolver: ") + s + ": " +
                          cudaGetErrorString(e));
}
static void ck(cublasStatus_t e, const char* s) {
    if (e != CUBLAS_STATUS_SUCCESS)
        throw SolverError(std::string("CUDA eigensolver cuBLAS ") + s +
                          " status=" + std::to_string(static_cast<int>(e)));
}
static void ck(cusparseStatus_t e, const char* s) {
    if (e != CUSPARSE_STATUS_SUCCESS)
        throw SolverError(std::string("CUDA eigensolver cuSPARSE ") + s +
                          " status=" + std::to_string(static_cast<int>(e)));
}
static void ck(cudssStatus_t e, const char* s) {
    if (e != CUDSS_STATUS_SUCCESS)
        throw SolverError(std::string("CUDA eigensolver cuDSS ") + s +
                          " status=" + std::to_string(static_cast<int>(e)));
}

// CSR arrays on device.
struct CsrDev {
    EigDevBuf<double> vals;
    EigDevBuf<int>    rptr;
    EigDevBuf<int>    cind;
    int n = 0, nnz = 0;
};

static CsrDev upload(const Eigen::SparseMatrix<double, Eigen::RowMajor>& m) {
    CsrDev d;
    d.n   = static_cast<int>(m.rows());
    d.nnz = static_cast<int>(m.nonZeros());
    d.vals = EigDevBuf<double>(static_cast<std::size_t>(d.nnz));
    d.rptr = EigDevBuf<int>(static_cast<std::size_t>(d.n + 1));
    d.cind = EigDevBuf<int>(static_cast<std::size_t>(d.nnz));
    d.vals.upload(m.valuePtr(),      static_cast<std::size_t>(d.nnz));
    d.rptr.upload(m.outerIndexPtr(), static_cast<std::size_t>(d.n + 1));
    d.cind.upload(m.innerIndexPtr(), static_cast<std::size_t>(d.nnz));
    return d;
}

} // anonymous namespace

// ── CudaEigenContext ──────────────────────────────────────────────────────────

struct CudaEigenContext {
    cudssHandle_t    cudss    = nullptr;
    cublasHandle_t   cublas   = nullptr;
    cusparseHandle_t cusparse = nullptr;
    std::string      device_name;
};

// ── Constructor / destructor / move ──────────────────────────────────────────

CudaEigensolverBackend::CudaEigensolverBackend(
    std::unique_ptr<CudaEigenContext> ctx) noexcept
    : ctx_(std::move(ctx)) {}

CudaEigensolverBackend::~CudaEigensolverBackend() {
    if (!ctx_) return;
    if (ctx_->cusparse) cusparseDestroy(ctx_->cusparse);
    if (ctx_->cublas)   cublasDestroy(ctx_->cublas);
    if (ctx_->cudss)    cudssDestroy(ctx_->cudss);
}

CudaEigensolverBackend::CudaEigensolverBackend(
    CudaEigensolverBackend&&) noexcept = default;
CudaEigensolverBackend& CudaEigensolverBackend::operator=(
    CudaEigensolverBackend&&) noexcept = default;

// ── Factory ───────────────────────────────────────────────────────────────────

std::optional<CudaEigensolverBackend>
CudaEigensolverBackend::try_create() noexcept {
    int cnt = 0;
    if (cudaGetDeviceCount(&cnt) != cudaSuccess || cnt == 0) return std::nullopt;
    if (cudaSetDevice(0) != cudaSuccess) return std::nullopt;

    auto ctx = std::make_unique<CudaEigenContext>();

    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, 0) == cudaSuccess)
        ctx->device_name = props.name;

    if (cudssCreate(&ctx->cudss) != CUDSS_STATUS_SUCCESS) return std::nullopt;
    if (cublasCreate(&ctx->cublas) != CUBLAS_STATUS_SUCCESS) {
        cudssDestroy(ctx->cudss); return std::nullopt;
    }
    if (cusparseCreate(&ctx->cusparse) != CUSPARSE_STATUS_SUCCESS) {
        cublasDestroy(ctx->cublas); cudssDestroy(ctx->cudss); return std::nullopt;
    }
    return CudaEigensolverBackend(std::move(ctx));
}

// ── Accessors ─────────────────────────────────────────────────────────────────

std::string CudaEigensolverBackend::name() const {
    return "CUDA cuDSS shift-invert Lanczos eigensolver";
}
std::string_view CudaEigensolverBackend::device_name() const noexcept {
    return ctx_->device_name;
}

// ── solve ─────────────────────────────────────────────────────────────────────

std::vector<EigenPair> CudaEigensolverBackend::solve(
    const Eigen::SparseMatrix<double>& K,
    const Eigen::SparseMatrix<double>& M,
    int nd, double sigma)
{
    const int n = static_cast<int>(K.rows());
    if (n < 1)
        throw SolverError("CUDA eigensolver: system has no free DOFs");
    if (nd < 1)
        throw SolverError("CUDA eigensolver: nd must be >= 1");
    if (nd > n) nd = n;

    // No-restart Lanczos converges more slowly than restarted ARPACK/Spectra.
    // Use a larger ncv to ensure the higher requested modes converge well.
    const int ncv = std::min(n, std::max(2 * nd + 20, 4 * nd));

    log_debug("[cuda-eig] n=" + std::to_string(n) +
              " nd=" + std::to_string(nd) +
              " ncv=" + std::to_string(ncv) +
              " sigma=" + std::to_string(sigma) +
              " device='" + ctx_->device_name + "'");

    // ── 1. Build C = K - sigma*M and upload K, M, C ──────────────────────────
    using RmSp = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    RmSp K_rm(K), M_rm(M);
    K_rm.makeCompressed();
    M_rm.makeCompressed();

    RmSp C_rm = K_rm - sigma * M_rm;
    C_rm.makeCompressed();

    CsrDev d_C = upload(C_rm);
    CsrDev d_M = upload(M_rm);

    const int C_nnz = d_C.nnz;
    const int M_nnz = d_M.nnz;

    // ── 2. cuDSS: factorize C (SPD first, LU fallback) ───────────────────────
    // stride is rounded up to the next even number so that every column of d_V
    // and d_W starts on a 16-byte aligned address (cuSPARSE uses double2 loads
    // that require 16-byte alignment; misaligned pointers cause device errors).
    const std::size_t stride = (static_cast<std::size_t>(n) + 1u) & ~1u;

    // ── Allocate ALL device buffers before cuDSS / cuSPARSE descriptors ───────
    // C++ destroys locals in reverse declaration order.  cuDSS matrix
    // descriptors (cudss_A/b/x_mat) hold device pointers set via
    // cudssMatrixSetValues — in particular cudss_b_mat points into d_W after
    // the last Lanczos solve.  If d_W were declared after the cuDSS
    // descriptors, it would be freed first, and cudssMatrixDestroy would
    // access freed GPU memory (double-free / SIGABRT in release builds).
    // Declaring all device buffers here guarantees they outlive every
    // descriptor that references them.
    EigDevBuf<double> d_z(stride);
    EigDevBuf<double> d_V(stride * static_cast<std::size_t>(ncv + 1));
    EigDevBuf<double> d_W(stride * static_cast<std::size_t>(ncv + 1));
    EigDevBuf<double> d_r(stride);
    EigDevBuf<double> d_co(static_cast<std::size_t>(ncv));

    EigCuDSSCfg cudss_cfg;
    {
        cudssAlgType_t alg = CUDSS_ALG_DEFAULT;
        cudssConfigSet(cudss_cfg.cfg, CUDSS_CONFIG_REORDERING_ALG,
                       &alg, sizeof(alg));
    }

    // All three cuDSS descriptors must outlive every SOLVE call.
    cudssMatrixType_t working_mtype = CUDSS_MTYPE_SPD;
    auto p_data = std::make_unique<EigCuDSSData>(ctx_->cudss);
    EigCuDSSMat cudss_A_mat, cudss_b_mat, cudss_x_mat;

    // Helper: attempt analysis + factorization with a given matrix type.
    // On success, moves the three descriptors into cudss_A_mat / _b_mat / _x_mat
    // so they remain alive through all subsequent SOLVE calls.
    auto try_factorize = [&](cudssMatrixType_t mtype) -> bool {
        cudssMatrixViewType_t mview =
            (mtype == CUDSS_MTYPE_GENERAL) ? CUDSS_MVIEW_FULL : CUDSS_MVIEW_UPPER;

        EigCuDSSMat A_mat, b_mat, x_mat;
        if (cudssMatrixCreateCsr(
                &A_mat.mat,
                static_cast<int64_t>(n), static_cast<int64_t>(n),
                static_cast<int64_t>(C_nnz),
                d_C.rptr.ptr, nullptr, d_C.cind.ptr, d_C.vals.ptr,
                CUDA_R_32I, CUDA_R_64F, mtype, mview, CUDSS_BASE_ZERO)
            != CUDSS_STATUS_SUCCESS) return false;

        // b and x point to d_z for ANALYSIS/FACTORIZATION (values unused).
        if (cudssMatrixCreateDn(&b_mat.mat, static_cast<int64_t>(n), 1,
                                static_cast<int64_t>(n), d_z.ptr,
                                CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR)
            != CUDSS_STATUS_SUCCESS) return false;
        if (cudssMatrixCreateDn(&x_mat.mat, static_cast<int64_t>(n), 1,
                                static_cast<int64_t>(n), d_z.ptr,
                                CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR)
            != CUDSS_STATUS_SUCCESS) return false;

        if (cudssExecute(ctx_->cudss, CUDSS_PHASE_ANALYSIS,
                         cudss_cfg.cfg, p_data->data,
                         A_mat.mat, x_mat.mat, b_mat.mat) != CUDSS_STATUS_SUCCESS)
            return false;
        if (cudssExecute(ctx_->cudss, CUDSS_PHASE_FACTORIZATION,
                         cudss_cfg.cfg, p_data->data,
                         A_mat.mat, x_mat.mat, b_mat.mat) != CUDSS_STATUS_SUCCESS)
            return false;

        // Transfer ownership: all three must remain alive through SOLVE calls.
        cudss_A_mat = std::move(A_mat);
        cudss_b_mat = std::move(b_mat);
        cudss_x_mat = std::move(x_mat);
        return true;
    };

    if (!try_factorize(CUDSS_MTYPE_SPD)) {
        log_debug("[cuda-eig] SPD factorization failed -- retrying with LU");
        p_data = std::make_unique<EigCuDSSData>(ctx_->cudss); // fresh data for LU
        if (!try_factorize(CUDSS_MTYPE_GENERAL))
            throw SolverError(
                "CUDA eigensolver: failed to factorize C = K - sigma*M "
                "(n=" + std::to_string(n) +
                ", sigma=" + std::to_string(sigma) + ")");
        working_mtype = CUDSS_MTYPE_GENERAL;
    }

    log_debug("[cuda-eig] factorized C with " +
              std::string(working_mtype == CUDSS_MTYPE_SPD ? "SPD Cholesky" : "LU"));

    // Helper: single-RHS solve z = C^{-1} * b, reusing the factorization.
    // Updates b and x pointers via cudssMatrixSetValues (descriptors remain alive).
    auto do_solve = [&](double* b_ptr, double* z_ptr) {
        ck(cudssMatrixSetValues(cudss_b_mat.mat, b_ptr),
           "cudssMatrixSetValues(b)");
        ck(cudssMatrixSetValues(cudss_x_mat.mat, z_ptr),
           "cudssMatrixSetValues(x)");
        ck(cudaMemset(z_ptr, 0, stride * sizeof(double)), "cudaMemset(z)");
        ck(cudssExecute(ctx_->cudss, CUDSS_PHASE_SOLVE,
                        cudss_cfg.cfg, p_data->data,
                        cudss_A_mat.mat, cudss_x_mat.mat, cudss_b_mat.mat),
           "cudssExecute(SOLVE)");
    };

    // ── 3. cuSPARSE: create M SpMV descriptor + buffer ───────────────────────
    SpMatD sp_M;
    ck(cusparseCreateCsr(
           &sp_M.d,
           static_cast<int64_t>(n), static_cast<int64_t>(n),
           static_cast<int64_t>(M_nnz),
           d_M.rptr.ptr, d_M.cind.ptr, d_M.vals.ptr,
           CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
           CUDA_R_64F),
       "cusparseCreateCsr(M)");

    // d_V, d_W, d_r, d_co already allocated above (before cuDSS descriptors).

    // Query SpMV buffer size using the first-column pointers as placeholders.
    const double sp1 = 1.0, sp0 = 0.0;
    DnVecD sv_tmp, sw_tmp;
    ck(cusparseCreateDnVec(&sv_tmp.d, static_cast<int64_t>(n), d_V.ptr, CUDA_R_64F),
       "cusparseCreateDnVec(v tmp)");
    ck(cusparseCreateDnVec(&sw_tmp.d, static_cast<int64_t>(n), d_W.ptr, CUDA_R_64F),
       "cusparseCreateDnVec(w tmp)");

    std::size_t spmv_sz = 0;
    ck(cusparseSpMV_bufferSize(
           ctx_->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
           &sp1, sp_M.d, sv_tmp.d, &sp0, sw_tmp.d,
           CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_sz),
       "cusparseSpMV_bufferSize");

    EigDevBuf<char> d_spbuf(std::max(spmv_sz, std::size_t{1}));

    // Persistent SpMV vector descriptors — updated per step with SetValues.
    DnVecD sp_v, sp_w;
    ck(cusparseCreateDnVec(&sp_v.d, static_cast<int64_t>(n), d_V.ptr, CUDA_R_64F),
       "cusparseCreateDnVec(v)");
    ck(cusparseCreateDnVec(&sp_w.d, static_cast<int64_t>(n), d_W.ptr, CUDA_R_64F),
       "cusparseCreateDnVec(w)");

    auto spmv_Mv = [&](double* v, double* w) {
        ck(cusparseDnVecSetValues(sp_v.d, v), "cusparseDnVecSetValues(v)");
        ck(cusparseDnVecSetValues(sp_w.d, w), "cusparseDnVecSetValues(w)");
        ck(cusparseSpMV(ctx_->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &sp1, sp_M.d, sp_v.d, &sp0, sp_w.d,
                        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spbuf.ptr),
           "cusparseSpMV");
    };

    // ── 4. Initial Lanczos vector (constant, M-normalized on CPU) ────────────
    {
        Eigen::VectorXd v0 = Eigen::VectorXd::Ones(n);
        double nm2 = v0.dot(M_rm * v0);
        if (nm2 <= 0.0)
            throw SolverError(
                "CUDA eigensolver: mass matrix M is not positive definite");
        v0 /= std::sqrt(nm2);
        d_V.upload(v0.data(), static_cast<std::size_t>(n));
    }
    spmv_Mv(d_V.ptr, d_W.ptr); // w_0 = M * v_0

    // ── 5. Lanczos loop ───────────────────────────────────────────────────────
    std::vector<double> alpha(static_cast<std::size_t>(ncv));
    std::vector<double> beta(static_cast<std::size_t>(ncv));

    const double c1 = 1.0, cn1 = -1.0, c0 = 0.0;
    int nstep = ncv;

    for (int k = 0; k < ncv; ++k) {
        double* vk = d_V.ptr + static_cast<std::size_t>(k) * stride;
        double* wk = d_W.ptr + static_cast<std::size_t>(k) * stride;

        // z = C^{-1} w_k
        do_solve(wk, d_z.ptr);

        // alpha_k = w_k · z  (= <v_k, A v_k>_M since w_k = M v_k)
        double alpha_k = 0.0;
        ck(cublasDdot(ctx_->cublas, n, wk, 1, d_z.ptr, 1, &alpha_k), "cublasDdot(alpha)");
        ck(cudaDeviceSynchronize(), "sync(alpha)");
        alpha[static_cast<std::size_t>(k)] = alpha_k;

        // r = z - alpha_k * v_k
        ck(cudaMemcpy(d_r.ptr, d_z.ptr,
                      stride * sizeof(double), cudaMemcpyDeviceToDevice),
           "memcpy z->r");
        double na = -alpha_k;
        ck(cublasDaxpy(ctx_->cublas, n, &na, vk, 1, d_r.ptr, 1), "cublasDaxpy(alpha)");

        // r -= beta_{k-1} * v_{k-1}
        if (k > 0) {
            double nb = -beta[static_cast<std::size_t>(k - 1)];
            double* vkm1 = d_V.ptr + static_cast<std::size_t>(k - 1) * stride;
            ck(cublasDaxpy(ctx_->cublas, n, &nb, vkm1, 1, d_r.ptr, 1),
               "cublasDaxpy(beta)");
        }

        // Double re-orthogonalization in M-inner product:
        //   coeffs = W[:,0:k]^T * r;   r -= V[:,0:k] * coeffs   (two passes)
        // lda must equal stride (not n) so cuBLAS correctly skips the padding
        // doubles at the end of each column.
        if (k > 0) {
            for (int pass = 0; pass < 2; ++pass) {
                ck(cublasDgemv(ctx_->cublas, CUBLAS_OP_T, n, k,
                               &c1, d_W.ptr, static_cast<int>(stride), d_r.ptr, 1,
                               &c0, d_co.ptr, 1),
                   "cublasDgemv(reorth T)");
                ck(cublasDgemv(ctx_->cublas, CUBLAS_OP_N, n, k,
                               &cn1, d_V.ptr, static_cast<int>(stride), d_co.ptr, 1,
                               &c1,  d_r.ptr, 1),
                   "cublasDgemv(reorth N)");
            }
        }

        // w_{k+1} = M * r  (last iter: reuse d_z as temp; only need the dot)
        double* w_next = (k < ncv - 1)
            ? d_W.ptr + static_cast<std::size_t>(k + 1) * stride
            : d_z.ptr;
        spmv_Mv(d_r.ptr, w_next);

        // beta_k = sqrt(r · w_{k+1}) = ||r||_M
        double beta_sq = 0.0;
        ck(cublasDdot(ctx_->cublas, n, d_r.ptr, 1, w_next, 1, &beta_sq),
           "cublasDdot(beta^2)");
        ck(cudaDeviceSynchronize(), "sync(beta)");

        if (beta_sq <= 0.0 || std::sqrt(beta_sq) < 1e-14) {
            nstep = k + 1;
            log_debug("[cuda-eig] lucky breakdown at step " + std::to_string(k));
            break;
        }

        const double bk = std::sqrt(beta_sq);
        beta[static_cast<std::size_t>(k)] = bk;

        if (k < ncv - 1) {
            double* v_next = d_V.ptr + static_cast<std::size_t>(k + 1) * stride;
            ck(cudaMemcpy(v_next, d_r.ptr, stride * sizeof(double),
                          cudaMemcpyDeviceToDevice), "memcpy r->v_next");
            double inv = 1.0 / bk;
            ck(cublasDscal(ctx_->cublas, n, &inv, v_next, 1), "cublasDscal(v)");
            ck(cublasDscal(ctx_->cublas, n, &inv, w_next, 1), "cublasDscal(w)");
        }
    }

    log_debug("[cuda-eig] Lanczos done, nstep=" + std::to_string(nstep));

    // ── 5b. Release cuDSS resources ────────────────────────────────────────────
    // Factorization is no longer needed.  Destroy descriptors before data.
    cudss_x_mat = EigCuDSSMat{};
    cudss_b_mat = EigCuDSSMat{};
    cudss_A_mat = EigCuDSSMat{};
    p_data.reset();

    // ── 6. Tridiagonal eigendecomposition on CPU ──────────────────────────────
    Eigen::MatrixXd T_mat = Eigen::MatrixXd::Zero(nstep, nstep);
    for (int i = 0; i < nstep; ++i)
        T_mat(i, i) = alpha[static_cast<std::size_t>(i)];
    for (int i = 0; i < nstep - 1; ++i) {
        T_mat(i, i + 1) = beta[static_cast<std::size_t>(i)];
        T_mat(i + 1, i) = beta[static_cast<std::size_t>(i)];
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_T(T_mat);
    if (eig_T.info() != Eigen::Success)
        throw SolverError(
            "CUDA eigensolver: tridiagonal eigendecomposition failed "
            "(nstep=" + std::to_string(nstep) + ")");

    // ── 7. Select nd Ritz pairs by largest |nu| ───────────────────────────────
    const int nd_actual = std::min(nd, nstep);
    const Eigen::VectorXd& nu_all = eig_T.eigenvalues();   // ascending

    std::vector<int> idx(static_cast<std::size_t>(nstep));
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + nd_actual, idx.end(),
                      [&](int a, int b) {
                          return std::abs(nu_all(a)) > std::abs(nu_all(b));
                      });

    // ── 8. Download V and compute Ritz vectors on CPU ─────────────────────────
    std::vector<double> V_host(stride * static_cast<std::size_t>(nstep));
    ck(cudaMemcpy(V_host.data(), d_V.ptr,
                  stride * static_cast<std::size_t>(nstep) * sizeof(double),
                  cudaMemcpyDeviceToHost),
       "cudaMemcpy V D->H");

    // stride may be > n (alignment padding); use OuterStride so Eigen reads
    // n rows per column but skips to the next column every stride elements.
    using OuterStr = Eigen::OuterStride<Eigen::Dynamic>;
    Eigen::Map<const Eigen::MatrixXd, Eigen::Unaligned, OuterStr>
        V_map(V_host.data(), n, nstep,
              OuterStr(static_cast<Eigen::Index>(stride)));
    const Eigen::MatrixXd& Y_all = eig_T.eigenvectors();

    // ── 9. Build output EigenPairs ────────────────────────────────────────────
    std::vector<EigenPair> results;
    results.reserve(static_cast<std::size_t>(nd_actual));

    for (int i = 0; i < nd_actual; ++i) {
        int j = idx[static_cast<std::size_t>(i)];
        double nu = nu_all(j);
        if (std::abs(nu) < 1e-20) continue; // degenerate; skip

        EigenPair ep;
        ep.eigenvalue  = sigma + 1.0 / nu;
        ep.eigenvector = V_map * Y_all.col(j); // Ritz vector (M-normalised)
        results.push_back(std::move(ep));
    }

    if (results.empty())
        throw SolverError(
            "CUDA eigensolver: no Ritz pairs converged "
            "(nd=" + std::to_string(nd) +
            ", nstep=" + std::to_string(nstep) +
            ", sigma=" + std::to_string(sigma) + ")");

    std::sort(results.begin(), results.end(),
              [](const EigenPair& a, const EigenPair& b) {
                  return a.eigenvalue < b.eigenvalue;
              });

    log_debug("[cuda-eig] returning " + std::to_string(results.size()) +
              " pairs, lambda[0]=" + std::to_string(results[0].eigenvalue));

    return results;
}

} // namespace vibetran
