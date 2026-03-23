// src/solver/cuda_eigensolver_backend.cu
// CUDA shift-and-invert Lanczos eigensolver for K φ = λ M φ.
//
// Algorithm overview (Implicitly Restarted M-inner-product Lanczos
//                    + Rayleigh-Ritz post-refinement):
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
//   3. Implicit restart loop (ARPACK-style):
//      a. Solve tridiagonal T eigenproblem on CPU.
//      b. Check convergence via Ritz estimates.
//      c. If not converged, apply p = ncv - k implicit QR shifts to T,
//         compress V and W on GPU via cuBLAS dgemm, update residual,
//         then re-expand Lanczos from step k to ncv.
//   4. Select nd Ritz pairs with largest |nu|; Ritz vectors = V * Y (CPU).
//   5. Rayleigh-Ritz post-refinement (2 iterations, cuDSS still live):
//      a. Apply A = C^{-1} M to each Ritz vector on GPU.
//      b. M-orthonormalize via Cholesky of G = Z^T M Z on CPU.
//      c. Project K onto the subspace: K_sub = Z_orth^T K Z_orth.
//      d. Solve the small dense eigenproblem K_sub y = lambda y.
//      e. Update X = Z_orth * Y, lambda = eigenvalues of K_sub.
//   6. Release cuDSS factorization.
//   7. Sort eigenvalues ascending and return.
//
// Note: <format> / std::format intentionally absent — nvcc uses its bundled
// g++-12 as host compiler, which does not ship <format>.  Use std::string
// concatenation and std::to_string() for all error / log messages.
#define HAVE_CUDA_EIGENSOLVER 1

#include "core/exceptions.hpp"
#include "core/logger.hpp"
#include "solver/cuda_eigensolver_backend.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudss.h>
#include <cusparse.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace vibestran {

// ── RAII helpers (anonymous namespace) ───────────────────────────────────────

namespace {

template <typename T> struct EigDevBuf {
  T *ptr = nullptr;

  EigDevBuf() = default;
  explicit EigDevBuf(std::size_t count) {
    if (count == 0)
      return;
    cudaError_t err =
        cudaMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T));
    if (err != cudaSuccess)
      throw SolverError(std::string("CUDA eigensolver: cudaMalloc(") +
                        std::to_string(count) +
                        "): " + cudaGetErrorString(err));
  }
  ~EigDevBuf() {
    if (ptr)
      cudaFree(ptr);
  }
  EigDevBuf(const EigDevBuf &) = delete;
  EigDevBuf &operator=(const EigDevBuf &) = delete;
  EigDevBuf(EigDevBuf &&o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
  EigDevBuf &operator=(EigDevBuf &&o) noexcept {
    if (this != &o) {
      if (ptr)
        cudaFree(ptr);
      ptr = o.ptr;
      o.ptr = nullptr;
    }
    return *this;
  }

  void upload(const T *h, std::size_t cnt) {
    if (cudaMemcpy(ptr, h, cnt * sizeof(T), cudaMemcpyHostToDevice) !=
        cudaSuccess)
      throw SolverError("CUDA eigensolver: H->D copy failed");
  }
  void download(T *h, std::size_t cnt) const {
    if (cudaMemcpy(h, ptr, cnt * sizeof(T), cudaMemcpyDeviceToHost) !=
        cudaSuccess)
      throw SolverError("CUDA eigensolver: D->H copy failed");
  }
};

struct EigCuDSSCfg {
  cudssConfig_t cfg = nullptr;
  EigCuDSSCfg() {
    if (cudssConfigCreate(&cfg) != CUDSS_STATUS_SUCCESS)
      throw SolverError("CUDA eigensolver: cudssConfigCreate failed");
  }
  ~EigCuDSSCfg() {
    if (cfg)
      cudssConfigDestroy(cfg);
  }
  EigCuDSSCfg(const EigCuDSSCfg &) = delete;
  EigCuDSSCfg &operator=(const EigCuDSSCfg &) = delete;
};

struct EigCuDSSData {
  cudssHandle_t handle;
  cudssData_t data = nullptr;
  explicit EigCuDSSData(cudssHandle_t h) : handle(h) {
    if (cudssDataCreate(h, &data) != CUDSS_STATUS_SUCCESS)
      throw SolverError("CUDA eigensolver: cudssDataCreate failed");
  }
  ~EigCuDSSData() {
    if (data)
      cudssDataDestroy(handle, data);
  }
  EigCuDSSData(const EigCuDSSData &) = delete;
  EigCuDSSData &operator=(const EigCuDSSData &) = delete;
};

struct EigCuDSSMat {
  cudssMatrix_t mat = nullptr;
  EigCuDSSMat() = default;
  ~EigCuDSSMat() {
    if (mat)
      cudssMatrixDestroy(mat);
  }
  EigCuDSSMat(const EigCuDSSMat &) = delete;
  EigCuDSSMat &operator=(const EigCuDSSMat &) = delete;
  EigCuDSSMat(EigCuDSSMat &&o) noexcept : mat(o.mat) { o.mat = nullptr; }
  EigCuDSSMat &operator=(EigCuDSSMat &&o) noexcept {
    if (this != &o) {
      if (mat)
        cudssMatrixDestroy(mat);
      mat = o.mat;
      o.mat = nullptr;
    }
    return *this;
  }
};

struct SpMatD {
  cusparseSpMatDescr_t d = nullptr;
  SpMatD() = default;
  ~SpMatD() {
    if (d)
      cusparseDestroySpMat(d);
  }
  SpMatD(const SpMatD &) = delete;
  SpMatD &operator=(const SpMatD &) = delete;
};

struct DnVecD {
  cusparseDnVecDescr_t d = nullptr;
  DnVecD() = default;
  ~DnVecD() {
    if (d)
      cusparseDestroyDnVec(d);
  }
  DnVecD(const DnVecD &) = delete;
  DnVecD &operator=(const DnVecD &) = delete;
};

static void ck(cudaError_t e, const char *s) {
  if (e != cudaSuccess)
    throw SolverError(std::string("CUDA eigensolver: ") + s + ": " +
                      cudaGetErrorString(e));
}
static void ck(cublasStatus_t e, const char *s) {
  if (e != CUBLAS_STATUS_SUCCESS)
    throw SolverError(std::string("CUDA eigensolver cuBLAS ") + s +
                      " status=" + std::to_string(static_cast<int>(e)));
}
static void ck(cusparseStatus_t e, const char *s) {
  if (e != CUSPARSE_STATUS_SUCCESS)
    throw SolverError(std::string("CUDA eigensolver cuSPARSE ") + s +
                      " status=" + std::to_string(static_cast<int>(e)));
}
static void ck(cudssStatus_t e, const char *s) {
  if (e != CUDSS_STATUS_SUCCESS)
    throw SolverError(std::string("CUDA eigensolver cuDSS ") + s +
                      " status=" + std::to_string(static_cast<int>(e)));
}

// CSR arrays on device.
struct CsrDev {
  EigDevBuf<double> vals;
  EigDevBuf<int> rptr;
  EigDevBuf<int> cind;
  int n = 0, nnz = 0;
};

static CsrDev upload(const Eigen::SparseMatrix<double, Eigen::RowMajor> &m) {
  CsrDev d;
  d.n = static_cast<int>(m.rows());
  d.nnz = static_cast<int>(m.nonZeros());
  d.vals = EigDevBuf<double>(static_cast<std::size_t>(d.nnz));
  d.rptr = EigDevBuf<int>(static_cast<std::size_t>(d.n + 1));
  d.cind = EigDevBuf<int>(static_cast<std::size_t>(d.nnz));
  d.vals.upload(m.valuePtr(), static_cast<std::size_t>(d.nnz));
  d.rptr.upload(m.outerIndexPtr(), static_cast<std::size_t>(d.n + 1));
  d.cind.upload(m.innerIndexPtr(), static_cast<std::size_t>(d.nnz));
  return d;
}

// ── IRL helper: apply one implicit QR shift to a symmetric tridiagonal T ─────
// Uses Givens rotations to compute T <- Q^T * T * Q where Q*R = T - mu*I.
// Accumulates Q into Q_total for basis compression.
// Closely follows Spectra's TridiagQR::compute() + matrix_QtHQ() + apply_YQ().
static void apply_tridiag_qr_shift(Eigen::MatrixXd &T, double mu,
                                   Eigen::MatrixXd &Q_total) {
  const int m = static_cast<int>(T.rows());
  if (m < 2)
    return;
  const int n1 = m - 1;
  const int n2 = m - 2;

  // ── Save original diagonal and subdiagonal of T ──────────────────────────
  Eigen::VectorXd T_diag = T.diagonal();
  Eigen::VectorXd T_subd = T.diagonal(-1);

  // Deflation of small subdiagonal elements (before QR)
  const double eps = std::numeric_limits<double>::epsilon();
  for (int i = 0; i < n1; ++i)
    if (std::abs(T_subd(i)) <=
        eps * (std::abs(T_diag(i)) + std::abs(T_diag(i + 1))))
      T_subd(i) = 0.0;

  // ── QR factorization of (T - mu*I) via Givens rotations ─────────────────
  // R is upper triangular with bands: R_diag (diagonal), R_supd (first super-
  // diagonal), R_supd2 (second superdiagonal, fill-in).
  // Only cos/sin sequences are needed for the subsequent transforms.
  Eigen::VectorXd rot_cos(n1), rot_sin(n1);
  Eigen::VectorXd R_diag(m), R_supd(n1);

  R_diag = T_diag.array() - mu;
  R_supd = T_subd; // starts as original subdiag

  for (int i = 0; i < n1; ++i) {
    // Compute Givens rotation to zero T_subd[i] (original subdiag)
    double a = R_diag(i);
    double b = T_subd(i);
    double r = std::hypot(a, b);
    double c, s;
    if (r < 1e-300) {
      c = 1.0;
      s = 0.0;
    } else {
      c = a / r;
      s = -b / r;
    }
    rot_cos(i) = c;
    rot_sin(i) = s;

    // Apply G_i^T to rows i, i+1 of R
    R_diag(i) = r;
    double Tii1 = R_supd(i);              // R[i, i+1] before update
    double Ti1i1 = R_diag(i + 1);         // R[i+1, i+1] before update
    R_supd(i) = c * Tii1 - s * Ti1i1;     // updated R[i, i+1]
    R_diag(i + 1) = s * Tii1 + c * Ti1i1; // updated R[i+1, i+1]
    if (i < n2) {
      // R[i+1, i+2] fill-in propagation
      R_supd(i + 1) *= c;
    }
  }

  // ── Apply Q^T T Q (matrix_QtHQ) ─────────────────────────────────────────
  // Initialise dest from saved original T values (lower subdiag only).
  T.setZero();
  T.diagonal() = T_diag;
  T.diagonal(-1) = T_subd;

  for (int i = 0; i < n1; ++i) {
    const double c = rot_cos(i);
    const double s = rot_sin(i);
    const double cs = c * s, c2 = c * c, s2 = s * s;
    const double x = T(i, i);
    const double y = T(i + 1, i);
    const double z = T(i + 1, i + 1);

    T(i, i) = c2 * x - 2.0 * cs * y + s2 * z;
    T(i + 1, i) = cs * (x - z) + (c2 - s2) * y;
    T(i + 1, i + 1) = s2 * x + 2.0 * cs * y + c2 * z;

    if (i < n2) {
      const double ci1 = rot_cos(i + 1);
      const double si1 = rot_sin(i + 1);
      const double o = -s * T_subd(i + 1);       // off-tridiagonal fill-in
      T(i + 2, i + 1) *= c;                      // w' = c * w
      T(i + 1, i) = ci1 * T(i + 1, i) - si1 * o; // absorb fill-in
    }
  }

  // Deflation of small subdiagonal elements (after QR)
  for (int i = 0; i < n1; ++i) {
    double d = std::abs(T(i, i)) + std::abs(T(i + 1, i + 1));
    if (std::abs(T(i + 1, i)) <= eps * d)
      T(i + 1, i) = 0.0;
  }

  // Copy lower subdiagonal to upper (symmetry)
  T.diagonal(1) = T.diagonal(-1);

  // ── Accumulate Q_total <- Q_total * Q ────────────────────────────────────
  // Q = G_0 * G_1 * ... * G_{m-2},  G_i = [[c, s], [-s, c]]
  for (int i = 0; i < n1; ++i) {
    const double c = rot_cos(i);
    const double s = rot_sin(i);
    for (int j = 0; j < m; ++j) {
      double qi = Q_total(j, i);
      double qi1 = Q_total(j, i + 1);
      Q_total(j, i) = c * qi - s * qi1;
      Q_total(j, i + 1) = s * qi + c * qi1;
    }
  }
}

// ── IRL helper: count converged Ritz pairs ──────────────────────────────────
// Convergence criterion from ARPACK/Spectra:
//   |ritz_est[i]| * f_norm < tol * max(eps^(2/3), |ritz_val[i]|)
// where ritz_est[i] = last row of the i-th eigenvector of T.
static int check_convergence(int nev, const Eigen::VectorXd &ritz_val,
                             const Eigen::VectorXd &ritz_est, double f_norm,
                             double tol) {
  const double eps23 =
      std::pow(std::numeric_limits<double>::epsilon(), 2.0 / 3.0);
  int nconv = 0;
  for (int i = 0; i < nev; ++i) {
    double thresh = tol * std::max(eps23, std::abs(ritz_val(i)));
    double resid = std::abs(ritz_est(i)) * f_norm;
    if (resid < thresh)
      ++nconv;
  }
  return nconv;
}

// ── IRL helper: ARPACK-style adjusted nev for restart ───────────────────────
// Reference: dsaup2.f lines 677-684, Spectra HermEigsBase.h nev_adjusted().
static int nev_adjusted(int nev, int ncv, int nconv,
                        const Eigen::VectorXd &ritz_est) {
  const double near_0 = std::numeric_limits<double>::min() * 10.0;
  int nev_new = nev;
  for (int i = nev; i < ncv; ++i)
    if (std::abs(ritz_est(i)) < near_0)
      ++nev_new;
  nev_new += std::min(nconv, (ncv - nev_new) / 2);
  if (nev_new == 1 && ncv >= 6)
    nev_new = ncv / 2;
  else if (nev_new == 1 && ncv > 2)
    nev_new = 2;
  if (nev_new > ncv - 1)
    nev_new = ncv - 1;
  return nev_new;
}

} // anonymous namespace

// ── CudaEigenContext
// ──────────────────────────────────────────────────────────

struct CudaEigenContext {
  cudssHandle_t cudss = nullptr;
  cublasHandle_t cublas = nullptr;
  cusparseHandle_t cusparse = nullptr;
  std::string device_name;
};

// ── Constructor / destructor / move ──────────────────────────────────────────

CudaEigensolverBackend::CudaEigensolverBackend(
    std::unique_ptr<CudaEigenContext> ctx) noexcept
    : ctx_(std::move(ctx)) {}

CudaEigensolverBackend::~CudaEigensolverBackend() {
  if (!ctx_)
    return;
  if (ctx_->cusparse)
    cusparseDestroy(ctx_->cusparse);
  if (ctx_->cublas)
    cublasDestroy(ctx_->cublas);
  if (ctx_->cudss)
    cudssDestroy(ctx_->cudss);
}

CudaEigensolverBackend::CudaEigensolverBackend(
    CudaEigensolverBackend &&) noexcept = default;
CudaEigensolverBackend &
CudaEigensolverBackend::operator=(CudaEigensolverBackend &&) noexcept = default;

// ── Factory
// ───────────────────────────────────────────────────────────────────

std::optional<CudaEigensolverBackend>
CudaEigensolverBackend::try_create() noexcept {
  int cnt = 0;
  if (cudaGetDeviceCount(&cnt) != cudaSuccess || cnt == 0)
    return std::nullopt;
  if (cudaSetDevice(0) != cudaSuccess)
    return std::nullopt;

  auto ctx = std::make_unique<CudaEigenContext>();

  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, 0) == cudaSuccess)
    ctx->device_name = props.name;

  if (cudssCreate(&ctx->cudss) != CUDSS_STATUS_SUCCESS)
    return std::nullopt;
  if (cublasCreate(&ctx->cublas) != CUBLAS_STATUS_SUCCESS) {
    cudssDestroy(ctx->cudss);
    return std::nullopt;
  }
  if (cusparseCreate(&ctx->cusparse) != CUSPARSE_STATUS_SUCCESS) {
    cublasDestroy(ctx->cublas);
    cudssDestroy(ctx->cudss);
    return std::nullopt;
  }
  return CudaEigensolverBackend(std::move(ctx));
}

// ── Accessors
// ─────────────────────────────────────────────────────────────────

std::string CudaEigensolverBackend::name() const {
  return "CUDA cuDSS shift-invert Lanczos eigensolver";
}
std::string_view CudaEigensolverBackend::device_name() const noexcept {
  return ctx_->device_name;
}

// ── solve
// ─────────────────────────────────────────────────────────────────────

std::vector<EigenPair>
CudaEigensolverBackend::solve(const Eigen::SparseMatrix<double> &K,
                              const Eigen::SparseMatrix<double> &M, int nd,
                              double sigma) {
  const int n = static_cast<int>(K.rows());
  if (n < 1)
    throw SolverError("CUDA eigensolver: system has no free DOFs");
  if (nd < 1)
    throw SolverError("CUDA eigensolver: nd must be >= 1");
  if (nd > n)
    nd = n;

  // Implicitly restarted Lanczos uses a smaller basis than the old
  // no-restart approach (which needed max(2*nd+20, 4*nd)).
  const int ncv = std::min(n, std::max(2 * nd + 10, 3 * nd));
  const int maxit = 300;
  const double tol = 1e-10;

  log_debug("[cuda-eig] n=" + std::to_string(n) + " nd=" + std::to_string(nd) +
            " ncv=" + std::to_string(ncv) + " sigma=" + std::to_string(sigma) +
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
  const auto ncv_sz = static_cast<std::size_t>(ncv);

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
  EigDevBuf<double> d_V(stride * (ncv_sz + 1));
  EigDevBuf<double> d_W(stride * (ncv_sz + 1));
  EigDevBuf<double> d_r(stride);
  EigDevBuf<double> d_co(ncv_sz);
  // IRL scratch buffers: Q matrix upload and dgemm temporary.
  EigDevBuf<double> d_Q(ncv_sz * ncv_sz);
  EigDevBuf<double> d_temp(stride * (ncv_sz + 1));

  EigCuDSSCfg cudss_cfg;
  {
    cudssAlgType_t alg = CUDSS_ALG_DEFAULT;
    cudssConfigSet(cudss_cfg.cfg, CUDSS_CONFIG_REORDERING_ALG, &alg,
                   sizeof(alg));
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
            &A_mat.mat, static_cast<int64_t>(n), static_cast<int64_t>(n),
            static_cast<int64_t>(C_nnz), d_C.rptr.ptr, nullptr, d_C.cind.ptr,
            d_C.vals.ptr, CUDA_R_32I, CUDA_R_64F, mtype, mview,
            CUDSS_BASE_ZERO) != CUDSS_STATUS_SUCCESS)
      return false;

    // b and x point to d_z for ANALYSIS/FACTORIZATION (values unused).
    if (cudssMatrixCreateDn(&b_mat.mat, static_cast<int64_t>(n), 1,
                            static_cast<int64_t>(n), d_z.ptr, CUDA_R_64F,
                            CUDSS_LAYOUT_COL_MAJOR) != CUDSS_STATUS_SUCCESS)
      return false;
    if (cudssMatrixCreateDn(&x_mat.mat, static_cast<int64_t>(n), 1,
                            static_cast<int64_t>(n), d_z.ptr, CUDA_R_64F,
                            CUDSS_LAYOUT_COL_MAJOR) != CUDSS_STATUS_SUCCESS)
      return false;

    if (cudssExecute(ctx_->cudss, CUDSS_PHASE_ANALYSIS, cudss_cfg.cfg,
                     p_data->data, A_mat.mat, x_mat.mat,
                     b_mat.mat) != CUDSS_STATUS_SUCCESS)
      return false;
    if (cudssExecute(ctx_->cudss, CUDSS_PHASE_FACTORIZATION, cudss_cfg.cfg,
                     p_data->data, A_mat.mat, x_mat.mat,
                     b_mat.mat) != CUDSS_STATUS_SUCCESS)
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
      throw SolverError("CUDA eigensolver: failed to factorize C = K - sigma*M "
                        "(n=" +
                        std::to_string(n) + ", sigma=" + std::to_string(sigma) +
                        ")");
    working_mtype = CUDSS_MTYPE_GENERAL;
  }

  log_debug(
      "[cuda-eig] factorized C with " +
      std::string(working_mtype == CUDSS_MTYPE_SPD ? "SPD Cholesky" : "LU"));

  // Helper: single-RHS solve z = C^{-1} * b, reusing the factorization.
  // Updates b and x pointers via cudssMatrixSetValues (descriptors remain
  // alive).
  auto do_solve = [&](double *b_ptr, double *z_ptr) {
    ck(cudssMatrixSetValues(cudss_b_mat.mat, b_ptr), "cudssMatrixSetValues(b)");
    ck(cudssMatrixSetValues(cudss_x_mat.mat, z_ptr), "cudssMatrixSetValues(x)");
    ck(cudaMemset(z_ptr, 0, stride * sizeof(double)), "cudaMemset(z)");
    ck(cudssExecute(ctx_->cudss, CUDSS_PHASE_SOLVE, cudss_cfg.cfg, p_data->data,
                    cudss_A_mat.mat, cudss_x_mat.mat, cudss_b_mat.mat),
       "cudssExecute(SOLVE)");
  };

  // ── 3. cuSPARSE: create M SpMV descriptor + buffer ───────────────────────
  SpMatD sp_M;
  ck(cusparseCreateCsr(&sp_M.d, static_cast<int64_t>(n),
                       static_cast<int64_t>(n), static_cast<int64_t>(M_nnz),
                       d_M.rptr.ptr, d_M.cind.ptr, d_M.vals.ptr,
                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
     "cusparseCreateCsr(M)");

  // Query SpMV buffer size using the first-column pointers as placeholders.
  const double sp1 = 1.0, sp0 = 0.0;
  DnVecD sv_tmp, sw_tmp;
  ck(cusparseCreateDnVec(&sv_tmp.d, static_cast<int64_t>(n), d_V.ptr,
                         CUDA_R_64F),
     "cusparseCreateDnVec(v tmp)");
  ck(cusparseCreateDnVec(&sw_tmp.d, static_cast<int64_t>(n), d_W.ptr,
                         CUDA_R_64F),
     "cusparseCreateDnVec(w tmp)");

  std::size_t spmv_sz = 0;
  ck(cusparseSpMV_bufferSize(ctx_->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             &sp1, sp_M.d, sv_tmp.d, &sp0, sw_tmp.d, CUDA_R_64F,
                             CUSPARSE_SPMV_ALG_DEFAULT, &spmv_sz),
     "cusparseSpMV_bufferSize");

  EigDevBuf<char> d_spbuf(std::max(spmv_sz, std::size_t{1}));

  // Persistent SpMV vector descriptors — updated per step with SetValues.
  DnVecD sp_v, sp_w;
  ck(cusparseCreateDnVec(&sp_v.d, static_cast<int64_t>(n), d_V.ptr, CUDA_R_64F),
     "cusparseCreateDnVec(v)");
  ck(cusparseCreateDnVec(&sp_w.d, static_cast<int64_t>(n), d_W.ptr, CUDA_R_64F),
     "cusparseCreateDnVec(w)");

  auto spmv_Mv = [&](double *v, double *w) {
    ck(cusparseDnVecSetValues(sp_v.d, v), "cusparseDnVecSetValues(v)");
    ck(cusparseDnVecSetValues(sp_w.d, w), "cusparseDnVecSetValues(w)");
    ck(cusparseSpMV(ctx_->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &sp1,
                    sp_M.d, sp_v.d, &sp0, sp_w.d, CUDA_R_64F,
                    CUSPARSE_SPMV_ALG_DEFAULT, d_spbuf.ptr),
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

  // ── 5. Lanczos iterations with implicit restart ─────────────────────────
  std::vector<double> alpha(ncv_sz);
  std::vector<double> beta(ncv_sz);

  const double c1 = 1.0, cn1 = -1.0, c0 = 0.0;

  // Lambda: run Lanczos steps from from_k to to_m-1 (inclusive).
  // Returns the actual number of completed steps (may be < to_m on breakdown).
  // After return, d_r holds the unnormalised residual from the last step.
  auto run_lanczos = [&](int from_k, int to_m) -> int {
    int nstep = to_m;
    for (int k = from_k; k < to_m; ++k) {
      double *vk = d_V.ptr + static_cast<std::size_t>(k) * stride;
      double *wk = d_W.ptr + static_cast<std::size_t>(k) * stride;

      // z = C^{-1} w_k
      do_solve(wk, d_z.ptr);

      // alpha_k = w_k · z
      double alpha_k = 0.0;
      ck(cublasDdot(ctx_->cublas, n, wk, 1, d_z.ptr, 1, &alpha_k),
         "cublasDdot(alpha)");
      ck(cudaDeviceSynchronize(), "sync(alpha)");
      alpha[static_cast<std::size_t>(k)] = alpha_k;

      // r = z - alpha_k * v_k
      ck(cudaMemcpy(d_r.ptr, d_z.ptr, stride * sizeof(double),
                    cudaMemcpyDeviceToDevice),
         "memcpy z->r");
      double na = -alpha_k;
      ck(cublasDaxpy(ctx_->cublas, n, &na, vk, 1, d_r.ptr, 1),
         "cublasDaxpy(alpha)");

      // r -= beta_{k-1} * v_{k-1}
      if (k > 0) {
        double nb = -beta[static_cast<std::size_t>(k - 1)];
        double *vkm1 = d_V.ptr + static_cast<std::size_t>(k - 1) * stride;
        ck(cublasDaxpy(ctx_->cublas, n, &nb, vkm1, 1, d_r.ptr, 1),
           "cublasDaxpy(beta)");
      }

      // Double re-orthogonalization in M-inner product
      if (k > 0) {
        for (int pass = 0; pass < 2; ++pass) {
          ck(cublasDgemv(ctx_->cublas, CUBLAS_OP_T, n, k, &c1, d_W.ptr,
                         static_cast<int>(stride), d_r.ptr, 1, &c0, d_co.ptr,
                         1),
             "cublasDgemv(reorth T)");
          ck(cublasDgemv(ctx_->cublas, CUBLAS_OP_N, n, k, &cn1, d_V.ptr,
                         static_cast<int>(stride), d_co.ptr, 1, &c1, d_r.ptr,
                         1),
             "cublasDgemv(reorth N)");
        }
      }

      // w_{k+1} = M * r
      double *w_next = (k < to_m - 1)
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

      if (k < to_m - 1) {
        double *v_next = d_V.ptr + static_cast<std::size_t>(k + 1) * stride;
        ck(cudaMemcpy(v_next, d_r.ptr, stride * sizeof(double),
                      cudaMemcpyDeviceToDevice),
           "memcpy r->v_next");
        double inv = 1.0 / bk;
        ck(cublasDscal(ctx_->cublas, n, &inv, v_next, 1), "cublasDscal(v)");
        ck(cublasDscal(ctx_->cublas, n, &inv, w_next, 1), "cublasDscal(w)");
      }
    }
    return nstep;
  };

  // Initial full Lanczos factorization: ncv steps from scratch.
  int nstep = run_lanczos(0, ncv);

  log_debug("[cuda-eig] initial Lanczos done, nstep=" + std::to_string(nstep));

  // ── 6. Implicit restart loop ────────────────────────────────────────────
  int nconv = 0;
  int restart_iter = 0;

  // Build tridiagonal T from alpha/beta.
  auto build_T = [&](int ns) -> Eigen::MatrixXd {
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(ns, ns);
    for (int i = 0; i < ns; ++i)
      T(i, i) = alpha[static_cast<std::size_t>(i)];
    for (int i = 0; i < ns - 1; ++i) {
      T(i, i + 1) = beta[static_cast<std::size_t>(i)];
      T(i + 1, i) = beta[static_cast<std::size_t>(i)];
    }
    return T;
  };

  for (restart_iter = 0; restart_iter < maxit; ++restart_iter) {
    // 6a. Solve tridiagonal eigenproblem
    Eigen::MatrixXd T_mat = build_T(nstep);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_T(T_mat);
    if (eig_T.info() != Eigen::Success)
      throw SolverError(
          "CUDA eigensolver: tridiagonal eigendecomposition failed "
          "(nstep=" +
          std::to_string(nstep) + ", restart=" + std::to_string(restart_iter) +
          ")");

    const Eigen::VectorXd &ritz_val_raw = eig_T.eigenvalues();
    const Eigen::MatrixXd &ritz_vec_raw = eig_T.eigenvectors();

    // Sort Ritz pairs by descending |nu| (wanted = largest magnitude first)
    std::vector<int> sort_idx(static_cast<std::size_t>(nstep));
    std::iota(sort_idx.begin(), sort_idx.end(), 0);
    std::sort(sort_idx.begin(), sort_idx.end(), [&](int a, int b) {
      return std::abs(ritz_val_raw(a)) > std::abs(ritz_val_raw(b));
    });

    Eigen::VectorXd ritz_val(nstep);
    Eigen::VectorXd ritz_est(nstep);
    for (int i = 0; i < nstep; ++i) {
      int si = sort_idx[static_cast<std::size_t>(i)];
      ritz_val(i) = ritz_val_raw(si);
      ritz_est(i) = ritz_vec_raw(nstep - 1, si); // last row
    }

    // 6b. Check convergence
    double f_norm = (nstep > 0 && nstep <= ncv)
                        ? beta[static_cast<std::size_t>(nstep - 1)]
                        : 0.0;
    nconv =
        check_convergence(std::min(nd, nstep), ritz_val, ritz_est, f_norm, tol);

    log_debug("[cuda-eig] restart iter=" + std::to_string(restart_iter) +
              " nconv=" + std::to_string(nconv) + "/" + std::to_string(nd) +
              " f_norm=" + std::to_string(f_norm));

    if (nconv >= nd || nstep < ncv)
      break; // converged or breakdown

    // 6c. Compute adjusted nev and number of shifts
    int k = nev_adjusted(nd, ncv, nconv, ritz_est);
    int p = ncv - k;

    // 6d. Collect p unwanted shifts (from the tail of sorted Ritz values),
    //     sorted by descending magnitude for numerical stability.
    Eigen::VectorXd shifts(p);
    for (int i = 0; i < p; ++i)
      shifts(i) = ritz_val(k + i);
    // Already sorted by descending |magnitude| from the sort above.

    // 6e. Apply p implicit QR shifts to T on CPU
    Eigen::MatrixXd Q_total = Eigen::MatrixXd::Identity(ncv, ncv);
    for (int i = 0; i < p; ++i)
      apply_tridiag_qr_shift(T_mat, shifts(i), Q_total);

    // Extract updated alpha/beta from the shifted T
    for (int i = 0; i < k; ++i)
      alpha[static_cast<std::size_t>(i)] = T_mat(i, i);
    for (int i = 0; i < k - 1; ++i)
      beta[static_cast<std::size_t>(i)] = T_mat(i + 1, i);

    // 6f. Compress V and W on GPU: V_new[:, 0:k+1] = V_old * Q[:, 0:k+1]
    // Upload Q_total (column-major) to device.
    // Eigen stores column-major by default, so Q_total.data() is fine.
    d_Q.upload(Q_total.data(), ncv_sz * ncv_sz);

    // V_new = V_old * Q  via dgemm (only first k+1 columns of Q needed)
    ck(cublasDgemm(ctx_->cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, k + 1, ncv, &c1,
                   d_V.ptr, static_cast<int>(stride), d_Q.ptr, ncv, &c0,
                   d_temp.ptr, static_cast<int>(stride)),
       "cublasDgemm(V compress)");
    ck(cudaMemcpy(d_V.ptr, d_temp.ptr,
                  stride * static_cast<std::size_t>(k + 1) * sizeof(double),
                  cudaMemcpyDeviceToDevice),
       "memcpy temp->V");

    // W_new = W_old * Q
    ck(cublasDgemm(ctx_->cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, k + 1, ncv, &c1,
                   d_W.ptr, static_cast<int>(stride), d_Q.ptr, ncv, &c0,
                   d_temp.ptr, static_cast<int>(stride)),
       "cublasDgemm(W compress)");
    ck(cudaMemcpy(d_W.ptr, d_temp.ptr,
                  stride * static_cast<std::size_t>(k + 1) * sizeof(double),
                  cudaMemcpyDeviceToDevice),
       "memcpy temp->W");

    // 6g. Update residual vector:
    //   f_new = f_old * Q(ncv-1, k-1) + V_new[:,k] * T_shifted(k, k-1)
    // d_r still holds f_old (unnormalised residual from last Lanczos step).
    // V_new[:,k] is now in d_V + k*stride from the dgemm above.
    double q_last = Q_total(ncv - 1, k - 1);
    double h_kk1 = T_mat(k, k - 1);

    // f_new = q_last * d_r + h_kk1 * V_new[:,k]
    // Use d_z as scratch to build f_new.
    // Start with d_z = h_kk1 * V_new[:,k]
    double *vk_ptr = d_V.ptr + static_cast<std::size_t>(k) * stride;
    ck(cudaMemcpy(d_z.ptr, vk_ptr, stride * sizeof(double),
                  cudaMemcpyDeviceToDevice),
       "memcpy V[:,k]->z");
    ck(cublasDscal(ctx_->cublas, n, &h_kk1, d_z.ptr, 1), "cublasDscal(h_kk1)");
    // d_z += q_last * d_r
    ck(cublasDaxpy(ctx_->cublas, n, &q_last, d_r.ptr, 1, d_z.ptr, 1),
       "cublasDaxpy(q_last)");
    // Copy f_new to d_r
    ck(cudaMemcpy(d_r.ptr, d_z.ptr, stride * sizeof(double),
                  cudaMemcpyDeviceToDevice),
       "memcpy z->r(f_new)");

    // Compute ||f_new||_M and normalise into v_k, w_k
    spmv_Mv(d_r.ptr, d_W.ptr + static_cast<std::size_t>(k) * stride);
    double beta_sq = 0.0;
    ck(cublasDdot(ctx_->cublas, n, d_r.ptr, 1,
                  d_W.ptr + static_cast<std::size_t>(k) * stride, 1, &beta_sq),
       "cublasDdot(f_new norm)");
    ck(cudaDeviceSynchronize(), "sync(f_new)");

    if (beta_sq <= 0.0 || std::sqrt(beta_sq) < 1e-14) {
      nstep = k;
      log_debug("[cuda-eig] breakdown after restart compression at k=" +
                std::to_string(k));
      break;
    }

    double bk = std::sqrt(beta_sq);
    beta[static_cast<std::size_t>(k - 1)] = bk;

    // v_k = f_new / bk, w_k already computed
    ck(cudaMemcpy(vk_ptr, d_r.ptr, stride * sizeof(double),
                  cudaMemcpyDeviceToDevice),
       "memcpy r->v_k");
    double inv = 1.0 / bk;
    ck(cublasDscal(ctx_->cublas, n, &inv, vk_ptr, 1), "cublasDscal(v_k)");
    ck(cublasDscal(ctx_->cublas, n, &inv,
                   d_W.ptr + static_cast<std::size_t>(k) * stride, 1),
       "cublasDscal(w_k)");

    // 6h. Re-expand Lanczos from step k to ncv
    nstep = run_lanczos(k, ncv);
  }

  log_debug("[cuda-eig] IRL done after " + std::to_string(restart_iter + 1) +
            " restart(s), nconv=" + std::to_string(nconv) +
            ", nstep=" + std::to_string(nstep));

  // ── 7. Final Ritz extraction (cuDSS still live for RR refinement) ──────
  Eigen::MatrixXd T_final = build_T(nstep);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_final(T_final);
  if (eig_final.info() != Eigen::Success)
    throw SolverError(
        "CUDA eigensolver: final tridiagonal eigendecomposition failed "
        "(nstep=" +
        std::to_string(nstep) + ")");

  const int nd_actual = std::min(nd, nstep);
  const Eigen::VectorXd &nu_all = eig_final.eigenvalues();

  std::vector<int> idx(static_cast<std::size_t>(nstep));
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(
      idx.begin(), idx.begin() + nd_actual, idx.end(),
      [&](int a, int b) { return std::abs(nu_all(a)) > std::abs(nu_all(b)); });

  // Download V and compute initial Ritz vectors on CPU.
  std::vector<double> V_host(stride * static_cast<std::size_t>(nstep));
  ck(cudaMemcpy(V_host.data(), d_V.ptr,
                stride * static_cast<std::size_t>(nstep) * sizeof(double),
                cudaMemcpyDeviceToHost),
     "cudaMemcpy V D->H");

  using OuterStr = Eigen::OuterStride<Eigen::Dynamic>;
  Eigen::Map<const Eigen::MatrixXd, Eigen::Unaligned, OuterStr> V_map(
      V_host.data(), n, nstep, OuterStr(static_cast<Eigen::Index>(stride)));
  const Eigen::MatrixXd &Y_all = eig_final.eigenvectors();

  // Collect wanted Ritz pairs; filter degenerate nu (|nu| < 1e-20).
  std::vector<int> good;
  good.reserve(static_cast<std::size_t>(nd_actual));
  for (int i = 0; i < nd_actual; ++i) {
    int j = idx[static_cast<std::size_t>(i)];
    if (std::abs(nu_all(j)) >= 1e-20)
      good.push_back(j);
  }
  const int nd_rr = static_cast<int>(good.size());

  Eigen::MatrixXd X_rr(n, nd_rr);
  Eigen::VectorXd lambda_rr(nd_rr);
  for (int i = 0; i < nd_rr; ++i) {
    int j = good[static_cast<std::size_t>(i)];
    X_rr.col(i) = V_map * Y_all.col(j);
    lambda_rr(i) = sigma + 1.0 / nu_all(j);
  }

  // ── 8. Rayleigh-Ritz post-refinement (cuDSS still live) ────────────────
  // Each iteration:
  //   a. Apply A = C^{-1} M to each column of X on GPU -> subspace Z.
  //   b. M-orthonormalize Z via Cholesky of G = Z^T M Z on CPU.
  //   c. Project K: K_sub = Z_orth^T K Z_orth  (M_sub = I by construction).
  //   d. Solve the small standard eigenproblem K_sub y = lambda y.
  //   e. Update X = Z_orth * Y,  lambda = eigenvalues of K_sub.
  {
    const int nrr = 2;
    // stride-sized single-vector GPU scratch buffers reuse the cuDSS/SpMV
    // lambdas which zero/access `stride` elements at a time.
    EigDevBuf<double> d_xi(stride);
    EigDevBuf<double> d_mxi(stride);
    EigDevBuf<double> d_zi(stride);

    for (int rr = 0; rr < nrr; ++rr) {
      // (a) Z[:,i] = C^{-1} M X[:,i]
      Eigen::MatrixXd Z(n, nd_rr);
      for (int i = 0; i < nd_rr; ++i) {
        ck(cudaMemcpy(d_xi.ptr, X_rr.col(i).data(), n * sizeof(double),
                      cudaMemcpyHostToDevice),
           "RR upload x_i");
        spmv_Mv(d_xi.ptr, d_mxi.ptr);
        do_solve(d_mxi.ptr, d_zi.ptr);
        ck(cudaMemcpy(Z.col(i).data(), d_zi.ptr, n * sizeof(double),
                      cudaMemcpyDeviceToHost),
           "RR download z_i");
      }

      // (b) M-orthonormalize Z: G = Z^T M Z = L L^T  =>  Z_orth = Z L^{-T}
      //     Solve L Z_orth^T = Z^T (forward substitution), then transpose.
      Eigen::MatrixXd MZ(n, nd_rr);
      for (int i = 0; i < nd_rr; ++i)
        MZ.col(i) = M_rm * Z.col(i);
      Eigen::MatrixXd G = Z.transpose() * MZ;

      Eigen::LLT<Eigen::MatrixXd> llt(G);
      if (llt.info() != Eigen::Success) {
        log_debug("[cuda-eig] RR iter=" + std::to_string(rr) +
                  ": M-Cholesky failed, stopping refinement");
        break;
      }

      Eigen::MatrixXd ZT = Z.transpose();
      llt.matrixL().solveInPlace(ZT); // ZT = L^{-1} Z^T
      Eigen::MatrixXd Z_orth = ZT.transpose();

      // (c) K_sub = Z_orth^T K Z_orth
      Eigen::MatrixXd KZ(n, nd_rr);
      for (int i = 0; i < nd_rr; ++i)
        KZ.col(i) = K_rm * Z_orth.col(i);
      Eigen::MatrixXd K_sub = Z_orth.transpose() * KZ;
      K_sub = 0.5 * (K_sub + K_sub.transpose()); // enforce symmetry

      // (d) Solve K_sub y = lambda y  (M_sub = I)
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_sub(K_sub);
      if (eig_sub.info() != Eigen::Success) {
        log_debug("[cuda-eig] RR iter=" + std::to_string(rr) +
                  ": dense eigensolver failed, stopping refinement");
        break;
      }

      // (e) Update subspace
      X_rr = Z_orth * eig_sub.eigenvectors();
      lambda_rr = eig_sub.eigenvalues();

      log_debug("[cuda-eig] RR iter=" + std::to_string(rr) +
                " lambda[0]=" + std::to_string(lambda_rr(0)));
    }
  }

  // ── 9. Release cuDSS resources ──────────────────────────────────────────
  // Factorization is no longer needed.  Destroy descriptors before data.
  cudss_x_mat = EigCuDSSMat{};
  cudss_b_mat = EigCuDSSMat{};
  cudss_A_mat = EigCuDSSMat{};
  p_data.reset();

  // ── 10. Build output EigenPairs from refined vectors ────────────────────
  std::vector<EigenPair> results;
  results.reserve(static_cast<std::size_t>(nd_rr));
  for (int i = 0; i < nd_rr; ++i) {
    EigenPair ep;
    ep.eigenvalue = lambda_rr(i);
    ep.eigenvector = X_rr.col(i);
    results.push_back(std::move(ep));
  }

  if (results.empty())
    throw SolverError("CUDA eigensolver: no Ritz pairs converged "
                      "(nd=" +
                      std::to_string(nd) + ", nstep=" + std::to_string(nstep) +
                      ", sigma=" + std::to_string(sigma) +
                      ", restarts=" + std::to_string(restart_iter) + ")");

  std::sort(results.begin(), results.end(),
            [](const EigenPair &a, const EigenPair &b) {
              return a.eigenvalue < b.eigenvalue;
            });

  log_debug("[cuda-eig] returning " + std::to_string(results.size()) +
            " pairs, lambda[0]=" + std::to_string(results[0].eigenvalue));

  return results;
}

} // namespace vibestran
