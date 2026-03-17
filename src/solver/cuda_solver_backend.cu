// src/solver/cuda_solver_backend.cu
// CUDA sparse solver backend using NVIDIA cuDSS (direct sparse solver library).
//
// Solver selection logic:
//   1. Try CUDSS_MTYPE_SPD (Cholesky, AMD reordering).
//      Optimal for SPD FEM stiffness matrices; less fill-in and workspace than
//      LU.
//   2. If Cholesky factorisation fails or the residual is too large, retry with
//      CUDSS_MTYPE_GENERAL (LU with partial pivoting, AMD reordering) which
//      handles non-SPD and mildly ill-conditioned matrices.
//
// Both paths allocate GPU memory, execute all cuDSS phases on-device
// (ANALYSIS → FACTORIZATION → SOLVE), then copy the solution host←device.
//
// For large problems where cuDSS's internal workspace exceeds device memory,
// single-precision mode (try_create(use_single_precision=true)) halves the
// device memory footprint by downcasting inputs to float before the solve and
// upcasting the result back to double.
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
#include <cudss.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace nastran {

// ── RAII helpers
// ──────────────────────────────────────────────────────────────

struct CudaContext {
  cudssHandle_t cudss = nullptr;
  std::string device_name;
  bool use_single_precision = false;
};

// RAII wrapper for device memory allocated with cudaMalloc.
template <typename T> struct DeviceBuffer {
  T *ptr = nullptr;

  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) {
    if (count == 0)
      return;
    cudaError_t err =
        cudaMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T));
    if (err != cudaSuccess)
      throw SolverError("CUDA solver: cudaMalloc failed for " +
                        std::to_string(count) +
                        " elements: " + cudaGetErrorString(err));
  }
  ~DeviceBuffer() {
    if (ptr)
      cudaFree(ptr);
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
  DeviceBuffer &operator=(DeviceBuffer &&o) noexcept {
    if (this != &o) {
      if (ptr)
        cudaFree(ptr);
      ptr = o.ptr;
      o.ptr = nullptr;
    }
    return *this;
  }

  void upload(const T *host, std::size_t count) {
    cudaError_t err =
        cudaMemcpy(ptr, host, count * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      throw SolverError(std::string("CUDA solver: cudaMemcpy H→D failed: ") +
                        cudaGetErrorString(err));
  }

  void download(T *host, std::size_t count) const {
    cudaError_t err =
        cudaMemcpy(host, ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
      throw SolverError(std::string("CUDA solver: cudaMemcpy D→H failed: ") +
                        cudaGetErrorString(err));
  }
};

// RAII wrappers for cuDSS objects.
struct CuDSSConfig {
  cudssConfig_t cfg = nullptr;
  CuDSSConfig() {
    if (cudssConfigCreate(&cfg) != CUDSS_STATUS_SUCCESS)
      throw SolverError("CUDA solver: cudssConfigCreate failed");
  }
  ~CuDSSConfig() {
    if (cfg)
      cudssConfigDestroy(cfg);
  }
  CuDSSConfig(const CuDSSConfig &) = delete;
  CuDSSConfig &operator=(const CuDSSConfig &) = delete;
};

struct CuDSSData {
  cudssHandle_t handle;
  cudssData_t data = nullptr;
  explicit CuDSSData(cudssHandle_t h) : handle(h) {
    if (cudssDataCreate(handle, &data) != CUDSS_STATUS_SUCCESS)
      throw SolverError("CUDA solver: cudssDataCreate failed");
  }
  ~CuDSSData() {
    if (data)
      cudssDataDestroy(handle, data);
  }
  CuDSSData(const CuDSSData &) = delete;
  CuDSSData &operator=(const CuDSSData &) = delete;
};

struct CuDSSMatrix {
  cudssMatrix_t mat = nullptr;
  CuDSSMatrix() = default;
  ~CuDSSMatrix() {
    if (mat)
      cudssMatrixDestroy(mat);
  }
  CuDSSMatrix(const CuDSSMatrix &) = delete;
  CuDSSMatrix &operator=(const CuDSSMatrix &) = delete;
};

// ── Constructor / destructor
// ──────────────────────────────────────────────────

CudaSolverBackend::CudaSolverBackend(std::unique_ptr<CudaContext> ctx) noexcept
    : ctx_(std::move(ctx)) {}

CudaSolverBackend::~CudaSolverBackend() {
  if (!ctx_)
    return;
  if (ctx_->cudss)
    cudssDestroy(ctx_->cudss);
}

CudaSolverBackend::CudaSolverBackend(CudaSolverBackend &&) noexcept = default;
CudaSolverBackend &
CudaSolverBackend::operator=(CudaSolverBackend &&) noexcept = default;

// ── Factory
// ───────────────────────────────────────────────────────────────────

std::optional<CudaSolverBackend>
CudaSolverBackend::try_create(bool use_single_precision) noexcept {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
    return std::nullopt;

  if (cudaSetDevice(0) != cudaSuccess)
    return std::nullopt;

  auto ctx = std::make_unique<CudaContext>();

  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, 0) == cudaSuccess)
    ctx->device_name = props.name;
  ctx->use_single_precision = use_single_precision;

  if (cudssCreate(&ctx->cudss) != CUDSS_STATUS_SUCCESS)
    return std::nullopt;

  return CudaSolverBackend(std::move(ctx));
}

// ── Accessors
// ─────────────────────────────────────────────────────────────────

std::string_view CudaSolverBackend::name() const noexcept {
  return "CUDA cuDSS sparse direct solver";
}

bool CudaSolverBackend::last_solve_used_cholesky() const noexcept {
  return last_cholesky_;
}

std::string_view CudaSolverBackend::device_name() const noexcept {
  return ctx_->device_name;
}

bool CudaSolverBackend::uses_single_precision() const noexcept {
  return ctx_->use_single_precision;
}

// ── Residual validation
// ───────────────────────────────────────────────────────

// Compute relative residual ||K*u - F||_2 / ||F||_2 on the CPU.
static double relative_residual(const SparseMatrixBuilder::CsrData &K,
                                const std::vector<double> &u,
                                const std::vector<double> &F) {
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
  if (rhs_sq == 0.0)
    return (res_sq == 0.0) ? 0.0 : 1.0;
  return std::sqrt(res_sq / rhs_sq);
}

// ── cudss_execute_phase: thin error-checking wrapper
// ─────────────────────────────────────────────────────

static cudssStatus_t cudss_execute(cudssHandle_t handle, cudssPhase_t phase,
                                   cudssConfig_t cfg, cudssData_t data,
                                   cudssMatrix_t A, cudssMatrix_t x,
                                   cudssMatrix_t b) {
  return cudssExecute(handle, phase, cfg, data, A, x, b);
}

// ── solve_cudss: run analysis + factorization + solve for a given matrix type
// Returns false (without throwing) for recoverable factorisation failures so
// the caller can retry with a different matrix type.
// Throws SolverError for unrecoverable errors (bad status, alloc failure, etc.)
template <typename Scalar>
static bool solve_cudss(cudssHandle_t handle, const std::string &device_name,
                        int n, int nnz, DeviceBuffer<Scalar> &d_values,
                        DeviceBuffer<int> &d_row_ptr,
                        DeviceBuffer<int> &d_col_ind, DeviceBuffer<Scalar> &d_F,
                        DeviceBuffer<Scalar> &d_u, cudssMatrixType_t mtype,
                        cudaDataType_t scalar_type, const std::string &label) {
  // Reset solution buffer.
  cudaMemset(d_u.ptr, 0, (std::size_t)n * sizeof(Scalar));

  CuDSSConfig cfg;
  CuDSSData solver_data(handle);

  // Configure reordering: CUDSS_ALG_DEFAULT uses AMD-like nested dissection.
  cudssAlgType_t reorder_alg = CUDSS_ALG_DEFAULT;
  cudssConfigSet(cfg.cfg, CUDSS_CONFIG_REORDERING_ALG, &reorder_alg,
                 sizeof(reorder_alg));

  // Build the sparse matrix descriptor.
  // Pass rowEnd=NULL: cuDSS treats rowStart as a standard n+1 CSR offsets
  // array. For SPD/symmetric types, use MVIEW_UPPER — cuDSS reads only the
  // upper triangle and ignores any lower-triangle entries present in the full
  // matrix. For GENERAL, use MVIEW_FULL.
  cudssMatrixViewType_t mview =
      (mtype == CUDSS_MTYPE_GENERAL) ? CUDSS_MVIEW_FULL : CUDSS_MVIEW_UPPER;

  CuDSSMatrix A_mat;
  cudssStatus_t st = cudssMatrixCreateCsr(&A_mat.mat,
                                          /*nrows=*/static_cast<int64_t>(n),
                                          /*ncols=*/static_cast<int64_t>(n),
                                          /*nnz=*/static_cast<int64_t>(nnz),
                                          /*rowStart=*/d_row_ptr.ptr,
                                          /*rowEnd=*/nullptr,
                                          /*colIndices=*/d_col_ind.ptr,
                                          /*values=*/d_values.ptr,
                                          /*indexType=*/CUDA_R_32I,
                                          /*valueType=*/scalar_type, mtype,
                                          mview, CUDSS_BASE_ZERO);
  if (st != CUDSS_STATUS_SUCCESS)
    throw SolverError("CUDA solver: cudssMatrixCreateCsr failed, status=" +
                      std::to_string(static_cast<int>(st)));

  // RHS dense vector.
  CuDSSMatrix b_mat;
  st = cudssMatrixCreateDn(&b_mat.mat,
                           /*nrows=*/static_cast<int64_t>(n),
                           /*ncols=*/1,
                           /*ld=*/static_cast<int64_t>(n), d_F.ptr, scalar_type,
                           CUDSS_LAYOUT_COL_MAJOR);
  if (st != CUDSS_STATUS_SUCCESS)
    throw SolverError("CUDA solver: cudssMatrixCreateDn (rhs) failed, status=" +
                      std::to_string(static_cast<int>(st)));

  // Solution dense vector.
  CuDSSMatrix x_mat;
  st = cudssMatrixCreateDn(&x_mat.mat, static_cast<int64_t>(n), 1,
                           static_cast<int64_t>(n), d_u.ptr, scalar_type,
                           CUDSS_LAYOUT_COL_MAJOR);
  if (st != CUDSS_STATUS_SUCCESS)
    throw SolverError("CUDA solver: cudssMatrixCreateDn (sol) failed, status=" +
                      std::to_string(static_cast<int>(st)));

  constexpr double kMiB = 1024.0 * 1024.0;

  // Helper: enable hybrid mode on cfg and re-run analysis.
  // Returns the status of the re-run analysis.
  auto enable_hybrid_and_reanalyse = [&]() -> cudssStatus_t {
    int hybrid_mode = 1;
    cudssConfigSet(cfg.cfg, CUDSS_CONFIG_HYBRID_MODE, &hybrid_mode,
                   sizeof(hybrid_mode));
    cudssStatus_t s =
        cudss_execute(handle, CUDSS_PHASE_ANALYSIS, cfg.cfg, solver_data.data,
                      A_mat.mat, x_mat.mat, b_mat.mat);
    if (s == CUDSS_STATUS_SUCCESS) {
      int64_t dev_mem_min = 0;
      cudssDataGet(handle, solver_data.data,
                   CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN, &dev_mem_min,
                   sizeof(dev_mem_min), nullptr);
      std::clog
          << "[cuda] cuDSS hybrid mode active: min device memory required="
          << dev_mem_min / kMiB << " MiB\n";
    }
    return s;
  };

  // ── Analysis pass 1 (no hybrid) ──────────────────────────────────────────
  // Run analysis without hybrid first to obtain memory estimates cheaply.
  // If it fails with ALLOC_FAILED the symbolic phase itself needs more memory
  // than is available on device — enable hybrid and retry immediately.
  // It's not clear that trying analysis without hybrid first is the right
  // choice, it may be better to always use hybrid, since it will attempt to use
  // all vram if possible.
  st = cudss_execute(handle, CUDSS_PHASE_ANALYSIS, cfg.cfg, solver_data.data,
                     A_mat.mat, x_mat.mat, b_mat.mat);
  if (st == CUDSS_STATUS_ALLOC_FAILED) {
    std::clog << "[cuda] cuDSS (" << label
              << "): analysis alloc failed -- "
                 "enabling hybrid mode and retrying\n";
    st = enable_hybrid_and_reanalyse();
    if (st != CUDSS_STATUS_SUCCESS)
      throw SolverError(
          "CUDA solver: cuDSS analysis failed even in hybrid mode (" + label +
          "), status=" + std::to_string(static_cast<int>(st)) +
          ". Retry with --cuda-single-precision to halve GPU memory usage.");
  } else if (st != CUDSS_STATUS_SUCCESS) {
    throw SolverError("CUDA solver: cuDSS analysis failed (" + label +
                      "), status=" + std::to_string(static_cast<int>(st)));
  } else {
    // Analysis succeeded — check if factorisation peak would exceed device
    // memory. Layout: [0]=device stable, [1]=device peak, [2]=host stable,
    // [3]=host peak
    int64_t mem_est[16] = {0};
    cudssDataGet(handle, solver_data.data, CUDSS_DATA_MEMORY_ESTIMATES,
                 &mem_est, sizeof(mem_est), nullptr);
    std::clog << "[cuda] cuDSS memory estimates (" << label << "):"
              << " device stable=" << mem_est[0] / kMiB << " MiB"
              << " peak=" << mem_est[1] / kMiB << " MiB"
              << ", host stable=" << mem_est[2] / kMiB << " MiB"
              << " peak=" << mem_est[3] / kMiB << " MiB\n";

    std::size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    const int64_t threshold = static_cast<int64_t>(free_bytes * 0.85);
    if (mem_est[1] > threshold) {
      std::clog << "[cuda] cuDSS (" << label << "): device peak estimate "
                << mem_est[1] / kMiB << " MiB exceeds 85% of free memory ("
                << free_bytes / kMiB << " MiB) -- enabling hybrid mode\n";
      st = enable_hybrid_and_reanalyse();
      if (st != CUDSS_STATUS_SUCCESS)
        throw SolverError("CUDA solver: cuDSS analysis (hybrid) failed (" +
                          label +
                          "), status=" + std::to_string(static_cast<int>(st)));
    }
  }

  // ── Factorisation ─────────────────────────────────────────────────────────
  st = cudss_execute(handle, CUDSS_PHASE_FACTORIZATION, cfg.cfg,
                     solver_data.data, A_mat.mat, x_mat.mat, b_mat.mat);
  if (st == CUDSS_STATUS_ALLOC_FAILED)
    throw SolverError(
        "CUDA solver: cuDSS factorisation alloc failed -- matrix too large for "
        "device workspace (n=" +
        std::to_string(n) + ", nnz=" + std::to_string(nnz) +
        "). Retry with --cuda-single-precision to halve GPU memory usage.");
  // Non-success for SPD factorisation likely means matrix is not positive
  // definite; return false so the caller can retry with GENERAL type.
  if (st != CUDSS_STATUS_SUCCESS) {
    std::clog << "[cuda] cuDSS " << label
              << " factorisation failed, status=" << static_cast<int>(st)
              << " -- will retry with LU\n";
    return false;
  }

  // ── Solve ─────────────────────────────────────────────────────────────────
  st = cudss_execute(handle, CUDSS_PHASE_SOLVE, cfg.cfg, solver_data.data,
                     A_mat.mat, x_mat.mat, b_mat.mat);
  if (st != CUDSS_STATUS_SUCCESS)
    throw SolverError("CUDA solver: cuDSS solve failed (" + label +
                      "), status=" + std::to_string(static_cast<int>(st)));

  (void)device_name; // used by caller in log messages
  return true;
}

// ── solve_single_precision
// ────────────────────────────────────────────────────

static std::vector<double>
solve_single_precision(cudssHandle_t handle, const std::string &device_name,
                       const SparseMatrixBuilder::CsrData &K,
                       const std::vector<double> &F) {
  const int n = K.n;
  const int nnz = K.nnz;

  std::vector<float> values_f(nnz);
  std::vector<float> F_f(n);
  for (int i = 0; i < nnz; ++i)
    values_f[i] = static_cast<float>(K.values[i]);
  for (int i = 0; i < n; ++i)
    F_f[i] = static_cast<float>(F[i]);

  std::size_t alloc_bytes = (std::size_t)nnz * sizeof(float)     // d_values
                            + (std::size_t)(n + 1) * sizeof(int) // d_row_ptr
                            + (std::size_t)nnz * sizeof(int)     // d_col_ind
                            + (std::size_t)n * sizeof(float)     // d_F
                            + (std::size_t)n * sizeof(float);    // d_u
  std::size_t free_bytes = 0, total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  std::clog << "[cuda] single-precision device memory: allocating "
            << alloc_bytes / (1024.0 * 1024.0) << " MiB"
            << " (free=" << free_bytes / (1024.0 * 1024.0) << " MiB"
            << ", total=" << total_bytes / (1024.0 * 1024.0) << " MiB)\n";

  DeviceBuffer<float> d_values(nnz);
  DeviceBuffer<int> d_row_ptr(n + 1);
  DeviceBuffer<int> d_col_ind(nnz);
  DeviceBuffer<float> d_F(n);
  DeviceBuffer<float> d_u(n);

  d_values.upload(values_f.data(), nnz);
  d_row_ptr.upload(K.row_ptr.data(), n + 1);
  d_col_ind.upload(K.col_ind.data(), nnz);
  d_F.upload(F_f.data(), n);

  bool ok = solve_cudss<float>(handle, device_name, n, nnz, d_values, d_row_ptr,
                               d_col_ind, d_F, d_u, CUDSS_MTYPE_SPD, CUDA_R_32F,
                               "SPD/float");
  if (!ok) {
    std::clog << "[cuda] single-precision SPD failed -- retrying with LU\n";
    ok = solve_cudss<float>(handle, device_name, n, nnz, d_values, d_row_ptr,
                            d_col_ind, d_F, d_u, CUDSS_MTYPE_GENERAL,
                            CUDA_R_32F, "LU/float");
    if (!ok)
      throw SolverError(
          "CUDA solver: single-precision LU factorisation failed -- "
          "stiffness matrix may be singular. Check boundary conditions (SPCs)");
  }

  std::vector<float> u_f(n);
  d_u.download(u_f.data(), n);

  std::vector<double> u(n);
  for (int i = 0; i < n; ++i)
    u[i] = static_cast<double>(u_f[i]);

  std::clog << "[cuda] single-precision solve: n=" << n << ", nnz=" << nnz
            << ", device='" << device_name << "'\n";
  return u;
}

// ── solve
// ─────────────────────────────────────────────────────────────────────

std::vector<double>
CudaSolverBackend::solve(const SparseMatrixBuilder::CsrData &K,
                         const std::vector<double> &F) {
  const int n = K.n;
  const int nnz = K.nnz;

  if (n == 0)
    throw SolverError("CUDA solver: stiffness matrix is empty -- no free DOFs");
  if (static_cast<int>(F.size()) != n)
    throw SolverError("CUDA solver: force vector size " +
                      std::to_string(F.size()) + " != matrix size " +
                      std::to_string(n));

  if (ctx_->use_single_precision) {
    last_cholesky_ = true;
    return solve_single_precision(ctx_->cudss, ctx_->device_name, K, F);
  }

  // ── Allocate and upload ───────────────────────────────────────────────────
  {
    double alloc_mib = ((std::size_t)nnz * sizeof(double)    // d_values
                        + (std::size_t)(n + 1) * sizeof(int) // d_row_ptr
                        + (std::size_t)nnz * sizeof(int)     // d_col_ind
                        + (std::size_t)n * sizeof(double)    // d_F
                        + (std::size_t)n * sizeof(double))   // d_u
                       / (1024.0 * 1024.0);
    std::size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    std::clog << "[cuda] device memory: allocating " << alloc_mib << " MiB"
              << " (free=" << free_bytes / (1024.0 * 1024.0) << " MiB"
              << ", total=" << total_bytes / (1024.0 * 1024.0) << " MiB)\n";
  }

  DeviceBuffer<double> d_values(nnz);
  DeviceBuffer<int> d_row_ptr(n + 1);
  DeviceBuffer<int> d_col_ind(nnz);
  DeviceBuffer<double> d_F(n);
  DeviceBuffer<double> d_u(n);

  d_values.upload(K.values.data(), nnz);
  d_row_ptr.upload(K.row_ptr.data(), n + 1);
  d_col_ind.upload(K.col_ind.data(), nnz);
  d_F.upload(F.data(), n);

  // ── Path 1: Cholesky (SPD) ────────────────────────────────────────────────
  std::vector<double> u(n, 0.0);

  bool chol_ok = solve_cudss<double>(ctx_->cudss, ctx_->device_name, n, nnz,
                                     d_values, d_row_ptr, d_col_ind, d_F, d_u,
                                     CUDSS_MTYPE_SPD, CUDA_R_64F, "SPD");
  if (chol_ok) {
    d_u.download(u.data(), n);
    double rel_res = relative_residual(K, u, F);
    if (rel_res > 1e-2) {
      std::clog << "[cuda] Cholesky residual " << rel_res
                << " > 1e-2 -- retrying with LU\n";
      chol_ok = false;
    }
  }

  if (chol_ok) {
    last_cholesky_ = true;
    std::clog << "[cuda] Cholesky solve: n=" << n << ", nnz=" << nnz
              << ", device='" << ctx_->device_name << "'\n";
    return u;
  }

  // ── Path 2: LU (GENERAL) fallback ────────────────────────────────────────
  bool lu_ok = solve_cudss<double>(ctx_->cudss, ctx_->device_name, n, nnz,
                                   d_values, d_row_ptr, d_col_ind, d_F, d_u,
                                   CUDSS_MTYPE_GENERAL, CUDA_R_64F, "LU");
  if (!lu_ok)
    throw SolverError(
        "CUDA solver: LU factorisation failed -- stiffness matrix may be "
        "singular. Check boundary conditions (SPCs)");

  d_u.download(u.data(), n);

  double rel_res_lu = relative_residual(K, u, F);
  if (rel_res_lu > 1e-6)
    throw SolverError(
        "CUDA solver: LU produced large residual " +
        std::to_string(rel_res_lu) +
        " -- stiffness matrix is singular or very ill-conditioned. "
        "Check boundary conditions (SPCs)");

  last_cholesky_ = false;
  std::clog << "[cuda] LU solve: n=" << n << ", nnz=" << nnz << ", device='"
            << ctx_->device_name << "'\n";
  return u;
}

} // namespace nastran
