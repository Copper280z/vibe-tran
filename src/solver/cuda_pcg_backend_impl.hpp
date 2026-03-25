// src/solver/cuda_pcg_backend_impl.hpp
// Shared CUDA PCG implementation used by the stable and mixed backends.
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
// PCG loop (executed in solver scalar type T):
//   r = F, z = M^{-1}r, p = z, rz = r·z
//   while not converged:
//     Ap = K*p  (cusparseSpMV)
//     alpha = rz / (p·Ap)
//     u += alpha*p, r -= alpha*Ap
//     z = M^{-1}r
//     rz_new = r·z
//     if ||r||_2 / ||b||_2 < tol: done
//     p = z + (rz_new/rz)*p
//     rz = rz_new
//
// Dense vector descriptors (vec_r, vec_tmp, vec_z) are created during
// preconditioner setup and reused in every apply call — cusparseSpSV requires
// the same descriptor handles in solve() as were used in analysis().
//
// Note: <format> / std::format intentionally avoided — nvcc uses its bundled
// g++-12 as host compiler, which does not provide <format>.
#pragma once

#include "core/exceptions.hpp"
#include "core/logger.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
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

using HostClock = std::chrono::steady_clock;
using HostTimePoint = HostClock::time_point;

static double host_elapsed_ms(HostTimePoint start, HostTimePoint end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static std::string format_ms(double ms) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f", ms);
    return std::string(buf);
}

static bool cuda_pcg_profile_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("VIBESTRAN_CUDA_PCG_PROFILE");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

static double compute_true_relative_residual(
    const vibestran::SparseMatrixBuilder::CsrData& K,
    std::span<const double> x,
    std::span<const double> b,
    std::vector<double>* residual_out = nullptr)
{
    const std::vector<double> Ax = K.multiply(x);
    double r_norm_sq = 0.0;
    double b_norm_sq = 0.0;

    if (residual_out)
        residual_out->resize(b.size());

    for (std::size_t i = 0; i < b.size(); ++i) {
        const double ri = b[i] - Ax[i];
        r_norm_sq += ri * ri;
        b_norm_sq += b[i] * b[i];
        if (residual_out)
            (*residual_out)[i] = ri;
    }

    return (b_norm_sq > 1e-300)
        ? std::sqrt(r_norm_sq / b_norm_sq)
        : std::sqrt(r_norm_sq);
}

static void add_in_place(std::vector<double>& y, std::span<const double> x) {
    for (std::size_t i = 0; i < y.size(); ++i)
        y[i] += x[i];
}

class CublasPointerModeGuard {
public:
    CublasPointerModeGuard(cublasHandle_t handle, cublasPointerMode_t new_mode)
        : handle_(handle)
    {
        if (cublasGetPointerMode(handle_, &old_mode_) != CUBLAS_STATUS_SUCCESS)
            throw vibestran::SolverError("CUDA PCG: cublasGetPointerMode failed");
        if (cublasSetPointerMode(handle_, new_mode) != CUBLAS_STATUS_SUCCESS)
            throw vibestran::SolverError("CUDA PCG: cublasSetPointerMode failed");
    }

    ~CublasPointerModeGuard() {
        if (handle_)
            (void)cublasSetPointerMode(handle_, old_mode_);
    }

    CublasPointerModeGuard(const CublasPointerModeGuard&) = delete;
    CublasPointerModeGuard& operator=(const CublasPointerModeGuard&) = delete;

private:
    cublasHandle_t handle_;
    cublasPointerMode_t old_mode_ = CUBLAS_POINTER_MODE_HOST;
};

struct TimingStats {
    double total_ms = 0.0;
    double min_ms = std::numeric_limits<double>::infinity();
    double max_ms = 0.0;
    int count = 0;

    void add(double ms) {
        total_ms += ms;
        if (ms < min_ms)
            min_ms = ms;
        if (ms > max_ms)
            max_ms = ms;
        ++count;
    }
};

static std::string format_timing_stats(const TimingStats& stats) {
    if (stats.count == 0)
        return "count=0";

    const double avg_ms = stats.total_ms / static_cast<double>(stats.count);
    return "count=" + std::to_string(stats.count) +
           ", total=" + format_ms(stats.total_ms) + " ms" +
           ", avg=" + format_ms(avg_ms) + " ms" +
           ", min=" + format_ms(stats.min_ms) + " ms" +
           ", max=" + format_ms(stats.max_ms) + " ms";
}

static void log_profile_timing_line(
    const std::string& prefix,
    const char* label,
    const TimingStats& stats)
{
    if (stats.count == 0)
        return;
    vibestran::log_debug(prefix + label + ": " + format_timing_stats(stats));
}

static void throw_if_cuda_failed(cudaError_t err, const char* op) {
    if (err != cudaSuccess) {
        throw vibestran::SolverError(
            std::string("CUDA PCG: ") + op + " failed: " +
            cudaGetErrorString(err));
    }
}

class MaybeCudaEventPair {
public:
    explicit MaybeCudaEventPair(bool enabled) : enabled_(enabled) {
        if (!enabled_)
            return;
        throw_if_cuda_failed(cudaEventCreate(&start_), "cudaEventCreate(start)");
        throw_if_cuda_failed(cudaEventCreate(&stop_), "cudaEventCreate(stop)");
    }

    ~MaybeCudaEventPair() {
        if (start_)
            (void)cudaEventDestroy(start_);
        if (stop_)
            (void)cudaEventDestroy(stop_);
    }

    MaybeCudaEventPair(const MaybeCudaEventPair&) = delete;
    MaybeCudaEventPair& operator=(const MaybeCudaEventPair&) = delete;

    template<typename Fn>
    void record(Fn&& fn) {
        if (!enabled_) {
            fn();
            return;
        }
        throw_if_cuda_failed(cudaEventRecord(start_), "cudaEventRecord(start)");
        fn();
        throw_if_cuda_failed(cudaEventRecord(stop_), "cudaEventRecord(stop)");
        recorded_ = true;
    }

    [[nodiscard]] bool has_pending() const noexcept { return recorded_; }

    void wait() const {
        if (!enabled_ || !recorded_)
            return;
        throw_if_cuda_failed(cudaEventSynchronize(stop_), "cudaEventSynchronize(stop)");
    }

    double consume_ms() {
        if (!enabled_ || !recorded_)
            return 0.0;
        float ms = 0.0f;
        throw_if_cuda_failed(cudaEventElapsedTime(&ms, start_, stop_), "cudaEventElapsedTime");
        recorded_ = false;
        return static_cast<double>(ms);
    }

private:
    bool enabled_ = false;
    bool recorded_ = false;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

struct PrecondSetupProfile {
    bool enabled = false;
    const char* name = "";
    std::string scalar_name;
    TimingStats total;
    TimingStats value_copy;
    TimingStats factor_buffer_size;
    TimingStats factor_analysis;
    TimingStats factor_numeric;
    TimingStats zero_pivot_check;
    TimingStats sptrsv_fwd_buffer_size;
    TimingStats sptrsv_fwd_analysis;
    TimingStats sptrsv_bwd_buffer_size;
    TimingStats sptrsv_bwd_analysis;

    void log_summary(int n, int nnz) const {
        if (!enabled)
            return;
        const std::string prefix =
            "[cuda-pcg][profile] " + std::string(name) + " setup (" + scalar_name +
            ", n=" + std::to_string(n) + ", nnz=" + std::to_string(nnz) + ") ";
        log_profile_timing_line(prefix, "total", total);
        log_profile_timing_line(prefix, "value_copy", value_copy);
        log_profile_timing_line(prefix, "factor_buffer_size", factor_buffer_size);
        log_profile_timing_line(prefix, "factor_analysis", factor_analysis);
        log_profile_timing_line(prefix, "factor_numeric", factor_numeric);
        log_profile_timing_line(prefix, "zero_pivot_check", zero_pivot_check);
        log_profile_timing_line(prefix, "sptrsv_fwd_buffer_size", sptrsv_fwd_buffer_size);
        log_profile_timing_line(prefix, "sptrsv_fwd_analysis", sptrsv_fwd_analysis);
        log_profile_timing_line(prefix, "sptrsv_bwd_buffer_size", sptrsv_bwd_buffer_size);
        log_profile_timing_line(prefix, "sptrsv_bwd_analysis", sptrsv_bwd_analysis);
    }
};

struct PreparePcgProfile {
    bool enabled = false;
    std::string scalar_name;
    TimingStats total;
    TimingStats allocate_core_buffers;
    TimingStats upload_matrix_diag;
    TimingStats expand_symmetric_host;
    TimingStats preconditioner_setup;
    TimingStats mat_descriptor_create;
    TimingStats vec_descriptor_create;
    TimingStats spmv_buffer_size;
    TimingStats scalar_buffer_alloc;

    void log_summary(int n, int nnz, bool lower_only, const char* precond_name) const {
        if (!enabled)
            return;
        const std::string prefix =
            "[cuda-pcg][profile] prepare (" + scalar_name +
            ", n=" + std::to_string(n) + ", nnz=" + std::to_string(nnz) +
            ", lower_only=" + std::string(lower_only ? "true" : "false") +
            ", precond=" + precond_name + ") ";
        log_profile_timing_line(prefix, "total", total);
        log_profile_timing_line(prefix, "allocate_core_buffers", allocate_core_buffers);
        log_profile_timing_line(prefix, "upload_matrix_diag", upload_matrix_diag);
        log_profile_timing_line(prefix, "expand_symmetric_host", expand_symmetric_host);
        log_profile_timing_line(prefix, "preconditioner_setup", preconditioner_setup);
        log_profile_timing_line(prefix, "mat_descriptor_create", mat_descriptor_create);
        log_profile_timing_line(prefix, "vec_descriptor_create", vec_descriptor_create);
        log_profile_timing_line(prefix, "spmv_buffer_size", spmv_buffer_size);
        log_profile_timing_line(prefix, "scalar_buffer_alloc", scalar_buffer_alloc);
    }
};

enum class SolveGpuPhase : std::size_t {
    DotRr0,
    ZeroSolution,
    PrecondForward,
    PrecondBackward,
    Jacobi,
    CopyInitialSearch,
    DotRz0,
    SpmvNonTranspose,
    SpmvTranspose,
    DiagSubtract,
    DotPAp,
    PrepareScalars,
    AxpySolution,
    AxpyResidual,
    DotRzNew,
    DotRr,
    FinalizeIteration,
    UpdateSearch,
    Count
};

static constexpr std::array<const char*, static_cast<std::size_t>(SolveGpuPhase::Count)>
    kSolveGpuPhaseNames = {
        "dot_rr0",
        "zero_solution",
        "sptrsv_forward",
        "sptrsv_backward",
        "jacobi",
        "copy_initial_search",
        "dot_rz0",
        "spmv_non_transpose",
        "spmv_transpose",
        "diag_subtract",
        "dot_pAp",
        "prepare_scalars",
        "axpy_solution",
        "axpy_residual",
        "dot_rz_new",
        "dot_rr",
        "finalize_iteration",
        "update_search",
};

struct SolveProfile {
    bool enabled = false;
    std::string solve_label;
    std::string scalar_name;
    std::array<TimingStats, static_cast<std::size_t>(SolveGpuPhase::Count)> gpu_phase_stats{};
    TimingStats rhs_upload;
    TimingStats solution_zero;
    TimingStats rr0_download;
    TimingStats rz0_download;
    TimingStats iter_state_download;
    TimingStats solution_download;
    TimingStats wall_total;

    void add_gpu(SolveGpuPhase phase, double ms) {
        if (!enabled)
            return;
        gpu_phase_stats[static_cast<std::size_t>(phase)].add(ms);
    }

    void log_summary(
        const char* precond_name,
        int n,
        int nnz,
        int iterations,
        double rel_res) const
    {
        if (!enabled)
            return;

        double gpu_total_ms = 0.0;
        for (const TimingStats& stats : gpu_phase_stats)
            gpu_total_ms += stats.total_ms;

        const std::string prefix =
            "[cuda-pcg][profile] solve '" + solve_label + "' (" + scalar_name +
            ", precond=" + precond_name +
            ", n=" + std::to_string(n) +
            ", nnz=" + std::to_string(nnz) +
            ", iters=" + std::to_string(iterations) +
            ", rel_res=" + std::to_string(rel_res) + ") ";

        vibestran::log_debug(
            prefix + "gpu_accounted=" + format_ms(gpu_total_ms) + " ms" +
            ", wall=" + format_ms(wall_total.total_ms) + " ms");
        log_profile_timing_line(prefix, "rhs_upload", rhs_upload);
        log_profile_timing_line(prefix, "solution_zero", solution_zero);
        log_profile_timing_line(prefix, "rr0_download", rr0_download);
        log_profile_timing_line(prefix, "rz0_download", rz0_download);
        log_profile_timing_line(prefix, "iter_state_download", iter_state_download);
        log_profile_timing_line(prefix, "solution_download", solution_download);

        std::vector<std::size_t> order(gpu_phase_stats.size());
        for (std::size_t i = 0; i < order.size(); ++i)
            order[i] = i;
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return gpu_phase_stats[a].total_ms > gpu_phase_stats[b].total_ms;
        });

        for (std::size_t idx : order) {
            const TimingStats& stats = gpu_phase_stats[idx];
            if (stats.count == 0)
                continue;
            log_profile_timing_line(prefix, kSolveGpuPhaseNames[idx], stats);
        }
    }
};

struct RefinementProfile {
    bool enabled = false;
    TimingStats total;
    TimingStats diag_build;
    TimingStats downcast_inputs;
    TimingStats prepare_system;
    TimingStats rhs_cast;
    TimingStats inner_solve_wall;
    TimingStats accumulate_solution;
    TimingStats true_residual;

    void log_summary(int n, int nnz, int total_inner_iters, double true_rel_res) const {
        if (!enabled)
            return;
        const std::string prefix =
            "[cuda-pcg][profile] float32 outer (n=" + std::to_string(n) +
            ", nnz=" + std::to_string(nnz) +
            ", total_inner_iters=" + std::to_string(total_inner_iters) +
            ", true_rel_res=" + std::to_string(true_rel_res) + ") ";
        log_profile_timing_line(prefix, "total", total);
        log_profile_timing_line(prefix, "diag_build", diag_build);
        log_profile_timing_line(prefix, "downcast_inputs", downcast_inputs);
        log_profile_timing_line(prefix, "prepare_system", prepare_system);
        log_profile_timing_line(prefix, "rhs_cast", rhs_cast);
        log_profile_timing_line(prefix, "inner_solve_wall", inner_solve_wall);
        log_profile_timing_line(prefix, "accumulate_solution", accumulate_solution);
        log_profile_timing_line(prefix, "true_residual", true_residual);
    }
};

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

static void throw_if_cublas_failed(cublasStatus_t status, const char* op) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw SolverError(
            std::string("CUDA PCG: ") + op +
            " failed, status=" + std::to_string(static_cast<int>(status)));
    }
}

// ── Scalar type traits ────────────────────────────────────────────────────────
// Dispatch to the correct cuBLAS and cuSPARSE type-specific functions.

template<typename T> struct ScalarTraits;

template<> struct ScalarTraits<double> {
    using dot_type = double;
    static constexpr cudaDataType_t cuda_dtype = CUDA_R_64F;
    static const char* name() { return "float64"; }

    static void dot(cublasHandle_t h, int n,
        const double* x, const double* y, double* r) {
        throw_if_cublas_failed(cublasDdot(h, n, x, 1, y, 1, r), "cublasDdot");
    }
    static void axpy(cublasHandle_t h, int n,
        const double* alpha, const double* x, double* y) {
        throw_if_cublas_failed(cublasDaxpy(h, n, alpha, x, 1, y, 1), "cublasDaxpy");
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
    using dot_type = float;
    static constexpr cudaDataType_t cuda_dtype = CUDA_R_32F;
    static const char* name() { return "float32"; }

    static void dot(cublasHandle_t h, int n,
        const float* x, const float* y, float* r) {
        throw_if_cublas_failed(cublasSdot(h, n, x, 1, y, 1, r), "cublasSdot");
    }
    static void axpy(cublasHandle_t h, int n,
        const float* alpha, const float* x, float* y) {
        throw_if_cublas_failed(cublasSaxpy(h, n, alpha, x, 1, y, 1), "cublasSaxpy");
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

static constexpr int kBlock = 256;

template<typename T>
__global__ static void jacobi_kernel(
    const T* __restrict__ r,
    const T* __restrict__ diag,
    T* __restrict__ z,
    int n)
{
    int i = static_cast<int>(blockIdx.x) * blockDim.x +
            static_cast<int>(threadIdx.x);
    if (i < n) z[i] = r[i] / diag[i];
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

template<typename T>
__global__ static void subtract_diag_product_kernel(
    T* __restrict__ y,
    const T* __restrict__ diag,
    const T* __restrict__ x,
    int n)
{
    int i = static_cast<int>(blockIdx.x) * blockDim.x +
            static_cast<int>(threadIdx.x);
    if (i < n) y[i] -= diag[i] * x[i];
}

struct IterationHostState {
    double rel = 1.0;
    double pAp = 0.0;
    int flags = 0;
};

static constexpr int kIterFlagBreakdown = 1;
static constexpr int kIterFlagConverged = 2;

template<typename T>
__global__ static void prepare_iteration_scalars_kernel(
    const typename ScalarTraits<T>::dot_type* __restrict__ rz,
    const typename ScalarTraits<T>::dot_type* __restrict__ pAp,
    T* __restrict__ alpha,
    T* __restrict__ neg_alpha,
    IterationHostState* __restrict__ state)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    const double p_ap = static_cast<double>(*pAp);
    state->pAp = p_ap;
    state->flags = 0;
    state->rel = 1.0;

    if (!(p_ap > 0.0)) {
        *alpha = T{0};
        *neg_alpha = T{0};
        state->flags |= kIterFlagBreakdown;
        return;
    }

    const double rz_host = static_cast<double>(*rz);
    const T alpha_value = static_cast<T>(rz_host / p_ap);
    *alpha = alpha_value;
    *neg_alpha = -alpha_value;
}

template<typename T>
__global__ static void finalize_iteration_kernel(
    typename ScalarTraits<T>::dot_type* __restrict__ rz,
    const typename ScalarTraits<T>::dot_type* __restrict__ rz_new,
    const typename ScalarTraits<T>::dot_type* __restrict__ rr,
    T* __restrict__ beta,
    double norm_b,
    double tolerance,
    IterationHostState* __restrict__ state)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    if ((state->flags & kIterFlagBreakdown) != 0) {
        state->rel = 1e300;
        *beta = T{0};
        return;
    }

    const double rr_host = static_cast<double>(*rr);
    state->rel = std::sqrt(std::abs(rr_host)) / norm_b;

    const double rz_host = static_cast<double>(*rz);
    const double rz_new_host = static_cast<double>(*rz_new);

    if (state->rel < tolerance) {
        state->flags |= kIterFlagConverged;
        *beta = T{0};
        *rz = *rz_new;
        return;
    }

    *beta = static_cast<T>(rz_new_host / rz_host);
    *rz = *rz_new;
}

template<typename T>
__global__ static void update_search_direction_kernel(
    const T* __restrict__ z,
    const T* __restrict__ beta,
    T* __restrict__ p,
    int n)
{
    int i = static_cast<int>(blockIdx.x) * blockDim.x +
            static_cast<int>(threadIdx.x);
    if (i < n)
        p[i] = z[i] + (*beta) * p[i];
}

template<typename T>
static void launch_jacobi(const T* r, const T* diag, T* z, int n) {
    jacobi_kernel<T><<<(n + kBlock - 1) / kBlock, kBlock>>>(r, diag, z, n);
}

template<typename T>
static void launch_axpby(const T* x, T a, T b, T* y, int n) {
    axpby_kernel<T><<<(n + kBlock - 1) / kBlock, kBlock>>>(x, a, b, y, n);
}

template<typename T>
static void launch_subtract_diag_product(T* y, const T* diag, const T* x, int n) {
    subtract_diag_product_kernel<T><<<(n + kBlock - 1) / kBlock, kBlock>>>(
        y, diag, x, n);
}

template<typename T>
static void launch_prepare_iteration_scalars(
    const typename ScalarTraits<T>::dot_type* rz,
    const typename ScalarTraits<T>::dot_type* pAp,
    T* alpha,
    T* neg_alpha,
    IterationHostState* state)
{
    prepare_iteration_scalars_kernel<T><<<1, 1>>>(rz, pAp, alpha, neg_alpha, state);
}

template<typename T>
static void launch_finalize_iteration(
    typename ScalarTraits<T>::dot_type* rz,
    const typename ScalarTraits<T>::dot_type* rz_new,
    const typename ScalarTraits<T>::dot_type* rr,
    T* beta,
    double norm_b,
    double tolerance,
    IterationHostState* state)
{
    finalize_iteration_kernel<T><<<1, 1>>>(
        rz, rz_new, rr, beta, norm_b, tolerance, state);
}

template<typename T>
static void launch_update_search_direction(
    const T* z,
    const T* beta,
    T* p,
    int n)
{
    update_search_direction_kernel<T><<<(n + kBlock - 1) / kBlock, kBlock>>>(
        z, beta, p, n);
}


template<typename T>
struct HostCsr {
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<T>   values;
    int nnz = 0;
};

template<typename T>
static HostCsr<T> expand_symmetric_host_csr(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind,
    const std::vector<T>& values)
{
    HostCsr<T> full;
    full.row_ptr.assign(static_cast<std::size_t>(n + 1), 0);

    int offdiag = 0;
    for (int row = 0; row < n; ++row) {
        for (int idx = row_ptr[static_cast<std::size_t>(row)];
             idx < row_ptr[static_cast<std::size_t>(row + 1)]; ++idx) {
            ++full.row_ptr[static_cast<std::size_t>(row + 1)];
            if (col_ind[static_cast<std::size_t>(idx)] != row) {
                ++full.row_ptr[static_cast<std::size_t>(
                    col_ind[static_cast<std::size_t>(idx)] + 1)];
                ++offdiag;
            }
        }
    }

    for (int i = 0; i < n; ++i)
        full.row_ptr[static_cast<std::size_t>(i + 1)] +=
            full.row_ptr[static_cast<std::size_t>(i)];

    full.nnz = static_cast<int>(values.size()) + offdiag;
    full.col_ind.resize(static_cast<std::size_t>(full.nnz));
    full.values.resize(static_cast<std::size_t>(full.nnz));

    std::vector<int> cursor(full.row_ptr.begin(), full.row_ptr.begin() + n);
    for (int row = 0; row < n; ++row) {
        for (int idx = row_ptr[static_cast<std::size_t>(row)];
             idx < row_ptr[static_cast<std::size_t>(row + 1)]; ++idx) {
            const int col = col_ind[static_cast<std::size_t>(idx)];
            const T value = values[static_cast<std::size_t>(idx)];

            int out = cursor[static_cast<std::size_t>(row)]++;
            full.col_ind[static_cast<std::size_t>(out)] = col;
            full.values[static_cast<std::size_t>(out)] = value;

            if (col != row) {
                out = cursor[static_cast<std::size_t>(col)]++;
                full.col_ind[static_cast<std::size_t>(out)] = row;
                full.values[static_cast<std::size_t>(out)] = value;
            }
        }
    }

    return full;
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

template<typename T>
struct PreparedPcgSystem {
    using DotT = typename ScalarTraits<T>::dot_type;

    int n = 0;
    int nnz = 0;
    bool lower_only = false;
    PrecondKind precond_kind = PrecondKind::Jacobi;

    PCGDeviceBuffer<T>   d_values;
    PCGDeviceBuffer<int> d_row_ptr;
    PCGDeviceBuffer<int> d_col_ind;
    PCGDeviceBuffer<T>   d_diag;

    PCGDeviceBuffer<T> d_u;
    PCGDeviceBuffer<T> d_r;
    PCGDeviceBuffer<T> d_z;
    PCGDeviceBuffer<T> d_p;
    PCGDeviceBuffer<T> d_Ap;

    std::unique_ptr<PCGDeviceBuffer<T>> d_ilu_values;
    std::unique_ptr<PCGDeviceBuffer<int>> d_ilu_row_ptr;
    std::unique_ptr<PCGDeviceBuffer<int>> d_ilu_col_ind;
    std::unique_ptr<TriangPrecond<T>> tp;

    cusparseSpMatDescr_t mat_K = nullptr;
    cusparseDnVecDescr_t vec_p = nullptr;
    cusparseDnVecDescr_t vec_Ap = nullptr;
    PCGDeviceBuffer<char> d_spmv_buf;

    PCGDeviceBuffer<DotT> d_rz;
    PCGDeviceBuffer<DotT> d_pAp;
    PCGDeviceBuffer<DotT> d_rz_new;
    PCGDeviceBuffer<DotT> d_rr;
    PCGDeviceBuffer<T> d_alpha;
    PCGDeviceBuffer<T> d_neg_alpha;
    PCGDeviceBuffer<T> d_beta;
    PCGDeviceBuffer<IterationHostState> d_iter_state;

    PreparedPcgSystem() = default;
    ~PreparedPcgSystem() {
        if (vec_p)  cusparseDestroyDnVec(vec_p);
        if (vec_Ap) cusparseDestroyDnVec(vec_Ap);
        if (mat_K)  cusparseDestroySpMat(mat_K);
    }

    PreparedPcgSystem(const PreparedPcgSystem&) = delete;
    PreparedPcgSystem& operator=(const PreparedPcgSystem&) = delete;
    PreparedPcgSystem(PreparedPcgSystem&& other) noexcept
        : n(other.n)
        , nnz(other.nnz)
        , lower_only(other.lower_only)
        , precond_kind(other.precond_kind)
        , d_values(std::move(other.d_values))
        , d_row_ptr(std::move(other.d_row_ptr))
        , d_col_ind(std::move(other.d_col_ind))
        , d_diag(std::move(other.d_diag))
        , d_u(std::move(other.d_u))
        , d_r(std::move(other.d_r))
        , d_z(std::move(other.d_z))
        , d_p(std::move(other.d_p))
        , d_Ap(std::move(other.d_Ap))
        , d_ilu_values(std::move(other.d_ilu_values))
        , d_ilu_row_ptr(std::move(other.d_ilu_row_ptr))
        , d_ilu_col_ind(std::move(other.d_ilu_col_ind))
        , tp(std::move(other.tp))
        , mat_K(other.mat_K)
        , vec_p(other.vec_p)
        , vec_Ap(other.vec_Ap)
        , d_spmv_buf(std::move(other.d_spmv_buf))
        , d_rz(std::move(other.d_rz))
        , d_pAp(std::move(other.d_pAp))
        , d_rz_new(std::move(other.d_rz_new))
        , d_rr(std::move(other.d_rr))
        , d_alpha(std::move(other.d_alpha))
        , d_neg_alpha(std::move(other.d_neg_alpha))
        , d_beta(std::move(other.d_beta))
        , d_iter_state(std::move(other.d_iter_state))
    {
        other.mat_K = nullptr;
        other.vec_p = nullptr;
        other.vec_Ap = nullptr;
    }

    PreparedPcgSystem& operator=(PreparedPcgSystem&& other) noexcept {
        if (this == &other)
            return *this;

        if (vec_p)  cusparseDestroyDnVec(vec_p);
        if (vec_Ap) cusparseDestroyDnVec(vec_Ap);
        if (mat_K)  cusparseDestroySpMat(mat_K);

        n = other.n;
        nnz = other.nnz;
        lower_only = other.lower_only;
        precond_kind = other.precond_kind;
        d_values = std::move(other.d_values);
        d_row_ptr = std::move(other.d_row_ptr);
        d_col_ind = std::move(other.d_col_ind);
        d_diag = std::move(other.d_diag);
        d_u = std::move(other.d_u);
        d_r = std::move(other.d_r);
        d_z = std::move(other.d_z);
        d_p = std::move(other.d_p);
        d_Ap = std::move(other.d_Ap);
        d_ilu_values = std::move(other.d_ilu_values);
        d_ilu_row_ptr = std::move(other.d_ilu_row_ptr);
        d_ilu_col_ind = std::move(other.d_ilu_col_ind);
        tp = std::move(other.tp);
        mat_K = other.mat_K;
        vec_p = other.vec_p;
        vec_Ap = other.vec_Ap;
        d_spmv_buf = std::move(other.d_spmv_buf);
        d_rz = std::move(other.d_rz);
        d_pAp = std::move(other.d_pAp);
        d_rz_new = std::move(other.d_rz_new);
        d_rr = std::move(other.d_rr);
        d_alpha = std::move(other.d_alpha);
        d_neg_alpha = std::move(other.d_neg_alpha);
        d_beta = std::move(other.d_beta);
        d_iter_state = std::move(other.d_iter_state);

        other.mat_K = nullptr;
        other.vec_p = nullptr;
        other.vec_Ap = nullptr;
        return *this;
    }

    [[nodiscard]] const char* precond_name() const noexcept {
        return precond_kind == PrecondKind::IC0 ? "IC0" :
               precond_kind == PrecondKind::ILU0 ? "ILU0" : "Jacobi";
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
    TriangPrecond<T>& tp,
    PrecondSetupProfile* profile = nullptr)
{
    using Tr = ScalarTraits<T>;
    constexpr double kMiB = 1024.0 * 1024.0;
    const bool profile_enabled = profile != nullptr && profile->enabled;
    const HostTimePoint total_start = HostClock::now();
    const auto finalize_profile = [&]() {
        if (!profile_enabled)
            return;
        profile->total.add(host_elapsed_ms(total_start, HostClock::now()));
        profile->log_summary(n, nnz);
    };

    tp.d_M_vals = PCGDeviceBuffer<T>(nnz);
    tp.d_tmp    = PCGDeviceBuffer<T>(n);

    {
        const HostTimePoint t0 = HostClock::now();
        cudaMemcpy(tp.d_M_vals.ptr, d_values,
                   static_cast<std::size_t>(nnz) * sizeof(T),
                   cudaMemcpyDeviceToDevice);
        if (profile_enabled)
            profile->value_copy.add(host_elapsed_ms(t0, HostClock::now()));
    }

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
    cusparseStatus_t cs = CUSPARSE_STATUS_SUCCESS;
    {
        const HostTimePoint t0 = HostClock::now();
        cs = Tr::csric02_buffer_size(
            cusparse, n, nnz, descr,
            tp.d_M_vals.ptr, d_row_ptr, d_col_ind, info, &buf_size);
        if (profile_enabled)
            profile->factor_buffer_size.add(host_elapsed_ms(t0, HostClock::now()));
    }
    if (cs != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroyCsric02Info(info);
        cusparseDestroyMatDescr(descr);
        finalize_profile();
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

    PCGDeviceBuffer<char> d_factor_buf(buf_size > 0 ? buf_size : 1);

    {
        MaybeCudaEventPair evt(profile_enabled);
        evt.record([&]() {
            Tr::csric02_analysis(cusparse, n, nnz, descr,
                tp.d_M_vals.ptr, d_row_ptr, d_col_ind, info,
                CUSPARSE_SOLVE_POLICY_NO_LEVEL, d_factor_buf.ptr);
        });
        if (profile_enabled) {
            evt.wait();
            profile->factor_analysis.add(evt.consume_ms());
        }
    }

    int structural_zero = -1;
    {
        const HostTimePoint t0 = HostClock::now();
        cusparseXcsric02_zeroPivot(cusparse, info, &structural_zero);
        if (profile_enabled)
            profile->zero_pivot_check.add(host_elapsed_ms(t0, HostClock::now()));
    }
    if (structural_zero >= 0 || cs != CUSPARSE_STATUS_SUCCESS) {
        vibestran::log_warn("[cuda-pcg] IC0: structural zero at row " +
                           std::to_string(structural_zero) + " -- retrying with ILU0");
        cusparseDestroyCsric02Info(info);
        cusparseDestroyMatDescr(descr);
        finalize_profile();
        return false;
    }

    {
        MaybeCudaEventPair evt(profile_enabled);
        evt.record([&]() {
            cs = Tr::csric02(cusparse, n, nnz, descr,
                tp.d_M_vals.ptr, d_row_ptr, d_col_ind, info,
                CUSPARSE_SOLVE_POLICY_NO_LEVEL, d_factor_buf.ptr);
        });
        if (profile_enabled) {
            evt.wait();
            profile->factor_numeric.add(evt.consume_ms());
        }
    }

    int numerical_zero = -1;
    {
        const HostTimePoint t0 = HostClock::now();
        cusparseXcsric02_zeroPivot(cusparse, info, &numerical_zero);
        if (profile_enabled)
            profile->zero_pivot_check.add(host_elapsed_ms(t0, HostClock::now()));
    }
    cusparseDestroyCsric02Info(info);
    cusparseDestroyMatDescr(descr);

    if (numerical_zero >= 0 || cs != CUSPARSE_STATUS_SUCCESS) {
        vibestran::log_warn("[cuda-pcg] IC0: numerical zero at row " +
                           std::to_string(numerical_zero) + " -- retrying with ILU0");
        finalize_profile();
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
    {
        const HostTimePoint t0 = HostClock::now();
        cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
            tp.mat_L, tp.vec_r, tp.vec_tmp,
            Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_L, &sz);
        if (profile_enabled)
            profile->sptrsv_fwd_buffer_size.add(host_elapsed_ms(t0, HostClock::now()));
    }
    vibestran::log_debug("[cuda-pcg] IC0 SpSV forward scratch=" +
                        std::to_string(sz / kMiB) + " MiB");
    tp.d_sv_L_buf = PCGDeviceBuffer<char>(sz > 0 ? sz : 1);
    {
        MaybeCudaEventPair evt(profile_enabled);
        evt.record([&]() {
            cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                tp.mat_L, tp.vec_r, tp.vec_tmp,
                Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_L, tp.d_sv_L_buf.ptr);
        });
        if (profile_enabled) {
            evt.wait();
            profile->sptrsv_fwd_analysis.add(evt.consume_ms());
        }
    }

    sz = 0;
    {
        const HostTimePoint t0 = HostClock::now();
        cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_TRANSPOSE, &one,
            tp.mat_L, tp.vec_tmp, tp.vec_z,
            Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_UT, &sz);
        if (profile_enabled)
            profile->sptrsv_bwd_buffer_size.add(host_elapsed_ms(t0, HostClock::now()));
    }
    vibestran::log_debug("[cuda-pcg] IC0 SpSV backward scratch=" +
                        std::to_string(sz / kMiB) + " MiB");
    tp.d_sv_UT_buf = PCGDeviceBuffer<char>(sz > 0 ? sz : 1);
    {
        MaybeCudaEventPair evt(profile_enabled);
        evt.record([&]() {
            cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_TRANSPOSE, &one,
                tp.mat_L, tp.vec_tmp, tp.vec_z,
                Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_UT, tp.d_sv_UT_buf.ptr);
        });
        if (profile_enabled) {
            evt.wait();
            profile->sptrsv_bwd_analysis.add(evt.consume_ms());
        }
    }

    log_mem("after IC0 setup");
    vibestran::log_debug("[cuda-pcg] IC0 preconditioner setup successful");
    finalize_profile();
    return true;
}

// ── ILU0 factorization and SpSV setup ────────────────────────────────────────
template<typename T>
static bool setup_ilu0(
    cusparseHandle_t cusparse,
    int n, int nnz,
    const int* d_row_ptr, const int* d_col_ind, const T* d_values,
    T* d_r, T* d_z,
    TriangPrecond<T>& tp,
    PrecondSetupProfile* profile = nullptr)
{
    using Tr = ScalarTraits<T>;
    constexpr double kMiB = 1024.0 * 1024.0;
    const bool profile_enabled = profile != nullptr && profile->enabled;
    const HostTimePoint total_start = HostClock::now();
    const auto finalize_profile = [&]() {
        if (!profile_enabled)
            return;
        profile->total.add(host_elapsed_ms(total_start, HostClock::now()));
        profile->log_summary(n, nnz);
    };

    tp.d_M_vals = PCGDeviceBuffer<T>(nnz);
    tp.d_tmp    = PCGDeviceBuffer<T>(n);

    {
        const HostTimePoint t0 = HostClock::now();
        cudaMemcpy(tp.d_M_vals.ptr, d_values,
                   static_cast<std::size_t>(nnz) * sizeof(T),
                   cudaMemcpyDeviceToDevice);
        if (profile_enabled)
            profile->value_copy.add(host_elapsed_ms(t0, HostClock::now()));
    }

    cusparseMatDescr_t descr = nullptr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    csrilu02Info_t info = nullptr;
    cusparseCreateCsrilu02Info(&info);

    int buf_size = 0;
    cusparseStatus_t cs = CUSPARSE_STATUS_SUCCESS;
    {
        const HostTimePoint t0 = HostClock::now();
        cs = Tr::csrilu02_buffer_size(
            cusparse, n, nnz, descr,
            tp.d_M_vals.ptr, d_row_ptr, d_col_ind, info, &buf_size);
        if (profile_enabled)
            profile->factor_buffer_size.add(host_elapsed_ms(t0, HostClock::now()));
    }
    if (cs != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroyCsrilu02Info(info);
        cusparseDestroyMatDescr(descr);
        finalize_profile();
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

    PCGDeviceBuffer<char> d_factor_buf(buf_size > 0 ? buf_size : 1);

    {
        MaybeCudaEventPair evt(profile_enabled);
        evt.record([&]() {
            Tr::csrilu02_analysis(cusparse, n, nnz, descr,
                tp.d_M_vals.ptr, d_row_ptr, d_col_ind, info,
                CUSPARSE_SOLVE_POLICY_NO_LEVEL, d_factor_buf.ptr);
        });
        if (profile_enabled) {
            evt.wait();
            profile->factor_analysis.add(evt.consume_ms());
        }
    }

    int structural_zero = -1;
    {
        const HostTimePoint t0 = HostClock::now();
        cusparseXcsrilu02_zeroPivot(cusparse, info, &structural_zero);
        if (profile_enabled)
            profile->zero_pivot_check.add(host_elapsed_ms(t0, HostClock::now()));
    }
    if (structural_zero >= 0) {
        vibestran::log_warn("[cuda-pcg] ILU0: structural zero at row " +
                           std::to_string(structural_zero) + " -- falling back to Jacobi");
        cusparseDestroyCsrilu02Info(info);
        cusparseDestroyMatDescr(descr);
        finalize_profile();
        return false;
    }

    {
        MaybeCudaEventPair evt(profile_enabled);
        evt.record([&]() {
            cs = Tr::csrilu02(cusparse, n, nnz, descr,
                tp.d_M_vals.ptr, d_row_ptr, d_col_ind, info,
                CUSPARSE_SOLVE_POLICY_NO_LEVEL, d_factor_buf.ptr);
        });
        if (profile_enabled) {
            evt.wait();
            profile->factor_numeric.add(evt.consume_ms());
        }
    }

    int numerical_zero = -1;
    {
        const HostTimePoint t0 = HostClock::now();
        cusparseXcsrilu02_zeroPivot(cusparse, info, &numerical_zero);
        if (profile_enabled)
            profile->zero_pivot_check.add(host_elapsed_ms(t0, HostClock::now()));
    }
    cusparseDestroyCsrilu02Info(info);
    cusparseDestroyMatDescr(descr);

    if (numerical_zero >= 0 || cs != CUSPARSE_STATUS_SUCCESS) {
        vibestran::log_warn("[cuda-pcg] ILU0: numerical zero at row " +
                           std::to_string(numerical_zero) + " -- falling back to Jacobi");
        finalize_profile();
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
    {
        const HostTimePoint t0 = HostClock::now();
        cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
            tp.mat_L, tp.vec_r, tp.vec_tmp,
            Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_L, &sz);
        if (profile_enabled)
            profile->sptrsv_fwd_buffer_size.add(host_elapsed_ms(t0, HostClock::now()));
    }
    vibestran::log_debug("[cuda-pcg] ILU0 SpSV forward scratch=" +
                        std::to_string(sz / kMiB) + " MiB");
    tp.d_sv_L_buf = PCGDeviceBuffer<char>(sz > 0 ? sz : 1);
    {
        MaybeCudaEventPair evt(profile_enabled);
        evt.record([&]() {
            cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                tp.mat_L, tp.vec_r, tp.vec_tmp,
                Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_L, tp.d_sv_L_buf.ptr);
        });
        if (profile_enabled) {
            evt.wait();
            profile->sptrsv_fwd_analysis.add(evt.consume_ms());
        }
    }

    sz = 0;
    {
        const HostTimePoint t0 = HostClock::now();
        cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
            tp.mat_U, tp.vec_tmp, tp.vec_z,
            Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_UT, &sz);
        if (profile_enabled)
            profile->sptrsv_bwd_buffer_size.add(host_elapsed_ms(t0, HostClock::now()));
    }
    vibestran::log_debug("[cuda-pcg] ILU0 SpSV backward scratch=" +
                        std::to_string(sz / kMiB) + " MiB");
    tp.d_sv_UT_buf = PCGDeviceBuffer<char>(sz > 0 ? sz : 1);
    {
        MaybeCudaEventPair evt(profile_enabled);
        evt.record([&]() {
            cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                tp.mat_U, tp.vec_tmp, tp.vec_z,
                Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, tp.sv_UT, tp.d_sv_UT_buf.ptr);
        });
        if (profile_enabled) {
            evt.wait();
            profile->sptrsv_bwd_analysis.add(evt.consume_ms());
        }
    }

    log_mem("after ILU0 setup");
    vibestran::log_debug("[cuda-pcg] ILU0 preconditioner setup successful");
    finalize_profile();
    return true;
}

// ── Typed PCG prepare + execute ──────────────────────────────────────────────
template<typename T>
static PreparedPcgSystem<T> prepare_pcg_system(
    cusparseHandle_t cusparse,
    int n, int nnz,
    const std::vector<T>& h_values,
    const std::vector<int>& h_row_ptr,
    const std::vector<int>& h_col_ind,
    const std::vector<T>& h_diag,
    bool lower_only)
{
    using Tr = ScalarTraits<T>;
    using DotT = typename PreparedPcgSystem<T>::DotT;
    constexpr double kMiB = 1024.0 * 1024.0;
    PreparePcgProfile profile;
    profile.enabled = cuda_pcg_profile_enabled();
    profile.scalar_name = Tr::name();
    const HostTimePoint total_start = HostClock::now();

    const std::size_t bytes_matrix  = static_cast<std::size_t>(nnz) * sizeof(T)
                                    + static_cast<std::size_t>(n + 1) * sizeof(int)
                                    + static_cast<std::size_t>(nnz) * sizeof(int);
    const std::size_t bytes_vectors = 6UL * static_cast<std::size_t>(n) * sizeof(T);
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

    PreparedPcgSystem<T> prepared;
    prepared.n = n;
    prepared.nnz = nnz;
    prepared.lower_only = lower_only;

    {
        const HostTimePoint t0 = HostClock::now();
        prepared.d_values = PCGDeviceBuffer<T>(nnz);
        prepared.d_row_ptr = PCGDeviceBuffer<int>(n + 1);
        prepared.d_col_ind = PCGDeviceBuffer<int>(nnz);
        prepared.d_diag = PCGDeviceBuffer<T>(n);
        prepared.d_u = PCGDeviceBuffer<T>(n);
        prepared.d_r = PCGDeviceBuffer<T>(n);
        prepared.d_z = PCGDeviceBuffer<T>(n);
        prepared.d_p = PCGDeviceBuffer<T>(n);
        prepared.d_Ap = PCGDeviceBuffer<T>(n);
        if (profile.enabled)
            profile.allocate_core_buffers.add(host_elapsed_ms(t0, HostClock::now()));
    }

    {
        const HostTimePoint t0 = HostClock::now();
        prepared.d_values.upload(h_values.data(), nnz);
        prepared.d_row_ptr.upload(h_row_ptr.data(), n + 1);
        prepared.d_col_ind.upload(h_col_ind.data(), nnz);
        prepared.d_diag.upload(h_diag.data(), n);
        if (profile.enabled)
            profile.upload_matrix_diag.add(host_elapsed_ms(t0, HostClock::now()));
    }

    {
        const HostTimePoint t0 = HostClock::now();
        auto try_tp = std::make_unique<TriangPrecond<T>>();
        PrecondSetupProfile ic0_profile;
        ic0_profile.enabled = profile.enabled;
        ic0_profile.name = "IC0";
        ic0_profile.scalar_name = Tr::name();
        if (setup_ic0<T>(cusparse, n, nnz,
                         prepared.d_row_ptr.ptr, prepared.d_col_ind.ptr, prepared.d_values.ptr,
                         prepared.d_r.ptr, prepared.d_z.ptr, *try_tp, &ic0_profile)) {
            prepared.precond_kind = PrecondKind::IC0;
            prepared.tp = std::move(try_tp);
        } else {
            auto try_ilu = std::make_unique<TriangPrecond<T>>();
            bool ilu_ok = false;
            if (lower_only) {
                const HostTimePoint expand_start = HostClock::now();
                HostCsr<T> full_host =
                    expand_symmetric_host_csr(n, h_row_ptr, h_col_ind, h_values);
                if (profile.enabled)
                    profile.expand_symmetric_host.add(
                        host_elapsed_ms(expand_start, HostClock::now()));
                prepared.d_ilu_values = std::make_unique<PCGDeviceBuffer<T>>(full_host.nnz);
                prepared.d_ilu_row_ptr = std::make_unique<PCGDeviceBuffer<int>>(n + 1);
                prepared.d_ilu_col_ind = std::make_unique<PCGDeviceBuffer<int>>(full_host.nnz);
                prepared.d_ilu_values->upload(full_host.values.data(), full_host.nnz);
                prepared.d_ilu_row_ptr->upload(full_host.row_ptr.data(), n + 1);
                prepared.d_ilu_col_ind->upload(full_host.col_ind.data(), full_host.nnz);
                PrecondSetupProfile ilu_profile;
                ilu_profile.enabled = profile.enabled;
                ilu_profile.name = "ILU0";
                ilu_profile.scalar_name = Tr::name();
                ilu_ok = setup_ilu0<T>(cusparse, n, full_host.nnz,
                                       prepared.d_ilu_row_ptr->ptr,
                                       prepared.d_ilu_col_ind->ptr,
                                       prepared.d_ilu_values->ptr,
                                       prepared.d_r.ptr, prepared.d_z.ptr,
                                       *try_ilu, &ilu_profile);
            } else {
                PrecondSetupProfile ilu_profile;
                ilu_profile.enabled = profile.enabled;
                ilu_profile.name = "ILU0";
                ilu_profile.scalar_name = Tr::name();
                ilu_ok = setup_ilu0<T>(cusparse, n, nnz,
                                       prepared.d_row_ptr.ptr,
                                       prepared.d_col_ind.ptr,
                                       prepared.d_values.ptr,
                                       prepared.d_r.ptr, prepared.d_z.ptr,
                                       *try_ilu, &ilu_profile);
            }
            if (ilu_ok) {
                prepared.precond_kind = PrecondKind::ILU0;
                prepared.tp = std::move(try_ilu);
            } else {
                vibestran::log_debug("[cuda-pcg] Using Jacobi preconditioner");
            }
        }
        if (profile.enabled)
            profile.preconditioner_setup.add(host_elapsed_ms(t0, HostClock::now()));
    }

    {
        const HostTimePoint t0 = HostClock::now();
        cusparseCreateCsr(&prepared.mat_K,
            static_cast<int64_t>(n), static_cast<int64_t>(n),
            static_cast<int64_t>(nnz),
            prepared.d_row_ptr.ptr, prepared.d_col_ind.ptr, prepared.d_values.ptr,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, Tr::cuda_dtype);
        if (profile.enabled)
            profile.mat_descriptor_create.add(host_elapsed_ms(t0, HostClock::now()));
    }

    {
        const HostTimePoint t0 = HostClock::now();
        cusparseCreateDnVec(&prepared.vec_p,  n, prepared.d_p.ptr,  Tr::cuda_dtype);
        cusparseCreateDnVec(&prepared.vec_Ap, n, prepared.d_Ap.ptr, Tr::cuda_dtype);
        if (profile.enabled)
            profile.vec_descriptor_create.add(host_elapsed_ms(t0, HostClock::now()));
    }

    const T one{1};
    const T zero{0};
    const T one_accum{1};

    std::size_t spmv_sz = 0;
    {
        const HostTimePoint t0 = HostClock::now();
        cusparseSpMV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, prepared.mat_K, prepared.vec_p, &zero, prepared.vec_Ap,
            Tr::cuda_dtype, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_sz);
        if (lower_only) {
            std::size_t spmv_trans_sz = 0;
            cusparseSpMV_bufferSize(cusparse, CUSPARSE_OPERATION_TRANSPOSE,
                &one, prepared.mat_K, prepared.vec_p, &one_accum, prepared.vec_Ap,
                Tr::cuda_dtype, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_trans_sz);
            if (spmv_trans_sz > spmv_sz)
                spmv_sz = spmv_trans_sz;
        }
        if (profile.enabled)
            profile.spmv_buffer_size.add(host_elapsed_ms(t0, HostClock::now()));
    }
    vibestran::log_debug("[cuda-pcg] SpMV scratch=" + std::to_string(spmv_sz / kMiB) + " MiB");
    log_mem("after all PCG allocs");
    {
        const HostTimePoint t0 = HostClock::now();
        prepared.d_spmv_buf = PCGDeviceBuffer<char>(spmv_sz > 0 ? spmv_sz : 1);

        prepared.d_rz = PCGDeviceBuffer<DotT>(1);
        prepared.d_pAp = PCGDeviceBuffer<DotT>(1);
        prepared.d_rz_new = PCGDeviceBuffer<DotT>(1);
        prepared.d_rr = PCGDeviceBuffer<DotT>(1);
        prepared.d_alpha = PCGDeviceBuffer<T>(1);
        prepared.d_neg_alpha = PCGDeviceBuffer<T>(1);
        prepared.d_beta = PCGDeviceBuffer<T>(1);
        prepared.d_iter_state = PCGDeviceBuffer<IterationHostState>(1);
        if (profile.enabled)
            profile.scalar_buffer_alloc.add(host_elapsed_ms(t0, HostClock::now()));
    }

    if (profile.enabled) {
        profile.total.add(host_elapsed_ms(total_start, HostClock::now()));
        profile.log_summary(n, nnz, lower_only, prepared.precond_name());
    }

    return prepared;
}

template<typename T>
static std::vector<double> run_pcg_prepared(
    cublasHandle_t cublas,
    cusparseHandle_t cusparse,
    const std::string& device_name,
    std::span<const T> h_F,
    double tolerance,
    int max_iters,
    int& out_iters,
    double& out_rel_res,
    PreparedPcgSystem<T>& prepared,
    std::string_view solve_label = "solve")
{
    using Tr = ScalarTraits<T>;
    using DotT = typename PreparedPcgSystem<T>::DotT;

    const int n = prepared.n;
    const int nnz = prepared.nnz;
    const T one{1};
    const T zero{0};
    const T one_accum{1};
    SolveProfile profile;
    profile.enabled = cuda_pcg_profile_enabled();
    profile.solve_label = std::string(solve_label);
    profile.scalar_name = Tr::name();
    const HostTimePoint wall_start = HostClock::now();

    MaybeCudaEventPair evt_dot_rr0(profile.enabled);
    MaybeCudaEventPair evt_zero_solution(profile.enabled);
    MaybeCudaEventPair evt_precond_fwd(profile.enabled);
    MaybeCudaEventPair evt_precond_bwd(profile.enabled);
    MaybeCudaEventPair evt_jacobi(profile.enabled);
    MaybeCudaEventPair evt_copy_initial_search(profile.enabled);
    MaybeCudaEventPair evt_dot_rz0(profile.enabled);
    MaybeCudaEventPair evt_spmv_n(profile.enabled);
    MaybeCudaEventPair evt_spmv_t(profile.enabled);
    MaybeCudaEventPair evt_diag_subtract(profile.enabled);
    MaybeCudaEventPair evt_dot_pAp(profile.enabled);
    MaybeCudaEventPair evt_prepare_scalars(profile.enabled);
    MaybeCudaEventPair evt_axpy_solution(profile.enabled);
    MaybeCudaEventPair evt_axpy_residual(profile.enabled);
    MaybeCudaEventPair evt_dot_rz_new(profile.enabled);
    MaybeCudaEventPair evt_dot_rr(profile.enabled);
    MaybeCudaEventPair evt_finalize(profile.enabled);
    MaybeCudaEventPair evt_update_search(profile.enabled);
    bool update_pending = false;

    const auto add_gpu_phase = [&](SolveGpuPhase phase, MaybeCudaEventPair& evt) {
        if (!profile.enabled || !evt.has_pending())
            return;
        profile.add_gpu(phase, evt.consume_ms());
    };

    auto apply_matrix = [&]() {
        cusparseStatus_t cs = CUSPARSE_STATUS_SUCCESS;
        evt_spmv_n.record([&]() {
            cs = cusparseSpMV(
                cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one, prepared.mat_K, prepared.vec_p, &zero, prepared.vec_Ap,
                Tr::cuda_dtype, CUSPARSE_SPMV_ALG_DEFAULT, prepared.d_spmv_buf.ptr);
        });
        if (cs != CUSPARSE_STATUS_SUCCESS)
            throw SolverError("CUDA PCG: cusparseSpMV failed");

        if (!prepared.lower_only)
            return;

        evt_spmv_t.record([&]() {
            cs = cusparseSpMV(
                cusparse, CUSPARSE_OPERATION_TRANSPOSE,
                &one, prepared.mat_K, prepared.vec_p, &one_accum, prepared.vec_Ap,
                Tr::cuda_dtype, CUSPARSE_SPMV_ALG_DEFAULT, prepared.d_spmv_buf.ptr);
        });
        if (cs != CUSPARSE_STATUS_SUCCESS)
            throw SolverError("CUDA PCG: cusparseSpMV transpose failed");

        evt_diag_subtract.record([&]() {
            launch_subtract_diag_product<T>(
                prepared.d_Ap.ptr, prepared.d_diag.ptr, prepared.d_p.ptr, n);
        });
    };

    auto apply_precond = [&]() {
        if (prepared.precond_kind != PrecondKind::Jacobi) {
            evt_precond_fwd.record([&]() {
                cusparseSpSV_solve(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one, prepared.tp->mat_L, prepared.tp->vec_r, prepared.tp->vec_tmp,
                    Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, prepared.tp->sv_L);
            });
            if (prepared.precond_kind == PrecondKind::IC0) {
                evt_precond_bwd.record([&]() {
                    cusparseSpSV_solve(cusparse, CUSPARSE_OPERATION_TRANSPOSE,
                        &one, prepared.tp->mat_L, prepared.tp->vec_tmp, prepared.tp->vec_z,
                        Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, prepared.tp->sv_UT);
                });
            } else {
                evt_precond_bwd.record([&]() {
                    cusparseSpSV_solve(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &one, prepared.tp->mat_U, prepared.tp->vec_tmp, prepared.tp->vec_z,
                        Tr::cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT, prepared.tp->sv_UT);
                });
            }
        } else {
            evt_jacobi.record([&]() {
                launch_jacobi<T>(prepared.d_r.ptr, prepared.d_diag.ptr, prepared.d_z.ptr, n);
            });
        }
    };

    {
        const HostTimePoint t0 = HostClock::now();
        prepared.d_r.upload(h_F.data(), static_cast<std::size_t>(n));
        if (profile.enabled)
            profile.rhs_upload.add(host_elapsed_ms(t0, HostClock::now()));
    }
    evt_zero_solution.record([&]() {
        prepared.d_u.zero(n);
    });

    CublasPointerModeGuard pointer_mode_guard(cublas, CUBLAS_POINTER_MODE_DEVICE);

    evt_dot_rr0.record([&]() {
        Tr::dot(cublas, n, prepared.d_r.ptr, prepared.d_r.ptr, prepared.d_rr.ptr);
    });
    DotT rr0_T{0};
    {
        const HostTimePoint t0 = HostClock::now();
        prepared.d_rr.download(&rr0_T, 1);
        if (profile.enabled)
            profile.rr0_download.add(host_elapsed_ms(t0, HostClock::now()));
    }
    add_gpu_phase(SolveGpuPhase::DotRr0, evt_dot_rr0);
    add_gpu_phase(SolveGpuPhase::ZeroSolution, evt_zero_solution);
    const double norm_b = std::sqrt(std::abs(static_cast<double>(rr0_T)));
    if (norm_b < 1e-300) {
        out_iters = 0;
        out_rel_res = 0.0;
        if (profile.enabled) {
            profile.wall_total.add(host_elapsed_ms(wall_start, HostClock::now()));
            profile.log_summary(prepared.precond_name(), n, nnz, out_iters, out_rel_res);
        }
        return std::vector<double>(n, 0.0);
    }

    apply_precond();

    evt_copy_initial_search.record([&]() {
        cudaMemcpy(prepared.d_p.ptr, prepared.d_z.ptr,
                   static_cast<std::size_t>(n) * sizeof(T),
                   cudaMemcpyDeviceToDevice);
    });

    evt_dot_rz0.record([&]() {
        Tr::dot(cublas, n, prepared.d_r.ptr, prepared.d_z.ptr, prepared.d_rz.ptr);
    });
    DotT rz_init_T{0};
    {
        const HostTimePoint t0 = HostClock::now();
        prepared.d_rz.download(&rz_init_T, 1);
        if (profile.enabled)
            profile.rz0_download.add(host_elapsed_ms(t0, HostClock::now()));
    }
    if (prepared.precond_kind != PrecondKind::Jacobi) {
        add_gpu_phase(SolveGpuPhase::PrecondForward, evt_precond_fwd);
        add_gpu_phase(SolveGpuPhase::PrecondBackward, evt_precond_bwd);
    } else {
        add_gpu_phase(SolveGpuPhase::Jacobi, evt_jacobi);
    }
    add_gpu_phase(SolveGpuPhase::CopyInitialSearch, evt_copy_initial_search);
    add_gpu_phase(SolveGpuPhase::DotRz0, evt_dot_rz0);
    const double rz_init = static_cast<double>(rz_init_T);
    if (rz_init <= 0.0)
        throw SolverError("CUDA PCG: non-positive initial r*z=" +
                          std::to_string(rz_init) +
                          " -- preconditioner is not SPD or matrix is singular");

    int iter = 0;
    double rel = 1.0;
    IterationHostState iter_state{};
    for (; iter < max_iters; ++iter) {
        try {
            apply_matrix();
        } catch (const SolverError& e) {
            throw SolverError(std::string(e.what()) + " at iteration " +
                              std::to_string(iter));
        }

        evt_dot_pAp.record([&]() {
            Tr::dot(cublas, n, prepared.d_p.ptr, prepared.d_Ap.ptr, prepared.d_pAp.ptr);
        });
        evt_prepare_scalars.record([&]() {
            launch_prepare_iteration_scalars<T>(
                prepared.d_rz.ptr, prepared.d_pAp.ptr,
                prepared.d_alpha.ptr, prepared.d_neg_alpha.ptr,
                prepared.d_iter_state.ptr);
        });

        evt_axpy_solution.record([&]() {
            Tr::axpy(cublas, n, prepared.d_alpha.ptr, prepared.d_p.ptr, prepared.d_u.ptr);
        });
        evt_axpy_residual.record([&]() {
            Tr::axpy(cublas, n, prepared.d_neg_alpha.ptr, prepared.d_Ap.ptr, prepared.d_r.ptr);
        });

        apply_precond();

        evt_dot_rz_new.record([&]() {
            Tr::dot(cublas, n, prepared.d_r.ptr, prepared.d_z.ptr, prepared.d_rz_new.ptr);
        });
        evt_dot_rr.record([&]() {
            Tr::dot(cublas, n, prepared.d_r.ptr, prepared.d_r.ptr, prepared.d_rr.ptr);
        });

        evt_finalize.record([&]() {
            launch_finalize_iteration<T>(
                prepared.d_rz.ptr, prepared.d_rz_new.ptr, prepared.d_rr.ptr,
                prepared.d_beta.ptr, norm_b, tolerance, prepared.d_iter_state.ptr);
        });
        {
            const HostTimePoint t0 = HostClock::now();
            prepared.d_iter_state.download(&iter_state, 1);
            if (profile.enabled)
                profile.iter_state_download.add(host_elapsed_ms(t0, HostClock::now()));
        }
        if (profile.enabled && update_pending) {
            add_gpu_phase(SolveGpuPhase::UpdateSearch, evt_update_search);
            update_pending = false;
        }
        add_gpu_phase(SolveGpuPhase::SpmvNonTranspose, evt_spmv_n);
        add_gpu_phase(SolveGpuPhase::SpmvTranspose, evt_spmv_t);
        add_gpu_phase(SolveGpuPhase::DiagSubtract, evt_diag_subtract);
        add_gpu_phase(SolveGpuPhase::DotPAp, evt_dot_pAp);
        add_gpu_phase(SolveGpuPhase::PrepareScalars, evt_prepare_scalars);
        add_gpu_phase(SolveGpuPhase::AxpySolution, evt_axpy_solution);
        add_gpu_phase(SolveGpuPhase::AxpyResidual, evt_axpy_residual);
        if (prepared.precond_kind != PrecondKind::Jacobi) {
            add_gpu_phase(SolveGpuPhase::PrecondForward, evt_precond_fwd);
            add_gpu_phase(SolveGpuPhase::PrecondBackward, evt_precond_bwd);
        } else {
            add_gpu_phase(SolveGpuPhase::Jacobi, evt_jacobi);
        }
        add_gpu_phase(SolveGpuPhase::DotRzNew, evt_dot_rz_new);
        add_gpu_phase(SolveGpuPhase::DotRr, evt_dot_rr);
        add_gpu_phase(SolveGpuPhase::FinalizeIteration, evt_finalize);
        rel = iter_state.rel;

        if ((iter_state.flags & kIterFlagBreakdown) != 0) {
            vibestran::log_warn(
                "[cuda-pcg] non-positive p*Ap=" + std::to_string(iter_state.pAp) +
                " at iteration " + std::to_string(iter) +
                " -- matrix may not be positive definite or"
                " preconditioner is ill-conditioned; stopping.");
            iter = max_iters;
            break;
        }

        if ((iter_state.flags & kIterFlagConverged) != 0) {
            ++iter;
            break;
        }

        evt_update_search.record([&]() {
            launch_update_search_direction<T>(
                prepared.d_z.ptr, prepared.d_beta.ptr, prepared.d_p.ptr, n);
        });
        update_pending = profile.enabled;
    }

    if (profile.enabled && update_pending) {
        evt_update_search.wait();
        add_gpu_phase(SolveGpuPhase::UpdateSearch, evt_update_search);
    }

    out_iters = iter;
    out_rel_res = rel;

    if (iter >= max_iters)
        throw SolverError(
            "CUDA PCG: did not converge after " + std::to_string(max_iters) +
            " iterations (relative residual " +
            std::to_string(out_rel_res) + " > tolerance " +
            std::to_string(tolerance) +
            "). Consider increasing max iterations or using a direct solver.");

    vibestran::log_info(
        std::string("[cuda-pcg] ") + prepared.precond_name() +
        " (" + Tr::name() + ") converged in " + std::to_string(out_iters) +
        " iterations, rel_res=" + std::to_string(out_rel_res) +
        ", n=" + std::to_string(n) + ", nnz=" + std::to_string(nnz) +
        ", device='" + device_name + "'");

    std::vector<double> u(static_cast<std::size_t>(n));
    if constexpr (std::is_same_v<T, double>) {
        const HostTimePoint t0 = HostClock::now();
        prepared.d_u.download(u.data(), n);
        if (profile.enabled)
            profile.solution_download.add(host_elapsed_ms(t0, HostClock::now()));
    } else {
        std::vector<T> u_T(static_cast<std::size_t>(n));
        const HostTimePoint t0 = HostClock::now();
        prepared.d_u.download(u_T.data(), n);
        if (profile.enabled)
            profile.solution_download.add(host_elapsed_ms(t0, HostClock::now()));
        for (int i = 0; i < n; ++i)
            u[static_cast<std::size_t>(i)] = static_cast<double>(u_T[static_cast<std::size_t>(i)]);
    }
    if (profile.enabled) {
        profile.wall_total.add(host_elapsed_ms(wall_start, HostClock::now()));
        profile.log_summary(prepared.precond_name(), n, nnz, out_iters, out_rel_res);
    }
    return u;
}

// Performs the full PCG iteration in scalar type T. K values and F must
// already be on the host in type T (caller handles downcasting for float32).
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
    const std::vector<T>& h_diag,
    bool lower_only,
    double tolerance,
    int max_iters,
    int& out_iters,
    double& out_rel_res)
{
    auto prepared = prepare_pcg_system<T>(
        cusparse, n, nnz, h_values, h_row_ptr, h_col_ind, h_diag, lower_only);
    return run_pcg_prepared<T>(
        cublas, cusparse, device_name, h_F, tolerance, max_iters,
        out_iters, out_rel_res, prepared, "solve");
}

// ── Context ───────────────────────────────────────────────────────────────────

struct CudaPCGContext {
    cublasHandle_t   cublas   = nullptr;
    cusparseHandle_t cusparse = nullptr;
    std::string      device_name;
    double           tolerance = 0.0;
    int              max_iters = 10000;

    int    last_iters   = 0;
    double last_rel_res = 0.0;
};

static std::unique_ptr<CudaPCGContext>
try_create_cuda_pcg_context(double tolerance, int max_iters) noexcept {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        return nullptr;
    if (cudaSetDevice(0) != cudaSuccess)
        return nullptr;

    auto ctx = std::make_unique<CudaPCGContext>();

    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, 0) == cudaSuccess)
        ctx->device_name = props.name;

    ctx->tolerance = tolerance;
    ctx->max_iters = (max_iters > 0) ? max_iters : 10000;

    if (cublasCreate(&ctx->cublas) != CUBLAS_STATUS_SUCCESS)
        return nullptr;
    if (cusparseCreate(&ctx->cusparse) != CUSPARSE_STATUS_SUCCESS) {
        cublasDestroy(ctx->cublas);
        return nullptr;
    }

    return ctx;
}

static void destroy_cuda_pcg_context(std::unique_ptr<CudaPCGContext>& ctx) noexcept {
    if (!ctx)
        return;
    if (ctx->cusparse)
        cusparseDestroy(ctx->cusparse);
    if (ctx->cublas)
        cublasDestroy(ctx->cublas);
}

static std::vector<double> build_jacobi_diagonal(
    const SparseMatrixBuilder::CsrData& K)
{
    const int n   = K.n;
    std::vector<double> diag_d(n, 0.0);
    for (int i = 0; i < n; ++i) {
        bool found_diag = false;
        for (int j = K.row_ptr[i]; j < K.row_ptr[i + 1]; ++j) {
            if (K.col_ind[j] == i) {
                double kii = K.values[j];
                if (kii == 0.0)
                    throw SolverError(
                        "CUDA PCG: zero diagonal at row " + std::to_string(i) +
                        " -- matrix is singular. Check boundary conditions.");
                diag_d[static_cast<std::size_t>(i)] = kii;
                found_diag = true;
                break;
            }
        }
        if (!found_diag)
            throw SolverError(
                "CUDA PCG: missing diagonal at row " + std::to_string(i) +
                " -- matrix is singular. Check boundary conditions.");
    }
    return diag_d;
}

static void validate_pcg_inputs(
    const SparseMatrixBuilder::CsrData& K,
    const std::vector<double>& F)
{
    const int n = K.n;
    if (n == 0)
        throw SolverError("CUDA PCG: stiffness matrix is empty -- no free DOFs");
    if (static_cast<int>(F.size()) != n)
        throw SolverError("CUDA PCG: force vector size " +
                          std::to_string(F.size()) + " != matrix size " +
                          std::to_string(n));
}

static std::vector<double> solve_cuda_pcg_stable(
    CudaPCGContext& ctx,
    const SparseMatrixBuilder::CsrData& K,
    const std::vector<double>& F)
{
    validate_pcg_inputs(K, F);
    const int n = K.n;
    const int nnz = K.nnz;
    const bool lower_only = K.stores_lower_triangle_only();
    const std::vector<double> diag_d = build_jacobi_diagonal(K);
    return solve_pcg<double>(
        ctx.cublas, ctx.cusparse, ctx.device_name,
        n, nnz, K.values, K.row_ptr, K.col_ind, F, diag_d,
        lower_only,
        ctx.tolerance, ctx.max_iters,
        ctx.last_iters, ctx.last_rel_res);
}

static std::vector<double> solve_cuda_pcg_mixed(
    CudaPCGContext& ctx,
    const SparseMatrixBuilder::CsrData& K,
    const std::vector<double>& F)
{
    validate_pcg_inputs(K, F);
    const int n = K.n;
    const int nnz = K.nnz;
    const bool lower_only = K.stores_lower_triangle_only();
    const bool profile_enabled = cuda_pcg_profile_enabled();

    const HostTimePoint diag_build_start = HostClock::now();
    const std::vector<double> diag_d = build_jacobi_diagonal(K);
    const double diag_build_ms = host_elapsed_ms(diag_build_start, HostClock::now());

    RefinementProfile profile;
    profile.enabled = profile_enabled;
    const HostTimePoint total_start = HostClock::now();
    if (profile.enabled)
        profile.diag_build.add(diag_build_ms);

    std::vector<float> vals_f(nnz), diag_f(n);
    const HostTimePoint downcast_start = HostClock::now();
    for (int i = 0; i < nnz; ++i)
        vals_f[static_cast<std::size_t>(i)] =
            static_cast<float>(K.values[static_cast<std::size_t>(i)]);
    for (int i = 0; i < n; ++i)
        diag_f[static_cast<std::size_t>(i)] =
            static_cast<float>(diag_d[static_cast<std::size_t>(i)]);
    if (profile.enabled)
        profile.downcast_inputs.add(host_elapsed_ms(downcast_start, HostClock::now()));

    constexpr int kMaxRefinementCorrections = 2;
    double inner_tolerance = ctx.tolerance * 100.0;
    if (inner_tolerance < 1e-4)
        inner_tolerance = 1e-4;
    if (inner_tolerance > 1e-2)
        inner_tolerance = 1e-2;

    const HostTimePoint prepare_start = HostClock::now();
    auto prepared = prepare_pcg_system<float>(
        ctx.cusparse, n, nnz, vals_f, K.row_ptr, K.col_ind, diag_f, lower_only);
    if (profile.enabled)
        profile.prepare_system.add(host_elapsed_ms(prepare_start, HostClock::now()));

    std::vector<double> solution(static_cast<std::size_t>(n), 0.0);
    std::vector<double> correction_rhs = F;
    std::vector<float> rhs_f(static_cast<std::size_t>(n));
    std::vector<double> residual;

    int total_inner_iters = 0;
    double true_rel_res = 0.0;
    bool have_prev_true_rel_res = false;
    double prev_true_rel_res = 0.0;

    for (int refinement_step = 0;
         refinement_step <= kMaxRefinementCorrections;
         ++refinement_step) {
        const HostTimePoint rhs_cast_start = HostClock::now();
        for (int i = 0; i < n; ++i) {
            rhs_f[static_cast<std::size_t>(i)] =
                static_cast<float>(correction_rhs[static_cast<std::size_t>(i)]);
        }
        if (profile.enabled)
            profile.rhs_cast.add(host_elapsed_ms(rhs_cast_start, HostClock::now()));

        int inner_iters = 0;
        double inner_rel_res = 0.0;
        const HostTimePoint inner_start = HostClock::now();
        const std::string solve_label =
            "refinement_step_" + std::to_string(refinement_step);
        auto delta = run_pcg_prepared<float>(
            ctx.cublas, ctx.cusparse, ctx.device_name,
            rhs_f,
            inner_tolerance, ctx.max_iters,
            inner_iters, inner_rel_res, prepared, solve_label);
        if (profile.enabled)
            profile.inner_solve_wall.add(host_elapsed_ms(inner_start, HostClock::now()));
        total_inner_iters += inner_iters;

        const HostTimePoint add_start = HostClock::now();
        add_in_place(solution, delta);
        if (profile.enabled)
            profile.accumulate_solution.add(host_elapsed_ms(add_start, HostClock::now()));

        const HostTimePoint true_res_start = HostClock::now();
        true_rel_res = compute_true_relative_residual(K, solution, F, &residual);
        if (profile.enabled)
            profile.true_residual.add(host_elapsed_ms(true_res_start, HostClock::now()));
        vibestran::log_debug(
            "[cuda-pcg-mixed] refinement step " + std::to_string(refinement_step) +
            ": inner_iters=" + std::to_string(inner_iters) +
            ", inner_rel_res=" + std::to_string(inner_rel_res) +
            ", true_rel_res=" + std::to_string(true_rel_res));

        if (true_rel_res <= ctx.tolerance)
            break;
        if (refinement_step == kMaxRefinementCorrections)
            break;
        if (have_prev_true_rel_res &&
            true_rel_res >= prev_true_rel_res * 0.9) {
            vibestran::log_warn(
                "[cuda-pcg-mixed] iterative refinement stalled at true residual " +
                std::to_string(true_rel_res) +
                " after " + std::to_string(refinement_step + 1) +
                " solve(s)");
            break;
        }

        prev_true_rel_res = true_rel_res;
        have_prev_true_rel_res = true;
        correction_rhs = residual;
    }

    ctx.last_iters = total_inner_iters;
    ctx.last_rel_res = true_rel_res;
    if (true_rel_res > ctx.tolerance) {
        vibestran::log_warn(
            "[cuda-pcg-mixed] true residual remained at " +
            std::to_string(true_rel_res) +
            " after iterative refinement (target " +
            std::to_string(ctx.tolerance) + ")");
    }
    if (profile.enabled) {
        profile.total.add(host_elapsed_ms(total_start, HostClock::now()));
        profile.log_summary(n, nnz, total_inner_iters, true_rel_res);
    }
    return solution;
}

} // namespace vibestran
