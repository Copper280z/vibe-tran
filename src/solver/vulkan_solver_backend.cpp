// src/solver/vulkan_solver_backend.cpp
#ifdef HAVE_VULKAN

#include "solver/vulkan_solver_backend.hpp"
#include "solver/vulkan_context.hpp"
#include "solver/solver_backend.hpp"
#include "vulkan_pcg_internal.hpp"
#include "vulkan_pipelines.hpp"
#include "core/types.hpp"
#include <chrono>
#include <format>
#include <iostream>

namespace nastran {

// ── Constructor / destructor ──────────────────────────────────────────────────

VulkanSolverBackend::VulkanSolverBackend(VulkanContext ctx, const VulkanSolverConfig& cfg)
    : ctx_(std::move(ctx)), cfg_(cfg) {}

VulkanSolverBackend::~VulkanSolverBackend() = default;

VulkanSolverBackend::VulkanSolverBackend(VulkanSolverBackend&&) noexcept = default;
VulkanSolverBackend& VulkanSolverBackend::operator=(VulkanSolverBackend&&) noexcept = default;

// ── Factory ───────────────────────────────────────────────────────────────────

std::optional<VulkanSolverBackend>
VulkanSolverBackend::try_create(const VulkanSolverConfig& cfg) {
    auto ctx = VulkanContext::create();
    if (!ctx) return std::nullopt;
    return VulkanSolverBackend(std::move(*ctx), cfg);
}

// ── Accessors ─────────────────────────────────────────────────────────────────

std::string_view VulkanSolverBackend::name() const noexcept {
    return "Vulkan PCG (Jacobi preconditioner)";
}

bool   VulkanSolverBackend::last_solve_was_full_gpu() const noexcept { return last_full_gpu_; }
int    VulkanSolverBackend::last_iteration_count()    const noexcept { return last_iters_; }
double VulkanSolverBackend::last_residual_norm()       const noexcept { return last_residual_; }

// ── VRAM checks ───────────────────────────────────────────────────────────────

// Returns estimated bytes needed for a float32 full-GPU solve.
static size_t vram_needed_f32(const SparseMatrixBuilder::CsrData& K) noexcept {
    size_t matrix = static_cast<size_t>(K.nnz)   * sizeof(float)
                  + static_cast<size_t>(K.nnz)   * sizeof(int)
                  + static_cast<size_t>(K.n + 1) * sizeof(int);
    size_t vecs   = 6ULL * static_cast<size_t>(K.n) * sizeof(float); // x,r,z,p,Ap,diag
    size_t parts  = static_cast<size_t>((static_cast<uint32_t>(K.n) + 255u) / 256u) * sizeof(float);
    return matrix + vecs + parts;
}

// Returns estimated bytes needed for a float64 full-GPU solve.
static size_t vram_needed_f64(const SparseMatrixBuilder::CsrData& K) noexcept {
    size_t matrix = static_cast<size_t>(K.nnz)   * sizeof(double)
                  + static_cast<size_t>(K.nnz)   * sizeof(int)
                  + static_cast<size_t>(K.n + 1) * sizeof(int);
    size_t vecs   = 6ULL * static_cast<size_t>(K.n) * sizeof(double);
    size_t parts  = static_cast<size_t>((static_cast<uint32_t>(K.n) + 255u) / 256u) * sizeof(double);
    return matrix + vecs + parts;
}

bool VulkanSolverBackend::fits_in_vram(const SparseMatrixBuilder::CsrData& K) const noexcept {
    size_t needed    = cfg_.use_double ? vram_needed_f64(K) : vram_needed_f32(K);
    size_t available = static_cast<size_t>(
        static_cast<double>(ctx_.device_info().vram_bytes) * (1.0 - cfg_.vram_headroom));
    return needed <= available;
}

// ── Logging helpers ───────────────────────────────────────────────────────────

static std::string mib_str(size_t bytes) {
    return std::format("{} MiB", bytes / (1024 * 1024));
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

// ── Lazy pipeline init ─────────────────────────────────────────────────────────

static Pipelines* ensure_pipelines(const VulkanContext& ctx,
                                    std::unique_ptr<Pipelines>& cache) {
    if (cache) return cache.get();
    // build_pipelines throws SolverError on failure
    cache.reset(build_pipelines(ctx));
    return cache.get();
}

// ── solve ─────────────────────────────────────────────────────────────────────

std::vector<double>
VulkanSolverBackend::solve(const SparseMatrixBuilder::CsrData& K,
                            const std::vector<double>& F) {
    const int n = K.n;
    if (n == 0)
        throw SolverError("Vulkan solver: stiffness matrix is empty — no free DOFs");
    if (static_cast<int>(F.size()) != n)
        throw SolverError(std::format(
            "Vulkan solver: force vector size {} != matrix size {}", F.size(), n));

    auto t_diag = std::chrono::steady_clock::now();

    // Check for missing diagonal (structural singularity / missing BCs).
    // The Jacobi preconditioner would divide by zero otherwise.
    for (int row = 0; row < n; ++row) {
        bool found = false;
        for (int idx = K.row_ptr[row]; idx < K.row_ptr[row + 1]; ++idx)
            if (K.col_ind[idx] == row) { found = true; break; }
        if (!found)
            throw SolverError(std::format(
                "Vulkan solver: zero diagonal at row {} — "
                "stiffness matrix appears singular. Check boundary conditions.", row));
    }
    std::clog << std::format("[vulkan] diagonal scan: {:.3f} ms\n", ms_since(t_diag));

    // ── Path selection ────────────────────────────────────────────────────

    // Small problems: GPU kernel launch and PCIe readback overhead per iteration
    // dominates actual compute time. Fall back to the CPU direct solver.
    if (n < cfg_.min_dofs_for_gpu) {
        std::clog << std::format(
            "[vulkan] {} DOFs → CPU fallback (below min_dofs_for_gpu={})\n",
            n, cfg_.min_dofs_for_gpu);
        last_full_gpu_ = false;
        last_iters_    = 1;
        last_residual_ = 0.0;
        EigenSolverBackend eigen;
        return eigen.solve(K, F);
    }

    // use_double requires shaderFloat64 device feature
    if (cfg_.use_double && !ctx_.device_info().supports_float64)
        throw SolverError(std::format(
            "Vulkan solver: use_double=true but GPU '{}' does not support shaderFloat64. "
            "Use --cpu for double-precision solve.",
            ctx_.device_info().device_name));

    auto t_pipelines = std::chrono::steady_clock::now();
    Pipelines* pl = ensure_pipelines(ctx_, pipelines_);
    std::clog << std::format("[vulkan] pipeline init: {:.3f} ms\n", ms_since(t_pipelines));

    const size_t vram_total = ctx_.device_info().vram_bytes;
    const size_t available  = static_cast<size_t>(
        static_cast<double>(vram_total) * (1.0 - cfg_.vram_headroom));

    const char* precision = cfg_.use_double ? "float64" : "float32";

    if (!cfg_.force_tiled && fits_in_vram(K)) {
        const size_t needed = cfg_.use_double ? vram_needed_f64(K) : vram_needed_f32(K);
        std::clog << std::format(
            "[vulkan] {} DOFs → full-GPU {} ({} / {} VRAM)\n",
            n, precision, mib_str(needed), mib_str(vram_total));

        last_full_gpu_ = true;
        auto t_solve = std::chrono::steady_clock::now();
        std::vector<double> result;
        if (cfg_.use_double) {
            result = solve_full_gpu_double(ctx_, *pl, K, F, cfg_, last_iters_, last_residual_);
        } else {
            // Attempt float32.  If it diverges/stagnates and the device supports
            // float64 (and the matrix fits in VRAM at float64 size), transparently
            // retry with the double-precision path rather than propagating an error.
            try {
                result = solve_full_gpu(ctx_, *pl, K, F, cfg_, last_iters_, last_residual_);
            } catch (const SolverError& e) {
                if (ctx_.device_info().supports_float64 &&
                    vram_needed_f64(K) <= available) {
                    std::clog << std::format(
                        "[vulkan] float32 failed ({}), retrying with float64\n", e.what());
                    result = solve_full_gpu_double(ctx_, *pl, K, F, cfg_, last_iters_, last_residual_);
                } else {
                    throw;
                }
            }
        }

        std::clog << std::format(
            "[vulkan] converged: {} iterations, residual = {:.2e}, solve = {:.3f} ms\n",
            last_iters_, last_residual_, ms_since(t_solve));
        return result;
    } else {
        if (cfg_.use_double)
            throw SolverError(std::format(
                "Vulkan solver: use_double=true but K ({}) exceeds available VRAM ({}) "
                "for a full-GPU float64 solve. Use --cpu.",
                mib_str(vram_needed_f64(K)), mib_str(available)));

        std::clog << std::format(
            "[vulkan] {} DOFs → tiled float32 (matrix {} exceeds VRAM budget {})\n",
            n, mib_str(vram_needed_f32(K)), mib_str(available));

        last_full_gpu_ = false;
        auto t_solve = std::chrono::steady_clock::now();
        auto result = solve_tiled(ctx_, *pl, K, F, cfg_,
                                   static_cast<uint64_t>(available),
                                   last_iters_, last_residual_);
        std::clog << std::format(
            "[vulkan] converged: {} iterations, residual = {:.2e}, solve = {:.3f} ms\n",
            last_iters_, last_residual_, ms_since(t_solve));
        return result;
    }
}

} // namespace nastran

#endif // HAVE_VULKAN
