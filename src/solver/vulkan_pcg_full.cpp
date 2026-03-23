// src/solver/vulkan_pcg_full.cpp
// Full-GPU Preconditioned Conjugate Gradient path.
// Used when the entire K matrix and working vectors fit in VRAM.
//
// All SpMV, dot products, vector ops run on GPU.
// Two scalar readbacks per CG iteration: pAp and rz_new.
// Throws SolverError on unrecoverable failure.
#ifdef HAVE_VULKAN

#include "solver/vulkan_buffer.hpp"
#include "solver/vulkan_context.hpp"
#include "solver/vulkan_solver_backend.hpp"
#include "vulkan_pipelines.hpp"
#include "core/types.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>
#include <vector>

namespace vibestran {

// ── Float conversion ──────────────────────────────────────────────────────────

static std::vector<float> to_float(const std::vector<double>& v) {
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<float>(v[i]);
    return out;
}

static std::vector<double> to_double(const std::vector<float>& v) {
    std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<double>(v[i]);
    return out;
}

// ── One-shot command buffer helpers ──────────────────────────────────────────

static VkCommandBuffer begin_one_shot(const VulkanContext& ctx) {
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = ctx.command_pool();
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(ctx.device(), &ai, &cmd);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    // Full memory barrier: make all writes from previous submissions (uploads,
    // compute dispatches) visible before any operation in this command buffer.
    // Vulkan does not guarantee cross-submission memory visibility via fences
    // alone — explicit barriers are required even on the same queue.
    VkMemoryBarrier mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);

    return cmd;
}

static void end_and_submit(const VulkanContext& ctx, VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    submit_and_wait(ctx, cmd); // throws on failure
    vkFreeCommandBuffers(ctx.device(), ctx.command_pool(), 1, &cmd);
}

// ── GPU dot product ───────────────────────────────────────────────────────────

static double gpu_dot(const VulkanContext& ctx, Pipelines& pl,
                       const VulkanBuffer& a_buf, const VulkanBuffer& b_buf,
                       VulkanBuffer& partial_buf, uint32_t n) {
    uint32_t num_wg = (n + 255) / 256;

    auto cmd = begin_one_shot(ctx);
    VkDescriptorSet ds = pl.alloc_set(PL_DOT_REDUCE);
    Pipelines::bind_buffer(ctx.device(), ds, 0, a_buf.handle(),       a_buf.size_bytes());
    Pipelines::bind_buffer(ctx.device(), ds, 1, b_buf.handle(),       b_buf.size_bytes());
    Pipelines::bind_buffer(ctx.device(), ds, 2, partial_buf.handle(), partial_buf.size_bytes());
    cmd_dot_reduce(cmd, pl, ds, n);
    end_and_submit(ctx, cmd);
    vkFreeDescriptorSets(ctx.device(), pl.pool, 1, &ds);

    auto partials = partial_buf.download<float>(ctx, num_wg);
    double sum = std::accumulate(partials.begin(), partials.end(), 0.0,
        [](double acc, float v) { return acc + static_cast<double>(v); });
    return sum;
}

// ── GPU axpby: y = alpha*x + beta*y ──────────────────────────────────────────

static void gpu_axpby(const VulkanContext& ctx, Pipelines& pl,
                       const VulkanBuffer& x_buf, const VulkanBuffer& y_buf,
                       float alpha, float beta, uint32_t n) {
    auto cmd = begin_one_shot(ctx);
    VkDescriptorSet ds = pl.alloc_set(PL_AXPBY);
    Pipelines::bind_buffer(ctx.device(), ds, 0, x_buf.handle(), x_buf.size_bytes());
    Pipelines::bind_buffer(ctx.device(), ds, 1, y_buf.handle(), y_buf.size_bytes());
    cmd_axpby(cmd, pl, ds, {alpha, beta, n}, n);
    end_and_submit(ctx, cmd);
    vkFreeDescriptorSets(ctx.device(), pl.pool, 1, &ds);
}

// ── GPU Jacobi preconditioner: z = r / diag ───────────────────────────────────

static void gpu_jacobi(const VulkanContext& ctx, Pipelines& pl,
                        const VulkanBuffer& diag_buf, const VulkanBuffer& r_buf,
                        const VulkanBuffer& z_buf, uint32_t n) {
    auto cmd = begin_one_shot(ctx);
    VkDescriptorSet ds = pl.alloc_set(PL_JACOBI);
    Pipelines::bind_buffer(ctx.device(), ds, 0, diag_buf.handle(), diag_buf.size_bytes());
    Pipelines::bind_buffer(ctx.device(), ds, 1, r_buf.handle(),    r_buf.size_bytes());
    Pipelines::bind_buffer(ctx.device(), ds, 2, z_buf.handle(),    z_buf.size_bytes());
    cmd_jacobi(cmd, pl, ds, n);
    end_and_submit(ctx, cmd);
    vkFreeDescriptorSets(ctx.device(), pl.pool, 1, &ds);
}

// ── GPU SpMV: y = K * x ───────────────────────────────────────────────────────

static void gpu_spmv(const VulkanContext& ctx, Pipelines& pl,
                      const VulkanBuffer& row_ptr_buf, const VulkanBuffer& col_ind_buf,
                      const VulkanBuffer& val_buf, const VulkanBuffer& x_buf,
                      const VulkanBuffer& y_buf, uint32_t n) {
    const uint32_t max_wg = ctx.device_info().max_workgroup_count_x;
    auto cmd = begin_one_shot(ctx);
    VkDescriptorSet ds = pl.alloc_set(PL_SPMV);
    Pipelines::bind_buffer(ctx.device(), ds, 0, row_ptr_buf.handle(), row_ptr_buf.size_bytes());
    Pipelines::bind_buffer(ctx.device(), ds, 1, col_ind_buf.handle(), col_ind_buf.size_bytes());
    Pipelines::bind_buffer(ctx.device(), ds, 2, val_buf.handle(),     val_buf.size_bytes());
    Pipelines::bind_buffer(ctx.device(), ds, 3, x_buf.handle(),       x_buf.size_bytes());
    Pipelines::bind_buffer(ctx.device(), ds, 4, y_buf.handle(),       y_buf.size_bytes());
    // Batch dispatch: SpMV uses one workgroup per row, which can exceed the
    // device's maxComputeWorkGroupCount[0] (65535 on AMD RDNA).  The shader's
    // row_start push constant lets us dispatch in batches.
    for (uint32_t row = 0; row < n; row += max_wg) {
        uint32_t batch = std::min(max_wg, n - row);
        PCSpmv pc{row, batch, n};
        cmd_spmv(cmd, pl, ds, pc, batch);
    }
    end_and_submit(ctx, cmd);
    vkFreeDescriptorSets(ctx.device(), pl.pool, 1, &ds);
}

// ── Buffer allocation helper ──────────────────────────────────────────────────

static VulkanBuffer make_buf(const VulkanContext& ctx, VkDeviceSize sz, const char* label) {
    auto b = VulkanBuffer::create(ctx, sz);
    if (!b) throw SolverError(std::format("Vulkan: {} buffer allocation failed", label));
    return std::move(*b);
}

// ── solve_full_gpu ────────────────────────────────────────────────────────────

std::vector<double>
solve_full_gpu(VulkanContext& ctx, Pipelines& pl,
               const SparseMatrixBuilder::CsrData& K,
               const std::vector<double>& F,
               const VulkanSolverConfig& cfg,
               int& out_iters, double& out_residual) {
    const uint32_t n      = static_cast<uint32_t>(K.n);
    const uint32_t nnz    = static_cast<uint32_t>(K.nnz);
    const uint32_t nwg    = (n + 255) / 256;

    // ── Extract diagonal ──────────────────────────────────────────────────
    std::vector<float> diag_f(n, 1.0f); // default 1 to avoid div-by-zero
    for (int row = 0; row < K.n; ++row)
        for (int idx = K.row_ptr[row]; idx < K.row_ptr[row + 1]; ++idx)
            if (K.col_ind[idx] == row) {
                diag_f[row] = static_cast<float>(K.values[idx]);
                break;
            }

    // ── Downcast K to float32 ─────────────────────────────────────────────
    std::vector<float> vals_f(nnz);
    for (uint32_t i = 0; i < nnz; ++i) vals_f[i] = static_cast<float>(K.values[i]);

    // ── Allocate device buffers ───────────────────────────────────────────
    auto row_ptr_buf = make_buf(ctx, (n + 1) * sizeof(int),    "row_ptr");
    auto col_ind_buf = make_buf(ctx,  nnz    * sizeof(int),    "col_ind");
    auto val_buf     = make_buf(ctx,  nnz    * sizeof(float),  "values");
    auto diag_buf    = make_buf(ctx,  n      * sizeof(float),  "diag");
    auto x_buf       = make_buf(ctx,  n      * sizeof(float),  "x");
    auto r_buf       = make_buf(ctx,  n      * sizeof(float),  "r");
    auto z_buf       = make_buf(ctx,  n      * sizeof(float),  "z");
    auto p_buf       = make_buf(ctx,  n      * sizeof(float),  "p");
    auto Ap_buf      = make_buf(ctx,  n      * sizeof(float),  "Ap");
    auto partial_buf = make_buf(ctx,  nwg    * sizeof(float),  "partials");

    // ── Upload initial data ───────────────────────────────────────────────
    const std::vector<float> zeros(n, 0.0f);
    const std::vector<float> F_f = to_float(F);

    row_ptr_buf.upload<int>  (ctx, std::span<const int>  (K.row_ptr));
    col_ind_buf.upload<int>  (ctx, std::span<const int>  (K.col_ind));
    val_buf.upload<float>    (ctx, std::span<const float>(vals_f));
    diag_buf.upload<float>   (ctx, std::span<const float>(diag_f));
    x_buf.upload<float>      (ctx, std::span<const float>(zeros));
    r_buf.upload<float>      (ctx, std::span<const float>(F_f));   // r = F (x=0)

    // z = M^{-1} * r
    gpu_jacobi(ctx, pl, diag_buf, r_buf, z_buf, n);

    // p = z  (download z, upload as p)
    {
        auto z_init = z_buf.download<float>(ctx, n);
        p_buf.upload<float>(ctx, std::span<const float>(z_init));
    }

    double rz_old = gpu_dot(ctx, pl, r_buf, z_buf, partial_buf, n);

    // Use the 2-norm of b for the convergence denominator.
    // The preconditioned residual norm sqrt(r^T M^{-1} r) is NOT monotonically
    // decreasing in PCG — it can oscillate significantly for ill-conditioned
    // systems (e.g. 3D elasticity with Jacobi preconditioning), causing the
    // stagnation detector to false-trigger.  The 2-norm ||r||_2 is much better
    // behaved and is the standard metric used in practical PCG implementations.
    double norm_b_sq = gpu_dot(ctx, pl, r_buf, r_buf, partial_buf, n); // r=b here
    double norm_b    = std::sqrt(norm_b_sq);
    if (norm_b < 1e-300) {
        out_iters    = 0;
        out_residual = 0.0;
        return std::vector<double>(n, 0.0);
    }

    out_iters    = 0;
    out_residual = 1.0; // ||b||/||b|| = 1 by construction

    // ── PCG iteration ─────────────────────────────────────────────────────
    // Stagnation detection uses the A-norm error decrease, which is
    // monotonically non-negative in CG (unlike the 2-norm residual which
    // oscillates).  Each iteration decreases ||e||_A^2 by rz_old^2 / pAp.
    // We track cumulative progress and check that recent iterations
    // contribute a meaningful fraction.
    double cum_anorm_progress    = 0.0;
    double checkpoint_progress   = 0.0;
    int    iters_since_checkpoint = 0;

    for (int iter = 0; iter < cfg.max_iterations; ++iter) {
        gpu_spmv(ctx, pl, row_ptr_buf, col_ind_buf, val_buf, p_buf, Ap_buf, n);

        double pAp = gpu_dot(ctx, pl, p_buf, Ap_buf, partial_buf, n);
        if (pAp <= 0.0)
            throw SolverError(std::format(
                "Vulkan PCG breakdown: p^T*A*p = {:.3e} <= 0 at iteration {} "
                "(matrix is not positive definite)", pAp, iter));

        float alpha = static_cast<float>(rz_old / pAp);

        gpu_axpby(ctx, pl, p_buf,  x_buf,  alpha,  1.0f, n); // x += alpha*p
        gpu_axpby(ctx, pl, Ap_buf, r_buf, -alpha,  1.0f, n); // r -= alpha*Ap

        gpu_jacobi(ctx, pl, diag_buf, r_buf, z_buf, n);

        double rz_new = gpu_dot(ctx, pl, r_buf, z_buf, partial_buf, n);

        // Convergence metric: ||r||_2 / ||b||_2
        double rr     = gpu_dot(ctx, pl, r_buf, r_buf, partial_buf, n);
        out_residual  = std::sqrt(std::abs(rr)) / norm_b;
        out_iters     = iter + 1;

        if (out_residual < cfg.tolerance) break;
        if (iter + 1 == cfg.max_iterations) break;

        // A-norm stagnation detection: the decrease in ||e||_A^2 per
        // iteration is rz_old^2 / pAp (always non-negative for SPD K).
        // If the last `stagnation_window` iterations contributed less than
        // `stagnation_threshold` of total cumulative progress, the solver
        // has hit the precision floor and further iterations are wasted.
        cum_anorm_progress += rz_old * rz_old / pAp;
        ++iters_since_checkpoint;
        if (iters_since_checkpoint >= cfg.stagnation_window) {
            double window_progress = cum_anorm_progress - checkpoint_progress;
            if (cum_anorm_progress > 0.0 &&
                window_progress < cfg.stagnation_threshold * cum_anorm_progress) {
                throw SolverError(std::format(
                    "Vulkan PCG stagnated: A-norm progress over last {} iterations was {:.1e}% "
                    "of total (precision floor reached). "
                    "Relative residual = {:.3e}. "
                    "Consider --cpu for higher precision.",
                    cfg.stagnation_window,
                    100.0 * window_progress / cum_anorm_progress,
                    out_residual));
            }
            checkpoint_progress = cum_anorm_progress;
            iters_since_checkpoint = 0;
        }

        float beta = static_cast<float>(rz_new / rz_old);
        gpu_axpby(ctx, pl, z_buf, p_buf, 1.0f, beta, n); // p = z + beta*p
        rz_old = rz_new;
    }

    if (out_residual >= cfg.tolerance)
        throw SolverError(std::format(
            "Vulkan PCG did not converge after {} iterations; "
            "relative residual = {:.3e} (tolerance = {:.3e}). "
            "Check boundary conditions or increase max_iterations.",
            out_iters, out_residual, cfg.tolerance));

    auto x_f = x_buf.download<float>(ctx, n);
    return to_double(x_f);
}

// ── solve_full_gpu_double ─────────────────────────────────────────────────────
// Identical algorithm to solve_full_gpu but all buffers and shader dispatches
// use float64.  Requires shaderFloat64 physical device feature (verified by
// the caller before dispatching here).

static double gpu_dot_d(const VulkanContext& ctx, Pipelines& pl,
                         const VulkanBuffer& a_buf, const VulkanBuffer& b_buf,
                         VulkanBuffer& partial_buf, uint32_t n) {
    uint32_t num_wg = (n + 255) / 256;

    auto cmd = begin_one_shot(ctx);
    VkDescriptorSet ds = pl.alloc_set(PL_DOT_REDUCE_D);
    Pipelines::bind_buffer(ctx.device(), ds, 0, a_buf.handle(),       a_buf.size_bytes());
    Pipelines::bind_buffer(ctx.device(), ds, 1, b_buf.handle(),       b_buf.size_bytes());
    Pipelines::bind_buffer(ctx.device(), ds, 2, partial_buf.handle(), partial_buf.size_bytes());
    cmd_dot_reduce_d(cmd, pl, ds, n);
    end_and_submit(ctx, cmd);
    vkFreeDescriptorSets(ctx.device(), pl.pool, 1, &ds);

    // Partials are already double — sum directly
    auto partials = partial_buf.download<double>(ctx, num_wg);
    return std::accumulate(partials.begin(), partials.end(), 0.0);
}

static void gpu_axpby_d(const VulkanContext& ctx, Pipelines& pl,
                         const VulkanBuffer& x_buf, const VulkanBuffer& y_buf,
                         double alpha, double beta, uint32_t n) {
    auto cmd = begin_one_shot(ctx);
    VkDescriptorSet ds = pl.alloc_set(PL_AXPBY_D);
    Pipelines::bind_buffer(ctx.device(), ds, 0, x_buf.handle(), x_buf.size_bytes());
    Pipelines::bind_buffer(ctx.device(), ds, 1, y_buf.handle(), y_buf.size_bytes());
    cmd_axpby_d(cmd, pl, ds, {alpha, beta, n, 0u}, n);
    end_and_submit(ctx, cmd);
    vkFreeDescriptorSets(ctx.device(), pl.pool, 1, &ds);
}

static void gpu_jacobi_d(const VulkanContext& ctx, Pipelines& pl,
                          const VulkanBuffer& diag_buf, const VulkanBuffer& r_buf,
                          const VulkanBuffer& z_buf, uint32_t n) {
    auto cmd = begin_one_shot(ctx);
    VkDescriptorSet ds = pl.alloc_set(PL_JACOBI_D);
    Pipelines::bind_buffer(ctx.device(), ds, 0, diag_buf.handle(), diag_buf.size_bytes());
    Pipelines::bind_buffer(ctx.device(), ds, 1, r_buf.handle(),    r_buf.size_bytes());
    Pipelines::bind_buffer(ctx.device(), ds, 2, z_buf.handle(),    z_buf.size_bytes());
    cmd_jacobi_d(cmd, pl, ds, n);
    end_and_submit(ctx, cmd);
    vkFreeDescriptorSets(ctx.device(), pl.pool, 1, &ds);
}

static void gpu_spmv_d(const VulkanContext& ctx, Pipelines& pl,
                        const VulkanBuffer& row_ptr_buf, const VulkanBuffer& col_ind_buf,
                        const VulkanBuffer& val_buf, const VulkanBuffer& x_buf,
                        const VulkanBuffer& y_buf, uint32_t n) {
    const uint32_t max_wg = ctx.device_info().max_workgroup_count_x;
    auto cmd = begin_one_shot(ctx);
    VkDescriptorSet ds = pl.alloc_set(PL_SPMV_D);
    Pipelines::bind_buffer(ctx.device(), ds, 0, row_ptr_buf.handle(), row_ptr_buf.size_bytes());
    Pipelines::bind_buffer(ctx.device(), ds, 1, col_ind_buf.handle(), col_ind_buf.size_bytes());
    Pipelines::bind_buffer(ctx.device(), ds, 2, val_buf.handle(),     val_buf.size_bytes());
    Pipelines::bind_buffer(ctx.device(), ds, 3, x_buf.handle(),       x_buf.size_bytes());
    Pipelines::bind_buffer(ctx.device(), ds, 4, y_buf.handle(),       y_buf.size_bytes());
    for (uint32_t row = 0; row < n; row += max_wg) {
        uint32_t batch = std::min(max_wg, n - row);
        PCSpmv pc{row, batch, n};
        cmd_spmv_d(cmd, pl, ds, pc, batch);
    }
    end_and_submit(ctx, cmd);
    vkFreeDescriptorSets(ctx.device(), pl.pool, 1, &ds);
}

std::vector<double>
solve_full_gpu_double(VulkanContext& ctx, Pipelines& pl,
                       const SparseMatrixBuilder::CsrData& K,
                       const std::vector<double>& F,
                       const VulkanSolverConfig& cfg,
                       int& out_iters, double& out_residual) {
    const uint32_t n   = static_cast<uint32_t>(K.n);
    const uint32_t nnz = static_cast<uint32_t>(K.nnz);
    const uint32_t nwg = (n + 255) / 256;

    // ── Extract diagonal ──────────────────────────────────────────────────
    std::vector<double> diag_d(n, 1.0);
    for (int row = 0; row < K.n; ++row)
        for (int idx = K.row_ptr[row]; idx < K.row_ptr[row + 1]; ++idx)
            if (K.col_ind[idx] == row) { diag_d[row] = K.values[idx]; break; }

    // ── Allocate device buffers (all double) ──────────────────────────────
    auto row_ptr_buf = make_buf(ctx, (n + 1)          * sizeof(int),    "row_ptr");
    auto col_ind_buf = make_buf(ctx,  nnz              * sizeof(int),    "col_ind");
    auto val_buf     = make_buf(ctx,  nnz              * sizeof(double), "values");
    auto diag_buf    = make_buf(ctx,  n                * sizeof(double), "diag");
    auto x_buf       = make_buf(ctx,  n                * sizeof(double), "x");
    auto r_buf       = make_buf(ctx,  n                * sizeof(double), "r");
    auto z_buf       = make_buf(ctx,  n                * sizeof(double), "z");
    auto p_buf       = make_buf(ctx,  n                * sizeof(double), "p");
    auto Ap_buf      = make_buf(ctx,  n                * sizeof(double), "Ap");
    auto partial_buf = make_buf(ctx,  nwg              * sizeof(double), "partials");

    // ── Upload initial data ───────────────────────────────────────────────
    const std::vector<double> zeros(n, 0.0);

    row_ptr_buf.upload<int>   (ctx, std::span<const int>   (K.row_ptr));
    col_ind_buf.upload<int>   (ctx, std::span<const int>   (K.col_ind));
    val_buf.upload<double>    (ctx, std::span<const double>(K.values));
    diag_buf.upload<double>   (ctx, std::span<const double>(diag_d));
    x_buf.upload<double>      (ctx, std::span<const double>(zeros));
    r_buf.upload<double>      (ctx, std::span<const double>(F));  // r = F (x=0)

    // z = M^{-1} * r
    gpu_jacobi_d(ctx, pl, diag_buf, r_buf, z_buf, n);

    // p = z
    {
        auto z_init = z_buf.download<double>(ctx, n);
        p_buf.upload<double>(ctx, std::span<const double>(z_init));
    }

    double rz_old = gpu_dot_d(ctx, pl, r_buf, z_buf, partial_buf, n);

    // Use 2-norm of b for convergence (see solve_full_gpu comment for rationale).
    double norm_b_sq = gpu_dot_d(ctx, pl, r_buf, r_buf, partial_buf, n); // r=b here
    double norm_b    = std::sqrt(norm_b_sq);
    if (norm_b < 1e-300) {
        out_iters    = 0;
        out_residual = 0.0;
        return std::vector<double>(n, 0.0);
    }

    out_iters    = 0;
    out_residual = 1.0;

    // ── PCG iteration ─────────────────────────────────────────────────────
    double cum_anorm_progress    = 0.0;
    double checkpoint_progress   = 0.0;
    int    iters_since_checkpoint = 0;

    for (int iter = 0; iter < cfg.max_iterations; ++iter) {
        gpu_spmv_d(ctx, pl, row_ptr_buf, col_ind_buf, val_buf, p_buf, Ap_buf, n);

        double pAp = gpu_dot_d(ctx, pl, p_buf, Ap_buf, partial_buf, n);
        if (pAp <= 0.0)
            throw SolverError(std::format(
                "Vulkan PCG (float64) breakdown: p^T*A*p = {:.3e} <= 0 at iteration {} "
                "(matrix is not positive definite)", pAp, iter));

        double alpha = rz_old / pAp;

        gpu_axpby_d(ctx, pl, p_buf,  x_buf,  alpha,  1.0, n); // x += alpha*p
        gpu_axpby_d(ctx, pl, Ap_buf, r_buf, -alpha,  1.0, n); // r -= alpha*Ap

        gpu_jacobi_d(ctx, pl, diag_buf, r_buf, z_buf, n);

        double rz_new = gpu_dot_d(ctx, pl, r_buf, z_buf, partial_buf, n);

        // Convergence metric: ||r||_2 / ||b||_2
        double rr     = gpu_dot_d(ctx, pl, r_buf, r_buf, partial_buf, n);
        out_residual  = std::sqrt(std::abs(rr)) / norm_b;
        out_iters     = iter + 1;

        if (out_residual < cfg.tolerance) break;
        if (iter + 1 == cfg.max_iterations) break;

        // A-norm stagnation detection (same as float32 path).
        cum_anorm_progress += rz_old * rz_old / pAp;
        ++iters_since_checkpoint;
        if (iters_since_checkpoint >= cfg.stagnation_window) {
            double window_progress = cum_anorm_progress - checkpoint_progress;
            if (cum_anorm_progress > 0.0 &&
                window_progress < cfg.stagnation_threshold * cum_anorm_progress) {
                throw SolverError(std::format(
                    "Vulkan PCG (float64) stagnated: A-norm progress over last {} iterations "
                    "was {:.1e}% of total (precision floor reached). "
                    "Relative residual = {:.3e}. "
                    "Check boundary conditions.",
                    cfg.stagnation_window,
                    100.0 * window_progress / cum_anorm_progress,
                    out_residual));
            }
            checkpoint_progress = cum_anorm_progress;
            iters_since_checkpoint = 0;
        }

        double beta = rz_new / rz_old;
        gpu_axpby_d(ctx, pl, z_buf, p_buf, 1.0, beta, n); // p = z + beta*p
        rz_old = rz_new;
    }

    if (out_residual >= cfg.tolerance)
        throw SolverError(std::format(
            "Vulkan PCG (float64) did not converge after {} iterations; "
            "relative residual = {:.3e} (tolerance = {:.3e}). "
            "Check boundary conditions or increase max_iterations.",
            out_iters, out_residual, cfg.tolerance));

    return x_buf.download<double>(ctx, n);
}

} // namespace vibestran

#endif // HAVE_VULKAN
