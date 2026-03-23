// src/solver/vulkan_pcg_tiled.cpp
// Tiled/streaming PCG path.
// Used when K does not fit in VRAM.
//
// Strategy:
//   - All scalar and vector operations run on CPU (O(n) memory, always fits).
//   - SpMV is GPU-accelerated by streaming K in horizontal row-band tiles.
//   - Each tile is uploaded once per CG iteration; only the p vector (O(n))
//     stays in VRAM across tiles.  Output Ap slice is downloaded per tile.
//
// Memory footprint: O(max_tile_nnz + n) device memory, not O(total_nnz).
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
#include <vector>

namespace vibestran {

// ── CPU scalar helpers ────────────────────────────────────────────────────────

static double dot_cpu(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

static void axpy_cpu(double alpha, const std::vector<double>& x, std::vector<double>& y) {
    for (size_t i = 0; i < x.size(); ++i) y[i] += alpha * x[i];
}

// y = alpha*x + beta*y
static void axpby_cpu(double alpha, const std::vector<double>& x,
                       double beta, std::vector<double>& y) {
    for (size_t i = 0; i < x.size(); ++i) y[i] = alpha * x[i] + beta * y[i];
}

static void jacobi_cpu(const std::vector<double>& diag,
                        const std::vector<double>& r, std::vector<double>& z) {
    for (size_t i = 0; i < r.size(); ++i) z[i] = r[i] / diag[i];
}

// ── Tile sizing ───────────────────────────────────────────────────────────────

static uint32_t compute_tile_rows(const SparseMatrixBuilder::CsrData& K,
                                   uint64_t avail_for_tiles) {
    double avg_nnz = static_cast<double>(K.nnz) / std::max(K.n, 1);
    // bytes per row: (float values + int col_ind + int row_ptr entry)
    uint64_t bytes_per_row = static_cast<uint64_t>(
        avg_nnz * (sizeof(float) + sizeof(int))) + sizeof(int);
    uint32_t rows = static_cast<uint32_t>(
        avail_for_tiles / std::max(bytes_per_row, uint64_t{1}));
    return std::max(rows, uint32_t{1});
}

// ── Buffer allocation helper ──────────────────────────────────────────────────

static VulkanBuffer make_buf(const VulkanContext& ctx, VkDeviceSize sz, const char* label) {
    auto b = VulkanBuffer::create(ctx, sz);
    if (!b) throw SolverError(std::format("Vulkan tiled: {} buffer allocation failed", label));
    return std::move(*b);
}

// ── solve_tiled ───────────────────────────────────────────────────────────────

std::vector<double>
solve_tiled(VulkanContext& ctx, Pipelines& pl,
            const SparseMatrixBuilder::CsrData& K,
            const std::vector<double>& F,
            const VulkanSolverConfig& cfg,
            uint64_t available_vram_bytes,
            int& out_iters, double& out_residual) {
    const int n = K.n;

    // ── Tile sizing ───────────────────────────────────────────────────────
    uint64_t p_buf_bytes    = static_cast<uint64_t>(n) * sizeof(float);
    uint64_t Ap_tile_bytes  = 0; // determined below after tile_rows is known
    uint64_t avail_for_tiles =
        (available_vram_bytes > p_buf_bytes + 64)
            ? available_vram_bytes - p_buf_bytes - 64
            : 1;

    uint32_t tile_rows = compute_tile_rows(K, avail_for_tiles);

    // Max tile nnz for buffer sizing
    uint32_t max_tile_nnz = 0;
    for (int r = 0; r < n; r += static_cast<int>(tile_rows)) {
        int rend = std::min(r + static_cast<int>(tile_rows), n);
        uint32_t nnz = static_cast<uint32_t>(K.row_ptr[rend] - K.row_ptr[r]);
        max_tile_nnz = std::max(max_tile_nnz, nnz);
    }
    Ap_tile_bytes = static_cast<uint64_t>(tile_rows) * sizeof(float);

    // ── Allocate persistent device buffers ────────────────────────────────
    auto p_buf        = make_buf(ctx, p_buf_bytes,                                      "p");
    auto Ap_tile_buf  = make_buf(ctx, Ap_tile_bytes,                                    "Ap_tile");
    auto row_ptr_tile = make_buf(ctx, (tile_rows + 1)              * sizeof(int),       "row_ptr_tile");
    auto col_ind_tile = make_buf(ctx, static_cast<VkDeviceSize>(max_tile_nnz) * sizeof(int),   "col_ind_tile");
    auto val_tile     = make_buf(ctx, static_cast<VkDeviceSize>(max_tile_nnz) * sizeof(float), "val_tile");

    // ── Extract diagonal for Jacobi preconditioner (on CPU) ───────────────
    std::vector<double> diag(n, 1.0);
    for (int row = 0; row < n; ++row)
        for (int idx = K.row_ptr[row]; idx < K.row_ptr[row + 1]; ++idx)
            if (K.col_ind[idx] == row) { diag[row] = K.values[idx]; break; }

    // ── Initialize PCG on CPU ─────────────────────────────────────────────
    std::vector<double> x(n, 0.0);
    std::vector<double> r(F);   // r = F (x=0)
    std::vector<double> z(n), p(n), Ap(n);

    jacobi_cpu(diag, r, z);
    p = z;

    double rz_old = dot_cpu(r, z);
    double norm_b = std::sqrt(dot_cpu(F, F));
    if (norm_b < 1e-300) norm_b = 1.0;

    out_iters    = 0;
    out_residual = 1.0; // ||b||/||b|| = 1 (since r=b at x=0)

    // ── PCG iteration ─────────────────────────────────────────────────────
    double cum_anorm_progress    = 0.0;
    double checkpoint_progress   = 0.0;
    int    iters_since_checkpoint = 0;

    for (int iter = 0; iter < cfg.max_iterations; ++iter) {
        // ── Tiled SpMV: Ap = K * p ────────────────────────────────────────
        // Upload p (float32) once per iteration
        std::vector<float> p_f(n);
        for (int i = 0; i < n; ++i) p_f[i] = static_cast<float>(p[i]);
        p_buf.upload<float>(ctx, std::span<const float>(p_f));

        int row = 0;
        while (row < n) {
            int rows_this_tile = std::min(static_cast<int>(tile_rows), n - row);
            int col_start      = K.row_ptr[row];
            int col_end        = K.row_ptr[row + rows_this_tile];
            int tile_nnz       = col_end - col_start;

            // Re-based row pointers (offset so first entry is 0)
            std::vector<int> tile_rp(rows_this_tile + 1);
            for (int rr = 0; rr <= rows_this_tile; ++rr)
                tile_rp[rr] = K.row_ptr[row + rr] - col_start;

            std::span<const int> tile_ci(K.col_ind.data() + col_start, tile_nnz);

            std::vector<float> tile_v(tile_nnz);
            for (int i = 0; i < tile_nnz; ++i)
                tile_v[i] = static_cast<float>(K.values[col_start + i]);

            row_ptr_tile.upload<int>  (ctx, std::span<const int>  (tile_rp));
            col_ind_tile.upload<int>  (ctx, tile_ci);
            val_tile.upload<float>    (ctx, std::span<const float>(tile_v));

            // SpMV dispatch for this tile (row_start=0, re-based row_ptr)
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

            VkDescriptorSet ds = pl.alloc_set(PL_SPMV);
            Pipelines::bind_buffer(ctx.device(), ds, 0, row_ptr_tile.handle(), (rows_this_tile + 1) * sizeof(int));
            Pipelines::bind_buffer(ctx.device(), ds, 1, col_ind_tile.handle(), tile_nnz * sizeof(int));
            Pipelines::bind_buffer(ctx.device(), ds, 2, val_tile.handle(),     tile_nnz * sizeof(float));
            Pipelines::bind_buffer(ctx.device(), ds, 3, p_buf.handle(),        p_buf.size_bytes());
            Pipelines::bind_buffer(ctx.device(), ds, 4, Ap_tile_buf.handle(),  rows_this_tile * sizeof(float));

            PCSpmv pc{0, static_cast<uint32_t>(rows_this_tile), static_cast<uint32_t>(n)};
            cmd_spmv(cmd, pl, ds, pc, static_cast<uint32_t>(rows_this_tile));
            vkEndCommandBuffer(cmd);

            submit_and_wait(ctx, cmd); // throws on failure
            vkFreeCommandBuffers(ctx.device(), ctx.command_pool(), 1, &cmd);
            vkFreeDescriptorSets(ctx.device(), pl.pool, 1, &ds);

            // Download Ap slice
            auto Ap_tile = Ap_tile_buf.download<float>(ctx, static_cast<size_t>(rows_this_tile));
            for (int i = 0; i < rows_this_tile; ++i)
                Ap[row + i] = static_cast<double>(Ap_tile[i]);

            row += rows_this_tile;
        }

        // ── CPU scalar PCG update ─────────────────────────────────────────
        double pAp = dot_cpu(p, Ap);
        if (pAp <= 0.0)
            throw SolverError(std::format(
                "Vulkan tiled PCG breakdown: p^T*A*p = {:.3e} <= 0 at iteration {} "
                "(matrix is not positive definite)", pAp, iter));

        double alpha = rz_old / pAp;
        axpy_cpu( alpha, p,  x);  // x += alpha*p
        axpy_cpu(-alpha, Ap, r);  // r -= alpha*Ap

        jacobi_cpu(diag, r, z);
        double rz_new = dot_cpu(r, z);
        // Convergence metric: ||r||_2 / ||b||_2
        out_residual  = std::sqrt(dot_cpu(r, r)) / norm_b;
        out_iters     = iter + 1;

        if (out_residual < cfg.tolerance) break;
        if (iter + 1 == cfg.max_iterations) break;

        // A-norm stagnation detection (same as full-GPU path).
        cum_anorm_progress += rz_old * rz_old / pAp;
        ++iters_since_checkpoint;
        if (iters_since_checkpoint >= cfg.stagnation_window) {
            double window_progress = cum_anorm_progress - checkpoint_progress;
            if (cum_anorm_progress > 0.0 &&
                window_progress < cfg.stagnation_threshold * cum_anorm_progress) {
                throw SolverError(std::format(
                    "Vulkan PCG (tiled) stagnated: A-norm progress over last {} iterations "
                    "was {:.1e}% of total (precision floor reached). "
                    "Relative residual = {:.3e}. "
                    "Consider --cpu for higher precision.",
                    cfg.stagnation_window,
                    100.0 * window_progress / cum_anorm_progress,
                    out_residual));
            }
            checkpoint_progress = cum_anorm_progress;
            iters_since_checkpoint = 0;
        }

        double beta = rz_new / rz_old;
        axpby_cpu(1.0, z, beta, p); // p = z + beta*p
        rz_old = rz_new;
    }

    if (out_residual >= cfg.tolerance)
        throw SolverError(std::format(
            "Vulkan PCG (tiled) did not converge after {} iterations; "
            "relative residual = {:.3e} (tolerance = {:.3e}). "
            "Check boundary conditions or increase max_iterations.",
            out_iters, out_residual, cfg.tolerance));

    return x;
}

} // namespace vibestran

#endif // HAVE_VULKAN
