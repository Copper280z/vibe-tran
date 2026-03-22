#pragma once
// include/solver/vulkan_solver_backend.hpp
// Vulkan compute backend: Jacobi-Preconditioned Conjugate Gradient (PCG).
//
// Two execution paths are selected automatically at solve() time:
//
//  Full-GPU path  — entire K matrix + working vectors fit in VRAM.
//                   SpMV, dot products, vector ops all run as GPU dispatches.
//                   Minimal CPU↔GPU sync (only two scalar readbacks per iter).
//
//  Tiled path     — K is too large for VRAM.
//                   Matrix is streamed in row-band tiles each CG iteration.
//                   GPU handles only SpMV; all scalar/vector ops run on CPU.
//                   Memory footprint is O(nnz_per_tile) not O(nnz_total).
//
// Use try_create() to construct — returns nullopt if Vulkan is unavailable
// so the caller can fall back to EigenSolverBackend without exception handling.

#ifdef HAVE_VULKAN

#include "solver/solver_backend.hpp"
#include "solver/vulkan_context.hpp"
#include <memory>
#include <optional>

namespace vibetran {

struct VulkanSolverConfig {
  int max_iterations = 10000;
  double tolerance =
      5e-5; // relative residual: ||r|| / ||b||
            // float32 GPU ops limit achievable precision to ~1e-7;
            // 1e-6 is safely above that floor and adequate for FEM use.
  double vram_headroom = 0.15; // fraction of VRAM reserved for driver
  bool force_tiled =
      false; // bypass VRAM check and always use tiled path (for testing)
  int stagnation_window =
      200; // throw if A-norm progress hasn't improved within this many iters
  double stagnation_threshold =
      0.01; // minimum fractional improvement required per window (1%)
  int min_dofs_for_gpu =
      50000; // fall back to Eigen CPU for problems smaller than this
  bool use_double = false; // use float64 GPU compute for higher precision
                           // (requires shaderFloat64)
};

class VulkanSolverBackend final : public SolverBackend {
public:
  ~VulkanSolverBackend() override;
  VulkanSolverBackend(VulkanSolverBackend &&) noexcept;
  VulkanSolverBackend &operator=(VulkanSolverBackend &&) noexcept;
  VulkanSolverBackend(const VulkanSolverBackend &) = delete;
  VulkanSolverBackend &operator=(const VulkanSolverBackend &) = delete;

  /// Factory — returns nullopt when Vulkan is unavailable.
  /// The caller should then instantiate EigenSolverBackend instead.
  [[nodiscard]] static std::optional<VulkanSolverBackend>
  try_create(const VulkanSolverConfig &cfg = {});

  /// Solve K*u = F using PCG.
  /// Throws SolverError on failure (consistent with EigenSolverBackend).
  [[nodiscard]] std::vector<double>
  solve(const SparseMatrixBuilder::CsrData &K,
        const std::vector<double> &F) override;

  [[nodiscard]] std::string_view name() const noexcept override;

  // ── Diagnostics (valid after each solve()) ─────────────────────────────
  /// True if the most recent solve used the full-GPU path.
  [[nodiscard]] bool last_solve_was_full_gpu() const noexcept;
  /// Number of PCG iterations in the most recent solve.
  [[nodiscard]] int last_iteration_count() const noexcept override;
  /// Relative residual norm at convergence (or at max_iterations).
  [[nodiscard]] double last_residual_norm() const noexcept;
  /// Alias for last_residual_norm() to satisfy SolverBackend interface.
  [[nodiscard]] double last_estimated_error() const noexcept override;

  // Opaque pipeline state — forward-declared here, defined in
  // vulkan_pipelines.hpp. Declared before private: so build_pipelines() (a free
  // function) can name this type.
  struct Pipelines;

private:
  explicit VulkanSolverBackend(VulkanContext ctx,
                               const VulkanSolverConfig &cfg);

  /// Returns true if K + working vectors fit in available VRAM.
  [[nodiscard]] bool
  fits_in_vram(const SparseMatrixBuilder::CsrData &K) const noexcept;

  VulkanContext ctx_;
  VulkanSolverConfig cfg_;
  bool last_full_gpu_{false};
  int last_iters_{0};
  double last_residual_{0.0};

  std::unique_ptr<Pipelines> pipelines_;
};

} // namespace vibetran

#endif // HAVE_VULKAN
