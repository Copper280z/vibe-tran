// src/solver/vulkan_pcg_internal.hpp  (internal, not installed)
// Forward declarations for the two PCG paths.
// Both functions throw SolverError on unrecoverable failure.
#pragma once
#ifdef HAVE_VULKAN

#include "solver/vulkan_solver_backend.hpp"
#include "vulkan_pipelines.hpp"
#include <vector>

namespace vibestran {

// Defined in vulkan_pcg_full.cpp
// Returns solution vector u.  Throws SolverError on failure.
std::vector<double> solve_full_gpu(VulkanContext& ctx, Pipelines& pl,
                                    const SparseMatrixBuilder::CsrData& K,
                                    const std::vector<double>& F,
                                    const VulkanSolverConfig& cfg,
                                    int& out_iters, double& out_residual);

// Float64 variant — requires shaderFloat64 device feature.
// Throws SolverError if the device does not support float64.
std::vector<double> solve_full_gpu_double(VulkanContext& ctx, Pipelines& pl,
                                           const SparseMatrixBuilder::CsrData& K,
                                           const std::vector<double>& F,
                                           const VulkanSolverConfig& cfg,
                                           int& out_iters, double& out_residual);

// Defined in vulkan_pcg_tiled.cpp
// Returns solution vector u.  Throws SolverError on failure.
std::vector<double> solve_tiled(VulkanContext& ctx, Pipelines& pl,
                                 const SparseMatrixBuilder::CsrData& K,
                                 const std::vector<double>& F,
                                 const VulkanSolverConfig& cfg,
                                 uint64_t available_vram_bytes,
                                 int& out_iters, double& out_residual);

} // namespace vibestran

#endif // HAVE_VULKAN
