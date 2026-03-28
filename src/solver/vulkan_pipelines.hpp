// src/solver/vulkan_pipelines.hpp  (internal, not installed)
// Shared pipeline state for the Vulkan PCG solver.
#pragma once
#ifdef HAVE_VULKAN

#include "solver/vulkan_context.hpp"
#include "solver/vulkan_solver_backend.hpp"
#include <array>
#include <cstdint>
#include <span>
#include <vector>
#include <vulkan/vulkan.h>

namespace vibestran {

// IDs used to index into the pipeline array.
// float32 pipelines (indices 0-3) and float64 pipelines (indices 4-7).
enum PipelineId : int {
    PL_SPMV          = 0,
    PL_DOT_REDUCE    = 1,
    PL_AXPBY         = 2,
    PL_JACOBI        = 3,
    // Double-precision variants (require shaderFloat64 device feature)
    PL_SPMV_D        = 4,
    PL_DOT_REDUCE_D  = 5,
    PL_AXPBY_D       = 6,
    PL_JACOBI_D      = 7,
    PL_COUNT         = 8,
};

using Pipelines = VulkanSolverBackend::Pipelines;

/// Holds all compute pipelines and descriptor infrastructure.
struct VulkanSolverBackend::Pipelines {
    std::array<VkShaderModule,        PL_COUNT> shader_modules{};
    std::array<VkDescriptorSetLayout, PL_COUNT> dsl{};
    std::array<VkPipelineLayout,      PL_COUNT> pl_layout{};
    std::array<VkPipeline,            PL_COUNT> pipeline{};
    VkDescriptorPool pool{VK_NULL_HANDLE};
    VkDevice         device{VK_NULL_HANDLE}; // non-owning

    ~Pipelines();

    /// Allocate a descriptor set for pipeline `id`.
    [[nodiscard]] VkDescriptorSet alloc_set(int id);

    /// Bind a storage buffer to a specific binding in a descriptor set.
    static void bind_buffer(VkDevice dev, VkDescriptorSet ds,
                             uint32_t binding, VkBuffer buf, VkDeviceSize size);
};

/// Build all pipelines from embedded SPIR-V.
/// Throws SolverError on failure.
[[nodiscard]] VulkanSolverBackend::Pipelines* build_pipelines(const VulkanContext& ctx);

// ── Push constant structs (must match shader layout) ─────────────────────────

struct PCSpmv   { uint32_t row_start; uint32_t row_count; uint32_t n; };
struct PCDot    { uint32_t n; };
struct PCAxpby  { float    alpha; float    beta; uint32_t n; };
struct PCJacobi { uint32_t n; };

// Double-precision axpby push constant.  Layout matches the GLSL block:
//   double alpha (offset 0), double beta (offset 8), uint n (offset 16),
//   uint _pad (offset 20) → total 24 bytes, aligned to 8.
struct PCAxpbyD { double alpha; double beta; uint32_t n; uint32_t _pad; };

// ── Dispatch helpers (float32) ────────────────────────────────────────────────

void cmd_spmv(VkCommandBuffer cmd, const Pipelines& pl,
              VkDescriptorSet ds, PCSpmv pc, uint32_t row_count);
void cmd_dot_reduce(VkCommandBuffer cmd, const Pipelines& pl,
                    VkDescriptorSet ds, uint32_t n);
void cmd_axpby(VkCommandBuffer cmd, const Pipelines& pl,
               VkDescriptorSet ds, PCAxpby pc, uint32_t n);
void cmd_jacobi(VkCommandBuffer cmd, const Pipelines& pl,
                VkDescriptorSet ds, uint32_t n);

// ── Dispatch helpers (float64) ────────────────────────────────────────────────

void cmd_spmv_d(VkCommandBuffer cmd, const Pipelines& pl,
                VkDescriptorSet ds, PCSpmv pc, uint32_t row_count);
void cmd_dot_reduce_d(VkCommandBuffer cmd, const Pipelines& pl,
                      VkDescriptorSet ds, uint32_t n);
void cmd_axpby_d(VkCommandBuffer cmd, const Pipelines& pl,
                 VkDescriptorSet ds, const PCAxpbyD& pc, uint32_t n);
void cmd_jacobi_d(VkCommandBuffer cmd, const Pipelines& pl,
                  VkDescriptorSet ds, uint32_t n);

} // namespace vibestran

#endif // HAVE_VULKAN
