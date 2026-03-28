// src/solver/vulkan_pipelines.cpp
// Pipeline creation from embedded SPIR-V + dispatch helpers.
#ifdef HAVE_VULKAN

#include "solver/vulkan_solver_backend.hpp"
#include "vulkan_pipelines.hpp"
#include "core/types.hpp"
#include <cstring>
#include <format>

// Embedded SPIR-V bytearrays generated at build time by glslc + xxd.
// Each header defines:  unsigned char <name>_spv[] = {...}; unsigned int <name>_spv_len = ...;
#include "spmv_spv.h"
#include "dot_reduce_spv.h"
#include "axpby_spv.h"
#include "jacobi_precond_spv.h"
#include "spmv_d_spv.h"
#include "dot_reduce_d_spv.h"
#include "axpby_d_spv.h"
#include "jacobi_precond_d_spv.h"

namespace vibestran {

// ── Pipelines destructor ──────────────────────────────────────────────────────

VulkanSolverBackend::Pipelines::~Pipelines() {
    if (device == VK_NULL_HANDLE) return;
    for (int i = 0; i < PL_COUNT; ++i) {
        if (pipeline[i]       != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline[i], nullptr);
        if (pl_layout[i]      != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pl_layout[i], nullptr);
        if (dsl[i]            != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, dsl[i], nullptr);
        if (shader_modules[i] != VK_NULL_HANDLE) vkDestroyShaderModule(device, shader_modules[i], nullptr);
    }
    if (pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, pool, nullptr);
}

// ── Descriptor helpers ─────────────────────────────────────────────────────────

VkDescriptorSet VulkanSolverBackend::Pipelines::alloc_set(int id) {
    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &dsl[id];
    VkDescriptorSet ds = VK_NULL_HANDLE;
    vkAllocateDescriptorSets(device, &ai, &ds);
    return ds;
}

void VulkanSolverBackend::Pipelines::bind_buffer(VkDevice dev, VkDescriptorSet ds,
                                                  uint32_t binding, VkBuffer buf,
                                                  VkDeviceSize size) {
    VkDescriptorBufferInfo bi{buf, 0, size};
    VkWriteDescriptorSet wr{};
    wr.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wr.dstSet          = ds;
    wr.dstBinding      = binding;
    wr.descriptorCount = 1;
    wr.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr.pBufferInfo     = &bi;
    vkUpdateDescriptorSets(dev, 1, &wr, 0, nullptr);
}

// ── Internal helpers ──────────────────────────────────────────────────────────

static VkShaderModule make_shader(VkDevice dev, const unsigned char* code, unsigned int len) {
    // SPIR-V requires 4-byte alignment; xxd output arrays may not be aligned.
    std::vector<uint32_t> aligned((len + 3) / 4);
    std::memcpy(aligned.data(), code, len);

    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = len;
    ci.pCode    = aligned.data();

    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(dev, &ci, nullptr, &mod) != VK_SUCCESS)
        throw SolverError("Vulkan: vkCreateShaderModule failed");
    return mod;
}

static VkDescriptorSetLayout make_dsl(VkDevice dev, uint32_t binding_count) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(binding_count);
    for (uint32_t i = 0; i < binding_count; ++i) {
        bindings[i].binding         = i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = binding_count;
    ci.pBindings    = bindings.data();
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(dev, &ci, nullptr, &dsl) != VK_SUCCESS)
        throw SolverError("Vulkan: vkCreateDescriptorSetLayout failed");
    return dsl;
}

static VkPipelineLayout make_pipeline_layout(VkDevice dev, VkDescriptorSetLayout dsl,
                                              uint32_t push_bytes) {
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
    VkPipelineLayoutCreateInfo ci{};
    ci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ci.setLayoutCount         = 1;
    ci.pSetLayouts            = &dsl;
    ci.pushConstantRangeCount = push_bytes ? 1u : 0u;
    ci.pPushConstantRanges    = push_bytes ? &pcr : nullptr;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(dev, &ci, nullptr, &pl) != VK_SUCCESS)
        throw SolverError("Vulkan: vkCreatePipelineLayout failed");
    return pl;
}

static VkPipeline make_compute_pipeline(VkDevice dev, VkShaderModule mod,
                                         VkPipelineLayout layout) {
    VkComputePipelineCreateInfo ci{};
    ci.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.layout       = layout;
    ci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = mod;
    ci.stage.pName  = "main";
    VkPipeline pl = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pl) != VK_SUCCESS)
        throw SolverError("Vulkan: vkCreateComputePipelines failed");
    return pl;
}

// ── build_pipelines ───────────────────────────────────────────────────────────

struct ShaderSpec {
    const unsigned char* code;
    unsigned int         len;
    uint32_t             bindings;
    uint32_t             push_bytes;
};

VulkanSolverBackend::Pipelines* build_pipelines(const VulkanContext& ctx) {
    // Binding counts:
    //   spmv / spmv_d:              RowPtr, ColInd, Values, X, Y  → 5
    //   dot_reduce / dot_reduce_d:  A, B, Partials                → 3
    //   axpby / axpby_d:            X, Y                          → 2
    //   jacobi_precond / _d:        Diag, R, Z                    → 3
    const ShaderSpec specs[PL_COUNT] = {
        // float32 pipelines
        { spmv_spv,             spmv_spv_len,             5, sizeof(PCSpmv)    },
        { dot_reduce_spv,       dot_reduce_spv_len,       3, sizeof(PCDot)     },
        { axpby_spv,            axpby_spv_len,            2, sizeof(PCAxpby)   },
        { jacobi_precond_spv,   jacobi_precond_spv_len,   3, sizeof(PCJacobi)  },
        // float64 pipelines
        { spmv_d_spv,           spmv_d_spv_len,           5, sizeof(PCSpmv)    },
        { dot_reduce_d_spv,     dot_reduce_d_spv_len,     3, sizeof(PCDot)     },
        { axpby_d_spv,          axpby_d_spv_len,          2, sizeof(PCAxpbyD)  },
        { jacobi_precond_d_spv, jacobi_precond_d_spv_len, 3, sizeof(PCJacobi)  },
    };

    auto* pl   = new VulkanSolverBackend::Pipelines();
    pl->device = ctx.device();

    for (int i = 0; i < PL_COUNT; ++i) {
        const auto& s = specs[i];
        pl->shader_modules[i] = make_shader(ctx.device(), s.code, s.len);
        pl->dsl[i]            = make_dsl(ctx.device(), s.bindings);
        pl->pl_layout[i]      = make_pipeline_layout(ctx.device(), pl->dsl[i], s.push_bytes);
        pl->pipeline[i]       = make_compute_pipeline(ctx.device(), pl->shader_modules[i], pl->pl_layout[i]);
    }

    // Descriptor pool
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512};
    VkDescriptorPoolCreateInfo pool_ci{};
    pool_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.maxSets       = 128;
    pool_ci.poolSizeCount = 1;
    pool_ci.pPoolSizes    = &pool_size;
    pool_ci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    if (vkCreateDescriptorPool(ctx.device(), &pool_ci, nullptr, &pl->pool) != VK_SUCCESS) {
        delete pl;
        throw SolverError("Vulkan: vkCreateDescriptorPool failed");
    }

    return pl;
}

// ── Dispatch helpers ──────────────────────────────────────────────────────────

void cmd_spmv(VkCommandBuffer cmd, const Pipelines& pl,
              VkDescriptorSet ds, PCSpmv pc, uint32_t row_count) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.pipeline[PL_SPMV]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pl.pl_layout[PL_SPMV], 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pl.pl_layout[PL_SPMV],
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, row_count, 1, 1);
}

void cmd_dot_reduce(VkCommandBuffer cmd, const Pipelines& pl,
                    VkDescriptorSet ds, uint32_t n) {
    PCDot pc{n};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.pipeline[PL_DOT_REDUCE]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pl.pl_layout[PL_DOT_REDUCE], 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pl.pl_layout[PL_DOT_REDUCE],
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (n + 255) / 256, 1, 1);
}

void cmd_axpby(VkCommandBuffer cmd, const Pipelines& pl,
               VkDescriptorSet ds, PCAxpby pc, uint32_t n) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.pipeline[PL_AXPBY]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pl.pl_layout[PL_AXPBY], 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pl.pl_layout[PL_AXPBY],
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (n + 255) / 256, 1, 1);
}

void cmd_jacobi(VkCommandBuffer cmd, const Pipelines& pl,
                VkDescriptorSet ds, uint32_t n) {
    PCJacobi pc{n};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.pipeline[PL_JACOBI]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pl.pl_layout[PL_JACOBI], 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pl.pl_layout[PL_JACOBI],
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (n + 255) / 256, 1, 1);
}

// ── Double-precision dispatch helpers ─────────────────────────────────────────

void cmd_spmv_d(VkCommandBuffer cmd, const Pipelines& pl,
                VkDescriptorSet ds, PCSpmv pc, uint32_t row_count) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.pipeline[PL_SPMV_D]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pl.pl_layout[PL_SPMV_D], 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pl.pl_layout[PL_SPMV_D],
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, row_count, 1, 1);
}

void cmd_dot_reduce_d(VkCommandBuffer cmd, const Pipelines& pl,
                      VkDescriptorSet ds, uint32_t n) {
    PCDot pc{n};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.pipeline[PL_DOT_REDUCE_D]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pl.pl_layout[PL_DOT_REDUCE_D], 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pl.pl_layout[PL_DOT_REDUCE_D],
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (n + 255) / 256, 1, 1);
}

void cmd_axpby_d(VkCommandBuffer cmd, const Pipelines& pl,
                 VkDescriptorSet ds, const PCAxpbyD& pc, uint32_t n) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.pipeline[PL_AXPBY_D]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pl.pl_layout[PL_AXPBY_D], 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pl.pl_layout[PL_AXPBY_D],
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (n + 255) / 256, 1, 1);
}

void cmd_jacobi_d(VkCommandBuffer cmd, const Pipelines& pl,
                  VkDescriptorSet ds, uint32_t n) {
    PCJacobi pc{n};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.pipeline[PL_JACOBI_D]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pl.pl_layout[PL_JACOBI_D], 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pl.pl_layout[PL_JACOBI_D],
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (n + 255) / 256, 1, 1);
}

} // namespace vibestran

#endif // HAVE_VULKAN
