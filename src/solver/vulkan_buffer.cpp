// src/solver/vulkan_buffer.cpp
#ifdef HAVE_VULKAN

#include "solver/vulkan_buffer.hpp"
#include "core/types.hpp"
#include <cstring>
#include <format>

namespace nastran {

// ── Destruction / move ────────────────────────────────────────────────────────

VulkanBuffer::~VulkanBuffer() {
    if (device_ == VK_NULL_HANDLE) return;
    if (buffer_ != VK_NULL_HANDLE) vkDestroyBuffer(device_, buffer_, nullptr);
    if (memory_ != VK_NULL_HANDLE) vkFreeMemory(device_, memory_, nullptr);
}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& o) noexcept
    : buffer_(o.buffer_), memory_(o.memory_), size_(o.size_), device_(o.device_)
{
    o.buffer_ = VK_NULL_HANDLE;
    o.memory_ = VK_NULL_HANDLE;
    o.device_ = VK_NULL_HANDLE;
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& o) noexcept {
    if (this != &o) { this->~VulkanBuffer(); new (this) VulkanBuffer(std::move(o)); }
    return *this;
}

// ── Internal helpers ──────────────────────────────────────────────────────────

struct RawBuffer {
    VkBuffer       buffer{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
};

/// Allocate a buffer + memory.  Returns {VK_NULL_HANDLE,VK_NULL_HANDLE} on failure.
static RawBuffer alloc_raw(const VulkanContext& ctx, VkDeviceSize size,
                            VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_props) {
    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = size;
    bci.usage       = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    RawBuffer result;
    if (vkCreateBuffer(ctx.device(), &bci, nullptr, &result.buffer) != VK_SUCCESS)
        return {};

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx.device(), result.buffer, &req);

    uint32_t mt = ctx.find_memory_type(req.memoryTypeBits, mem_props);
    if (mt == VK_MAX_MEMORY_TYPES) {
        vkDestroyBuffer(ctx.device(), result.buffer, nullptr);
        return {};
    }

    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = mt;

    if (vkAllocateMemory(ctx.device(), &ai, nullptr, &result.memory) != VK_SUCCESS) {
        vkDestroyBuffer(ctx.device(), result.buffer, nullptr);
        return {};
    }
    vkBindBufferMemory(ctx.device(), result.buffer, result.memory, 0);
    return result;
}

static void free_raw(VkDevice dev, RawBuffer& rb) {
    if (rb.buffer != VK_NULL_HANDLE) vkDestroyBuffer(dev, rb.buffer, nullptr);
    if (rb.memory != VK_NULL_HANDLE) vkFreeMemory(dev, rb.memory, nullptr);
    rb = {};
}

// ── Public factory ─────────────────────────────────────────────────────────────

std::optional<VulkanBuffer>
VulkanBuffer::create(const VulkanContext& ctx, VkDeviceSize size_bytes) noexcept {
    RawBuffer rb = alloc_raw(ctx, size_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT   |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (rb.buffer == VK_NULL_HANDLE) return std::nullopt;

    VulkanBuffer buf;
    buf.buffer_ = rb.buffer;
    buf.memory_ = rb.memory;
    buf.size_   = size_bytes;
    buf.device_ = ctx.device();
    return buf;
}

// ── submit_and_wait ────────────────────────────────────────────────────────────

void submit_and_wait(const VulkanContext& ctx, VkCommandBuffer cmd) {
    VkFenceCreateInfo fence_ci{};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence  = VK_NULL_HANDLE;
    if (vkCreateFence(ctx.device(), &fence_ci, nullptr, &fence) != VK_SUCCESS)
        throw SolverError("Vulkan: vkCreateFence failed");

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;

    VkResult r = vkQueueSubmit(ctx.compute_queue(), 1, &si, fence);
    if (r == VK_SUCCESS)
        r = vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(ctx.device(), fence, nullptr);
    if (r != VK_SUCCESS)
        throw SolverError(std::format("Vulkan: queue submit/wait failed: {}", static_cast<int>(r)));
}

// ── Upload ─────────────────────────────────────────────────────────────────────

void VulkanBuffer::upload_raw(const VulkanContext& ctx, const void* data, VkDeviceSize size) {
    RawBuffer stg = alloc_raw(ctx, size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (stg.buffer == VK_NULL_HANDLE)
        throw SolverError("Vulkan: staging buffer allocation failed during upload");

    void* mapped = nullptr;
    if (vkMapMemory(ctx.device(), stg.memory, 0, size, 0, &mapped) != VK_SUCCESS) {
        free_raw(ctx.device(), stg);
        throw SolverError("Vulkan: vkMapMemory failed during upload");
    }
    std::memcpy(mapped, data, static_cast<size_t>(size));
    vkUnmapMemory(ctx.device(), stg.memory);

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
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(cmd, stg.buffer, buffer_, 1, &region);
    vkEndCommandBuffer(cmd);

    submit_and_wait(ctx, cmd); // throws on failure
    vkFreeCommandBuffers(ctx.device(), ctx.command_pool(), 1, &cmd);
    free_raw(ctx.device(), stg);
}

// ── Download ───────────────────────────────────────────────────────────────────

std::vector<std::byte> VulkanBuffer::download_raw(const VulkanContext& ctx, VkDeviceSize size) {
    RawBuffer stg = alloc_raw(ctx, size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (stg.buffer == VK_NULL_HANDLE)
        throw SolverError("Vulkan: staging buffer allocation failed during download");

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

    // Ensure any preceding compute/transfer writes to this buffer (from a
    // prior submission) are visible before the copy reads from it.
    VkMemoryBarrier mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);

    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(cmd, buffer_, stg.buffer, 1, &region);
    vkEndCommandBuffer(cmd);

    submit_and_wait(ctx, cmd); // throws on failure
    vkFreeCommandBuffers(ctx.device(), ctx.command_pool(), 1, &cmd);

    void* mapped = nullptr;
    if (vkMapMemory(ctx.device(), stg.memory, 0, size, 0, &mapped) != VK_SUCCESS) {
        free_raw(ctx.device(), stg);
        throw SolverError("Vulkan: vkMapMemory failed during download");
    }
    std::vector<std::byte> out(static_cast<size_t>(size));
    std::memcpy(out.data(), mapped, static_cast<size_t>(size));
    vkUnmapMemory(ctx.device(), stg.memory);
    free_raw(ctx.device(), stg);
    return out;
}

} // namespace nastran

#endif // HAVE_VULKAN
