#pragma once
// include/solver/vulkan_buffer.hpp
// RAII GPU buffer: device-local storage with upload/download via staging.
//
// create() returns std::optional; upload/download throw SolverError on failure
// (consistent with the project convention: failures here are unrecoverable).

#ifdef HAVE_VULKAN

#include "solver/vulkan_context.hpp"
#include <cstring>
#include <optional>
#include <span>
#include <vector>
#include <vulkan/vulkan.h>

namespace vibestran {

/// A device-local Vulkan buffer.  Move-only, destroyed on scope exit.
class VulkanBuffer {
public:
    ~VulkanBuffer();
    VulkanBuffer(VulkanBuffer&&) noexcept;
    VulkanBuffer& operator=(VulkanBuffer&&) noexcept;
    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;

    /// Allocate a device-local buffer of `size_bytes`.
    /// Returns nullopt if allocation fails.
    [[nodiscard]] static std::optional<VulkanBuffer>
        create(const VulkanContext& ctx, VkDeviceSize size_bytes) noexcept;

    /// Upload host data to the device buffer via a staging buffer.
    /// Throws SolverError on failure.
    template <typename T>
    void upload(const VulkanContext& ctx, std::span<const T> data) {
        upload_raw(ctx, data.data(),
                   static_cast<VkDeviceSize>(data.size_bytes()));
    }

    /// Download count elements of type T from this buffer into a host vector.
    /// Throws SolverError on failure.
    template <typename T>
    [[nodiscard]] std::vector<T> download(const VulkanContext& ctx, size_t count) {
        auto raw = download_raw(ctx, count * sizeof(T));
        std::vector<T> out(count);
        std::memcpy(out.data(), raw.data(), count * sizeof(T));
        return out;
    }

    [[nodiscard]] VkBuffer     handle()     const noexcept { return buffer_; }
    [[nodiscard]] VkDeviceSize size_bytes() const noexcept { return size_;   }

private:
    VulkanBuffer() = default;

    void upload_raw(const VulkanContext& ctx, const void* data, VkDeviceSize size);
    [[nodiscard]] std::vector<std::byte> download_raw(const VulkanContext& ctx, VkDeviceSize size);

    VkBuffer       buffer_{VK_NULL_HANDLE};
    VkDeviceMemory memory_{VK_NULL_HANDLE};
    VkDeviceSize   size_{0};
    VkDevice       device_{VK_NULL_HANDLE}; // non-owning
};

/// Execute a single-use command buffer and wait for completion.
/// Throws SolverError if queue submission fails.
void submit_and_wait(const VulkanContext& ctx, VkCommandBuffer cmd);

} // namespace vibestran

#endif // HAVE_VULKAN
