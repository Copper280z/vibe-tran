#pragma once
// include/solver/vulkan_context.hpp
// RAII wrapper for Vulkan instance, physical device, logical device,
// compute queue, and command pool.
//
// VulkanContext::create() returns std::optional — callers can test availability
// without exception handling.  All other failures throw SolverError (consistent
// with the project convention for unrecoverable errors).

#ifdef HAVE_VULKAN

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vulkan/vulkan.h>

namespace nastran {

struct VulkanDeviceInfo {
    uint64_t    vram_bytes;       // Largest device-local memory heap
    std::string device_name;      // Human-readable GPU name
    bool        supports_float64; // shaderFloat64 feature available
    uint32_t    max_workgroup_count_x; // maxComputeWorkGroupCount[0]
};

/// Owns all global Vulkan objects needed for compute:
///   VkInstance, VkPhysicalDevice, VkDevice, compute VkQueue, VkCommandPool.
///
/// Construct via VulkanContext::create(). Move-only.
class VulkanContext {
public:
    ~VulkanContext();
    VulkanContext(VulkanContext&&) noexcept;
    VulkanContext& operator=(VulkanContext&&) noexcept;
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    /// Factory — returns nullopt when Vulkan is unavailable or no suitable
    /// GPU is found.  Never throws.
    [[nodiscard]] static std::optional<VulkanContext> create() noexcept;

    [[nodiscard]] const VulkanDeviceInfo& device_info()         const noexcept;
    [[nodiscard]] VkDevice               device()               const noexcept;
    [[nodiscard]] VkPhysicalDevice       physical_device()      const noexcept; // only used in tests
    [[nodiscard]] VkQueue                compute_queue()        const noexcept;
    [[nodiscard]] VkCommandPool          command_pool()         const noexcept;
    [[nodiscard]] uint32_t               compute_queue_family() const noexcept; // only used in tests

    /// Find a memory type index satisfying the required type bits and
    /// property flags. Returns VK_MAX_MEMORY_TYPES on failure.
    [[nodiscard]] uint32_t find_memory_type(uint32_t type_filter,
                                            VkMemoryPropertyFlags props) const noexcept;

private:
    VulkanContext() = default;

    VkInstance       instance_{VK_NULL_HANDLE};
    VkPhysicalDevice phys_dev_{VK_NULL_HANDLE};
    VkDevice         device_{VK_NULL_HANDLE};
    VkQueue          compute_queue_{VK_NULL_HANDLE};
    VkCommandPool    command_pool_{VK_NULL_HANDLE};
    uint32_t         compute_queue_family_{0};
    VulkanDeviceInfo device_info_{};
};

} // namespace nastran

#endif // HAVE_VULKAN
