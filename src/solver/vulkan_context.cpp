// src/solver/vulkan_context.cpp
#ifdef HAVE_VULKAN

#include "solver/vulkan_context.hpp"
#include <algorithm>
#include <vector>

namespace vibestran {

// ── Destruction ─────────────────────────────────────────────────────────────

VulkanContext::~VulkanContext() {
    if (command_pool_ != VK_NULL_HANDLE)
        vkDestroyCommandPool(device_, command_pool_, nullptr);
    if (device_ != VK_NULL_HANDLE)
        vkDestroyDevice(device_, nullptr);
    if (instance_ != VK_NULL_HANDLE)
        vkDestroyInstance(instance_, nullptr);
}

VulkanContext::VulkanContext(VulkanContext&& o) noexcept
    : instance_(o.instance_)
    , phys_dev_(o.phys_dev_)
    , device_(o.device_)
    , compute_queue_(o.compute_queue_)
    , command_pool_(o.command_pool_)
    , compute_queue_family_(o.compute_queue_family_)
    , device_info_(std::move(o.device_info_))
{
    o.instance_      = VK_NULL_HANDLE;
    o.device_        = VK_NULL_HANDLE;
    o.command_pool_  = VK_NULL_HANDLE;
}

VulkanContext& VulkanContext::operator=(VulkanContext&& o) noexcept {
    if (this != &o) {
        this->~VulkanContext();
        new (this) VulkanContext(std::move(o));
    }
    return *this;
}

// ── Accessors ────────────────────────────────────────────────────────────────

const VulkanDeviceInfo& VulkanContext::device_info()     const noexcept { return device_info_; }
VkDevice                VulkanContext::device()          const noexcept { return device_; }
VkPhysicalDevice        VulkanContext::physical_device() const noexcept { return phys_dev_; }
VkQueue                 VulkanContext::compute_queue()   const noexcept { return compute_queue_; }
VkCommandPool           VulkanContext::command_pool()    const noexcept { return command_pool_; }
uint32_t                VulkanContext::compute_queue_family() const noexcept { return compute_queue_family_; }

uint32_t VulkanContext::find_memory_type(uint32_t type_filter,
                                         VkMemoryPropertyFlags props) const noexcept {
    VkPhysicalDeviceMemoryProperties mem_props{};
    vkGetPhysicalDeviceMemoryProperties(phys_dev_, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return VK_MAX_MEMORY_TYPES;
}

// ── Factory ──────────────────────────────────────────────────────────────────

std::optional<VulkanContext> VulkanContext::create() noexcept {
    VulkanContext ctx;

    // ── Instance ─────────────────────────────────────────────────────────────
    VkApplicationInfo app_info{};
    app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName   = "Vibestran";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion         = VK_API_VERSION_1_1;

    VkInstanceCreateInfo inst_ci{};
    inst_ci.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_ci.pApplicationInfo = &app_info;

    if (vkCreateInstance(&inst_ci, nullptr, &ctx.instance_) != VK_SUCCESS)
        return std::nullopt;

    // ── Physical device selection ─────────────────────────────────────────────
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(ctx.instance_, &dev_count, nullptr);
    if (dev_count == 0) {
        vkDestroyInstance(ctx.instance_, nullptr);
        return std::nullopt;
    }
    std::vector<VkPhysicalDevice> devs(dev_count);
    vkEnumeratePhysicalDevices(ctx.instance_, &dev_count, devs.data());

    struct Candidate {
        VkPhysicalDevice dev;
        uint32_t         queue_family;
        uint64_t         vram_bytes;
        bool             is_discrete;
        bool             float64;
    };
    std::vector<Candidate> candidates;

    for (auto phys : devs) {
        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &qf_count, nullptr);
        std::vector<VkQueueFamilyProperties> qf_props(qf_count);
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &qf_count, qf_props.data());

        uint32_t compute_qf = UINT32_MAX;
        for (uint32_t i = 0; i < qf_count; ++i) {
            if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                compute_qf = i;
                break;
            }
        }
        if (compute_qf == UINT32_MAX) continue;

        VkPhysicalDeviceMemoryProperties mem_props{};
        vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
        uint64_t vram = 0;
        for (uint32_t h = 0; h < mem_props.memoryHeapCount; ++h)
            if (mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                vram = std::max(vram, mem_props.memoryHeaps[h].size);

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(phys, &props);
        bool discrete = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);

        VkPhysicalDeviceFeatures feats{};
        vkGetPhysicalDeviceFeatures(phys, &feats);

        candidates.push_back({phys, compute_qf, vram, discrete,
                               feats.shaderFloat64 == VK_TRUE});
    }

    if (candidates.empty()) {
        vkDestroyInstance(ctx.instance_, nullptr);
        return std::nullopt;
    }

    std::stable_sort(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) {
            if (a.is_discrete != b.is_discrete) return a.is_discrete > b.is_discrete;
            return a.vram_bytes > b.vram_bytes;
        });

    const Candidate& best = candidates[0];
    ctx.phys_dev_             = best.dev;
    ctx.compute_queue_family_ = best.queue_family;

    // ── Logical device ────────────────────────────────────────────────────────
    float priority = 1.0f;
    VkDeviceQueueCreateInfo q_ci{};
    q_ci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    q_ci.queueFamilyIndex = ctx.compute_queue_family_;
    q_ci.queueCount       = 1;
    q_ci.pQueuePriorities = &priority;

    // Enable shaderFloat64 when the physical device supports it.
    // Without this, using double-precision shaders is a spec violation
    // (undefined behaviour — silently zeroes or corrupts results at scale).
    VkPhysicalDeviceFeatures enabled_features{};
    enabled_features.shaderFloat64 = best.float64 ? VK_TRUE : VK_FALSE;

    VkDeviceCreateInfo dev_ci{};
    dev_ci.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dev_ci.queueCreateInfoCount = 1;
    dev_ci.pQueueCreateInfos    = &q_ci;
    dev_ci.pEnabledFeatures     = &enabled_features;

    if (vkCreateDevice(ctx.phys_dev_, &dev_ci, nullptr, &ctx.device_) != VK_SUCCESS) {
        vkDestroyInstance(ctx.instance_, nullptr);
        return std::nullopt;
    }

    vkGetDeviceQueue(ctx.device_, ctx.compute_queue_family_, 0, &ctx.compute_queue_);

    // ── Command pool ──────────────────────────────────────────────────────────
    VkCommandPoolCreateInfo pool_ci{};
    pool_ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.queueFamilyIndex = ctx.compute_queue_family_;
    pool_ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(ctx.device_, &pool_ci, nullptr, &ctx.command_pool_) != VK_SUCCESS) {
        vkDestroyDevice(ctx.device_, nullptr);
        vkDestroyInstance(ctx.instance_, nullptr);
        return std::nullopt;
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.phys_dev_, &props);
    ctx.device_info_.device_name           = props.deviceName;
    ctx.device_info_.vram_bytes            = best.vram_bytes;
    ctx.device_info_.supports_float64      = best.float64;
    ctx.device_info_.max_workgroup_count_x = props.limits.maxComputeWorkGroupCount[0];

    return ctx;
}

} // namespace vibestran

#endif // HAVE_VULKAN
