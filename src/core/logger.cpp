// src/core/logger.cpp
// spdlog-backed implementation of the Vibestran logging interface.

#include "core/logger.hpp"
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <memory>
#include <vector>

namespace vibestran {

void init_logger(const std::filesystem::path& log_file) {
    std::vector<spdlog::sink_ptr> sinks;

    // Console sink: plain message, no timestamp or level prefix, to match
    // the existing output appearance.
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::debug);
    console_sink->set_pattern("%v");
    sinks.push_back(console_sink);

    // Optional file sink: full timestamp + level for traceability.
    if (!log_file.empty()) {
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
            log_file.string(), /*truncate=*/true);
        file_sink->set_level(spdlog::level::debug);
        file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
        sinks.push_back(file_sink);
    }

    auto logger = std::make_shared<spdlog::logger>(
        "vibestran", sinks.begin(), sinks.end());
    logger->set_level(spdlog::level::debug);
    spdlog::set_default_logger(logger);
}

void log_trace(const std::string& msg) { spdlog::trace("{}", msg); }
void log_debug(const std::string& msg) { spdlog::debug("{}", msg); }
void log_info (const std::string& msg) { spdlog::info("{}", msg); }
void log_warn (const std::string& msg) { spdlog::warn("{}", msg); }
void log_error(const std::string& msg) { spdlog::error("{}", msg); }

} // namespace vibestran
