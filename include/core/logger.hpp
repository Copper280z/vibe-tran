#pragma once
// include/core/logger.hpp
// Minimal logging interface for Vibetran.
//
// This header is intentionally kept free of spdlog includes so it can be
// safely included from CUDA .cu files (nvcc / g++-12 host compiler).
//
// Usage in regular .cpp files:
//   #include "core/logger.hpp"
//   #include <spdlog/spdlog.h>   // for format-string API
//   spdlog::info("Reading: {}", path.string());
//   spdlog::debug("[subcase {}] solve: {:.3f} ms", id, ms);
//
// Usage in CUDA .cu files:
//   #include "core/logger.hpp"
//   vibetran::log_debug("[cuda] allocating " + std::to_string(mb) + " MiB");
//
// All log functions are thread-safe (spdlog async-safe sinks).

#include <filesystem>
#include <string>

namespace vibetran {

/// Initialize the global spdlog logger. Must be called once from main()
/// before any logging. Safe to call multiple times (re-initializes).
/// @param log_file  If non-empty, all log output is also written to this file
///                  in addition to the console (stdout).
void init_logger(const std::filesystem::path& log_file = {});

/// Plain-string log functions. Safe to call from CUDA host code.
void log_trace(const std::string& msg);
void log_debug(const std::string& msg);
void log_info (const std::string& msg);
void log_warn (const std::string& msg);
void log_error(const std::string& msg);

} // namespace vibetran
