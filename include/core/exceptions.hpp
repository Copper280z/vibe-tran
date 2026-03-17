#pragma once
// include/core/exceptions.hpp
// Solver and parser exception types.
//
// Intentionally kept free of <format> and other C++20-only headers so that
// this file can be included from CUDA compilation units, which are compiled
// with GCC 12 (the maximum version supported by CUDA 12.x).

#include <stdexcept>
#include <string>

namespace nastran {

class SolverError : public std::runtime_error {
public:
    explicit SolverError(const std::string& msg) : std::runtime_error(msg) {}
};

class ParseError : public std::runtime_error {
public:
    explicit ParseError(const std::string& msg) : std::runtime_error(msg) {}
};

} // namespace nastran
