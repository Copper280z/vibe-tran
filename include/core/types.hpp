#pragma once
// include/core/types.hpp
// Strong typedefs and fundamental types for the FE solver.
// Using strong types prevents accidental mix-ups between node IDs, DOF indices,
// etc.

#include "core/exceptions.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <numbers>
#include <span>
#include <string>
#include <vector>

namespace vibestran {

// ── Strong integer wrappers ──────────────────────────────────────────────────

/// Nastran grid point (node) ID as it appears in the BDF
struct NodeId {
  int value{0};
  constexpr explicit NodeId(int v) noexcept : value(v) {}
  constexpr bool operator==(const NodeId &) const noexcept = default;
  constexpr auto operator<=>(const NodeId &) const noexcept = default;
};

/// Nastran element ID as it appears in the BDF
struct ElementId {
  int value{0};
  constexpr explicit ElementId(int v) noexcept : value(v) {}
  constexpr bool operator==(const ElementId &) const noexcept = default;
  constexpr auto operator<=>(const ElementId &) const noexcept = default;
};

/// Nastran property ID (PID)
struct PropertyId {
  int value{0};
  constexpr explicit PropertyId(int v) noexcept : value(v) {}
  constexpr bool operator==(const PropertyId &) const noexcept = default;
  constexpr auto operator<=>(const PropertyId &) const noexcept = default;
};

/// Nastran material ID (MID)
struct MaterialId {
  int value{0};
  constexpr explicit MaterialId(int v) noexcept : value(v) {}
  constexpr bool operator==(const MaterialId &) const noexcept = default;
  constexpr auto operator<=>(const MaterialId &) const noexcept = default;
};

/// Nastran coordinate system ID
struct CoordId {
  int value{0};
  constexpr explicit CoordId(int v) noexcept : value(v) {}
  constexpr bool operator==(const CoordId &) const noexcept = default;
  constexpr auto operator<=>(const CoordId &) const noexcept = default;
  static constexpr CoordId basic() noexcept { return CoordId{0}; }
};

/// Load set ID
struct LoadSetId {
  int value{0};
  constexpr explicit LoadSetId(int v) noexcept : value(v) {}
  constexpr bool operator==(const LoadSetId &) const noexcept = default;
  constexpr auto operator<=>(const LoadSetId &) const noexcept = default;
};

/// SPC set ID
struct SpcSetId {
  int value{0};
  constexpr explicit SpcSetId(int v) noexcept : value(v) {}
  constexpr bool operator==(const SpcSetId &) const noexcept = default;
  constexpr auto operator<=>(const SpcSetId &) const noexcept = default;
};

/// MPC set ID
struct MpcSetId {
  int value{0};
  constexpr explicit MpcSetId(int v) noexcept : value(v) {}
  constexpr bool operator==(const MpcSetId &) const noexcept = default;
  constexpr auto operator<=>(const MpcSetId &) const noexcept = default;
};

// ── DOF encoding ─────────────────────────────────────────────────────────────

/// Nastran DOF component (1-6: T1,T2,T3,R1,R2,R3)
enum class DofComponent : int {
  T1 = 1,
  T2 = 2,
  T3 = 3,
  R1 = 4,
  R2 = 5,
  R3 = 6
};

/// Bitmask of DOF components (as used in SPC, etc.)
/// Bit 0 = T1, bit 1 = T2, ..., bit 5 = R3
struct DofSet {
  uint8_t mask{0};

  static DofSet from_nastran_string(std::string_view s) {
    DofSet ds;
    for (char c : s) {
      if (c < '1' || c > '6')
        throw std::invalid_argument(
            std::format("Invalid DOF character: '{}'", c));
      ds.mask |= static_cast<uint8_t>(1 << (c - '1'));
    }
    return ds;
  }

  static DofSet from_int(int n) {
    DofSet ds;
    int tmp = n;
    while (tmp > 0) {
      int digit = tmp % 10;
      if (digit >= 1 && digit <= 6)
        ds.mask |= static_cast<uint8_t>(1 << (digit - 1));
      tmp /= 10;
    }
    return ds;
  }

  [[nodiscard]] bool has(DofComponent c) const noexcept {
    return (mask >> (static_cast<int>(c) - 1)) & 1;
  }

  [[nodiscard]] bool has(int one_based) const noexcept {
    return (mask >> (one_based - 1)) & 1;
  }

  static DofSet all() noexcept {
    DofSet ds;
    ds.mask = 0x3F;
    return ds;
  }
  static DofSet none() noexcept { return {}; }
  static DofSet translations() noexcept {
    DofSet ds;
    ds.mask = 0x07;
    return ds;
  }
};

// ── 3D vector
// ─────────────────────────────────────────────────────────────────

struct Vec3 {
  double x{0}, y{0}, z{0};
  constexpr Vec3() = default;
  constexpr Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

  constexpr Vec3 operator+(const Vec3 &o) const noexcept {
    return {x + o.x, y + o.y, z + o.z};
  }
  constexpr Vec3 operator-(const Vec3 &o) const noexcept {
    return {x - o.x, y - o.y, z - o.z};
  }
  constexpr Vec3 operator*(double s) const noexcept {
    return {x * s, y * s, z * s};
  }
  constexpr double dot(const Vec3 &o) const noexcept {
    return x * o.x + y * o.y + z * o.z;
  }
  constexpr Vec3 cross(const Vec3 &o) const noexcept {
    return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
  }
  [[nodiscard]] double norm() const noexcept {
    return std::sqrt(x * x + y * y + z * z);
  }
  [[nodiscard]] Vec3 normalized() const {
    double n = norm();
    if (n < 1e-15)
      throw std::runtime_error("Cannot normalize zero vector");
    return {x / n, y / n, z / n};
  }
  constexpr bool operator==(const Vec3 &) const noexcept = default;
};

// ── Solution type
// ─────────────────────────────────────────────────────────────

enum class SolutionType {
  LinearStatic = 101,
  Modal        = 103,
};

// ── Solver error
// ──────────────────────────────────────────────────────────────

// SolverError and ParseError are defined in exceptions.hpp (included above).
// They are re-exported from this header so existing code that includes
// types.hpp continues to work without changes.

} // namespace vibestran

// Hash support for strong ID types
namespace std {
template <> struct hash<vibestran::NodeId> {
  size_t operator()(vibestran::NodeId id) const noexcept {
    return hash<int>{}(id.value);
  }
};
template <> struct hash<vibestran::ElementId> {
  size_t operator()(vibestran::ElementId id) const noexcept {
    return hash<int>{}(id.value);
  }
};
template <> struct hash<vibestran::PropertyId> {
  size_t operator()(vibestran::PropertyId id) const noexcept {
    return hash<int>{}(id.value);
  }
};
template <> struct hash<vibestran::MaterialId> {
  size_t operator()(vibestran::MaterialId id) const noexcept {
    return hash<int>{}(id.value);
  }
};
template <> struct hash<vibestran::CoordId> {
  size_t operator()(vibestran::CoordId id) const noexcept {
    return hash<int>{}(id.value);
  }
};
template <> struct hash<vibestran::MpcSetId> {
  size_t operator()(vibestran::MpcSetId id) const noexcept {
    return hash<int>{}(id.value);
  }
};
} // namespace std
