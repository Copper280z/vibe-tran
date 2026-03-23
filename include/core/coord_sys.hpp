#pragma once
// include/core/coord_sys.hpp
// Coordinate system definitions and transform utilities.
//
// Supports:
//   CORD2R / CORD1R — Rectangular (constant rotation matrix)
//   CORD2C / CORD1C — Cylindrical (position-dependent rotation matrix)
//   CORD2S          — Spherical   (position-dependent rotation matrix)
//
// All systems are represented by an origin + 3×3 orthonormal axis matrix
// (axes[0]=e_x, axes[1]=e_y, axes[2]=e_z of the local frame expressed in
// basic Cartesian).  For CORD2x, the defining points A, B, C must be
// transformed to basic before building axes.
//
// Transform convention (matches Nastran):
//   p_basic = origin + R * p_local_cart
// where p_local_cart is the Cartesian equivalent of the local coordinates
// (for cylindrical/spherical, the angular coords are converted to Cartesian
// in the local frame first).

#include "core/types.hpp"
#include <array>

namespace vibestran {

enum class CoordType {
    Rectangular,  // CORD2R / CORD1R
    Cylindrical,  // CORD2C / CORD1C
    Spherical,    // CORD2S
};

/// 3×3 rotation matrix (row-major): rot[row][col]
struct Mat3 {
    double m[3][3]{};
    double& operator()(int r, int c)       { return m[r][c]; }
    double  operator()(int r, int c) const { return m[r][c]; }
    static Mat3 identity() {
        Mat3 M;
        M(0,0) = M(1,1) = M(2,2) = 1.0;
        return M;
    }
};

struct CoordSys {
    CoordId  id{0};
    CoordType type{CoordType::Rectangular};
    CoordId  rid{0};  // reference coordinate system (used during parsing only)

    Vec3 origin{0, 0, 0};  // origin in basic Cartesian

    // Orthonormal column vectors: axes[0]=e_x, axes[1]=e_y, axes[2]=e_z
    // Each expressed as a Vec3 in basic Cartesian.
    std::array<Vec3, 3> axes{Vec3{1,0,0}, Vec3{0,1,0}, Vec3{0,0,1}};

    // For CORD1x: node IDs used as defining points (resolved post-parse).
    int def_node_a{0}, def_node_b{0}, def_node_c{0};
    bool is_cord1{false};  // true → defining points are node IDs

    // Defining points in the RID coordinate system (before resolution),
    // or in basic after resolution.
    Vec3 pt_a{0,0,0};  // origin  (A)
    Vec3 pt_b{0,0,0};  // point on Z axis (B)
    Vec3 pt_c{0,0,0};  // point in XZ plane (C)
};

/// Build a CoordSys from three defining points already expressed in basic
/// Cartesian: A = origin, B = point on Z axis, C = point in XZ plane.
/// Sets origin and axes; leaves id and type unchanged.
void build_axes(CoordSys& cs, const Vec3& a_basic, const Vec3& b_basic, const Vec3& c_basic);

/// Transform a position from the local coordinates of cs to basic Cartesian.
/// For rectangular: p_basic = origin + R * p_local (p_local is Cartesian).
/// For cylindrical: p_local = (r, theta_deg, z).
/// For spherical:   p_local = (rho, theta_deg, phi_deg).
Vec3 to_basic(const CoordSys& cs, const Vec3& p_local);

/// Compute the 3×3 rotation matrix T3 that maps a vector expressed in the
/// local frame at basic position basic_pos to basic Cartesian.
/// For rectangular, basic_pos is ignored (T3 = R = axes matrix).
/// For cylindrical/spherical, T3 is position-dependent.
///
/// Convention: v_basic = T3 * v_local
Mat3 rotation_matrix(const CoordSys& cs, const Vec3& basic_pos);

/// Apply a Mat3 rotation: result[i] = sum_j M[i][j] * v[j]
Vec3 apply_rotation(const Mat3& M, const Vec3& v);

} // namespace vibestran
