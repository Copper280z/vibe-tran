// src/core/coord_sys.cpp
// Coordinate system transform implementations.

#include "core/coord_sys.hpp"
#include <cmath>
#include <stdexcept>

namespace vibestran {

void build_axes(CoordSys& cs, const Vec3& a_basic, const Vec3& b_basic, const Vec3& c_basic) {
    // Z axis = (B - A) normalized
    Vec3 ez = (b_basic - a_basic).normalized();
    // Temporary X direction = (C - A)
    Vec3 ac = (c_basic - a_basic).normalized();
    // Y axis = Z × (C-A) direction, then re-normalize
    Vec3 ey_tmp = ez.cross(ac);
    double ey_len = ey_tmp.norm();
    if (ey_len < 1e-15)
        throw std::runtime_error("Coordinate system defining points are collinear");
    Vec3 ey = Vec3{ey_tmp.x / ey_len, ey_tmp.y / ey_len, ey_tmp.z / ey_len};
    // X axis = Y × Z (completes right-handed system)
    Vec3 ex = ey.cross(ez);
    double ex_len = ex.norm();
    ex = Vec3{ex.x / ex_len, ex.y / ex_len, ex.z / ex_len};

    cs.origin = a_basic;
    cs.axes[0] = ex;
    cs.axes[1] = ey;
    cs.axes[2] = ez;
}

Vec3 to_basic(const CoordSys& cs, const Vec3& p_local) {
    // Convert p_local to local Cartesian (x,y,z) in the local frame
    Vec3 p_cart;
    switch (cs.type) {
    case CoordType::Rectangular:
        p_cart = p_local;
        break;
    case CoordType::Cylindrical: {
        // p_local = (r, theta_deg, z)  — Nastran uses degrees
        double r     = p_local.x;
        double theta = p_local.y * (std::numbers::pi / 180.0);
        double z     = p_local.z;
        p_cart = Vec3{r * std::cos(theta), r * std::sin(theta), z};
        break;
    }
    case CoordType::Spherical: {
        // p_local = (rho, theta_deg, phi_deg)
        // Nastran spherical: theta=azimuth from X, phi=polar from Z
        double rho   = p_local.x;
        double theta = p_local.y * (std::numbers::pi / 180.0);
        double phi   = p_local.z * (std::numbers::pi / 180.0);
        p_cart = Vec3{
            rho * std::sin(phi) * std::cos(theta),
            rho * std::sin(phi) * std::sin(theta),
            rho * std::cos(phi)
        };
        break;
    }
    }

    // p_basic = origin + R * p_cart  where R columns = axes
    const auto& ax = cs.axes;
    return Vec3{
        cs.origin.x + ax[0].x * p_cart.x + ax[1].x * p_cart.y + ax[2].x * p_cart.z,
        cs.origin.y + ax[0].y * p_cart.x + ax[1].y * p_cart.y + ax[2].y * p_cart.z,
        cs.origin.z + ax[0].z * p_cart.x + ax[1].z * p_cart.y + ax[2].z * p_cart.z
    };
}

Mat3 rotation_matrix(const CoordSys& cs, const Vec3& basic_pos) {
    // R_frame: columns are cs.axes[0], cs.axes[1], cs.axes[2]
    // R_frame(i,j) = cs.axes[j].component_i
    auto& ax = cs.axes;

    switch (cs.type) {
    case CoordType::Rectangular: {
        // Constant rotation matrix: columns = local axes in basic
        Mat3 R;
        R(0,0) = ax[0].x; R(0,1) = ax[1].x; R(0,2) = ax[2].x;
        R(1,0) = ax[0].y; R(1,1) = ax[1].y; R(1,2) = ax[2].y;
        R(2,0) = ax[0].z; R(2,1) = ax[1].z; R(2,2) = ax[2].z;
        return R;
    }
    case CoordType::Cylindrical: {
        // Transform basic_pos to local Cartesian
        // p_local_cart = R^T * (basic_pos - origin)
        Vec3 dp{basic_pos.x - cs.origin.x,
                basic_pos.y - cs.origin.y,
                basic_pos.z - cs.origin.z};
        // local x = dot(ax[0], dp), local y = dot(ax[1], dp), local z = dot(ax[2], dp)
        double lx = ax[0].x*dp.x + ax[0].y*dp.y + ax[0].z*dp.z;
        double ly = ax[1].x*dp.x + ax[1].y*dp.y + ax[1].z*dp.z;
        double theta = std::atan2(ly, lx);
        double ct = std::cos(theta), st = std::sin(theta);
        // Local rotation due to cylindrical coords:
        // e_r = cos(θ)e_x_local + sin(θ)e_y_local
        // e_θ = -sin(θ)e_x_local + cos(θ)e_y_local
        // e_z = e_z_local
        // T_local = [[ct, -st, 0],[st, ct, 0],[0, 0, 1]]
        // T3 = R_frame * T_local  (where R_frame has columns = ax[0], ax[1], ax[2])
        // T3(i,j) = sum_k R_frame(i,k) * T_local(k,j)
        // R_frame: R_frame(i,k) = ax[k].component_i
        Mat3 T;
        // Column 0 (e_r in basic): R_frame * [ct, st, 0]^T
        for (int i = 0; i < 3; ++i) {
            double ri0 = (i==0)?ax[0].x:(i==1)?ax[0].y:ax[0].z;
            double ri1 = (i==0)?ax[1].x:(i==1)?ax[1].y:ax[1].z;
            double ri2 = (i==0)?ax[2].x:(i==1)?ax[2].y:ax[2].z;
            T(i, 0) = ri0 * ct + ri1 * st;   // e_r
            T(i, 1) = ri0 * (-st) + ri1 * ct; // e_θ
            T(i, 2) = ri2;                     // e_z
        }
        return T;
    }
    case CoordType::Spherical: {
        Vec3 dp{basic_pos.x - cs.origin.x,
                basic_pos.y - cs.origin.y,
                basic_pos.z - cs.origin.z};
        double lx = ax[0].x*dp.x + ax[0].y*dp.y + ax[0].z*dp.z;
        double ly = ax[1].x*dp.x + ax[1].y*dp.y + ax[1].z*dp.z;
        double lz = ax[2].x*dp.x + ax[2].y*dp.y + ax[2].z*dp.z;
        double theta = std::atan2(ly, lx);
        double rho_xy = std::sqrt(lx*lx + ly*ly);
        double phi = std::atan2(rho_xy, lz);  // polar from Z
        double ct = std::cos(theta), st = std::sin(theta);
        double cp = std::cos(phi),   sp = std::sin(phi);
        // Spherical local basis in local Cartesian frame:
        // e_rho = [sp*ct, sp*st, cp]
        // e_phi = [cp*ct, cp*st, -sp]
        // e_theta = [-st, ct, 0]
        // T_local has these as columns
        // T3 = R_frame * T_local
        Mat3 T;
        for (int i = 0; i < 3; ++i) {
            double ri0 = (i==0)?ax[0].x:(i==1)?ax[0].y:ax[0].z;
            double ri1 = (i==0)?ax[1].x:(i==1)?ax[1].y:ax[1].z;
            double ri2 = (i==0)?ax[2].x:(i==1)?ax[2].y:ax[2].z;
            T(i, 0) = ri0*(sp*ct) + ri1*(sp*st) + ri2*cp;    // e_rho
            T(i, 1) = ri0*(cp*ct) + ri1*(cp*st) + ri2*(-sp); // e_phi
            T(i, 2) = ri0*(-st)   + ri1*(ct);                 // e_theta
        }
        return T;
    }
    }
    return Mat3::identity(); // unreachable
}

// cppcheck-suppress unusedFunction -- called from linear_static.cpp
Vec3 apply_rotation(const Mat3& M, const Vec3& v) {
    return Vec3{
        M(0,0)*v.x + M(0,1)*v.y + M(0,2)*v.z,
        M(1,0)*v.x + M(1,1)*v.y + M(1,2)*v.z,
        M(2,0)*v.x + M(2,1)*v.y + M(2,2)*v.z
    };
}

} // namespace vibestran
