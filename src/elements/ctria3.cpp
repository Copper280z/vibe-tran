// src/elements/ctria3.cpp
// CTRIA3: 3-node constant-strain triangle shell element.
// Membrane: CST (plane-stress), closed-form integration
// Bending:  DKT (Discrete Kirchhoff Triangle) — simplified implementation
//           using constant curvature approximation for this linear element.

#include "elements/ctria3.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <format>

namespace vibestran {

CTria3::CTria3(ElementId eid, PropertyId pid,
               std::array<NodeId, 3> node_ids,
               const Model& model)
    : eid_(eid), pid_(pid), nodes_(node_ids), model_(model) {}

const PShell& CTria3::pshell() const {
    const auto& prop = model_.property(pid_);
    if (!std::holds_alternative<PShell>(prop))
        throw SolverError(std::format("CTRIA3 {}: property {} is not PSHELL", eid_.value, pid_.value));
    return std::get<PShell>(prop);
}

const Mat1& CTria3::material() const {
    return model_.material(pshell().mid1);
}

double CTria3::thickness() const { return pshell().t; }

std::array<Vec3, 3> CTria3::node_coords() const {
    return {model_.node(nodes_[0]).position,
            model_.node(nodes_[1]).position,
            model_.node(nodes_[2]).position};
}

Eigen::Matrix3d CTria3::plane_stress_D() const {
    const Mat1& m = material();
    double c = m.E / (1.0 - m.nu * m.nu);
    Eigen::Matrix3d D;
    D << c,      c*m.nu, 0,
         c*m.nu, c,      0,
         0,      0,      c*(1.0-m.nu)/2.0;
    return D;
}

Eigen::MatrixXd CTria3::membrane_stiffness() const {
    auto c = node_coords();
    double x1 = c[0].x, y1 = c[0].y;
    double x2 = c[1].x, y2 = c[1].y;
    double x3 = c[2].x, y3 = c[2].y;

    // Area via cross product
    double A2 = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1);
    if (std::abs(A2) < 1e-15)
        throw SolverError(std::format("CTRIA3 {}: zero area", eid_.value));
    double A = 0.5 * A2;

    // Constant strain-displacement matrix B_m [3x6]
    // Strains: εxx, εyy, γxy
    // DOFs per node: u, v (2 per node, 6 total for membrane)
    double b1 = y2 - y3, b2 = y3 - y1, b3 = y1 - y2;
    double d1 = x3 - x2, d2 = x1 - x3, d3 = x2 - x1;

    Eigen::MatrixXd Bm(3, 6);
    Bm.setZero();
    Bm(0,0)=b1; Bm(0,2)=b2; Bm(0,4)=b3;
    Bm(1,1)=d1; Bm(1,3)=d2; Bm(1,5)=d3;
    Bm(2,0)=d1; Bm(2,1)=b1;
    Bm(2,2)=d2; Bm(2,3)=b2;
    Bm(2,4)=d3; Bm(2,5)=b3;
    Bm /= (2.0 * A);

    double t = thickness();
    Eigen::Matrix3d D = plane_stress_D();
    return t * A * Bm.transpose() * D * Bm;
}

Eigen::MatrixXd CTria3::bending_stiffness() const {
    // Simplified DKT: constant curvature triangle bending element.
    // Uses the standard thin-plate stiffness formulation.
    // For each node: DOFs are w (out-of-plane), θx, θy
    // Total bending DOFs: 9 (3 nodes × 3)
    //
    // Curvature-displacement: κ = B_b * u_b
    // B_b is constant (same as CST approach for bending)

    auto c = node_coords();
    double x1=c[0].x, y1=c[0].y;
    double x2=c[1].x, y2=c[1].y;
    double x3=c[2].x, y3=c[2].y;

    double A2 = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1);
    double A  = 0.5 * std::abs(A2);

    // Side vectors
    double b1 = y2-y3, b2 = y3-y1, b3 = y1-y2;
    double d1 = x3-x2, d2 = x1-x3, d3 = x2-x1;

    // Bending B matrix [3x9]: curvatures from [w1,θx1,θy1, w2,θx2,θy2, w3,θx3,θy3]
    // Using standard DKT approximation:
    // κxx = dθx/dx, κyy = dθy/dy, κxy = dθx/dy + dθy/dx
    Eigen::MatrixXd Bb(3, 9);
    Bb.setZero();

    double inv2A = 1.0 / (2.0 * A);

    // Node 1: cols 0,1,2 → w1,θx1,θy1
    // Node 2: cols 3,4,5 → w2,θx2,θy2
    // Node 3: cols 6,7,8 → w3,θx3,θy3
    // dθx/dx contribution (κxx)
    Bb(0,1) = b1 * inv2A;  Bb(0,4) = b2 * inv2A;  Bb(0,7) = b3 * inv2A;
    // dθy/dy contribution (κyy)
    Bb(1,2) = d1 * inv2A;  Bb(1,5) = d2 * inv2A;  Bb(1,8) = d3 * inv2A;
    // dθx/dy + dθy/dx (κxy)
    Bb(2,1) = d1 * inv2A;  Bb(2,4) = d2 * inv2A;  Bb(2,7) = d3 * inv2A;
    Bb(2,2) = b1 * inv2A;  Bb(2,5) = b2 * inv2A;  Bb(2,8) = b3 * inv2A;

    double t = thickness();
    double Db_scale = t*t*t / 12.0;
    Eigen::Matrix3d Db = Db_scale * plane_stress_D();
    return A * Bb.transpose() * Db * Bb;
}

LocalKe CTria3::stiffness_matrix() const {
    LocalKe Ke = LocalKe::Zero(NUM_DOFS, NUM_DOFS);

    // Membrane stiffness (DOF layout per node: u=6i, v=6i+1)
    Eigen::MatrixXd Km = membrane_stiffness(); // 6x6
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            Ke(6*i+0, 6*j+0) += Km(2*i+0, 2*j+0);
            Ke(6*i+0, 6*j+1) += Km(2*i+0, 2*j+1);
            Ke(6*i+1, 6*j+0) += Km(2*i+1, 2*j+0);
            Ke(6*i+1, 6*j+1) += Km(2*i+1, 2*j+1);
        }

    // Bending stiffness (DOF layout per node: w=6i+2, θx=6i+3, θy=6i+4)
    // Local bending DOFs: w=3i+0, θx=3i+1, θy=3i+2
    Eigen::MatrixXd Kb = bending_stiffness(); // 9x9
    const int bend_map[3] = {2, 3, 4}; // local bend dof → offset in 6-DOF node block
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    Ke(6*i+bend_map[a], 6*j+bend_map[b]) += Kb(3*i+a, 3*j+b);

    // Drilling stiffness stabilization for θz DOF
    double drill = 1e-6 * Ke.diagonal().maxCoeff();
    if (drill < 1e-10) drill = 1.0;
    for (int i = 0; i < 3; ++i)
        Ke(6*i+5, 6*i+5) += drill;

    return Ke;
}

LocalKe CTria3::mass_matrix() const {
    LocalKe Me = LocalKe::Zero(NUM_DOFS, NUM_DOFS);
    const Mat1& mat = material();
    const double rho = mat.rho;
    if (rho == 0.0) return Me;
    const double t = thickness();

    // Closed-form consistent mass matrix for CST triangle.
    // Nodal mass integrals: integral(Li^2 dA) = A/6, integral(Li*Lj dA) = A/12 (i≠j)
    // Compute actual 3D area via cross product (handles arbitrary orientation).
    auto c = node_coords();
    Vec3 e1{c[1].x-c[0].x, c[1].y-c[0].y, c[1].z-c[0].z};
    Vec3 e2{c[2].x-c[0].x, c[2].y-c[0].y, c[2].z-c[0].z};
    double A = 0.5 * e1.cross(e2).norm();

    double m_diag = rho * t * A / 6.0;
    double m_off  = rho * t * A / 12.0;

    double r_diag = rho * t * t * t / 12.0 * A / 6.0;
    double r_off  = rho * t * t * t / 12.0 * A / 12.0;
    double drill  = rho * t * t * t / 1200.0 * A / 6.0; // small drilling mass

    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            double m_trans = (a == b) ? m_diag : m_off;
            double m_rot   = (a == b) ? r_diag : r_off;
            // Translational (T1,T2,T3): indices 6a+0,1,2
            for (int d = 0; d < 3; ++d)
                Me(6*a+d, 6*b+d) = m_trans;
            // Rotational bending (R1,R2): indices 6a+3,4
            for (int d = 0; d < 2; ++d)
                Me(6*a+3+d, 6*b+3+d) = m_rot;
        }
        // Drilling (R3): diagonal only
        Me(6*a+5, 6*a+5) = drill;
    }
    return Me;
}

LocalFe CTria3::thermal_load(std::span<const double> temperatures, double t_ref) const {
    LocalFe fe = LocalFe::Zero(NUM_DOFS);
    const double alpha = material().A;
    if (alpha == 0.0) return fe;

    auto c = node_coords();
    double x1=c[0].x, y1=c[0].y;
    double x2=c[1].x, y2=c[1].y;
    double x3=c[2].x, y3=c[2].y;

    double A2 = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1);
    // Use signed area for consistent B matrix sign (matches membrane_stiffness)
    double A  = 0.5 * A2;
    if (std::abs(A2) < 1e-15) return fe;

    // Average temperature over element
    double T_avg = (temperatures[0] + temperatures[1] + temperatures[2]) / 3.0;
    double dT = T_avg - t_ref;
    if (std::abs(dT) < 1e-15) return fe;

    double b1 = y2-y3, b2 = y3-y1, b3 = y1-y2;
    double d1 = x3-x2, d2 = x1-x3, d3 = x2-x1;

    Eigen::MatrixXd Bm(3, 6);
    Bm.setZero();
    Bm(0,0)=b1; Bm(0,2)=b2; Bm(0,4)=b3;
    Bm(1,1)=d1; Bm(1,3)=d2; Bm(1,5)=d3;
    Bm(2,0)=d1; Bm(2,1)=b1;
    Bm(2,2)=d2; Bm(2,3)=b2;
    Bm(2,4)=d3; Bm(2,5)=b3;
    Bm /= (2.0 * A);  // same sign as membrane_stiffness()

    Eigen::Matrix3d D = plane_stress_D();
    double t = thickness();
    Eigen::Vector3d eps_th(alpha*dT, alpha*dT, 0.0);
    // Integration weight = |A| (physical area, always positive)
    Eigen::VectorXd fe_mem = t * std::abs(A) * Bm.transpose() * D * eps_th;

    for (int n = 0; n < 3; ++n) {
        fe(6*n+0) += fe_mem(2*n+0);
        fe(6*n+1) += fe_mem(2*n+1);
    }
    return fe;
}

std::vector<EqIndex> CTria3::global_dof_indices(const DofMap& dof_map) const {
    std::vector<EqIndex> result;
    result.reserve(NUM_DOFS);
    for (NodeId nid : nodes_) {
        const auto& blk = dof_map.block(nid);
        for (int d = 0; d < 6; ++d)
            result.push_back(blk.eq[d]);
    }
    return result;
}

} // namespace vibestran
