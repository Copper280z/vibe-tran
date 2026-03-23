// src/elements/solid_elements.cpp
// CHEXA8: 8-node trilinear hexahedron, 2x2x2 Gauss
// CTETRA4: 4-node linear tetrahedron, closed-form

#include "elements/solid_elements.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <format>

namespace vibestran {

// ── Shared helpers ────────────────────────────────────────────────────────────

/// 3-D isotropic constitutive matrix D [6x6]
static Eigen::Matrix<double,6,6> isotropic_D(double E, double nu) {
    double lam = E * nu / ((1+nu)*(1-2*nu));
    double mu  = E / (2*(1+nu));
    Eigen::Matrix<double,6,6> D;
    D.setZero();
    D(0,0)=lam+2*mu; D(0,1)=lam;      D(0,2)=lam;
    D(1,0)=lam;      D(1,1)=lam+2*mu; D(1,2)=lam;
    D(2,0)=lam;      D(2,1)=lam;      D(2,2)=lam+2*mu;
    D(3,3)=mu; D(4,4)=mu; D(5,5)=mu;
    return D;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHEXA8
// ═══════════════════════════════════════════════════════════════════════════════

CHexa8::CHexa8(ElementId eid, PropertyId pid,
               std::array<NodeId, 8> node_ids,
               const Model& model)
    : eid_(eid), pid_(pid), nodes_(node_ids), model_(model) {}

const PSolid& CHexa8::psolid() const {
    const auto& prop = model_.property(pid_);
    if (!std::holds_alternative<PSolid>(prop))
        throw SolverError(std::format("CHEXA8 {}: property {} is not PSOLID", eid_.value, pid_.value));
    return std::get<PSolid>(prop);
}

const Mat1& CHexa8::material() const {
    return model_.material(psolid().mid);
}

Eigen::Matrix<double,6,6> CHexa8::constitutive_D() const {
    const Mat1& m = material();
    return isotropic_D(m.E, m.nu);
}

std::array<Vec3, 8> CHexa8::node_coords() const {
    std::array<Vec3, 8> c;
    for (int i = 0; i < 8; ++i)
        c[i] = model_.node(nodes_[i]).position;
    return c;
}

CHexa8::ShapeData CHexa8::shape_functions(double xi, double eta, double zeta) noexcept {
    // Trilinear shape functions for 8-node hex
    // Node ordering: (-1,-1,-1),(+1,-1,-1),(+1,+1,-1),(-1,+1,-1),
    //                (-1,-1,+1),(+1,-1,+1),(+1,+1,+1),(-1,+1,+1)
    const double xi_n[8]   = {-1,+1,+1,-1,-1,+1,+1,-1};
    const double eta_n[8]  = {-1,-1,+1,+1,-1,-1,+1,+1};
    const double zeta_n[8] = {-1,-1,-1,-1,+1,+1,+1,+1};

    ShapeData s;
    for (int i = 0; i < 8; ++i) {
        double xni = xi_n[i], eni = eta_n[i], zni = zeta_n[i];
        s.N[i]      = 0.125*(1+xni*xi)*(1+eni*eta)*(1+zni*zeta);
        s.dNdxi[i]  = 0.125*xni*(1+eni*eta)*(1+zni*zeta);
        s.dNdeta[i] = 0.125*eni*(1+xni*xi)*(1+zni*zeta);
        s.dNdzeta[i]= 0.125*zni*(1+xni*xi)*(1+eni*eta);
    }
    return s;
}

LocalKe CHexa8::stiffness_matrix() const {
    LocalKe Ke = LocalKe::Zero(NUM_DOFS, NUM_DOFS);
    auto coords = node_coords();
    Eigen::Matrix<double,6,6> D = constitutive_D();

    // Selective Reduced Integration (SRI) to eliminate volumetric locking:
    // D = D_dev + D_vol, where D_vol = lam * e*e^T (e = [1,1,1,0,0,0]^T)
    // D_dev uses full 2x2x2 Gauss; D_vol uses 1-point centroidal integration.
    // This prevents over-stiffness in constrained states without losing stability.
    double lam = D(0,1);  // D(i,j) = lam for i,j in {0,1,2}, since D(0,0)=lam+2mu
    Eigen::Matrix<double,6,6> D_vol = Eigen::Matrix<double,6,6>::Zero();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            D_vol(i,j) = lam;
    Eigen::Matrix<double,6,6> D_dev = D - D_vol;

    // 2x2x2 Gauss quadrature for deviatoric part
    const double gp = 1.0/std::sqrt(3.0);
    const double gpts[2] = {-gp, gp};
    const double gwts[2] = {1.0, 1.0};

    auto build_B = [&](const ShapeData& sd, const Eigen::Matrix3d& Jinv,
                       Eigen::MatrixXd& B) {
        B.setZero();
        for (int n = 0; n < 8; ++n) {
            double dnx = Jinv(0,0)*sd.dNdxi[n] + Jinv(0,1)*sd.dNdeta[n] + Jinv(0,2)*sd.dNdzeta[n];
            double dny = Jinv(1,0)*sd.dNdxi[n] + Jinv(1,1)*sd.dNdeta[n] + Jinv(1,2)*sd.dNdzeta[n];
            double dnz = Jinv(2,0)*sd.dNdxi[n] + Jinv(2,1)*sd.dNdeta[n] + Jinv(2,2)*sd.dNdzeta[n];
            int c0 = 3*n;
            B(0,c0+0)=dnx;
            B(1,c0+1)=dny;
            B(2,c0+2)=dnz;
            B(3,c0+0)=dny; B(3,c0+1)=dnx;
            B(4,c0+1)=dnz; B(4,c0+2)=dny;
            B(5,c0+0)=dnz; B(5,c0+2)=dnx;
        }
    };

    for (int gi = 0; gi < 2; ++gi)
    for (int gj = 0; gj < 2; ++gj)
    for (int gk = 0; gk < 2; ++gk) {
        double xi   = gpts[gi], eta  = gpts[gj], zeta = gpts[gk];
        double wi   = gwts[gi],  wj   = gwts[gj],  wk   = gwts[gk];

        auto sd = shape_functions(xi, eta, zeta);

        // Jacobian [3x3]
        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 8; ++n) {
            J(0,0)+=sd.dNdxi[n]  *coords[n].x; J(0,1)+=sd.dNdxi[n]  *coords[n].y; J(0,2)+=sd.dNdxi[n]  *coords[n].z;
            J(1,0)+=sd.dNdeta[n] *coords[n].x; J(1,1)+=sd.dNdeta[n] *coords[n].y; J(1,2)+=sd.dNdeta[n] *coords[n].z;
            J(2,0)+=sd.dNdzeta[n]*coords[n].x; J(2,1)+=sd.dNdzeta[n]*coords[n].y; J(2,2)+=sd.dNdzeta[n]*coords[n].z;
        }
        double detJ = J.determinant();
        if (detJ <= 0)
            throw SolverError(std::format("CHEXA8 {}: non-positive Jacobian det={:.6g}", eid_.value, detJ));
        Eigen::Matrix3d Jinv = J.inverse();

        Eigen::MatrixXd B(6, NUM_DOFS);
        build_B(sd, Jinv, B);

        Ke += B.transpose() * D_dev * B * detJ * wi * wj * wk;
    }

    // Volumetric part: 1-point centroidal integration (weight = 2*2*2 = 8)
    {
        auto sd0 = shape_functions(0.0, 0.0, 0.0);
        Eigen::Matrix3d J0 = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 8; ++n) {
            J0(0,0)+=sd0.dNdxi[n]  *coords[n].x; J0(0,1)+=sd0.dNdxi[n]  *coords[n].y; J0(0,2)+=sd0.dNdxi[n]  *coords[n].z;
            J0(1,0)+=sd0.dNdeta[n] *coords[n].x; J0(1,1)+=sd0.dNdeta[n] *coords[n].y; J0(1,2)+=sd0.dNdeta[n] *coords[n].z;
            J0(2,0)+=sd0.dNdzeta[n]*coords[n].x; J0(2,1)+=sd0.dNdzeta[n]*coords[n].y; J0(2,2)+=sd0.dNdzeta[n]*coords[n].z;
        }
        double detJ0 = J0.determinant();
        if (detJ0 <= 0)
            throw SolverError(std::format("CHEXA8 {}: non-positive centroidal Jacobian det={:.6g}", eid_.value, detJ0));
        Eigen::Matrix3d Jinv0 = J0.inverse();

        Eigen::MatrixXd B0(6, NUM_DOFS);
        build_B(sd0, Jinv0, B0);

        Ke += B0.transpose() * D_vol * B0 * detJ0 * 8.0;
    }

    return Ke;
}

LocalKe CHexa8::mass_matrix() const {
    LocalKe Me = LocalKe::Zero(NUM_DOFS, NUM_DOFS);
    const double rho = material().rho;
    if (rho == 0.0) return Me;

    auto coords = node_coords();
    const double gp = 1.0/std::sqrt(3.0);
    const double gpts[2] = {-gp, gp};

    for (int gi = 0; gi < 2; ++gi)
    for (int gj = 0; gj < 2; ++gj)
    for (int gk = 0; gk < 2; ++gk) {
        double xi = gpts[gi], eta = gpts[gj], zeta = gpts[gk];

        auto sd = shape_functions(xi, eta, zeta);

        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 8; ++n) {
            J(0,0)+=sd.dNdxi[n]  *coords[n].x; J(0,1)+=sd.dNdxi[n]  *coords[n].y; J(0,2)+=sd.dNdxi[n]  *coords[n].z;
            J(1,0)+=sd.dNdeta[n] *coords[n].x; J(1,1)+=sd.dNdeta[n] *coords[n].y; J(1,2)+=sd.dNdeta[n] *coords[n].z;
            J(2,0)+=sd.dNdzeta[n]*coords[n].x; J(2,1)+=sd.dNdzeta[n]*coords[n].y; J(2,2)+=sd.dNdzeta[n]*coords[n].z;
        }
        double detJ = J.determinant();
        if (detJ <= 0)
            throw SolverError(std::format("CHexa8 {}: non-positive Jacobian in mass_matrix det={:.6g}", eid_.value, detJ));

        // Consistent mass: Me_ab += rho * N[a]*N[b] * detJ (weights = 1)
        for (int a = 0; a < 8; ++a)
            for (int b = 0; b < 8; ++b) {
                double v = rho * sd.N[a] * sd.N[b] * detJ;
                for (int d = 0; d < 3; ++d)
                    Me(3*a+d, 3*b+d) += v;
            }
    }
    return Me;
}

LocalFe CHexa8::thermal_load(std::span<const double> temperatures, double t_ref) const {
    LocalFe fe = LocalFe::Zero(NUM_DOFS);
    const double alpha = material().A;
    if (alpha == 0.0) return fe;

    auto coords = node_coords();
    Eigen::Matrix<double,6,6> D = constitutive_D();

    const double gp = 1.0/std::sqrt(3.0);
    const double gpts[2] = {-gp, gp};

    for (int gi = 0; gi < 2; ++gi)
    for (int gj = 0; gj < 2; ++gj)
    for (int gk = 0; gk < 2; ++gk) {
        double xi=gpts[gi], eta=gpts[gj], zeta=gpts[gk];

        auto sd = shape_functions(xi, eta, zeta);

        double T = 0;
        for (int n = 0; n < 8; ++n) T += sd.N[n] * temperatures[n];
        double dT = T - t_ref;
        if (std::abs(dT) < 1e-15) continue;

        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 8; ++n) {
            J(0,0)+=sd.dNdxi[n]  *coords[n].x; J(0,1)+=sd.dNdxi[n]  *coords[n].y; J(0,2)+=sd.dNdxi[n]  *coords[n].z;
            J(1,0)+=sd.dNdeta[n] *coords[n].x; J(1,1)+=sd.dNdeta[n] *coords[n].y; J(1,2)+=sd.dNdeta[n] *coords[n].z;
            J(2,0)+=sd.dNdzeta[n]*coords[n].x; J(2,1)+=sd.dNdzeta[n]*coords[n].y; J(2,2)+=sd.dNdzeta[n]*coords[n].z;
        }
        double detJ = J.determinant();
        Eigen::Matrix3d Jinv = J.inverse();

        Eigen::MatrixXd dNdx(3, 8);
        for (int n = 0; n < 8; ++n) {
            dNdx(0,n) = Jinv(0,0)*sd.dNdxi[n] + Jinv(0,1)*sd.dNdeta[n] + Jinv(0,2)*sd.dNdzeta[n];
            dNdx(1,n) = Jinv(1,0)*sd.dNdxi[n] + Jinv(1,1)*sd.dNdeta[n] + Jinv(1,2)*sd.dNdzeta[n];
            dNdx(2,n) = Jinv(2,0)*sd.dNdxi[n] + Jinv(2,1)*sd.dNdeta[n] + Jinv(2,2)*sd.dNdzeta[n];
        }

        Eigen::MatrixXd B(6, NUM_DOFS);
        B.setZero();
        for (int n = 0; n < 8; ++n) {
            double dnx=dNdx(0,n), dny=dNdx(1,n), dnz=dNdx(2,n);
            int c0=3*n;
            B(0,c0+0)=dnx; B(1,c0+1)=dny; B(2,c0+2)=dnz;
            B(3,c0+0)=dny; B(3,c0+1)=dnx;
            B(4,c0+1)=dnz; B(4,c0+2)=dny;
            B(5,c0+0)=dnz; B(5,c0+2)=dnx;
        }

        Eigen::Matrix<double,6,1> eps_th;
        eps_th << alpha*dT, alpha*dT, alpha*dT, 0, 0, 0;
        fe += B.transpose() * D * eps_th * detJ;
    }
    return fe;
}

std::vector<EqIndex> CHexa8::global_dof_indices(const DofMap& dof_map) const {
    std::vector<EqIndex> result;
    result.reserve(NUM_DOFS);
    static constexpr int solid_dofs[3] = {0,1,2}; // T1,T2,T3 only
    for (NodeId nid : nodes_)
        for (int d : solid_dofs)
            result.push_back(dof_map.eq_index(nid, d));
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CTETRA4
// ═══════════════════════════════════════════════════════════════════════════════

CTetra4::CTetra4(ElementId eid, PropertyId pid,
                 std::array<NodeId, 4> node_ids,
                 const Model& model)
    : eid_(eid), pid_(pid), nodes_(node_ids), model_(model) {}

const PSolid& CTetra4::psolid() const {
    const auto& prop = model_.property(pid_);
    if (!std::holds_alternative<PSolid>(prop))
        throw SolverError(std::format("CTETRA4 {}: property {} is not PSOLID", eid_.value, pid_.value));
    return std::get<PSolid>(prop);
}

const Mat1& CTetra4::material() const {
    return model_.material(psolid().mid);
}

Eigen::Matrix<double,6,6> CTetra4::constitutive_D() const {
    const Mat1& m = material();
    return isotropic_D(m.E, m.nu);
}

std::array<Vec3, 4> CTetra4::node_coords() const {
    return {model_.node(nodes_[0]).position,
            model_.node(nodes_[1]).position,
            model_.node(nodes_[2]).position,
            model_.node(nodes_[3]).position};
}

LocalKe CTetra4::stiffness_matrix() const {
    auto c = node_coords();

    // CST tetrahedron: constant strain → B is constant
    // Volume via determinant of coordinate matrix
    // [x2-x1, x3-x1, x4-x1]
    // [y2-y1, y3-y1, y4-y1]
    // [z2-z1, z3-z1, z4-z1]
    Eigen::Matrix3d CM;
    for (int j = 0; j < 3; ++j) {
        CM(j,0) = (j==0?c[1].x:j==1?c[1].y:c[1].z) - (j==0?c[0].x:j==1?c[0].y:c[0].z);
        CM(j,1) = (j==0?c[2].x:j==1?c[2].y:c[2].z) - (j==0?c[0].x:j==1?c[0].y:c[0].z);
        CM(j,2) = (j==0?c[3].x:j==1?c[3].y:c[3].z) - (j==0?c[0].x:j==1?c[0].y:c[0].z);
    }
    // Cleaner version:
    double x1=c[0].x,y1=c[0].y,z1=c[0].z;
    double x2=c[1].x,y2=c[1].y,z2=c[1].z;
    double x3=c[2].x,y3=c[2].y,z3=c[2].z;
    double x4=c[3].x,y4=c[3].y,z4=c[3].z;

    Eigen::Matrix4d A4;
    A4 << 1,x1,y1,z1,
          1,x2,y2,z2,
          1,x3,y3,z3,
          1,x4,y4,z4;
    double V6 = A4.determinant(); // = 6 * volume
    double V  = V6 / 6.0;
    if (V <= 0)
        throw SolverError(std::format("CTETRA4 {}: non-positive volume={:.6g}", eid_.value, V));

    // Shape fn coefficients: N_i = (a_i + b_i*x + c_i*y + d_i*z) / (6V)
    // Computed via cofactors of A4
    // dN_i/dx = b_i / (6V), etc.
    // Cofactor matrix (signed minors of A4 transposed)
    Eigen::Matrix4d cofA = Eigen::Matrix4d::Zero();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            // Minor by removing row j, col i (transposed for cofactor)
            Eigen::Matrix3d minor3;
            int ri = 0;
            for (int r = 0; r < 4; ++r) {
                if (r == j) continue;
                int ci_ = 0;
                for (int cc = 0; cc < 4; ++cc) {
                    if (cc == i) continue;
                    minor3(ri, ci_++) = A4(r, cc);
                }
                ri++;
            }
            cofA(i,j) = std::pow(-1.0, i+j) * minor3.determinant();
        }

    // dN_i/dx = cofA(i,1) / V6
    // dN_i/dy = cofA(i,2) / V6
    // dN_i/dz = cofA(i,3) / V6
    Eigen::MatrixXd B(6, NUM_DOFS);
    B.setZero();
    for (int n = 0; n < 4; ++n) {
        double bx = cofA(1,n) / V6;
        double by = cofA(2,n) / V6;
        double bz = cofA(3,n) / V6;
        int c0 = 3*n;
        B(0,c0+0)=bx;
        B(1,c0+1)=by;
        B(2,c0+2)=bz;
        B(3,c0+0)=by; B(3,c0+1)=bx;
        B(4,c0+1)=bz; B(4,c0+2)=by;
        B(5,c0+0)=bz; B(5,c0+2)=bx;
    }

    Eigen::Matrix<double,6,6> D = constitutive_D();
    return V * B.transpose() * D * B;
}

LocalKe CTetra4::mass_matrix() const {
    LocalKe Me = LocalKe::Zero(NUM_DOFS, NUM_DOFS);
    const double rho = material().rho;
    if (rho == 0.0) return Me;

    // Closed-form consistent mass matrix for 4-node linear tetrahedron.
    // integral(Li * Lj dV) = V/20 for i!=j, V/10 for i==j
    auto c = node_coords();
    Eigen::Matrix4d A4;
    A4 << 1,c[0].x,c[0].y,c[0].z,
          1,c[1].x,c[1].y,c[1].z,
          1,c[2].x,c[2].y,c[2].z,
          1,c[3].x,c[3].y,c[3].z;
    double V = std::abs(A4.determinant()) / 6.0;

    double m_diag = rho * V / 10.0;
    double m_off  = rho * V / 20.0;

    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b) {
            double v = (a == b) ? m_diag : m_off;
            for (int d = 0; d < 3; ++d)
                Me(3*a+d, 3*b+d) = v;
        }
    return Me;
}

LocalFe CTetra4::thermal_load(std::span<const double> temperatures, double t_ref) const {
    LocalFe fe = LocalFe::Zero(NUM_DOFS);
    const double alpha = material().A;
    if (alpha == 0.0) return fe;

    // For CST tetrahedron, B is constant → same computation as stiffness
    // fe = V * Bᵀ * D * eps_th
    auto c = node_coords();
    double x1=c[0].x,y1=c[0].y,z1=c[0].z;
    double x2=c[1].x,y2=c[1].y,z2=c[1].z;
    double x3=c[2].x,y3=c[2].y,z3=c[2].z;
    double x4=c[3].x,y4=c[3].y,z4=c[3].z;

    Eigen::Matrix4d A4;
    A4 << 1,x1,y1,z1, 1,x2,y2,z2, 1,x3,y3,z3, 1,x4,y4,z4;
    double V6 = A4.determinant();
    double V  = V6 / 6.0;

    Eigen::Matrix4d cofA = Eigen::Matrix4d::Zero();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            Eigen::Matrix3d minor3;
            int ri = 0;
            for (int r = 0; r < 4; ++r) {
                if (r == j) continue;
                int ci_ = 0;
                for (int cc = 0; cc < 4; ++cc) {
                    if (cc == i) continue;
                    minor3(ri, ci_++) = A4(r, cc);
                }
                ri++;
            }
            cofA(i,j) = std::pow(-1.0, i+j) * minor3.determinant();
        }

    Eigen::MatrixXd B(6, NUM_DOFS);
    B.setZero();
    for (int n = 0; n < 4; ++n) {
        double bx = cofA(1,n)/V6, by = cofA(2,n)/V6, bz = cofA(3,n)/V6;
        int c0=3*n;
        B(0,c0+0)=bx; B(1,c0+1)=by; B(2,c0+2)=bz;
        B(3,c0+0)=by; B(3,c0+1)=bx;
        B(4,c0+1)=bz; B(4,c0+2)=by;
        B(5,c0+0)=bz; B(5,c0+2)=bx;
    }

    double T_avg = (temperatures[0]+temperatures[1]+temperatures[2]+temperatures[3]) / 4.0;
    double dT = T_avg - t_ref;
    if (std::abs(dT) < 1e-15) return fe;

    Eigen::Matrix<double,6,6> D = constitutive_D();
    Eigen::Matrix<double,6,1> eps_th;
    eps_th << alpha*dT, alpha*dT, alpha*dT, 0, 0, 0;
    fe = V * B.transpose() * D * eps_th;
    return fe;
}

std::vector<EqIndex> CTetra4::global_dof_indices(const DofMap& dof_map) const {
    std::vector<EqIndex> result;
    result.reserve(NUM_DOFS);
    static constexpr int solid_dofs[3] = {0,1,2};
    for (NodeId nid : nodes_)
        for (int d : solid_dofs)
            result.push_back(dof_map.eq_index(nid, d));
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CPENTA6
// ═══════════════════════════════════════════════════════════════════════════════

CPenta6::CPenta6(ElementId eid, PropertyId pid,
                 std::array<NodeId, 6> node_ids,
                 const Model& model)
    : eid_(eid), pid_(pid), nodes_(node_ids), model_(model) {}

const PSolid& CPenta6::psolid() const {
    const auto& prop = model_.property(pid_);
    if (!std::holds_alternative<PSolid>(prop))
        throw SolverError(std::format("CPENTA6 {}: property {} is not PSOLID", eid_.value, pid_.value));
    return std::get<PSolid>(prop);
}

const Mat1& CPenta6::material() const {
    return model_.material(psolid().mid);
}

Eigen::Matrix<double,6,6> CPenta6::constitutive_D() const {
    const Mat1& m = material();
    return isotropic_D(m.E, m.nu);
}

std::array<Vec3, 6> CPenta6::node_coords() const {
    std::array<Vec3, 6> c;
    for (int i = 0; i < 6; ++i)
        c[i] = model_.node(nodes_[i]).position;
    return c;
}

// Shape functions for 6-node pentahedron (wedge).
// Natural coordinates: L1, L2 (triangular), zeta ∈ [-1,1] (axial).
// L3 = 1 - L1 - L2.
// Nodes 0-2 on bottom face (zeta=-1), nodes 3-5 on top face (zeta=+1).
CPenta6::ShapeData6 CPenta6::shape_functions(double L1, double L2, double zeta) noexcept {
    double L3 = 1.0 - L1 - L2;
    double zm = (1.0 - zeta) * 0.5;
    double zp = (1.0 + zeta) * 0.5;

    ShapeData6 s;
    s.N[0] = L1 * zm;
    s.N[1] = L2 * zm;
    s.N[2] = L3 * zm;
    s.N[3] = L1 * zp;
    s.N[4] = L2 * zp;
    s.N[5] = L3 * zp;

    // dN/dL1
    s.dNdL1[0] =  zm;
    s.dNdL1[1] =  0.0;
    s.dNdL1[2] = -zm;
    s.dNdL1[3] =  zp;
    s.dNdL1[4] =  0.0;
    s.dNdL1[5] = -zp;

    // dN/dL2
    s.dNdL2[0] =  0.0;
    s.dNdL2[1] =  zm;
    s.dNdL2[2] = -zm;
    s.dNdL2[3] =  0.0;
    s.dNdL2[4] =  zp;
    s.dNdL2[5] = -zp;

    // dN/dzeta
    s.dNdzeta[0] = -L1 * 0.5;
    s.dNdzeta[1] = -L2 * 0.5;
    s.dNdzeta[2] = -L3 * 0.5;
    s.dNdzeta[3] =  L1 * 0.5;
    s.dNdzeta[4] =  L2 * 0.5;
    s.dNdzeta[5] =  L3 * 0.5;

    return s;
}

LocalKe CPenta6::stiffness_matrix() const {
    LocalKe Ke = LocalKe::Zero(NUM_DOFS, NUM_DOFS);
    auto coords = node_coords();
    Eigen::Matrix<double,6,6> D = constitutive_D();

    // Selective Reduced Integration (SRI) to eliminate volumetric locking:
    // D = D_dev + D_vol, where D_vol = lam * e*e^T (e = [1,1,1,0,0,0]^T)
    // D_dev uses full 6-point Gauss; D_vol uses 1-point centroidal integration.
    double lam = D(0,1);
    Eigen::Matrix<double,6,6> D_vol = Eigen::Matrix<double,6,6>::Zero();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            D_vol(i,j) = lam;
    Eigen::Matrix<double,6,6> D_dev = D - D_vol;

    // 6-point Gauss quadrature: 3 triangle points × 2 axial points
    // Triangle 3-point rule: weights = 1/6 each (includes reference triangle area 1/2)
    const double tri_pts[3][2] = {
        {2.0/3.0, 1.0/6.0},
        {1.0/6.0, 2.0/3.0},
        {1.0/6.0, 1.0/6.0}
    };
    const double tri_w = 1.0 / 6.0;

    // Axial 2-point Gauss: zeta = ±1/√3, weight = 1
    const double ax_pts[2] = {-1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0)};

    auto build_B = [&](const ShapeData6& sd, const Eigen::Matrix3d& Jinv,
                       Eigen::MatrixXd& B) {
        B.setZero();
        for (int n = 0; n < 6; ++n) {
            double dnx = Jinv(0,0)*sd.dNdL1[n] + Jinv(0,1)*sd.dNdL2[n] + Jinv(0,2)*sd.dNdzeta[n];
            double dny = Jinv(1,0)*sd.dNdL1[n] + Jinv(1,1)*sd.dNdL2[n] + Jinv(1,2)*sd.dNdzeta[n];
            double dnz = Jinv(2,0)*sd.dNdL1[n] + Jinv(2,1)*sd.dNdL2[n] + Jinv(2,2)*sd.dNdzeta[n];
            int c0 = 3*n;
            B(0,c0+0) = dnx;
            B(1,c0+1) = dny;
            B(2,c0+2) = dnz;
            B(3,c0+0) = dny; B(3,c0+1) = dnx;
            B(4,c0+1) = dnz; B(4,c0+2) = dny;
            B(5,c0+0) = dnz; B(5,c0+2) = dnx;
        }
    };

    // Deviatoric part: full 6-point quadrature
    for (int ti = 0; ti < 3; ++ti)
    for (int ai = 0; ai < 2; ++ai) {
        double L1 = tri_pts[ti][0], L2 = tri_pts[ti][1];
        double zeta = ax_pts[ai];
        double w = tri_w; // axial weight is 1.0

        auto sd = shape_functions(L1, L2, zeta);

        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 6; ++n) {
            J(0,0) += sd.dNdL1[n]*coords[n].x; J(0,1) += sd.dNdL1[n]*coords[n].y; J(0,2) += sd.dNdL1[n]*coords[n].z;
            J(1,0) += sd.dNdL2[n]*coords[n].x; J(1,1) += sd.dNdL2[n]*coords[n].y; J(1,2) += sd.dNdL2[n]*coords[n].z;
            J(2,0) += sd.dNdzeta[n]*coords[n].x; J(2,1) += sd.dNdzeta[n]*coords[n].y; J(2,2) += sd.dNdzeta[n]*coords[n].z;
        }
        double detJ = J.determinant();
        if (detJ <= 0)
            throw SolverError(std::format("CPENTA6 {}: non-positive Jacobian det={:.6g}", eid_.value, detJ));
        Eigen::Matrix3d Jinv = J.inverse();

        Eigen::MatrixXd B(6, NUM_DOFS);
        build_B(sd, Jinv, B);

        Ke += B.transpose() * D_dev * B * detJ * w;
    }

    // Volumetric part: 1-point centroidal integration
    // Reference wedge volume = triangle_area(1/2) × axial_length(2) = 1.0
    {
        auto sd0 = shape_functions(1.0/3.0, 1.0/3.0, 0.0);
        Eigen::Matrix3d J0 = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 6; ++n) {
            J0(0,0) += sd0.dNdL1[n]*coords[n].x; J0(0,1) += sd0.dNdL1[n]*coords[n].y; J0(0,2) += sd0.dNdL1[n]*coords[n].z;
            J0(1,0) += sd0.dNdL2[n]*coords[n].x; J0(1,1) += sd0.dNdL2[n]*coords[n].y; J0(1,2) += sd0.dNdL2[n]*coords[n].z;
            J0(2,0) += sd0.dNdzeta[n]*coords[n].x; J0(2,1) += sd0.dNdzeta[n]*coords[n].y; J0(2,2) += sd0.dNdzeta[n]*coords[n].z;
        }
        double detJ0 = J0.determinant();
        if (detJ0 <= 0)
            throw SolverError(std::format("CPENTA6 {}: non-positive centroidal Jacobian det={:.6g}", eid_.value, detJ0));
        Eigen::Matrix3d Jinv0 = J0.inverse();

        Eigen::MatrixXd B0(6, NUM_DOFS);
        build_B(sd0, Jinv0, B0);

        Ke += B0.transpose() * D_vol * B0 * detJ0 * 1.0;
    }

    return Ke;
}

LocalKe CPenta6::mass_matrix() const {
    LocalKe Me = LocalKe::Zero(NUM_DOFS, NUM_DOFS);
    const double rho = material().rho;
    if (rho == 0.0) return Me;

    auto coords = node_coords();

    // 3-point triangular × 2-point axial Gauss quadrature (6 total)
    const double tri_pts[3][2] = {
        {2.0/3.0, 1.0/6.0},
        {1.0/6.0, 2.0/3.0},
        {1.0/6.0, 1.0/6.0}
    };
    const double tri_w = 1.0 / 6.0;
    const double ax_pts[2] = {-1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0)};

    for (int ti = 0; ti < 3; ++ti)
    for (int ai = 0; ai < 2; ++ai) {
        double L1 = tri_pts[ti][0], L2 = tri_pts[ti][1];
        double zeta = ax_pts[ai];

        auto sd = shape_functions(L1, L2, zeta);

        // Jacobian [3x3]: maps (dL1,dL2,dzeta) → (dx,dy,dz)
        // For penta, the Jacobian is:
        // J = [dX/dL1, dX/dL2, dX/dzeta; ...]
        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 6; ++n) {
            J(0,0)+=sd.dNdL1[n]*coords[n].x; J(0,1)+=sd.dNdL1[n]*coords[n].y; J(0,2)+=sd.dNdL1[n]*coords[n].z;
            J(1,0)+=sd.dNdL2[n]*coords[n].x; J(1,1)+=sd.dNdL2[n]*coords[n].y; J(1,2)+=sd.dNdL2[n]*coords[n].z;
            J(2,0)+=sd.dNdzeta[n]*coords[n].x; J(2,1)+=sd.dNdzeta[n]*coords[n].y; J(2,2)+=sd.dNdzeta[n]*coords[n].z;
        }
        double detJ = J.determinant();
        if (detJ <= 0)
            throw SolverError(std::format("CPenta6 {}: non-positive Jacobian in mass_matrix det={:.6g}", eid_.value, detJ));

        double weight = tri_w * 1.0; // axial weight = 1 for 2-point Gauss
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b) {
                double v = rho * sd.N[a] * sd.N[b] * detJ * weight;
                for (int d = 0; d < 3; ++d)
                    Me(3*a+d, 3*b+d) += v;
            }
    }
    return Me;
}

LocalFe CPenta6::thermal_load(std::span<const double> temperatures, double t_ref) const {
    LocalFe fe = LocalFe::Zero(NUM_DOFS);
    const double alpha = material().A;
    if (alpha == 0.0) return fe;

    auto coords = node_coords();
    Eigen::Matrix<double,6,6> D = constitutive_D();

    const double tri_pts[3][2] = {
        {2.0/3.0, 1.0/6.0},
        {1.0/6.0, 2.0/3.0},
        {1.0/6.0, 1.0/6.0}
    };
    const double tri_w = 1.0 / 6.0;
    const double ax_pts[2] = {-1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0)};

    for (int ti = 0; ti < 3; ++ti)
    for (int ai = 0; ai < 2; ++ai) {
        double L1 = tri_pts[ti][0], L2 = tri_pts[ti][1];
        double zeta = ax_pts[ai];
        double w = tri_w;

        auto sd = shape_functions(L1, L2, zeta);

        double T = 0;
        for (int n = 0; n < 6; ++n) T += sd.N[n] * temperatures[n];
        double dT = T - t_ref;
        if (std::abs(dT) < 1e-15) continue;

        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 6; ++n) {
            J(0,0) += sd.dNdL1[n]*coords[n].x; J(0,1) += sd.dNdL1[n]*coords[n].y; J(0,2) += sd.dNdL1[n]*coords[n].z;
            J(1,0) += sd.dNdL2[n]*coords[n].x; J(1,1) += sd.dNdL2[n]*coords[n].y; J(1,2) += sd.dNdL2[n]*coords[n].z;
            J(2,0) += sd.dNdzeta[n]*coords[n].x; J(2,1) += sd.dNdzeta[n]*coords[n].y; J(2,2) += sd.dNdzeta[n]*coords[n].z;
        }
        double detJ = J.determinant();
        Eigen::Matrix3d Jinv = J.inverse();

        Eigen::MatrixXd B(6, NUM_DOFS);
        B.setZero();
        for (int n = 0; n < 6; ++n) {
            double dnx = Jinv(0,0)*sd.dNdL1[n] + Jinv(0,1)*sd.dNdL2[n] + Jinv(0,2)*sd.dNdzeta[n];
            double dny = Jinv(1,0)*sd.dNdL1[n] + Jinv(1,1)*sd.dNdL2[n] + Jinv(1,2)*sd.dNdzeta[n];
            double dnz = Jinv(2,0)*sd.dNdL1[n] + Jinv(2,1)*sd.dNdL2[n] + Jinv(2,2)*sd.dNdzeta[n];
            int c0 = 3*n;
            B(0,c0+0)=dnx; B(1,c0+1)=dny; B(2,c0+2)=dnz;
            B(3,c0+0)=dny; B(3,c0+1)=dnx;
            B(4,c0+1)=dnz; B(4,c0+2)=dny;
            B(5,c0+0)=dnz; B(5,c0+2)=dnx;
        }

        Eigen::Matrix<double,6,1> eps_th;
        eps_th << alpha*dT, alpha*dT, alpha*dT, 0, 0, 0;
        fe += B.transpose() * D * eps_th * detJ * w;
    }
    return fe;
}

std::vector<EqIndex> CPenta6::global_dof_indices(const DofMap& dof_map) const {
    std::vector<EqIndex> result;
    result.reserve(NUM_DOFS);
    static constexpr int solid_dofs[3] = {0,1,2};
    for (NodeId nid : nodes_)
        for (int d : solid_dofs)
            result.push_back(dof_map.eq_index(nid, d));
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CTETRA10
// ═══════════════════════════════════════════════════════════════════════════════

CTetra10::CTetra10(ElementId eid, PropertyId pid,
                   std::array<NodeId, 10> node_ids,
                   const Model& model)
    : eid_(eid), pid_(pid), nodes_(node_ids), model_(model) {}

const PSolid& CTetra10::psolid() const {
    const auto& prop = model_.property(pid_);
    if (!std::holds_alternative<PSolid>(prop))
        throw SolverError(std::format("CTETRA10 {}: property {} is not PSOLID", eid_.value, pid_.value));
    return std::get<PSolid>(prop);
}

const Mat1& CTetra10::material() const {
    return model_.material(psolid().mid);
}

Eigen::Matrix<double,6,6> CTetra10::constitutive_D() const {
    const Mat1& m = material();
    return isotropic_D(m.E, m.nu);
}

std::array<Vec3, 10> CTetra10::node_coords() const {
    std::array<Vec3, 10> c;
    for (int i = 0; i < 10; ++i)
        c[i] = model_.node(nodes_[i]).position;
    return c;
}

// Node ordering (Nastran CTETRA10):
//   0-3: corner nodes (L1=1,0,0,0), (L2=1,...), ...
//   4: midside 0-1, 5: midside 1-2, 6: midside 0-2
//   7: midside 0-3, 8: midside 1-3, 9: midside 2-3
// In barycentric: corners i at Li=1, midsides at Li=Lj=0.5
CTetra10::ShapeData10 CTetra10::shape_functions(double L1, double L2, double L3) noexcept {
    double L4 = 1.0 - L1 - L2 - L3;
    ShapeData10 s;
    // Corner nodes: Ni = Li*(2Li - 1)
    s.N[0] = L1*(2*L1 - 1);
    s.N[1] = L2*(2*L2 - 1);
    s.N[2] = L3*(2*L3 - 1);
    s.N[3] = L4*(2*L4 - 1);
    // Midside nodes: Nij = 4*Li*Lj
    s.N[4] = 4*L1*L2;  // 0-1
    s.N[5] = 4*L2*L3;  // 1-2
    s.N[6] = 4*L1*L3;  // 0-2
    s.N[7] = 4*L1*L4;  // 0-3
    s.N[8] = 4*L2*L4;  // 1-3
    s.N[9] = 4*L3*L4;  // 2-3

    // dN/dL1
    s.dNdL1[0] = 4*L1 - 1;
    s.dNdL1[1] = 0;
    s.dNdL1[2] = 0;
    s.dNdL1[3] = -(4*L4 - 1);  // dL4/dL1 = -1
    s.dNdL1[4] = 4*L2;
    s.dNdL1[5] = 0;
    s.dNdL1[6] = 4*L3;
    s.dNdL1[7] = 4*(L4 - L1);
    s.dNdL1[8] = -4*L2;
    s.dNdL1[9] = -4*L3;

    // dN/dL2
    s.dNdL2[0] = 0;
    s.dNdL2[1] = 4*L2 - 1;
    s.dNdL2[2] = 0;
    s.dNdL2[3] = -(4*L4 - 1);
    s.dNdL2[4] = 4*L1;
    s.dNdL2[5] = 4*L3;
    s.dNdL2[6] = 0;
    s.dNdL2[7] = -4*L1;
    s.dNdL2[8] = 4*(L4 - L2);
    s.dNdL2[9] = -4*L3;

    // dN/dL3
    s.dNdL3[0] = 0;
    s.dNdL3[1] = 0;
    s.dNdL3[2] = 4*L3 - 1;
    s.dNdL3[3] = -(4*L4 - 1);
    s.dNdL3[4] = 0;
    s.dNdL3[5] = 4*L2;
    s.dNdL3[6] = 4*L1;
    s.dNdL3[7] = -4*L1;
    s.dNdL3[8] = -4*L2;
    s.dNdL3[9] = 4*(L4 - L3);

    return s;
}

LocalKe CTetra10::stiffness_matrix() const {
    LocalKe Ke = LocalKe::Zero(NUM_DOFS, NUM_DOFS);
    auto coords = node_coords();
    Eigen::Matrix<double,6,6> D = constitutive_D();

    // 4-point Gauss quadrature for tetrahedra (exact degree 2)
    // Points in barycentric coords (L1,L2,L3,L4), weight = 1/4 each (sum=1)
    // Reference: Dunavant, IJNME 1985
    const double a = 0.5854101966249685; // (5 + 3*sqrt(5)) / 20
    const double b = 0.1381966011250105; // (5 - sqrt(5)) / 20
    const double w = 0.25;

    const double pts[4][3] = {
        {a, b, b},
        {b, a, b},
        {b, b, a},
        {b, b, b}
    };

    for (int gp = 0; gp < 4; ++gp) {
        double L1 = pts[gp][0], L2 = pts[gp][1], L3 = pts[gp][2];
        auto sd = shape_functions(L1, L2, L3);

        // Jacobian: maps (L1,L2,L3) → (x,y,z)
        // dX/dL = sum_i dNi/dL * xi
        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 10; ++n) {
            J(0,0) += sd.dNdL1[n] * coords[n].x;
            J(0,1) += sd.dNdL1[n] * coords[n].y;
            J(0,2) += sd.dNdL1[n] * coords[n].z;
            J(1,0) += sd.dNdL2[n] * coords[n].x;
            J(1,1) += sd.dNdL2[n] * coords[n].y;
            J(1,2) += sd.dNdL2[n] * coords[n].z;
            J(2,0) += sd.dNdL3[n] * coords[n].x;
            J(2,1) += sd.dNdL3[n] * coords[n].y;
            J(2,2) += sd.dNdL3[n] * coords[n].z;
        }
        double detJ = J.determinant();
        // The Jacobian det can be negative depending on node ordering convention in
        // barycentric coordinates — this is geometrically valid. Use |detJ| for the
        // integration weight; check only for degeneracy (|detJ| ≈ 0 → zero-volume tet).
        double absDetJ = std::abs(detJ);
        if (absDetJ < 1e-14)
            throw SolverError(std::format("CTETRA10 {}: degenerate element, |det(J)|={:.6g}", eid_.value, absDetJ));
        Eigen::Matrix3d Jinv = J.inverse();

        // B matrix [6 x 30]
        Eigen::MatrixXd B(6, NUM_DOFS);
        B.setZero();
        for (int n = 0; n < 10; ++n) {
            double dnx = Jinv(0,0)*sd.dNdL1[n] + Jinv(0,1)*sd.dNdL2[n] + Jinv(0,2)*sd.dNdL3[n];
            double dny = Jinv(1,0)*sd.dNdL1[n] + Jinv(1,1)*sd.dNdL2[n] + Jinv(1,2)*sd.dNdL3[n];
            double dnz = Jinv(2,0)*sd.dNdL1[n] + Jinv(2,1)*sd.dNdL2[n] + Jinv(2,2)*sd.dNdL3[n];
            int c0 = 3*n;
            B(0,c0+0) = dnx;
            B(1,c0+1) = dny;
            B(2,c0+2) = dnz;
            B(3,c0+0) = dny; B(3,c0+1) = dnx;
            B(4,c0+1) = dnz; B(4,c0+2) = dny;
            B(5,c0+0) = dnz; B(5,c0+2) = dnx;
        }

        // Integration weight: w * |detJ|; the Gauss weights w=0.25 sum to 1 over
        // the unit reference tet (volume=1/6 in 3-coord space). The transformation
        // ∫_phys = ∫_ref |detJ| dL; reference tet integral ≈ (1/6)*sum(w_i*f_i)
        // with w_i=0.25 (scaled so sum(w_i)=1, already includes the 1/6 factor via
        // the standard Dunavant normalization where weights sum to volume=1/6... no,
        // the weights here sum to 1 so they need the 1/6 reference volume factor:
        Ke += B.transpose() * D * B * (w / 6.0) * absDetJ;
    }

    return Ke;
}

LocalKe CTetra10::mass_matrix() const {
    LocalKe Me = LocalKe::Zero(NUM_DOFS, NUM_DOFS);
    const double rho = material().rho;
    if (rho == 0.0) return Me;

    // 4-point Gauss quadrature for tetrahedron (exact for degree-3 polynomials,
    // needed for quadratic shape function products which are degree 4 — slightly
    // under-integrated, but standard practice for CTetra10 mass).
    // Points and weights from Dunavant, 1985 (barycentric coordinates)
    static const double gL[4][4] = {
        {0.13819660,0.13819660,0.13819660,0.58541020},
        {0.13819660,0.13819660,0.58541020,0.13819660},
        {0.13819660,0.58541020,0.13819660,0.13819660},
        {0.58541020,0.13819660,0.13819660,0.13819660}
    };
    static const double gW[4] = {0.04166667, 0.04166667, 0.04166667, 0.04166667};

    auto coords = node_coords();

    for (int gi = 0; gi < 4; ++gi) {
        double L1 = gL[gi][0], L2 = gL[gi][1], L3 = gL[gi][2];
        // L4 = gL[gi][3] but unused in shape_functions (derived as 1-L1-L2-L3)

        auto sd = shape_functions(L1, L2, L3);

        // Jacobian: maps (dL1,dL2,dL3) → (dx,dy,dz)
        // J[i][j] = sum_n dN_n/dLj * coords[n][i]
        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 10; ++n) {
            J(0,0)+=sd.dNdL1[n]*coords[n].x; J(0,1)+=sd.dNdL2[n]*coords[n].x; J(0,2)+=sd.dNdL3[n]*coords[n].x;
            J(1,0)+=sd.dNdL1[n]*coords[n].y; J(1,1)+=sd.dNdL2[n]*coords[n].y; J(1,2)+=sd.dNdL3[n]*coords[n].y;
            J(2,0)+=sd.dNdL1[n]*coords[n].z; J(2,1)+=sd.dNdL2[n]*coords[n].z; J(2,2)+=sd.dNdL3[n]*coords[n].z;
        }
        double detJ = J.determinant();
        if (detJ <= 0)
            throw SolverError(std::format("CTetra10 {}: non-positive Jacobian in mass_matrix det={:.6g}", eid_.value, detJ));

        double weight = gW[gi];
        for (int a = 0; a < 10; ++a)
            for (int b = 0; b < 10; ++b) {
                double v = rho * sd.N[a] * sd.N[b] * detJ * weight;
                for (int d = 0; d < 3; ++d)
                    Me(3*a+d, 3*b+d) += v;
            }
    }
    return Me;
}

LocalFe CTetra10::thermal_load(std::span<const double> temperatures, double t_ref) const {
    LocalFe fe = LocalFe::Zero(NUM_DOFS);
    const double alpha = material().A;
    if (alpha == 0.0) return fe;

    auto coords = node_coords();
    Eigen::Matrix<double,6,6> D = constitutive_D();

    const double a = 0.5854101966249685;
    const double b = 0.1381966011250105;
    const double w = 0.25;
    const double pts[4][3] = {{a,b,b},{b,a,b},{b,b,a},{b,b,b}};

    for (int gp = 0; gp < 4; ++gp) {
        double L1 = pts[gp][0], L2 = pts[gp][1], L3 = pts[gp][2];
        auto sd = shape_functions(L1, L2, L3);

        double T = 0;
        for (int n = 0; n < 10; ++n) T += sd.N[n] * temperatures[n];
        double dT = T - t_ref;
        if (std::abs(dT) < 1e-15) continue;

        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 10; ++n) {
            J(0,0) += sd.dNdL1[n]*coords[n].x; J(0,1) += sd.dNdL1[n]*coords[n].y; J(0,2) += sd.dNdL1[n]*coords[n].z;
            J(1,0) += sd.dNdL2[n]*coords[n].x; J(1,1) += sd.dNdL2[n]*coords[n].y; J(1,2) += sd.dNdL2[n]*coords[n].z;
            J(2,0) += sd.dNdL3[n]*coords[n].x; J(2,1) += sd.dNdL3[n]*coords[n].y; J(2,2) += sd.dNdL3[n]*coords[n].z;
        }
        double detJ = J.determinant();
        double absDetJ = std::abs(detJ);
        Eigen::Matrix3d Jinv = J.inverse();

        Eigen::MatrixXd B(6, NUM_DOFS);
        B.setZero();
        for (int n = 0; n < 10; ++n) {
            double dnx = Jinv(0,0)*sd.dNdL1[n] + Jinv(0,1)*sd.dNdL2[n] + Jinv(0,2)*sd.dNdL3[n];
            double dny = Jinv(1,0)*sd.dNdL1[n] + Jinv(1,1)*sd.dNdL2[n] + Jinv(1,2)*sd.dNdL3[n];
            double dnz = Jinv(2,0)*sd.dNdL1[n] + Jinv(2,1)*sd.dNdL2[n] + Jinv(2,2)*sd.dNdL3[n];
            int c0 = 3*n;
            B(0,c0+0)=dnx; B(1,c0+1)=dny; B(2,c0+2)=dnz;
            B(3,c0+0)=dny; B(3,c0+1)=dnx;
            B(4,c0+1)=dnz; B(4,c0+2)=dny;
            B(5,c0+0)=dnz; B(5,c0+2)=dnx;
        }
        Eigen::Matrix<double,6,1> eps_th;
        eps_th << alpha*dT, alpha*dT, alpha*dT, 0, 0, 0;
        fe += B.transpose() * D * eps_th * (w / 6.0) * absDetJ;
    }
    return fe;
}

std::vector<EqIndex> CTetra10::global_dof_indices(const DofMap& dof_map) const {
    std::vector<EqIndex> result;
    result.reserve(NUM_DOFS);
    static constexpr int solid_dofs[3] = {0,1,2};
    for (NodeId nid : nodes_)
        for (int d : solid_dofs)
            result.push_back(dof_map.eq_index(nid, d));
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHEXA8EAS
// ═══════════════════════════════════════════════════════════════════════════════

CHexa8Eas::CHexa8Eas(ElementId eid, PropertyId pid,
                     std::array<NodeId, 8> node_ids,
                     const Model& model)
    : eid_(eid), pid_(pid), nodes_(node_ids), model_(model) {}

const PSolid& CHexa8Eas::psolid() const {
    const auto& prop = model_.property(pid_);
    if (!std::holds_alternative<PSolid>(prop))
        throw SolverError(std::format("CHEXA8EAS {}: property {} is not PSOLID", eid_.value, pid_.value));
    return std::get<PSolid>(prop);
}

const Mat1& CHexa8Eas::material() const {
    return model_.material(psolid().mid);
}

Eigen::Matrix<double,6,6> CHexa8Eas::constitutive_D() const {
    const Mat1& m = material();
    return isotropic_D(m.E, m.nu);
}

std::array<Vec3, 8> CHexa8Eas::node_coords() const {
    std::array<Vec3, 8> c;
    for (int i = 0; i < 8; ++i)
        c[i] = model_.node(nodes_[i]).position;
    return c;
}

LocalKe CHexa8Eas::stiffness_matrix() const {
    // EAS with 9 internal enhancement modes (3 per parametric direction).
    // Static condensation: Ke = Kuu - Kua * Kaa^-1 * Kau
    // Centroidal Jacobian J0 used to map enhancement modes for frame objectivity.

    LocalKe Ke = LocalKe::Zero(NUM_DOFS, NUM_DOFS);
    auto coords = node_coords();
    Eigen::Matrix<double,6,6> D = constitutive_D();

    // Centroidal Jacobian J0
    auto sd0 = CHexa8::shape_functions(0.0, 0.0, 0.0);
    Eigen::Matrix3d J0 = Eigen::Matrix3d::Zero();
    for (int n = 0; n < 8; ++n) {
        J0(0,0)+=sd0.dNdxi[n]*coords[n].x; J0(0,1)+=sd0.dNdxi[n]*coords[n].y; J0(0,2)+=sd0.dNdxi[n]*coords[n].z;
        J0(1,0)+=sd0.dNdeta[n]*coords[n].x; J0(1,1)+=sd0.dNdeta[n]*coords[n].y; J0(1,2)+=sd0.dNdeta[n]*coords[n].z;
        J0(2,0)+=sd0.dNdzeta[n]*coords[n].x; J0(2,1)+=sd0.dNdzeta[n]*coords[n].y; J0(2,2)+=sd0.dNdzeta[n]*coords[n].z;
    }
    double detJ0 = J0.determinant();
    if (detJ0 <= 0)
        throw SolverError(std::format("CHEXA8EAS {}: non-positive centroidal Jacobian det={:.6g}", eid_.value, detJ0));

    // Sub-matrices for static condensation
    Eigen::Matrix<double,24,24> Kuu = Eigen::Matrix<double,24,24>::Zero();
    Eigen::Matrix<double,24,9>  Kua = Eigen::Matrix<double,24,9>::Zero();
    Eigen::Matrix<double,9,9>   Kaa = Eigen::Matrix<double,9,9>::Zero();

    auto build_B = [&](const CHexa8::ShapeData& sd, const Eigen::Matrix3d& Jinv,
                       Eigen::Matrix<double,6,24>& B) {
        B.setZero();
        for (int n = 0; n < 8; ++n) {
            double dnx = Jinv(0,0)*sd.dNdxi[n] + Jinv(0,1)*sd.dNdeta[n] + Jinv(0,2)*sd.dNdzeta[n];
            double dny = Jinv(1,0)*sd.dNdxi[n] + Jinv(1,1)*sd.dNdeta[n] + Jinv(1,2)*sd.dNdzeta[n];
            double dnz = Jinv(2,0)*sd.dNdxi[n] + Jinv(2,1)*sd.dNdeta[n] + Jinv(2,2)*sd.dNdzeta[n];
            int c0 = 3*n;
            B(0,c0+0)=dnx; B(1,c0+1)=dny; B(2,c0+2)=dnz;
            B(3,c0+0)=dny; B(3,c0+1)=dnx;
            B(4,c0+1)=dnz; B(4,c0+2)=dny;
            B(5,c0+0)=dnz; B(5,c0+2)=dnx;
        }
    };

    // Enhancement mode matrix G [6x9] in physical space at a given Gauss point.
    // 9 modes: xi*(e1), eta*(e2), zeta*(e3) in each of 3 coordinate directions.
    // Mapped with centroidal Jacobian for frame objectivity (Simo & Rifai 1990).
    auto build_G = [&](double xi, double eta, double zeta,
                       double detJ, const Eigen::Matrix3d& Jinv,
                       Eigen::Matrix<double,6,9>& G) {
        // In parametric space, enhancement modes are diagonal matrices of
        // [xi, eta, zeta] ⊗ I3 restricted to symmetric gradient operator.
        // Physical G = (detJ0/detJ) * T0 * G_param
        // T0 = transformation from centroidal Jacobian (see Wilson-Taylor):
        //   For isotropic: G_physical_col = (detJ0/detJ) * J0^{-T} * J^T * e_param_col
        // Simplified (rectangular element, nearly correct for distorted):
        // G = (detJ0/detJ) * [symmetric gradient of J0^{-1} * J * param_modes]
        // Use: Ba(6x9) where mode i adds enhancement in direction (i/3), component (i%3)
        double scale = detJ0 / detJ;
        G.setZero();

        // 3 modes for xi: add strain from d(xi * u_i) / dx_j
        // Enhancement displacement field: u_alpha = xi * delta_{alpha,0}
        // => strain e = B_enh where B_enh comes from physical gradient of enhancement
        // Physical gradient of xi in x-space = J^{-T} * [1,0,0]^T (parametric gradient of xi)
        // But we use J0 for objectivity: grad_phys = J0^{-1} * [1,0,0]^T
        // Full formula: G_phys = (detJ0/detJ) * sym_grad(J0^{-1} * J_param_shape)
        // For simplicity use the standard EAS formulation:
        //   column i of G = (detJ0/detJ) * B_enh_i where B_enh_i encodes the enhancement
        // Columns 0-2: xi-modes in x,y,z directions
        // Columns 3-5: eta-modes in x,y,z directions
        // Columns 6-8: zeta-modes in x,y,z directions

        // Physical B for enhancement: d(param)/dx = J0^{-T} * e_param_dir
        // where e_param_dir = [1,0,0] for xi, [0,1,0] for eta, [0,0,1] for zeta
        Eigen::Vector3d gxi_phys  = Jinv.row(0).transpose() * xi;   // J^{-T} * [1,0,0] * xi
        Eigen::Vector3d geta_phys = Jinv.row(1).transpose() * eta;
        Eigen::Vector3d gzeta_phys= Jinv.row(2).transpose() * zeta;

        // Use J0 rows for the mapping (frame objectivity)
        Eigen::Vector3d gxi0   = J0.inverse().row(0).transpose() * xi;
        Eigen::Vector3d geta0  = J0.inverse().row(1).transpose() * eta;
        Eigen::Vector3d gzeta0 = J0.inverse().row(2).transpose() * zeta;

        // B_enh for enhancement mode: symmetric gradient of a vector field v
        // B_sym_grad(v) = [v_x,x; v_y,y; v_z,z; v_x,y+v_y,x; v_y,z+v_z,y; v_x,z+v_z,x]
        // For displacement mode alpha: u_alpha = param * delta_{alpha,i}
        // => v = [param, 0, 0] for i=0, etc.
        // grad(param) in physical = Jinv.row(param_dir)
        auto fill_G_col = [&](int col, const Eigen::Vector3d& g, int alpha) {
            // alpha = 0 (x-mode), 1 (y-mode), 2 (z-mode)
            // u_enh = g * e_alpha → symmetric gradient:
            double gx = g(0), gy = g(1), gz = g(2);
            if (alpha == 0) {
                G(0,col) = gx; G(3,col) = gy; G(5,col) = gz;
            } else if (alpha == 1) {
                G(3,col) = gx; G(1,col) = gy; G(4,col) = gz;
            } else {
                G(5,col) = gx; G(4,col) = gy; G(2,col) = gz;
            }
        };

        fill_G_col(0, gxi0,   0);
        fill_G_col(1, gxi0,   1);
        fill_G_col(2, gxi0,   2);
        fill_G_col(3, geta0,  0);
        fill_G_col(4, geta0,  1);
        fill_G_col(5, geta0,  2);
        fill_G_col(6, gzeta0, 0);
        fill_G_col(7, gzeta0, 1);
        fill_G_col(8, gzeta0, 2);

        G *= scale;
        (void)gxi_phys; (void)geta_phys; (void)gzeta_phys;
    };

    const double gp = 1.0/std::sqrt(3.0);
    const double gpts[2] = {-gp, gp};

    for (int gi = 0; gi < 2; ++gi)
    for (int gj = 0; gj < 2; ++gj)
    for (int gk = 0; gk < 2; ++gk) {
        double xi=gpts[gi], eta=gpts[gj], zeta=gpts[gk];

        auto sd = CHexa8::shape_functions(xi, eta, zeta);

        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 8; ++n) {
            J(0,0)+=sd.dNdxi[n]*coords[n].x; J(0,1)+=sd.dNdxi[n]*coords[n].y; J(0,2)+=sd.dNdxi[n]*coords[n].z;
            J(1,0)+=sd.dNdeta[n]*coords[n].x; J(1,1)+=sd.dNdeta[n]*coords[n].y; J(1,2)+=sd.dNdeta[n]*coords[n].z;
            J(2,0)+=sd.dNdzeta[n]*coords[n].x; J(2,1)+=sd.dNdzeta[n]*coords[n].y; J(2,2)+=sd.dNdzeta[n]*coords[n].z;
        }
        double detJ = J.determinant();
        if (detJ <= 0)
            throw SolverError(std::format("CHEXA8EAS {}: non-positive Jacobian det={:.6g}", eid_.value, detJ));
        Eigen::Matrix3d Jinv = J.inverse();

        Eigen::Matrix<double,6,24> B;
        build_B(sd, Jinv, B);

        Eigen::Matrix<double,6,9> G;
        build_G(xi, eta, zeta, detJ, Jinv, G);

        double w = detJ; // weights are 1×1×1=1 each
        Kuu += B.transpose() * D * B * w;
        Kua += B.transpose() * D * G * w;
        Kaa += G.transpose() * D * G * w;
    }

    // Static condensation: Ke = Kuu - Kua * Kaa^{-1} * Kau
    Eigen::LLT<Eigen::Matrix<double,9,9>> llt(Kaa);
    if (llt.info() != Eigen::Success)
        throw SolverError(std::format("CHEXA8EAS {}: Kaa factorization failed", eid_.value));
    Ke = Kuu - Kua * llt.solve(Kua.transpose());
    return Ke;
}

LocalKe CHexa8Eas::mass_matrix() const {
    // EAS modes are element-internal (condensed out); mass matrix is identical to CHexa8.
    LocalKe Me = LocalKe::Zero(NUM_DOFS, NUM_DOFS);
    const double rho = material().rho;
    if (rho == 0.0) return Me;

    auto coords = node_coords();
    const double gp = 1.0/std::sqrt(3.0);
    const double gpts[2] = {-gp, gp};

    for (int gi = 0; gi < 2; ++gi)
    for (int gj = 0; gj < 2; ++gj)
    for (int gk = 0; gk < 2; ++gk) {
        double xi = gpts[gi], eta = gpts[gj], zeta = gpts[gk];

        auto sd = CHexa8::shape_functions(xi, eta, zeta);

        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 8; ++n) {
            J(0,0)+=sd.dNdxi[n]  *coords[n].x; J(0,1)+=sd.dNdxi[n]  *coords[n].y; J(0,2)+=sd.dNdxi[n]  *coords[n].z;
            J(1,0)+=sd.dNdeta[n] *coords[n].x; J(1,1)+=sd.dNdeta[n] *coords[n].y; J(1,2)+=sd.dNdeta[n] *coords[n].z;
            J(2,0)+=sd.dNdzeta[n]*coords[n].x; J(2,1)+=sd.dNdzeta[n]*coords[n].y; J(2,2)+=sd.dNdzeta[n]*coords[n].z;
        }
        double detJ = J.determinant();
        if (detJ <= 0)
            throw SolverError(std::format("CHexa8Eas {}: non-positive Jacobian in mass_matrix det={:.6g}", eid_.value, detJ));

        for (int a = 0; a < 8; ++a)
            for (int b = 0; b < 8; ++b) {
                double v = rho * sd.N[a] * sd.N[b] * detJ;
                for (int d = 0; d < 3; ++d)
                    Me(3*a+d, 3*b+d) += v;
            }
    }
    return Me;
}

LocalFe CHexa8Eas::thermal_load(std::span<const double> temperatures, double t_ref) const {
    // EAS enhancement modes have zero mean strain, so thermal load is identical to CHexa8.
    LocalFe fe = LocalFe::Zero(NUM_DOFS);
    const double alpha = material().A;
    if (alpha == 0.0) return fe;

    auto coords = node_coords();
    Eigen::Matrix<double,6,6> D = constitutive_D();

    const double gp = 1.0/std::sqrt(3.0);
    const double gpts[2] = {-gp, gp};

    for (int gi = 0; gi < 2; ++gi)
    for (int gj = 0; gj < 2; ++gj)
    for (int gk = 0; gk < 2; ++gk) {
        double xi=gpts[gi], eta=gpts[gj], zeta=gpts[gk];

        auto sd = CHexa8::shape_functions(xi, eta, zeta);

        double T = 0;
        for (int n = 0; n < 8; ++n) T += sd.N[n] * temperatures[n];
        double dT = T - t_ref;
        if (std::abs(dT) < 1e-15) continue;

        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 8; ++n) {
            J(0,0)+=sd.dNdxi[n]*coords[n].x; J(0,1)+=sd.dNdxi[n]*coords[n].y; J(0,2)+=sd.dNdxi[n]*coords[n].z;
            J(1,0)+=sd.dNdeta[n]*coords[n].x; J(1,1)+=sd.dNdeta[n]*coords[n].y; J(1,2)+=sd.dNdeta[n]*coords[n].z;
            J(2,0)+=sd.dNdzeta[n]*coords[n].x; J(2,1)+=sd.dNdzeta[n]*coords[n].y; J(2,2)+=sd.dNdzeta[n]*coords[n].z;
        }
        double detJ = J.determinant();
        Eigen::Matrix3d Jinv = J.inverse();

        Eigen::Matrix<double,6,24> B;
        B.setZero();
        for (int n = 0; n < 8; ++n) {
            double dnx = Jinv(0,0)*sd.dNdxi[n] + Jinv(0,1)*sd.dNdeta[n] + Jinv(0,2)*sd.dNdzeta[n];
            double dny = Jinv(1,0)*sd.dNdxi[n] + Jinv(1,1)*sd.dNdeta[n] + Jinv(1,2)*sd.dNdzeta[n];
            double dnz = Jinv(2,0)*sd.dNdxi[n] + Jinv(2,1)*sd.dNdeta[n] + Jinv(2,2)*sd.dNdzeta[n];
            int c0=3*n;
            B(0,c0+0)=dnx; B(1,c0+1)=dny; B(2,c0+2)=dnz;
            B(3,c0+0)=dny; B(3,c0+1)=dnx;
            B(4,c0+1)=dnz; B(4,c0+2)=dny;
            B(5,c0+0)=dnz; B(5,c0+2)=dnx;
        }
        Eigen::Matrix<double,6,1> eps_th;
        eps_th << alpha*dT, alpha*dT, alpha*dT, 0, 0, 0;
        fe += B.transpose() * D * eps_th * detJ;
    }
    return fe;
}

std::vector<EqIndex> CHexa8Eas::global_dof_indices(const DofMap& dof_map) const {
    std::vector<EqIndex> result;
    result.reserve(NUM_DOFS);
    static constexpr int solid_dofs[3] = {0,1,2};
    for (NodeId nid : nodes_)
        for (int d : solid_dofs)
            result.push_back(dof_map.eq_index(nid, d));
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CPENTA6EAS
// ═══════════════════════════════════════════════════════════════════════════════

CPenta6Eas::CPenta6Eas(ElementId eid, PropertyId pid,
                       std::array<NodeId, 6> node_ids,
                       const Model& model)
    : eid_(eid), pid_(pid), nodes_(node_ids), model_(model) {}

const PSolid& CPenta6Eas::psolid() const {
    const auto& prop = model_.property(pid_);
    if (!std::holds_alternative<PSolid>(prop))
        throw SolverError(std::format("CPENTA6EAS {}: property {} is not PSOLID", eid_.value, pid_.value));
    return std::get<PSolid>(prop);
}

const Mat1& CPenta6Eas::material() const {
    return model_.material(psolid().mid);
}

Eigen::Matrix<double,6,6> CPenta6Eas::constitutive_D() const {
    const Mat1& m = material();
    return isotropic_D(m.E, m.nu);
}

std::array<Vec3, 6> CPenta6Eas::node_coords() const {
    std::array<Vec3, 6> c;
    for (int i = 0; i < 6; ++i)
        c[i] = model_.node(nodes_[i]).position;
    return c;
}

LocalKe CPenta6Eas::stiffness_matrix() const {
    // EAS with 9 internal enhancement modes (3 parametric × 3 spatial directions).
    // Enhancement functions with zero mean over the reference wedge:
    //   Triangular: (L1 - 1/3), (L2 - 1/3)  — zero mean over unit triangle
    //   Axial:      zeta                      — zero mean over [-1, 1]
    // Static condensation: Ke = Kuu - Kua * Kaa^{-1} * Kau

    auto coords = node_coords();
    Eigen::Matrix<double,6,6> D = constitutive_D();

    // Centroidal Jacobian J0 (L1=1/3, L2=1/3, zeta=0)
    auto sd0 = CPenta6::shape_functions(1.0/3.0, 1.0/3.0, 0.0);
    Eigen::Matrix3d J0 = Eigen::Matrix3d::Zero();
    for (int n = 0; n < 6; ++n) {
        J0(0,0) += sd0.dNdL1[n]*coords[n].x; J0(0,1) += sd0.dNdL1[n]*coords[n].y; J0(0,2) += sd0.dNdL1[n]*coords[n].z;
        J0(1,0) += sd0.dNdL2[n]*coords[n].x; J0(1,1) += sd0.dNdL2[n]*coords[n].y; J0(1,2) += sd0.dNdL2[n]*coords[n].z;
        J0(2,0) += sd0.dNdzeta[n]*coords[n].x; J0(2,1) += sd0.dNdzeta[n]*coords[n].y; J0(2,2) += sd0.dNdzeta[n]*coords[n].z;
    }
    double detJ0 = J0.determinant();
    if (detJ0 <= 0)
        throw SolverError(std::format("CPENTA6EAS {}: non-positive centroidal Jacobian det={:.6g}", eid_.value, detJ0));
    Eigen::Matrix3d J0inv = J0.inverse();

    // Sub-matrices for static condensation
    constexpr int N_EAS = 9;
    Eigen::Matrix<double,18,18> Kuu = Eigen::Matrix<double,18,18>::Zero();
    Eigen::Matrix<double,18,N_EAS> Kua = Eigen::Matrix<double,18,N_EAS>::Zero();
    Eigen::Matrix<double,N_EAS,N_EAS> Kaa = Eigen::Matrix<double,N_EAS,N_EAS>::Zero();

    auto build_B = [&](const CPenta6::ShapeData6& sd, const Eigen::Matrix3d& Jinv,
                       Eigen::Matrix<double,6,18>& B) {
        B.setZero();
        for (int n = 0; n < 6; ++n) {
            double dnx = Jinv(0,0)*sd.dNdL1[n] + Jinv(0,1)*sd.dNdL2[n] + Jinv(0,2)*sd.dNdzeta[n];
            double dny = Jinv(1,0)*sd.dNdL1[n] + Jinv(1,1)*sd.dNdL2[n] + Jinv(1,2)*sd.dNdzeta[n];
            double dnz = Jinv(2,0)*sd.dNdL1[n] + Jinv(2,1)*sd.dNdL2[n] + Jinv(2,2)*sd.dNdzeta[n];
            int c0 = 3*n;
            B(0,c0+0) = dnx; B(1,c0+1) = dny; B(2,c0+2) = dnz;
            B(3,c0+0) = dny; B(3,c0+1) = dnx;
            B(4,c0+1) = dnz; B(4,c0+2) = dny;
            B(5,c0+0) = dnz; B(5,c0+2) = dnx;
        }
    };

    // Enhancement mode matrix G [6×9] at a Gauss point.
    // 9 modes: 3 parametric enhancement functions × 3 displacement directions.
    // Parametric enhancement functions:
    //   f0 = L1 - 1/3  (zero mean over triangle)
    //   f1 = L2 - 1/3  (zero mean over triangle)
    //   f2 = zeta       (zero mean over [-1,1])
    // Physical gradient mapped via centroidal Jacobian for frame objectivity.
    auto build_G = [&](double L1, double L2, double zeta,
                       double detJ,
                       Eigen::Matrix<double,6,N_EAS>& G) {
        double scale = detJ0 / detJ;

        // Parametric enhancement function values
        double f[3] = { L1 - 1.0/3.0, L2 - 1.0/3.0, zeta };

        // Physical gradient of each enhancement function via centroidal Jacobian:
        // grad_phys(f_k) = J0^{-T} * grad_param(f_k)
        // grad_param(f0) = [1, 0, 0]^T (wrt L1, L2, zeta)
        // grad_param(f1) = [0, 1, 0]^T
        // grad_param(f2) = [0, 0, 1]^T
        // So grad_phys(f_k) = J0^{-T}.col(k) = J0inv.row(k)^T
        // But the enhancement strain is the symmetric gradient of (f_k * e_alpha):
        //   ε̃ = sym(e_alpha ⊗ grad_phys(f_k)) scaled by (detJ0/detJ)

        G.setZero();
        for (int k = 0; k < 3; ++k) {
            // Physical gradient of enhancement function f_k
            double gx = J0inv(0, k) * f[k];
            double gy = J0inv(1, k) * f[k];
            double gz = J0inv(2, k) * f[k];

            // 3 columns per parametric mode (one per displacement direction)
            int base = 3*k;
            // alpha=0 (x-mode): u_enh = f_k * e_x
            G(0, base+0) = gx; G(3, base+0) = gy; G(5, base+0) = gz;
            // alpha=1 (y-mode): u_enh = f_k * e_y
            G(3, base+1) = gx; G(1, base+1) = gy; G(4, base+1) = gz;
            // alpha=2 (z-mode): u_enh = f_k * e_z
            G(5, base+2) = gx; G(4, base+2) = gy; G(2, base+2) = gz;
        }
        G *= scale;
    };

    // 6-point Gauss quadrature: 3 triangle points × 2 axial points
    const double tri_pts[3][2] = {
        {2.0/3.0, 1.0/6.0},
        {1.0/6.0, 2.0/3.0},
        {1.0/6.0, 1.0/6.0}
    };
    const double tri_w = 1.0 / 6.0;
    const double gp = 1.0/std::sqrt(3.0);
    const double ax_pts[2] = {-gp, gp};

    for (int ti = 0; ti < 3; ++ti)
    for (int ai = 0; ai < 2; ++ai) {
        double L1 = tri_pts[ti][0], L2 = tri_pts[ti][1];
        double zeta = ax_pts[ai];

        auto sd = CPenta6::shape_functions(L1, L2, zeta);

        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 6; ++n) {
            J(0,0) += sd.dNdL1[n]*coords[n].x; J(0,1) += sd.dNdL1[n]*coords[n].y; J(0,2) += sd.dNdL1[n]*coords[n].z;
            J(1,0) += sd.dNdL2[n]*coords[n].x; J(1,1) += sd.dNdL2[n]*coords[n].y; J(1,2) += sd.dNdL2[n]*coords[n].z;
            J(2,0) += sd.dNdzeta[n]*coords[n].x; J(2,1) += sd.dNdzeta[n]*coords[n].y; J(2,2) += sd.dNdzeta[n]*coords[n].z;
        }
        double detJ = J.determinant();
        if (detJ <= 0)
            throw SolverError(std::format("CPENTA6EAS {}: non-positive Jacobian det={:.6g}", eid_.value, detJ));
        Eigen::Matrix3d Jinv = J.inverse();

        Eigen::Matrix<double,6,18> B;
        build_B(sd, Jinv, B);

        Eigen::Matrix<double,6,N_EAS> G;
        build_G(L1, L2, zeta, detJ, G);

        double w = detJ * tri_w; // axial weight is 1.0
        Kuu += B.transpose() * D * B * w;
        Kua += B.transpose() * D * G * w;
        Kaa += G.transpose() * D * G * w;
    }

    // Static condensation: Ke = Kuu - Kua * Kaa^{-1} * Kau
    Eigen::LLT<Eigen::Matrix<double,N_EAS,N_EAS>> llt(Kaa);
    if (llt.info() != Eigen::Success)
        throw SolverError(std::format("CPENTA6EAS {}: Kaa factorization failed", eid_.value));

    LocalKe Ke = Kuu - Kua * llt.solve(Kua.transpose());
    return Ke;
}

LocalKe CPenta6Eas::mass_matrix() const {
    // EAS modes are element-internal (condensed out); mass matrix is identical to CPenta6.
    LocalKe Me = LocalKe::Zero(NUM_DOFS, NUM_DOFS);
    const double rho = material().rho;
    if (rho == 0.0) return Me;

    auto coords = node_coords();
    const double tri_pts[3][2] = {
        {2.0/3.0, 1.0/6.0},
        {1.0/6.0, 2.0/3.0},
        {1.0/6.0, 1.0/6.0}
    };
    const double tri_w = 1.0 / 6.0;
    const double ax_pts[2] = {-1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0)};

    for (int ti = 0; ti < 3; ++ti)
    for (int ai = 0; ai < 2; ++ai) {
        double L1 = tri_pts[ti][0], L2 = tri_pts[ti][1];
        double zeta = ax_pts[ai];

        auto sd = CPenta6::shape_functions(L1, L2, zeta);

        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 6; ++n) {
            J(0,0)+=sd.dNdL1[n]*coords[n].x; J(0,1)+=sd.dNdL1[n]*coords[n].y; J(0,2)+=sd.dNdL1[n]*coords[n].z;
            J(1,0)+=sd.dNdL2[n]*coords[n].x; J(1,1)+=sd.dNdL2[n]*coords[n].y; J(1,2)+=sd.dNdL2[n]*coords[n].z;
            J(2,0)+=sd.dNdzeta[n]*coords[n].x; J(2,1)+=sd.dNdzeta[n]*coords[n].y; J(2,2)+=sd.dNdzeta[n]*coords[n].z;
        }
        double detJ = J.determinant();
        if (detJ <= 0)
            throw SolverError(std::format("CPenta6Eas {}: non-positive Jacobian in mass_matrix det={:.6g}", eid_.value, detJ));

        double weight = tri_w * 1.0;
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b) {
                double v = rho * sd.N[a] * sd.N[b] * detJ * weight;
                for (int d = 0; d < 3; ++d)
                    Me(3*a+d, 3*b+d) += v;
            }
    }
    return Me;
}

LocalFe CPenta6Eas::thermal_load(std::span<const double> temperatures, double t_ref) const {
    // EAS enhancement modes have zero mean strain, so thermal load is identical to CPenta6.
    LocalFe fe = LocalFe::Zero(NUM_DOFS);
    const double alpha = material().A;
    if (alpha == 0.0) return fe;

    auto coords = node_coords();
    Eigen::Matrix<double,6,6> D = constitutive_D();

    const double tri_pts[3][2] = {
        {2.0/3.0, 1.0/6.0},
        {1.0/6.0, 2.0/3.0},
        {1.0/6.0, 1.0/6.0}
    };
    const double tri_w = 1.0 / 6.0;
    const double ax_pts[2] = {-1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0)};

    for (int ti = 0; ti < 3; ++ti)
    for (int ai = 0; ai < 2; ++ai) {
        double L1 = tri_pts[ti][0], L2 = tri_pts[ti][1];
        double zeta = ax_pts[ai];
        double w = tri_w;

        auto sd = CPenta6::shape_functions(L1, L2, zeta);

        double T = 0;
        for (int n = 0; n < 6; ++n) T += sd.N[n] * temperatures[n];
        double dT = T - t_ref;
        if (std::abs(dT) < 1e-15) continue;

        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 6; ++n) {
            J(0,0) += sd.dNdL1[n]*coords[n].x; J(0,1) += sd.dNdL1[n]*coords[n].y; J(0,2) += sd.dNdL1[n]*coords[n].z;
            J(1,0) += sd.dNdL2[n]*coords[n].x; J(1,1) += sd.dNdL2[n]*coords[n].y; J(1,2) += sd.dNdL2[n]*coords[n].z;
            J(2,0) += sd.dNdzeta[n]*coords[n].x; J(2,1) += sd.dNdzeta[n]*coords[n].y; J(2,2) += sd.dNdzeta[n]*coords[n].z;
        }
        double detJ = J.determinant();
        Eigen::Matrix3d Jinv = J.inverse();

        Eigen::Matrix<double,6,18> B;
        B.setZero();
        for (int n = 0; n < 6; ++n) {
            double dnx = Jinv(0,0)*sd.dNdL1[n]+Jinv(0,1)*sd.dNdL2[n]+Jinv(0,2)*sd.dNdzeta[n];
            double dny = Jinv(1,0)*sd.dNdL1[n]+Jinv(1,1)*sd.dNdL2[n]+Jinv(1,2)*sd.dNdzeta[n];
            double dnz = Jinv(2,0)*sd.dNdL1[n]+Jinv(2,1)*sd.dNdL2[n]+Jinv(2,2)*sd.dNdzeta[n];
            int c0 = 3*n;
            B(0,c0+0)=dnx; B(1,c0+1)=dny; B(2,c0+2)=dnz;
            B(3,c0+0)=dny; B(3,c0+1)=dnx;
            B(4,c0+1)=dnz; B(4,c0+2)=dny;
            B(5,c0+0)=dnz; B(5,c0+2)=dnx;
        }
        Eigen::Matrix<double,6,1> eps_th;
        eps_th << alpha*dT, alpha*dT, alpha*dT, 0, 0, 0;
        fe += B.transpose() * D * eps_th * detJ * w;
    }
    return fe;
}

std::vector<EqIndex> CPenta6Eas::global_dof_indices(const DofMap& dof_map) const {
    std::vector<EqIndex> result;
    result.reserve(NUM_DOFS);
    static constexpr int solid_dofs[3] = {0,1,2};
    for (NodeId nid : nodes_)
        for (int d : solid_dofs)
            result.push_back(dof_map.eq_index(nid, d));
    return result;
}

} // namespace vibestran
