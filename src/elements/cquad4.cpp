// src/elements/cquad4.cpp
// CQUAD4 isoparametric quadrilateral shell element.
// Membrane: Q4 bilinear plane-stress, 2x2 Gauss
// Bending: Mindlin-Reissner DKQ, 2x2 Gauss
// The element frame is aligned with the element plane.

#include "elements/cquad4.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <format>

namespace nastran {

// Gauss points for 2x2 quadrature
static constexpr double GP2 = 1.0 / std::numbers::sqrt3; // ≈ 0.5773502692
static const double GAUSS2[2] = {-GP2, GP2};
static const double GAUSS2_W[2] = {1.0, 1.0};

CQuad4::CQuad4(ElementId eid, PropertyId pid,
               std::array<NodeId, 4> node_ids,
               const Model& model)
    : eid_(eid), pid_(pid), nodes_(node_ids), model_(model) {}

const PShell& CQuad4::pshell() const {
    const auto& prop = model_.property(pid_);
    if (!std::holds_alternative<PShell>(prop))
        throw SolverError(std::format("CQUAD4 {}: property {} is not PSHELL", eid_.value, pid_.value));
    return std::get<PShell>(prop);
}

const Mat1& CQuad4::material() const {
    return model_.material(pshell().mid1);
}

double CQuad4::thickness() const {
    return pshell().t;
}

std::array<Vec3, 4> CQuad4::node_coords() const {
    std::array<Vec3, 4> coords;
    for (int i = 0; i < 4; ++i)
        coords[i] = model_.node(nodes_[i]).position;
    return coords;
}

CQuad4::ShapeData CQuad4::shape_functions(double xi, double eta) noexcept {
    ShapeData s;
    // Q4 bilinear shape functions
    s.N[0] = 0.25 * (1-xi) * (1-eta);
    s.N[1] = 0.25 * (1+xi) * (1-eta);
    s.N[2] = 0.25 * (1+xi) * (1+eta);
    s.N[3] = 0.25 * (1-xi) * (1+eta);

    s.dNdxi[0]  = -0.25 * (1-eta);
    s.dNdxi[1]  =  0.25 * (1-eta);
    s.dNdxi[2]  =  0.25 * (1+eta);
    s.dNdxi[3]  = -0.25 * (1+eta);

    s.dNdeta[0] = -0.25 * (1-xi);
    s.dNdeta[1] = -0.25 * (1+xi);
    s.dNdeta[2] =  0.25 * (1+xi);
    s.dNdeta[3] =  0.25 * (1-xi);
    return s;
}

Eigen::Matrix3d CQuad4::membrane_D() const {
    const Mat1& mat = material();
    double E  = mat.E;
    double nu = mat.nu;
    double c  = E / (1.0 - nu*nu);
    Eigen::Matrix3d D;
    D << c,    c*nu, 0,
         c*nu, c,    0,
         0,    0,    c*(1-nu)/2.0;
    return D;
}

Eigen::Matrix3d CQuad4::bending_D() const {
    // Same constitutive law, scaled by t^3/12
    double t = thickness();
    return (t*t*t / 12.0) * membrane_D();
}

LocalKe CQuad4::stiffness_matrix() const {
    LocalKe Ke = LocalKe::Zero(NUM_DOFS, NUM_DOFS);
    auto coords = node_coords();
    const double t = thickness();

    Eigen::Matrix3d Dm = membrane_D();
    Eigen::Matrix3d Db = bending_D();

    // Selective Reduced Integration for membrane:
    // Normal strains (ε_xx, ε_yy) use full 2x2 Gauss to prevent hourglass modes.
    // Shear strain (γ_xy) uses 1-point centroidal integration to eliminate membrane
    // shear locking in bending, which otherwise causes severe under-prediction of
    // in-plane bending deflections.
    Eigen::Matrix3d Dm_normal = Dm;
    Dm_normal(2,2) = 0.0;  // zero out shear-shear term for 2x2 integration
    Eigen::Matrix3d Dm_shear = Eigen::Matrix3d::Zero();
    Dm_shear(2,2) = Dm(2,2);  // only in-plane shear stiffness for centroidal integration

    // ── Membrane part (DOFs: u1,v1, u2,v2, u3,v3, u4,v4 → local indices 0,1,6,7,12,13,18,19)
    // We compute membrane and bending separately then overlay into 24x24 Ke.

    for (int gi = 0; gi < 2; ++gi) {
        for (int gj = 0; gj < 2; ++gj) {
            double xi  = GAUSS2[gi];
            double eta = GAUSS2[gj];
            double wi  = GAUSS2_W[gi];
            double wj  = GAUSS2_W[gj];

            auto sd = shape_functions(xi, eta);

            // Jacobian J = dX/d(xi,eta) [2x2]
            // X = sum N_I * x_I, Y = sum N_I * y_I
            Eigen::Matrix2d J = Eigen::Matrix2d::Zero();
            for (int n = 0; n < 4; ++n) {
                J(0,0) += sd.dNdxi[n]  * coords[n].x;
                J(0,1) += sd.dNdxi[n]  * coords[n].y;
                J(1,0) += sd.dNdeta[n] * coords[n].x;
                J(1,1) += sd.dNdeta[n] * coords[n].y;
            }
            double detJ = J.determinant();
            if (detJ <= 0)
                throw SolverError(std::format("CQUAD4 {}: negative Jacobian det={:.6g}", eid_.value, detJ));
            Eigen::Matrix2d Jinv = J.inverse();

            // Shape fn physical derivatives
            // [dN/dx; dN/dy] = Jinv * [dN/dxi; dN/deta]
            Eigen::MatrixXd dNdx(2, 4);
            for (int n = 0; n < 4; ++n) {
                dNdx(0,n) = Jinv(0,0)*sd.dNdxi[n] + Jinv(0,1)*sd.dNdeta[n];
                dNdx(1,n) = Jinv(1,0)*sd.dNdxi[n] + Jinv(1,1)*sd.dNdeta[n];
            }

            // ── Membrane strain-displacement B_m [3x8]
            Eigen::MatrixXd Bm(3, 8);
            Bm.setZero();
            for (int n = 0; n < 4; ++n) {
                Bm(0, 2*n)   = dNdx(0,n);  // ε_xx = du/dx
                Bm(1, 2*n+1) = dNdx(1,n);  // ε_yy = dv/dy
                Bm(2, 2*n)   = dNdx(1,n);  // γ_xy = du/dy + dv/dx
                Bm(2, 2*n+1) = dNdx(0,n);
            }

            // Membrane normal stiffness contribution (2x2 Gauss, no shear)
            Eigen::MatrixXd Km_contrib = t * Bm.transpose() * Dm_normal * Bm * detJ * wi * wj;

            // Map membrane DOFs into global 24-DOF Ke
            // Node i has DOFs: u=6i+0, v=6i+1, w=6i+2, θx=6i+3, θy=6i+4, θz=6i+5
            // Membrane uses u,v (DOFs 0,1 per node)
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    Ke(6*i+0, 6*j+0) += Km_contrib(2*i+0, 2*j+0);
                    Ke(6*i+0, 6*j+1) += Km_contrib(2*i+0, 2*j+1);
                    Ke(6*i+1, 6*j+0) += Km_contrib(2*i+1, 2*j+0);
                    Ke(6*i+1, 6*j+1) += Km_contrib(2*i+1, 2*j+1);
                }
            }

            // ── Bending (Mindlin): uses w, θx, θy (DOFs 2,3,4 per node)
            // Bending curvature: κ = [∂θx/∂x, ∂θy/∂y, ∂θx/∂y + ∂θy/∂x]
            // Here θx = rotation about x axis, θy = rotation about y axis
            Eigen::MatrixXd Bb(3, 12); // 3 curvature strains x 4 nodes x 3 bending DOF
            Bb.setZero();
            for (int n = 0; n < 4; ++n) {
                // DOF layout per node in bending subspace: θx(0), θy(1), w(2)
                Bb(0, 3*n+0) = dNdx(0,n);  // ∂θx/∂x
                Bb(1, 3*n+1) = dNdx(1,n);  // ∂θy/∂y
                Bb(2, 3*n+0) = dNdx(1,n);  // ∂θx/∂y + ∂θy/∂x
                Bb(2, 3*n+1) = dNdx(0,n);
            }

            Eigen::MatrixXd Kb_contrib = Bb.transpose() * Db * Bb * detJ * wi * wj;

            // Map bending DOFs (w=2, θx=3, θy=4 per node) into global Ke
            // Local bending dof for node i: θx=3*i+0, θy=3*i+1, (w not in Bb above)
            // Ke structure: node i: [u=0,v=1,w=2,θx=3,θy=4,θz=5]
            // Bending DOF mapping: bending_dof 3*i+0 → Ke row 6*i+3, 3*i+1 → 6*i+4
            for (int i = 0; i < 4; ++i) {
                int ki_x = 6*i+3, ki_y = 6*i+4;
                for (int j = 0; j < 4; ++j) {
                    int kj_x = 6*j+3, kj_y = 6*j+4;
                    Ke(ki_x, kj_x) += Kb_contrib(3*i+0, 3*j+0);
                    Ke(ki_x, kj_y) += Kb_contrib(3*i+0, 3*j+1);
                    Ke(ki_y, kj_x) += Kb_contrib(3*i+1, 3*j+0);
                    Ke(ki_y, kj_y) += Kb_contrib(3*i+1, 3*j+1);
                }
            }

            // ── Transverse shear: included at full 2x2
            // Note: reduced integration for thin-plate transverse shear (Mindlin Ks) is a
            // future improvement; for current tests (in-plane bending) Ks is not exercised.
            // Shear strains: γ_xz = dw/dx - θx, γ_yz = dw/dy - θy
            // (Mindlin assumption)
            double kappa  = pshell().tst; // shear correction factor (~5/6)
            double G      = material().G > 0 ? material().G
                                             : material().E / (2*(1+material().nu));
            double Gts    = kappa * G * t;

            Eigen::MatrixXd Bs(2, 12); // 2 shear strains x 12 DOF (w,θx,θy for 4 nodes)
            Bs.setZero();
            for (int n = 0; n < 4; ++n) {
                // shear DOF per node local: 0=θx, 1=θy, 2=w
                Bs(0, 3*n+2) = dNdx(0,n);    // dw/dx
                Bs(0, 3*n+0) = -sd.N[n];     // -θx
                Bs(1, 3*n+2) = dNdx(1,n);    // dw/dy
                Bs(1, 3*n+1) = -sd.N[n];     // -θy
            }

            Eigen::Matrix2d Ds_mat;
            Ds_mat << Gts, 0, 0, Gts;

            Eigen::MatrixXd Ks_contrib = Bs.transpose() * Ds_mat * Bs * detJ * wi * wj;

            // Map: θx→3, θy→4, w→2 per node
            const int bmap[3] = {3, 4, 2}; // local bend dof → Ke dof offset
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            Ke(6*i+bmap[a], 6*j+bmap[b]) += Ks_contrib(3*i+a, 3*j+b);
                        }
                    }
                }
            }
        }
    }

    // ── Membrane shear SRI: 1-point centroidal integration for γ_xy term.
    // Evaluated at (xi=0,eta=0) with quadrature weight 4.0 (full reference element area).
    {
        auto sd0 = shape_functions(0.0, 0.0);
        Eigen::Matrix2d J0 = Eigen::Matrix2d::Zero();
        for (int n = 0; n < 4; ++n) {
            J0(0,0) += sd0.dNdxi[n]  * coords[n].x;
            J0(0,1) += sd0.dNdxi[n]  * coords[n].y;
            J0(1,0) += sd0.dNdeta[n] * coords[n].x;
            J0(1,1) += sd0.dNdeta[n] * coords[n].y;
        }
        double detJ0 = J0.determinant();
        Eigen::Matrix2d Jinv0 = J0.inverse();

        Eigen::MatrixXd dNdx0(2, 4);
        for (int n = 0; n < 4; ++n) {
            dNdx0(0,n) = Jinv0(0,0)*sd0.dNdxi[n] + Jinv0(0,1)*sd0.dNdeta[n];
            dNdx0(1,n) = Jinv0(1,0)*sd0.dNdxi[n] + Jinv0(1,1)*sd0.dNdeta[n];
        }

        Eigen::MatrixXd Bm0(3, 8);
        Bm0.setZero();
        for (int n = 0; n < 4; ++n) {
            Bm0(0, 2*n)   = dNdx0(0,n);
            Bm0(1, 2*n+1) = dNdx0(1,n);
            Bm0(2, 2*n)   = dNdx0(1,n);
            Bm0(2, 2*n+1) = dNdx0(0,n);
        }

        Eigen::MatrixXd Km_shear = t * Bm0.transpose() * Dm_shear * Bm0 * detJ0 * 4.0;

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                Ke(6*i+0, 6*j+0) += Km_shear(2*i+0, 2*j+0);
                Ke(6*i+0, 6*j+1) += Km_shear(2*i+0, 2*j+1);
                Ke(6*i+1, 6*j+0) += Km_shear(2*i+1, 2*j+0);
                Ke(6*i+1, 6*j+1) += Km_shear(2*i+1, 2*j+1);
            }
        }
    }

    // Add small drilling stiffness (θz) to prevent singularity
    // This is a common numerical stabilization for shells
    double drill_stiff = 1e-6 * Ke.diagonal().maxCoeff();
    if (drill_stiff < 1e-10) drill_stiff = 1.0;
    for (int i = 0; i < 4; ++i)
        Ke(6*i+5, 6*i+5) += drill_stiff;

    return Ke;
}

LocalFe CQuad4::thermal_load(std::span<const double> temperatures, double t_ref) const {
    LocalFe fe = LocalFe::Zero(NUM_DOFS);
    auto coords = node_coords();
    const double thickness = this->thickness();
    const Mat1& mat = material();
    const double alpha = mat.A;
    if (alpha == 0.0) return fe; // no thermal expansion

    Eigen::Matrix3d Dm = membrane_D();

    // Thermal strain vector: ε_th = α * ΔT * {1, 1, 0}
    // fe = ∫ Bᵀ * D * ε_th dA * t
    for (int gi = 0; gi < 2; ++gi) {
        for (int gj = 0; gj < 2; ++gj) {
            double xi  = GAUSS2[gi];
            double eta = GAUSS2[gj];
            double wi  = GAUSS2_W[gi];
            double wj  = GAUSS2_W[gj];

            auto sd = shape_functions(xi, eta);

            // Temperature at this point
            double T = 0;
            for (int n = 0; n < 4; ++n) T += sd.N[n] * temperatures[n];
            double dT = T - t_ref;
            if (std::abs(dT) < 1e-15) continue;

            // Jacobian
            Eigen::Matrix2d J = Eigen::Matrix2d::Zero();
            for (int n = 0; n < 4; ++n) {
                J(0,0) += sd.dNdxi[n]  * coords[n].x;
                J(0,1) += sd.dNdxi[n]  * coords[n].y;
                J(1,0) += sd.dNdeta[n] * coords[n].x;
                J(1,1) += sd.dNdeta[n] * coords[n].y;
            }
            double detJ = J.determinant();
            Eigen::Matrix2d Jinv = J.inverse();

            Eigen::MatrixXd dNdx(2, 4);
            for (int n = 0; n < 4; ++n) {
                dNdx(0,n) = Jinv(0,0)*sd.dNdxi[n] + Jinv(0,1)*sd.dNdeta[n];
                dNdx(1,n) = Jinv(1,0)*sd.dNdxi[n] + Jinv(1,1)*sd.dNdeta[n];
            }

            Eigen::MatrixXd Bm(3, 8);
            Bm.setZero();
            for (int n = 0; n < 4; ++n) {
                Bm(0, 2*n)   = dNdx(0,n);
                Bm(1, 2*n+1) = dNdx(1,n);
                Bm(2, 2*n)   = dNdx(1,n);
                Bm(2, 2*n+1) = dNdx(0,n);
            }

            Eigen::Vector3d eps_th(alpha*dT, alpha*dT, 0);
            Eigen::VectorXd fe_mem = thickness * Bm.transpose() * Dm * eps_th * detJ * wi * wj;

            // Map into full 24-DOF vector (u,v components only)
            for (int n = 0; n < 4; ++n) {
                fe(6*n+0) += fe_mem(2*n+0);
                fe(6*n+1) += fe_mem(2*n+1);
            }
        }
    }
    return fe;
}

std::vector<EqIndex> CQuad4::global_dof_indices(const DofMap& dof_map) const {
    std::vector<EqIndex> result;
    result.reserve(NUM_DOFS);
    for (NodeId nid : nodes_) {
        const auto& blk = dof_map.block(nid);
        for (int d = 0; d < 6; ++d)
            result.push_back(blk.eq[d]);
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CQuad4Mitc4
// ═══════════════════════════════════════════════════════════════════════════════

CQuad4Mitc4::CQuad4Mitc4(ElementId eid, PropertyId pid,
                          std::array<NodeId, 4> node_ids,
                          const Model& model)
    : eid_(eid), pid_(pid), nodes_(node_ids), model_(model) {}

const PShell& CQuad4Mitc4::pshell() const {
    const auto& prop = model_.property(pid_);
    if (!std::holds_alternative<PShell>(prop))
        throw SolverError(std::format("CQuad4Mitc4 {}: property {} is not PSHELL", eid_.value, pid_.value));
    return std::get<PShell>(prop);
}

const Mat1& CQuad4Mitc4::material() const {
    return model_.material(pshell().mid1);
}

double CQuad4Mitc4::thickness() const { return pshell().t; }

std::array<Vec3, 4> CQuad4Mitc4::node_coords() const {
    std::array<Vec3, 4> coords;
    for (int i = 0; i < 4; ++i)
        coords[i] = model_.node(nodes_[i]).position;
    return coords;
}

Eigen::Matrix3d CQuad4Mitc4::membrane_D() const {
    const Mat1& mat = material();
    double E = mat.E, nu = mat.nu;
    double c = E / (1.0 - nu*nu);
    Eigen::Matrix3d D;
    D << c, c*nu, 0, c*nu, c, 0, 0, 0, c*(1-nu)/2.0;
    return D;
}

Eigen::Matrix3d CQuad4Mitc4::bending_D() const {
    double t = thickness();
    return (t*t*t / 12.0) * membrane_D();
}

LocalKe CQuad4Mitc4::stiffness_matrix() const {
    // Membrane + bending: identical to CQuad4.
    // Transverse shear: MITC4 tying-point interpolation (Bathe & Dvorkin 1985).

    LocalKe Ke = LocalKe::Zero(NUM_DOFS, NUM_DOFS);
    auto coords = node_coords();
    const double t = thickness();
    double kappa = pshell().tst;
    const Mat1& mat = material();
    double G = mat.G > 0 ? mat.G : mat.E / (2*(1+mat.nu));
    double Gts = kappa * G * t;

    Eigen::Matrix3d Dm = membrane_D();
    Eigen::Matrix3d Db = bending_D();

    // Selective reduced integration for membrane (same as CQuad4)
    Eigen::Matrix3d Dm_normal = Dm; Dm_normal(2,2) = 0.0;
    Eigen::Matrix3d Dm_shear = Eigen::Matrix3d::Zero(); Dm_shear(2,2) = Dm(2,2);

    // ── MITC4 tying point shear strains ──────────────────────────────────────
    // Tying points A (xi=0,eta=-1), B (xi=1,eta=0), C (xi=0,eta=1), D (xi=-1,eta=0)
    // γ_ξζ(ξ,η) = (1-η)/2 * γ_A + (1+η)/2 * γ_C
    // γ_ηζ(ξ,η) = (1-ξ)/2 * γ_D + (1+ξ)/2 * γ_B
    // Each tying-point shear strain is:
    //   γ_A = dw/dξ|_A * dξ/dx - θx * N evaluated at A ... etc.
    // We build Bs_mitc [2×12] (shear strains in nat coords at integration point)
    // then transform to physical strains.

    // Helper: compute covariant shear strains at a tying point (xi0, eta0)
    // Returns [γ_ξζ, γ_ηζ] in covariant (natural) coordinates
    // γ_ξζ = sum_n(dN/dξ * w_n) - sum_n(N_n * θx_n * g_ξ·ex + N_n * θy_n * g_ξ·ey)
    // For flat element: γ_ξζ_cov = sum_n[ dN/dξ * w_n - N_n*(Jξx*θx_n + Jξy*θy_n) ]
    //   where g_ξ = [Jξx, Jξy, Jξz]
    // Local DOF order in shear subspace: [θx,θy,w] per node → indices 0,1,2

    auto covariant_shear = [&](double xi0, double eta0)
        -> std::pair<Eigen::Matrix<double,1,12>, Eigen::Matrix<double,1,12>> {
        auto sd = CQuad4::shape_functions(xi0, eta0);
        // Jacobian tangent vectors g_ξ and g_η
        double gxi_x=0,gxi_y=0, geta_x=0,geta_y=0;
        for (int n = 0; n < 4; ++n) {
            gxi_x  += sd.dNdxi[n]  * coords[n].x;
            gxi_y  += sd.dNdxi[n]  * coords[n].y;
            geta_x += sd.dNdeta[n] * coords[n].x;
            geta_y += sd.dNdeta[n] * coords[n].y;
        }
        // γ_ξζ_cov[n] = dNdξ_n * w - N_n*(gxi_x*θx + gxi_y*θy)
        // γ_ηζ_cov[n] = dNdη_n * w - N_n*(geta_x*θx + geta_y*θy)
        // Local shear DOF per node: [θx=0, θy=1, w=2]
        Eigen::Matrix<double,1,12> Bxi, Beta;
        Bxi.setZero(); Beta.setZero();
        for (int n = 0; n < 4; ++n) {
            Bxi( 0, 3*n+2) =  sd.dNdxi[n];
            Bxi( 0, 3*n+0) = -sd.N[n] * gxi_x;
            Bxi( 0, 3*n+1) = -sd.N[n] * gxi_y;
            Beta(0, 3*n+2) =  sd.dNdeta[n];
            Beta(0, 3*n+0) = -sd.N[n] * geta_x;
            Beta(0, 3*n+1) = -sd.N[n] * geta_y;
        }
        return {Bxi, Beta};
    };

    // Evaluate covariant shear rows at 4 tying points
    auto [BxiA, BetaA] = covariant_shear( 0.0, -1.0); // A
    auto [BxiC, BetaC] = covariant_shear( 0.0,  1.0); // C
    auto [BxiB, BetaB] = covariant_shear( 1.0,  0.0); // B
    auto [BxiD, BetaD] = covariant_shear(-1.0,  0.0); // D

    // ── Membrane and bending: 2×2 Gauss ──────────────────────────────────────
    for (int gi = 0; gi < 2; ++gi) {
    for (int gj = 0; gj < 2; ++gj) {
        double xi  = GAUSS2[gi], eta = GAUSS2[gj];
        double wi  = GAUSS2_W[gi], wj = GAUSS2_W[gj];

        auto sd = CQuad4::shape_functions(xi, eta);

        Eigen::Matrix2d J = Eigen::Matrix2d::Zero();
        for (int n = 0; n < 4; ++n) {
            J(0,0) += sd.dNdxi[n]  * coords[n].x;
            J(0,1) += sd.dNdxi[n]  * coords[n].y;
            J(1,0) += sd.dNdeta[n] * coords[n].x;
            J(1,1) += sd.dNdeta[n] * coords[n].y;
        }
        double detJ = J.determinant();
        if (detJ <= 0)
            throw SolverError(std::format("CQuad4Mitc4 {}: negative Jacobian det={:.6g}", eid_.value, detJ));
        Eigen::Matrix2d Jinv = J.inverse();

        Eigen::MatrixXd dNdx(2,4);
        for (int n = 0; n < 4; ++n) {
            dNdx(0,n) = Jinv(0,0)*sd.dNdxi[n] + Jinv(0,1)*sd.dNdeta[n];
            dNdx(1,n) = Jinv(1,0)*sd.dNdxi[n] + Jinv(1,1)*sd.dNdeta[n];
        }

        // Membrane [3×8]
        Eigen::MatrixXd Bm(3,8); Bm.setZero();
        for (int n = 0; n < 4; ++n) {
            Bm(0,2*n)  =dNdx(0,n); Bm(1,2*n+1)=dNdx(1,n);
            Bm(2,2*n)  =dNdx(1,n); Bm(2,2*n+1)=dNdx(0,n);
        }
        Eigen::MatrixXd Km_n = t * Bm.transpose() * Dm_normal * Bm * detJ * wi * wj;
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
            Ke(6*i+0,6*j+0)+=Km_n(2*i,2*j); Ke(6*i+0,6*j+1)+=Km_n(2*i,2*j+1);
            Ke(6*i+1,6*j+0)+=Km_n(2*i+1,2*j); Ke(6*i+1,6*j+1)+=Km_n(2*i+1,2*j+1);
        }

        // Bending [3×12]
        Eigen::MatrixXd Bb(3,12); Bb.setZero();
        for (int n = 0; n < 4; ++n) {
            Bb(0,3*n+0)=dNdx(0,n); Bb(1,3*n+1)=dNdx(1,n);
            Bb(2,3*n+0)=dNdx(1,n); Bb(2,3*n+1)=dNdx(0,n);
        }
        Eigen::MatrixXd Kb = Bb.transpose() * Db * Bb * detJ * wi * wj;
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
            Ke(6*i+3,6*j+3)+=Kb(3*i+0,3*j+0); Ke(6*i+3,6*j+4)+=Kb(3*i+0,3*j+1);
            Ke(6*i+4,6*j+3)+=Kb(3*i+1,3*j+0); Ke(6*i+4,6*j+4)+=Kb(3*i+1,3*j+1);
        }

        // ── MITC4 transverse shear ────────────────────────────────────────
        // Interpolate tying-point covariant shear strains at (xi, eta)
        // γ_ξζ = (1-eta)/2 * row_A + (1+eta)/2 * row_C  [1×12]
        // γ_ηζ = (1-xi)/2  * row_D + (1+xi)/2  * row_B  [1×12]
        Eigen::Matrix<double,1,12> Bcov_xi  = 0.5*(1-eta)*BxiA + 0.5*(1+eta)*BxiC;
        Eigen::Matrix<double,1,12> Bcov_eta = 0.5*(1-xi)*BetaD  + 0.5*(1+xi)*BetaB;

        // Transform covariant → physical shear strains
        // [γ_xz, γ_yz]^T = J_inv^T * [γ_ξζ, γ_ηζ]^T  (2D metric)
        // Physical shear: γ_phys = Jinv * γ_cov
        // Both rows give a 1×12 B; stack into [2×12]
        Eigen::Matrix<double,2,12> Bs_cov;
        Bs_cov.row(0) = Bcov_xi;
        Bs_cov.row(1) = Bcov_eta;

        // Physical: Bs_phys = Jinv * Bs_cov
        Eigen::Matrix<double,2,12> Bs = Jinv * Bs_cov;

        Eigen::Matrix2d Ds_mat;
        Ds_mat << Gts, 0, 0, Gts;

        Eigen::Matrix<double,12,12> Ks = Bs.transpose() * Ds_mat * Bs * detJ * wi * wj;

        // Map: θx→3, θy→4, w→2 per node
        const int bmap[3] = {3, 4, 2};
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
            for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b)
                Ke(6*i+bmap[a], 6*j+bmap[b]) += Ks(3*i+a, 3*j+b);
    }}

    // Membrane shear SRI at centroid
    {
        auto sd0 = CQuad4::shape_functions(0.0, 0.0);
        Eigen::Matrix2d J0 = Eigen::Matrix2d::Zero();
        for (int n = 0; n < 4; ++n) {
            J0(0,0) += sd0.dNdxi[n]*coords[n].x; J0(0,1) += sd0.dNdxi[n]*coords[n].y;
            J0(1,0) += sd0.dNdeta[n]*coords[n].x; J0(1,1) += sd0.dNdeta[n]*coords[n].y;
        }
        double detJ0 = J0.determinant();
        Eigen::Matrix2d Jinv0 = J0.inverse();
        Eigen::MatrixXd dNdx0(2,4);
        for (int n = 0; n < 4; ++n) {
            dNdx0(0,n) = Jinv0(0,0)*sd0.dNdxi[n] + Jinv0(0,1)*sd0.dNdeta[n];
            dNdx0(1,n) = Jinv0(1,0)*sd0.dNdxi[n] + Jinv0(1,1)*sd0.dNdeta[n];
        }
        Eigen::MatrixXd Bm0(3,8); Bm0.setZero();
        for (int n = 0; n < 4; ++n) {
            Bm0(0,2*n)=dNdx0(0,n); Bm0(1,2*n+1)=dNdx0(1,n);
            Bm0(2,2*n)=dNdx0(1,n); Bm0(2,2*n+1)=dNdx0(0,n);
        }
        Eigen::MatrixXd Km_shear = t * Bm0.transpose() * Dm_shear * Bm0 * detJ0 * 4.0;
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
            Ke(6*i+0,6*j+0)+=Km_shear(2*i,2*j); Ke(6*i+0,6*j+1)+=Km_shear(2*i,2*j+1);
            Ke(6*i+1,6*j+0)+=Km_shear(2*i+1,2*j); Ke(6*i+1,6*j+1)+=Km_shear(2*i+1,2*j+1);
        }
    }

    // Drilling stabilization
    double drill_stiff = 1e-6 * Ke.diagonal().maxCoeff();
    if (drill_stiff < 1e-10) drill_stiff = 1.0;
    for (int i = 0; i < 4; ++i)
        Ke(6*i+5, 6*i+5) += drill_stiff;

    return Ke;
}

LocalFe CQuad4Mitc4::thermal_load(std::span<const double> temperatures, double t_ref) const {
    // Identical to CQuad4 — MITC4 only changes transverse shear stiffness, not thermal load.
    LocalFe fe = LocalFe::Zero(NUM_DOFS);
    auto coords = node_coords();
    const double th = thickness();
    const Mat1& mat = material();
    const double alpha = mat.A;
    if (alpha == 0.0) return fe;

    Eigen::Matrix3d Dm = membrane_D();

    for (int gi = 0; gi < 2; ++gi) {
    for (int gj = 0; gj < 2; ++gj) {
        double xi=GAUSS2[gi], eta=GAUSS2[gj], wi=GAUSS2_W[gi], wj=GAUSS2_W[gj];
        auto sd = CQuad4::shape_functions(xi, eta);

        double T = 0;
        for (int n = 0; n < 4; ++n) T += sd.N[n] * temperatures[n];
        double dT = T - t_ref;
        if (std::abs(dT) < 1e-15) continue;

        Eigen::Matrix2d J = Eigen::Matrix2d::Zero();
        for (int n = 0; n < 4; ++n) {
            J(0,0)+=sd.dNdxi[n]*coords[n].x; J(0,1)+=sd.dNdxi[n]*coords[n].y;
            J(1,0)+=sd.dNdeta[n]*coords[n].x; J(1,1)+=sd.dNdeta[n]*coords[n].y;
        }
        double detJ = J.determinant();
        Eigen::Matrix2d Jinv = J.inverse();

        Eigen::MatrixXd dNdx(2,4);
        for (int n = 0; n < 4; ++n) {
            dNdx(0,n)=Jinv(0,0)*sd.dNdxi[n]+Jinv(0,1)*sd.dNdeta[n];
            dNdx(1,n)=Jinv(1,0)*sd.dNdxi[n]+Jinv(1,1)*sd.dNdeta[n];
        }
        Eigen::MatrixXd Bm(3,8); Bm.setZero();
        for (int n = 0; n < 4; ++n) {
            Bm(0,2*n)=dNdx(0,n); Bm(1,2*n+1)=dNdx(1,n);
            Bm(2,2*n)=dNdx(1,n); Bm(2,2*n+1)=dNdx(0,n);
        }
        Eigen::Vector3d eps_th(alpha*dT, alpha*dT, 0);
        Eigen::VectorXd fe_mem = th * Bm.transpose() * Dm * eps_th * detJ * wi * wj;
        for (int n = 0; n < 4; ++n) {
            fe(6*n+0) += fe_mem(2*n+0);
            fe(6*n+1) += fe_mem(2*n+1);
        }
    }}
    return fe;
}

std::vector<EqIndex> CQuad4Mitc4::global_dof_indices(const DofMap& dof_map) const {
    std::vector<EqIndex> result;
    result.reserve(NUM_DOFS);
    for (NodeId nid : nodes_) {
        const auto& blk = dof_map.block(nid);
        for (int d = 0; d < 6; ++d)
            result.push_back(blk.eq[d]);
    }
    return result;
}

} // namespace nastran
