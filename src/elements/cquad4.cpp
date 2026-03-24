// src/elements/cquad4.cpp

#include "elements/cquad4.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <format>

namespace vibestran {

namespace {

static constexpr double GP2 = 1.0 / std::numbers::sqrt3;
static constexpr double GAUSS2[2] = {-GP2, GP2};
static constexpr double GAUSS2_W[2] = {1.0, 1.0};
static constexpr double DRILL_INERTIA_SCALE = 1e-4;
static constexpr std::array<int, 2> MEMBRANE_DOF_MAP{0, 1};
static constexpr std::array<int, 3> PLATE_DOF_MAP{3, 4, 2}; // [theta_x, theta_y, w]

struct ShellFrame {
  Vec3 e1, e2, e3;
  std::array<double, 4> xl{};
  std::array<double, 4> yl{};
  Eigen::Matrix<double, 24, 24> T{Eigen::Matrix<double, 24, 24>::Zero()};
};

struct QuadPointData {
  double xi{0.0};
  double eta{0.0};
  CQuad4::ShapeData shape;
  Eigen::Matrix2d J{Eigen::Matrix2d::Zero()};
  Eigen::Matrix2d Jinv{Eigen::Matrix2d::Zero()};
  Eigen::Matrix<double, 2, 4> dNdx{Eigen::Matrix<double, 2, 4>::Zero()};
  Eigen::Vector2d g_r{Eigen::Vector2d::Zero()};
  Eigen::Vector2d g_s{Eigen::Vector2d::Zero()};
  Eigen::Matrix2d contravariant{Eigen::Matrix2d::Zero()};
  double detJ{0.0};
};

struct ShellSection {
  Eigen::Matrix3d membrane_D{Eigen::Matrix3d::Zero()};
  Eigen::Matrix3d bending_material_D{Eigen::Matrix3d::Zero()};
  Eigen::Matrix3d coupling_D{Eigen::Matrix3d::Zero()};
  Eigen::Matrix2d shear_material_D{Eigen::Matrix2d::Zero()};

  Eigen::Matrix3d A{Eigen::Matrix3d::Zero()};
  Eigen::Matrix3d B{Eigen::Matrix3d::Zero()};
  Eigen::Matrix3d D{Eigen::Matrix3d::Zero()};
  Eigen::Matrix2d T{Eigen::Matrix2d::Zero()};

  Eigen::Vector3d membrane_alpha{Eigen::Vector3d::Zero()};
  double thickness{0.0};
  double mass_per_area{0.0};
  double rotary_mass_per_area{0.0};
};

using MembraneB = Eigen::Matrix<double, 3, 8>;
using BendingB = Eigen::Matrix<double, 3, 12>;
using ShearB = Eigen::Matrix<double, 2, 12>;
using Row8 = Eigen::Matrix<double, 1, 8>;
using Row12 = Eigen::Matrix<double, 1, 12>;

struct CovariantMembraneRows {
  Row8 rr{Row8::Zero()};
  Row8 ss{Row8::Zero()};
  Row8 rs{Row8::Zero()};
};

struct Mitc4PlusMembraneData {
  CovariantMembraneRows top;
  CovariantMembraneRows bottom;
  CovariantMembraneRows right;
  CovariantMembraneRows left;
  CovariantMembraneRows center;
  double a_top{0.0};
  double a_bottom{0.0};
  double a_right{0.0};
  double a_left{0.0};
  double a_center{0.0};
};

struct Mitc4PlusShearData {
  Row12 xi_bottom{Row12::Zero()};
  Row12 xi_top{Row12::Zero()};
  Row12 eta_right{Row12::Zero()};
  Row12 eta_left{Row12::Zero()};
  Row12 xi_center{Row12::Zero()};
  Row12 eta_center{Row12::Zero()};
};

const PShell &lookup_pshell(const Model &model, ElementId eid, PropertyId pid,
                            std::string_view owner) {
  const auto &prop = model.property(pid);
  if (!std::holds_alternative<PShell>(prop)) {
    throw SolverError(std::format("{} {}: property {} is not PSHELL", owner,
                                  eid.value, pid.value));
  }
  return std::get<PShell>(prop);
}

std::array<Vec3, 4> lookup_node_coords(const Model &model,
                                       const std::array<NodeId, 4> &nodes) {
  std::array<Vec3, 4> coords;
  for (int i = 0; i < 4; ++i)
    coords[i] = model.node(nodes[i]).position;
  return coords;
}

double compute_drilling_stiffness(const LocalKe &Ke) {
  double max_rot_diag = 0.0;
  for (int i = 0; i < 4; ++i) {
    max_rot_diag =
        std::max(max_rot_diag, std::abs(Ke(6 * i + 3, 6 * i + 3)));
    max_rot_diag =
        std::max(max_rot_diag, std::abs(Ke(6 * i + 4, 6 * i + 4)));
  }

  if (max_rot_diag > 0.0)
    return 1e-6 * max_rot_diag;

  double max_diag = Ke.diagonal().cwiseAbs().maxCoeff();
  if (max_diag < 1e-10) {
    throw SolverError(std::format(
        "CQUAD4: stiffness matrix diagonal is near-zero (max={:.3e}); "
        "check material, thickness, and geometry",
        max_diag));
  }
  return 1e-12 * max_diag;
}

ShellFrame compute_shell_frame(const std::array<Vec3, 4> &g) {
  ShellFrame fr;
  const Vec3 v12 = g[1] - g[0];
  const Vec3 v14 = g[3] - g[0];
  fr.e3 = v12.cross(v14).normalized();
  fr.e1 = v12.normalized();
  fr.e2 = fr.e3.cross(fr.e1);

  for (int n = 0; n < 4; ++n) {
    fr.xl[n] = g[n].dot(fr.e1);
    fr.yl[n] = g[n].dot(fr.e2);
  }

  Eigen::Matrix3d R;
  R << fr.e1.x, fr.e1.y, fr.e1.z, fr.e2.x, fr.e2.y, fr.e2.z, fr.e3.x, fr.e3.y,
      fr.e3.z;

  for (int n = 0; n < 4; ++n) {
    fr.T.template block<3, 3>(6 * n, 6 * n) = R;
    auto Tr = fr.T.template block<3, 3>(6 * n + 3, 6 * n + 3);
    Tr.row(0) = -R.row(1);
    Tr.row(1) = R.row(0);
    Tr.row(2) = R.row(2);
  }
  return fr;
}

Eigen::Matrix3d plane_stress_D(const Mat1 &mat) {
  const double E = mat.E;
  const double nu = mat.nu;
  const double c = E / (1.0 - nu * nu);
  Eigen::Matrix3d D;
  D << c, c * nu, 0.0, c * nu, c, 0.0, 0.0, 0.0, c * (1.0 - nu) / 2.0;
  return D;
}

Eigen::Matrix2d transverse_shear_D(const Mat1 &mat) {
  const double G =
      (mat.G > 0.0) ? mat.G : mat.E / (2.0 * (1.0 + mat.nu));
  Eigen::Matrix2d D = Eigen::Matrix2d::Zero();
  D(0, 0) = G;
  D(1, 1) = G;
  return D;
}

const Mat1 &resolve_pshell_material(const Model &model, ElementId eid,
                                    PropertyId pid, MaterialId preferred,
                                    MaterialId fallback,
                                    std::string_view role) {
  const MaterialId effective =
      (preferred.value != 0) ? preferred : fallback;
  if (effective.value == 0) {
    throw SolverError(std::format("CQUAD4 {}: PSHELL {} has no {} material",
                                  eid.value, pid.value, role));
  }
  return model.material(effective);
}

ShellSection build_shell_section(const Model &model, ElementId eid,
                                 PropertyId pid) {
  const PShell &ps = lookup_pshell(model, eid, pid, "CQUAD4");

  const Mat1 &membrane = model.material(ps.mid1);
  const Mat1 &bending =
      resolve_pshell_material(model, eid, pid, ps.mid2, ps.mid1, "bending");
  const MaterialId bending_mid =
      (ps.mid2.value != 0) ? ps.mid2 : ps.mid1;
  const Mat1 &shear = resolve_pshell_material(model, eid, pid, ps.mid3,
                                              bending_mid, "shear");

  ShellSection section;
  section.thickness = ps.t;
  section.membrane_D = plane_stress_D(membrane);
  section.bending_material_D = plane_stress_D(bending);
  section.shear_material_D = transverse_shear_D(shear);
  if (ps.mid4.value != 0)
    section.coupling_D = plane_stress_D(model.material(ps.mid4));

  const double ib = ps.twelveI_t3 * ps.t * ps.t * ps.t / 12.0;
  section.A = ps.t * section.membrane_D;
  section.B = ps.t * ps.t * section.coupling_D;
  section.D = ib * section.bending_material_D;
  section.T = (ps.tst * ps.t) * section.shear_material_D;
  section.membrane_alpha << membrane.A, membrane.A, 0.0;
  section.mass_per_area = membrane.rho * ps.t + ps.nsm;
  section.rotary_mass_per_area = membrane.rho * ps.t * ps.t * ps.t / 12.0;
  return section;
}

QuadPointData evaluate_quad_point(ElementId eid, const std::array<double, 4> &xl,
                                  const std::array<double, 4> &yl, double xi,
                                  double eta) {
  QuadPointData q;
  q.xi = xi;
  q.eta = eta;
  q.shape = CQuad4::shape_functions(xi, eta);

  for (int n = 0; n < 4; ++n) {
    q.J(0, 0) += q.shape.dNdxi[n] * xl[n];
    q.J(0, 1) += q.shape.dNdxi[n] * yl[n];
    q.J(1, 0) += q.shape.dNdeta[n] * xl[n];
    q.J(1, 1) += q.shape.dNdeta[n] * yl[n];
  }

  q.detJ = q.J.determinant();
  if (q.detJ <= 0.0) {
    throw SolverError(std::format("CQUAD4 {}: negative Jacobian det={:.6g}",
                                  eid.value, q.detJ));
  }

  q.Jinv = q.J.inverse();
  q.contravariant = q.Jinv;
  q.g_r << q.J(0, 0), q.J(0, 1);
  q.g_s << q.J(1, 0), q.J(1, 1);
  for (int n = 0; n < 4; ++n) {
    q.dNdx(0, n) =
        q.Jinv(0, 0) * q.shape.dNdxi[n] + q.Jinv(0, 1) * q.shape.dNdeta[n];
    q.dNdx(1, n) =
        q.Jinv(1, 0) * q.shape.dNdxi[n] + q.Jinv(1, 1) * q.shape.dNdeta[n];
  }
  return q;
}

MembraneB standard_membrane_B(const QuadPointData &q) {
  MembraneB B = MembraneB::Zero();
  for (int n = 0; n < 4; ++n) {
    B(0, 2 * n) = q.dNdx(0, n);
    B(1, 2 * n + 1) = q.dNdx(1, n);
    B(2, 2 * n) = q.dNdx(1, n);
    B(2, 2 * n + 1) = q.dNdx(0, n);
  }
  return B;
}

BendingB bending_B(const QuadPointData &q) {
  BendingB B = BendingB::Zero();
  for (int n = 0; n < 4; ++n) {
    B(0, 3 * n) = q.dNdx(0, n);
    B(1, 3 * n + 1) = q.dNdx(1, n);
    B(2, 3 * n) = q.dNdx(1, n);
    B(2, 3 * n + 1) = q.dNdx(0, n);
  }
  return B;
}

ShearB mindlin_shear_B(const QuadPointData &q) {
  ShearB B = ShearB::Zero();
  for (int n = 0; n < 4; ++n) {
    B(0, 3 * n + 2) = q.dNdx(0, n);
    B(0, 3 * n) = -q.shape.N[n];
    B(1, 3 * n + 2) = q.dNdx(1, n);
    B(1, 3 * n + 1) = -q.shape.N[n];
  }
  return B;
}

CovariantMembraneRows direct_covariant_membrane_rows(
    ElementId eid, const std::array<double, 4> &xl,
    const std::array<double, 4> &yl, double xi, double eta) {
  const QuadPointData q = evaluate_quad_point(eid, xl, yl, xi, eta);

  CovariantMembraneRows rows;
  for (int n = 0; n < 4; ++n) {
    rows.rr(2 * n) = q.g_r.x() * q.shape.dNdxi[n];
    rows.rr(2 * n + 1) = q.g_r.y() * q.shape.dNdxi[n];

    rows.ss(2 * n) = q.g_s.x() * q.shape.dNdeta[n];
    rows.ss(2 * n + 1) = q.g_s.y() * q.shape.dNdeta[n];

    rows.rs(2 * n) =
        0.5 *
        (q.g_r.x() * q.shape.dNdeta[n] + q.g_s.x() * q.shape.dNdxi[n]);
    rows.rs(2 * n + 1) =
        0.5 *
        (q.g_r.y() * q.shape.dNdeta[n] + q.g_s.y() * q.shape.dNdxi[n]);
  }

  return rows;
}

Mitc4PlusMembraneData build_mitc4plus_membrane_data(
    ElementId eid, const std::array<double, 4> &xl,
    const std::array<double, 4> &yl) {
  Mitc4PlusMembraneData data;

  data.top = direct_covariant_membrane_rows(eid, xl, yl, 0.0, 1.0);
  data.bottom = direct_covariant_membrane_rows(eid, xl, yl, 0.0, -1.0);
  data.right = direct_covariant_membrane_rows(eid, xl, yl, 1.0, 0.0);
  data.left = direct_covariant_membrane_rows(eid, xl, yl, -1.0, 0.0);
  data.center = direct_covariant_membrane_rows(eid, xl, yl, 0.0, 0.0);

  Eigen::Vector2d x_r = Eigen::Vector2d::Zero();
  Eigen::Vector2d x_s = Eigen::Vector2d::Zero();
  Eigen::Vector2d x_d = Eigen::Vector2d::Zero();
  static constexpr int R_SIGN[4] = {-1, 1, 1, -1};
  static constexpr int S_SIGN[4] = {-1, -1, 1, 1};

  for (int n = 0; n < 4; ++n) {
    const Eigen::Vector2d xn{xl[n], yl[n]};
    x_r += 0.25 * static_cast<double>(R_SIGN[n]) * xn;
    x_s += 0.25 * static_cast<double>(S_SIGN[n]) * xn;
    x_d +=
        0.25 * static_cast<double>(R_SIGN[n] * S_SIGN[n]) * xn;
  }

  Eigen::Matrix2d xr_xs;
  xr_xs.col(0) = x_r;
  xr_xs.col(1) = x_s;
  const double dual_det = xr_xs.determinant();
  if (std::abs(dual_det) < 1e-12) {
    throw SolverError(std::format(
        "CQUAD4 {}: MITC4+ characteristic geometry is singular", eid.value));
  }

  const Eigen::Matrix2d dual = xr_xs.inverse();
  const Eigen::Vector2d m_r = dual.row(0).transpose();
  const Eigen::Vector2d m_s = dual.row(1).transpose();

  const double c_r = x_d.dot(m_r);
  const double c_s = x_d.dot(m_s);
  const double d = c_r * c_r + c_s * c_s - 1.0;
  if (std::abs(d) < 1e-12) {
    throw SolverError(std::format(
        "CQUAD4 {}: MITC4+ distortion denominator is near zero", eid.value));
  }

  data.a_top = c_r * (c_r - 1.0) / (2.0 * d);
  data.a_bottom = c_r * (c_r + 1.0) / (2.0 * d);
  data.a_right = c_s * (c_s - 1.0) / (2.0 * d);
  data.a_left = c_s * (c_s + 1.0) / (2.0 * d);
  data.a_center = 2.0 * c_r * c_s / d;

  return data;
}

MembraneB mitc4plus_membrane_B(const Mitc4PlusMembraneData &data,
                               const QuadPointData &q) {
  const double r = q.xi;
  const double s = q.eta;

  const Row8 e_rr =
      0.5 * (1.0 - 2.0 * data.a_top + s + 2.0 * data.a_top * s * s) *
          data.top.rr +
      0.5 *
          (1.0 - 2.0 * data.a_bottom - s +
           2.0 * data.a_bottom * s * s) *
          data.bottom.rr +
      data.a_right * (-1.0 + s * s) * data.right.ss +
      data.a_left * (-1.0 + s * s) * data.left.ss +
      data.a_center * (-1.0 + s * s) * data.center.rs;

  const Row8 e_ss =
      data.a_top * (-1.0 + r * r) * data.top.rr +
      data.a_bottom * (-1.0 + r * r) * data.bottom.rr +
      0.5 * (1.0 - 2.0 * data.a_right + r + 2.0 * data.a_right * r * r) *
          data.right.ss +
      0.5 * (1.0 - 2.0 * data.a_left - r + 2.0 * data.a_left * r * r) *
          data.left.ss +
      data.a_center * (-1.0 + r * r) * data.center.rs;

  const Row8 eps_rs =
      0.25 * (r + 4.0 * data.a_top * r * s) * data.top.rr +
      0.25 * (-r + 4.0 * data.a_bottom * r * s) * data.bottom.rr +
      0.25 * (s + 4.0 * data.a_right * r * s) * data.right.ss +
      0.25 * (-s + 4.0 * data.a_left * r * s) * data.left.ss +
      (1.0 + data.a_center * r * s) * data.center.rs;

  const double a = q.contravariant(0, 0);
  const double b = q.contravariant(1, 0);
  const double c = q.contravariant(0, 1);
  const double d = q.contravariant(1, 1);

  MembraneB B = MembraneB::Zero();
  B.row(0) = a * a * e_rr + c * c * e_ss + 2.0 * a * c * eps_rs;
  B.row(1) = b * b * e_rr + d * d * e_ss + 2.0 * b * d * eps_rs;
  B.row(2) =
      2.0 * a * b * e_rr + 2.0 * c * d * e_ss +
      2.0 * (a * d + b * c) * eps_rs;
  return B;
}

Mitc4PlusShearData build_mitc4plus_shear_data(
    const std::array<double, 4> &xl, const std::array<double, 4> &yl) {
  auto covariant_shear = [&](double xi0, double eta0)
      -> std::pair<Row12, Row12> {
    const auto shape = CQuad4::shape_functions(xi0, eta0);
    double gxi_x = 0.0, gxi_y = 0.0, geta_x = 0.0, geta_y = 0.0;
    for (int n = 0; n < 4; ++n) {
      gxi_x += shape.dNdxi[n] * xl[n];
      gxi_y += shape.dNdxi[n] * yl[n];
      geta_x += shape.dNdeta[n] * xl[n];
      geta_y += shape.dNdeta[n] * yl[n];
    }

    Row12 Bxi = Row12::Zero();
    Row12 Beta = Row12::Zero();
    for (int n = 0; n < 4; ++n) {
      Bxi(3 * n + 2) = shape.dNdxi[n];
      Bxi(3 * n) = -shape.N[n] * gxi_x;
      Bxi(3 * n + 1) = -shape.N[n] * gxi_y;

      Beta(3 * n + 2) = shape.dNdeta[n];
      Beta(3 * n) = -shape.N[n] * geta_x;
      Beta(3 * n + 1) = -shape.N[n] * geta_y;
    }
    return {Bxi, Beta};
  };

  Mitc4PlusShearData data;
  {
    auto [row_xi, row_eta] = covariant_shear(0.0, -1.0);
    data.xi_bottom = row_xi;
    (void)row_eta;
  }
  {
    auto [row_xi, row_eta] = covariant_shear(0.0, 1.0);
    data.xi_top = row_xi;
    (void)row_eta;
  }
  {
    auto [row_xi, row_eta] = covariant_shear(1.0, 0.0);
    data.eta_right = row_eta;
    (void)row_xi;
  }
  {
    auto [row_xi, row_eta] = covariant_shear(-1.0, 0.0);
    data.eta_left = row_eta;
    (void)row_xi;
  }
  {
    auto [row_xi, row_eta] = covariant_shear(0.0, 0.0);
    data.xi_center = row_xi;
    data.eta_center = row_eta;
  }
  return data;
}

ShearB mitc4plus_shear_B(const Mitc4PlusShearData &data,
                         const QuadPointData &q) {
  const Row12 cov_xi =
      0.5 * (1.0 - q.eta) * data.xi_bottom +
      0.5 * (1.0 + q.eta) * data.xi_top +
      q.xi * (data.xi_center - 0.5 * (data.xi_bottom + data.xi_top));

  const Row12 cov_eta =
      0.5 * (1.0 - q.xi) * data.eta_left +
      0.5 * (1.0 + q.xi) * data.eta_right +
      q.eta * (data.eta_center - 0.5 * (data.eta_left + data.eta_right));

  Eigen::Matrix<double, 2, 12> cov = Eigen::Matrix<double, 2, 12>::Zero();
  cov.row(0) = cov_xi;
  cov.row(1) = cov_eta;
  return q.Jinv * cov;
}

Eigen::Matrix<double, 24, 1>
to_local_displacements(const ShellFrame &frame,
                       std::span<const double> global_displacements) {
  if (global_displacements.size() != 24) {
    throw SolverError(std::format(
        "CQUAD4 recovery expects 24 DOFs, got {}", global_displacements.size()));
  }

  Eigen::Matrix<double, 24, 1> u_global = Eigen::Matrix<double, 24, 1>::Zero();
  for (int i = 0; i < 24; ++i)
    u_global(i) = global_displacements[static_cast<size_t>(i)];
  return frame.T * u_global;
}

Eigen::Matrix<double, 8, 1>
gather_membrane_dofs(const Eigen::Matrix<double, 24, 1> &u_local) {
  Eigen::Matrix<double, 8, 1> u = Eigen::Matrix<double, 8, 1>::Zero();
  for (int n = 0; n < 4; ++n) {
    u(2 * n) = u_local(6 * n);
    u(2 * n + 1) = u_local(6 * n + 1);
  }
  return u;
}

Eigen::Matrix<double, 12, 1>
gather_plate_dofs(const Eigen::Matrix<double, 24, 1> &u_local) {
  Eigen::Matrix<double, 12, 1> u = Eigen::Matrix<double, 12, 1>::Zero();
  for (int n = 0; n < 4; ++n) {
    u(3 * n) = u_local(6 * n + 3);
    u(3 * n + 1) = u_local(6 * n + 4);
    u(3 * n + 2) = u_local(6 * n + 2);
  }
  return u;
}

template <std::size_t DofPerNode>
void assemble_square_submatrix(LocalKe &Ke, const Eigen::MatrixXd &submatrix,
                               const std::array<int, DofPerNode> &dof_map) {
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (std::size_t a = 0; a < DofPerNode; ++a) {
        for (std::size_t b = 0; b < DofPerNode; ++b) {
          Ke(6 * i + dof_map[a], 6 * j + dof_map[b]) +=
              submatrix(static_cast<int>(DofPerNode) * i + static_cast<int>(a),
                        static_cast<int>(DofPerNode) * j + static_cast<int>(b));
        }
      }
    }
  }
}

void assemble_coupling_submatrix(LocalKe &Ke,
                                 const Eigen::Matrix<double, 8, 12> &submatrix) {
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 3; ++b) {
          Ke(6 * i + MEMBRANE_DOF_MAP[a], 6 * j + PLATE_DOF_MAP[b]) +=
              submatrix(2 * i + a, 3 * j + b);
        }
      }
    }
  }
}

void assemble_coupling_transpose_submatrix(
    LocalKe &Ke, const Eigen::Matrix<double, 8, 12> &submatrix) {
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 3; ++b) {
          Ke(6 * j + PLATE_DOF_MAP[b], 6 * i + MEMBRANE_DOF_MAP[a]) +=
              submatrix(2 * i + a, 3 * j + b);
        }
      }
    }
  }
}

void assemble_membrane_force(LocalFe &fe, const Eigen::Matrix<double, 8, 1> &f) {
  for (int n = 0; n < 4; ++n) {
    fe(6 * n) += f(2 * n);
    fe(6 * n + 1) += f(2 * n + 1);
  }
}

LocalKe compute_shell_stiffness(ElementId eid, PropertyId pid,
                                const std::array<NodeId, 4> &nodes,
                                const Model &model,
                                ShellFormulation formulation) {
  LocalKe Ke = LocalKe::Zero(24, 24);
  const auto coords = lookup_node_coords(model, nodes);
  const ShellFrame frame = compute_shell_frame(coords);
  const ShellSection section = build_shell_section(model, eid, pid);

  const Mitc4PlusMembraneData mitc_membrane =
      (formulation == ShellFormulation::MITC4)
          ? build_mitc4plus_membrane_data(eid, frame.xl, frame.yl)
          : Mitc4PlusMembraneData{};
  const Mitc4PlusShearData mitc_shear =
      (formulation == ShellFormulation::MITC4)
          ? build_mitc4plus_shear_data(frame.xl, frame.yl)
          : Mitc4PlusShearData{};

  const Eigen::Matrix3d A_normal = [&]() {
    Eigen::Matrix3d A = section.A;
    A(2, 2) = 0.0;
    return A;
  }();
  const Eigen::Matrix3d A_shear = [&]() {
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    A(2, 2) = section.A(2, 2);
    return A;
  }();

  for (int gi = 0; gi < 2; ++gi) {
    for (int gj = 0; gj < 2; ++gj) {
      const QuadPointData q =
          evaluate_quad_point(eid, frame.xl, frame.yl, GAUSS2[gi], GAUSS2[gj]);
      const double weight = q.detJ * GAUSS2_W[gi] * GAUSS2_W[gj];

      const MembraneB Bm =
          (formulation == ShellFormulation::MITC4)
              ? mitc4plus_membrane_B(mitc_membrane, q)
              : standard_membrane_B(q);
      const BendingB Bb = bending_B(q);
      const ShearB Bs =
          (formulation == ShellFormulation::MITC4)
              ? mitc4plus_shear_B(mitc_shear, q)
              : mindlin_shear_B(q);

      const Eigen::Matrix<double, 8, 8> Kmm =
          Bm.transpose() *
          ((formulation == ShellFormulation::MITC4) ? section.A : A_normal) *
          Bm * weight;
      const Eigen::Matrix<double, 12, 12> Kbb =
          Bb.transpose() * section.D * Bb * weight;
      const Eigen::Matrix<double, 12, 12> Kss =
          Bs.transpose() * section.T * Bs * weight;

      assemble_square_submatrix(Ke, Kmm, MEMBRANE_DOF_MAP);
      assemble_square_submatrix(Ke, Kbb, PLATE_DOF_MAP);
      assemble_square_submatrix(Ke, Kss, PLATE_DOF_MAP);

      if (!section.B.isZero(0.0)) {
        const Eigen::Matrix<double, 8, 12> Kmb =
            Bm.transpose() * section.B * Bb * weight;
        assemble_coupling_submatrix(Ke, Kmb);
        assemble_coupling_transpose_submatrix(Ke, Kmb);
      }
    }
  }

  if (formulation == ShellFormulation::MINDLIN &&
      std::abs(section.A(2, 2)) > 0.0) {
    const QuadPointData q0 =
        evaluate_quad_point(eid, frame.xl, frame.yl, 0.0, 0.0);
    const MembraneB Bm0 = standard_membrane_B(q0);
    const Eigen::Matrix<double, 8, 8> Kxy =
        Bm0.transpose() * A_shear * Bm0 * q0.detJ * 4.0;
    assemble_square_submatrix(Ke, Kxy, MEMBRANE_DOF_MAP);
  }

  const double max_diag = Ke.diagonal().cwiseAbs().maxCoeff();
  if (max_diag < 1e-10) {
    throw SolverError(std::format(
        "CQUAD4 {}: stiffness matrix diagonal is near-zero (max={:.3e}); "
        "check material, thickness, and geometry",
        eid.value, max_diag));
  }

  const double drill_stiffness = compute_drilling_stiffness(Ke);
  for (int i = 0; i < 4; ++i)
    Ke(6 * i + 5, 6 * i + 5) += drill_stiffness;

  return frame.T.transpose() * Ke * frame.T;
}

LocalKe compute_shell_mass(ElementId eid, PropertyId pid,
                           const std::array<NodeId, 4> &nodes,
                           const Model &model) {
  LocalKe Me = LocalKe::Zero(24, 24);
  const auto coords = lookup_node_coords(model, nodes);
  const ShellFrame frame = compute_shell_frame(coords);
  const ShellSection section = build_shell_section(model, eid, pid);

  if (section.mass_per_area == 0.0 && section.rotary_mass_per_area == 0.0)
    return Me;

  const double drill_mass_scale =
      DRILL_INERTIA_SCALE * section.rotary_mass_per_area / 100.0;

  for (int gi = 0; gi < 2; ++gi) {
    for (int gj = 0; gj < 2; ++gj) {
      const QuadPointData q =
          evaluate_quad_point(eid, frame.xl, frame.yl, GAUSS2[gi], GAUSS2[gj]);
      const double weight = q.detJ * GAUSS2_W[gi] * GAUSS2_W[gj];

      for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
          const double mab = q.shape.N[a] * q.shape.N[b] * weight;
          for (int d = 0; d < 3; ++d)
            Me(6 * a + d, 6 * b + d) += section.mass_per_area * mab;
          for (int d = 0; d < 2; ++d)
            Me(6 * a + 3 + d, 6 * b + 3 + d) +=
                section.rotary_mass_per_area * mab;
        }
        Me(6 * a + 5, 6 * a + 5) +=
            drill_mass_scale * q.shape.N[a] * q.shape.N[a] * weight;
      }
    }
  }

  return frame.T.transpose() * Me * frame.T;
}

LocalFe compute_shell_thermal_load(ElementId eid, PropertyId pid,
                                   const std::array<NodeId, 4> &nodes,
                                   const Model &model,
                                   std::span<const double> temperatures,
                                   double t_ref,
                                   ShellFormulation formulation) {
  LocalFe fe = LocalFe::Zero(24);
  const auto coords = lookup_node_coords(model, nodes);
  const ShellFrame frame = compute_shell_frame(coords);
  const ShellSection section = build_shell_section(model, eid, pid);

  if (section.membrane_alpha.isZero(0.0))
    return fe;

  const Mitc4PlusMembraneData mitc_membrane =
      (formulation == ShellFormulation::MITC4)
          ? build_mitc4plus_membrane_data(eid, frame.xl, frame.yl)
          : Mitc4PlusMembraneData{};

  for (int gi = 0; gi < 2; ++gi) {
    for (int gj = 0; gj < 2; ++gj) {
      const QuadPointData q =
          evaluate_quad_point(eid, frame.xl, frame.yl, GAUSS2[gi], GAUSS2[gj]);
      const double weight = q.detJ * GAUSS2_W[gi] * GAUSS2_W[gj];

      double temperature = 0.0;
      for (int n = 0; n < 4; ++n)
        temperature += q.shape.N[n] * temperatures[static_cast<size_t>(n)];
      const double dT = temperature - t_ref;
      if (std::abs(dT) < 1e-15)
        continue;

      const MembraneB Bm =
          (formulation == ShellFormulation::MITC4)
              ? mitc4plus_membrane_B(mitc_membrane, q)
              : standard_membrane_B(q);
      const Eigen::Vector3d thermal_resultant =
          section.A * (section.membrane_alpha * dT);
      const Eigen::Matrix<double, 8, 1> f_mem =
          Bm.transpose() * thermal_resultant * weight;
      assemble_membrane_force(fe, f_mem);
    }
  }

  return frame.T.transpose() * fe;
}

CQuad4::CentroidResponse
recover_shell_centroid_response(ElementId eid, PropertyId pid,
                                const std::array<NodeId, 4> &nodes,
                                const Model &model,
                                std::span<const double> global_displacements,
                                double avg_temperature,
                                double reference_temperature) {
  const auto coords = lookup_node_coords(model, nodes);
  const ShellFrame frame = compute_shell_frame(coords);
  const ShellSection section = build_shell_section(model, eid, pid);
  const PShell &ps = lookup_pshell(model, eid, pid, "CQUAD4");
  const ShellFormulation formulation = ps.shell_form;

  const auto u_local = to_local_displacements(frame, global_displacements);
  const auto u_membrane = gather_membrane_dofs(u_local);
  const auto u_plate = gather_plate_dofs(u_local);

  const QuadPointData q0 =
      evaluate_quad_point(eid, frame.xl, frame.yl, 0.0, 0.0);
  const Mitc4PlusMembraneData mitc_membrane =
      (formulation == ShellFormulation::MITC4)
          ? build_mitc4plus_membrane_data(eid, frame.xl, frame.yl)
          : Mitc4PlusMembraneData{};

  const MembraneB Bm =
      (formulation == ShellFormulation::MITC4)
          ? mitc4plus_membrane_B(mitc_membrane, q0)
          : standard_membrane_B(q0);
  const BendingB Bb = bending_B(q0);

  CQuad4::CentroidResponse response;
  response.membrane_strain = Bm * u_membrane;
  response.curvature = Bb * u_plate;

  const double dT = avg_temperature - reference_temperature;
  const Eigen::Vector3d mechanical_membrane =
      response.membrane_strain - section.membrane_alpha * dT;

  response.membrane_stress = section.membrane_D * mechanical_membrane;
  response.membrane_resultant =
      section.A * mechanical_membrane + section.B * response.curvature;
  response.bending_moment =
      section.B.transpose() * mechanical_membrane +
      section.D * response.curvature;
  return response;
}

} // namespace

CQuad4::CQuad4(ElementId eid, PropertyId pid, std::array<NodeId, 4> node_ids,
               const Model &model)
    : eid_(eid), pid_(pid), nodes_(node_ids), model_(model) {}

const PShell &CQuad4::pshell() const {
  return lookup_pshell(model_, eid_, pid_, "CQUAD4");
}

const Mat1 &CQuad4::material() const { return model_.material(pshell().mid1); }

double CQuad4::thickness() const { return pshell().t; }

std::array<Vec3, 4> CQuad4::node_coords() const {
  return lookup_node_coords(model_, nodes_);
}

CQuad4::ShapeData CQuad4::shape_functions(double xi, double eta) noexcept {
  ShapeData s;
  s.N[0] = 0.25 * (1 - xi) * (1 - eta);
  s.N[1] = 0.25 * (1 + xi) * (1 - eta);
  s.N[2] = 0.25 * (1 + xi) * (1 + eta);
  s.N[3] = 0.25 * (1 - xi) * (1 + eta);

  s.dNdxi[0] = -0.25 * (1 - eta);
  s.dNdxi[1] = 0.25 * (1 - eta);
  s.dNdxi[2] = 0.25 * (1 + eta);
  s.dNdxi[3] = -0.25 * (1 + eta);

  s.dNdeta[0] = -0.25 * (1 - xi);
  s.dNdeta[1] = -0.25 * (1 + xi);
  s.dNdeta[2] = 0.25 * (1 + xi);
  s.dNdeta[3] = 0.25 * (1 - xi);
  return s;
}

Eigen::Matrix3d CQuad4::membrane_D() const {
  return build_shell_section(model_, eid_, pid_).membrane_D;
}

Eigen::Matrix3d CQuad4::bending_D() const {
  return build_shell_section(model_, eid_, pid_).D;
}

LocalKe CQuad4::stiffness_matrix() const {
  return compute_shell_stiffness(eid_, pid_, nodes_, model_,
                                 ShellFormulation::MINDLIN);
}

LocalKe CQuad4::mass_matrix() const {
  return compute_shell_mass(eid_, pid_, nodes_, model_);
}

LocalFe CQuad4::thermal_load(std::span<const double> temperatures,
                             double t_ref) const {
  return compute_shell_thermal_load(eid_, pid_, nodes_, model_, temperatures,
                                    t_ref, ShellFormulation::MINDLIN);
}

std::vector<EqIndex> CQuad4::global_dof_indices(const DofMap &dof_map) const {
  std::vector<EqIndex> result;
  result.reserve(24);
  for (NodeId nid : nodes_) {
    const auto &blk = dof_map.block(nid);
    for (int d = 0; d < 6; ++d)
      result.push_back(blk.eq[d]);
  }
  return result;
}

CQuad4::CentroidResponse CQuad4::recover_centroid_response(
    ElementId eid, PropertyId pid, std::array<NodeId, NUM_NODES> node_ids,
    const Model &model, std::span<const double> global_displacements,
    double avg_temperature, double reference_temperature) {
  return recover_shell_centroid_response(eid, pid, node_ids, model,
                                         global_displacements, avg_temperature,
                                         reference_temperature);
}

CQuad4Mitc4::CQuad4Mitc4(ElementId eid, PropertyId pid,
                         std::array<NodeId, 4> node_ids, const Model &model)
    : eid_(eid), pid_(pid), nodes_(node_ids), model_(model) {}

const PShell &CQuad4Mitc4::pshell() const {
  return lookup_pshell(model_, eid_, pid_, "CQuad4Mitc4");
}

const Mat1 &CQuad4Mitc4::material() const {
  return model_.material(pshell().mid1);
}

double CQuad4Mitc4::thickness() const { return pshell().t; }

std::array<Vec3, 4> CQuad4Mitc4::node_coords() const {
  return lookup_node_coords(model_, nodes_);
}

Eigen::Matrix3d CQuad4Mitc4::membrane_D() const {
  return build_shell_section(model_, eid_, pid_).membrane_D;
}

Eigen::Matrix3d CQuad4Mitc4::bending_D() const {
  return build_shell_section(model_, eid_, pid_).D;
}

LocalKe CQuad4Mitc4::stiffness_matrix() const {
  return compute_shell_stiffness(eid_, pid_, nodes_, model_,
                                 ShellFormulation::MITC4);
}

LocalKe CQuad4Mitc4::mass_matrix() const {
  return compute_shell_mass(eid_, pid_, nodes_, model_);
}

LocalFe CQuad4Mitc4::thermal_load(std::span<const double> temperatures,
                                  double t_ref) const {
  return compute_shell_thermal_load(eid_, pid_, nodes_, model_, temperatures,
                                    t_ref, ShellFormulation::MITC4);
}

std::vector<EqIndex>
CQuad4Mitc4::global_dof_indices(const DofMap &dof_map) const {
  std::vector<EqIndex> result;
  result.reserve(24);
  for (NodeId nid : nodes_) {
    const auto &blk = dof_map.block(nid);
    for (int d = 0; d < 6; ++d)
      result.push_back(blk.eq[d]);
  }
  return result;
}

} // namespace vibestran
