// src/elements/line_elements.cpp

#include "elements/line_elements.hpp"
#include "core/coord_sys.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <numbers>

namespace vibestran {

namespace {

constexpr double kFrameTolerance = 1e-12;

struct LineSection {
  double E{0.0};
  double G{0.0};
  double rho{0.0};
  double alpha{0.0};
  double ref_temp{0.0};
  double A{0.0};
  double I1{0.0};
  double I2{0.0};
  double I12{0.0};
  double J{0.0};
  double nsm{0.0};
};

struct LineFrame {
  double length{0.0};
  Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};
  Eigen::Matrix<double, 12, 12> lambda{Eigen::Matrix<double, 12, 12>::Identity()};
};

struct HermiteData {
  std::array<double, 4> N{};
  std::array<double, 4> dN_dx{};
  std::array<double, 4> d2N_dx2{};
};

[[nodiscard]] std::string uppercase_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), ::toupper);
  return s;
}

[[nodiscard]] Vec3 normalized_or_throw(const Vec3 &v, const std::string &ctx) {
  const double n = v.norm();
  if (n < kFrameTolerance)
    throw SolverError(std::format("{}: direction vector has near-zero norm",
                                  ctx));
  return v * (1.0 / n);
}

[[nodiscard]] Vec3 choose_reference_axis(const Vec3 &x_axis) {
  Vec3 ref{0.0, 0.0, 1.0};
  if (std::abs(x_axis.dot(ref)) > 0.95)
    ref = Vec3{0.0, 1.0, 0.0};
  if (std::abs(x_axis.dot(ref)) > 0.95)
    ref = Vec3{1.0, 0.0, 0.0};
  return ref;
}

[[nodiscard]] Vec3 orientation_vector_in_basic(const Model &model, const Vec3 &v,
                                               const CoordId cid,
                                               const Vec3 &position,
                                               const std::string &ctx) {
  if (cid.value == 0)
    return v;
  const auto cs_it = model.coord_systems.find(cid);
  if (cs_it == model.coord_systems.end()) {
    throw SolverError(std::format(
        "{}: coordinate system {} not found for orientation vector", ctx,
        cid.value));
  }
  return apply_rotation(rotation_matrix(cs_it->second, position), v);
}

[[nodiscard]] LineFrame build_line_frame(
    const std::array<NodeId, 2> &nodes, const Model &model,
    const std::optional<Vec3> &orientation, const std::optional<NodeId> &g0,
    const std::optional<CoordId> &cid, const std::string &ctx,
    const bool allow_coincident, const bool coincident_orientation_is_x) {
  const Vec3 x1 = model.node(nodes[0]).position;
  const Vec3 x2 = model.node(nodes[1]).position;
  Vec3 axis = x2 - x1;
  double length = axis.norm();

  Vec3 ref_vec;
  if (g0.has_value()) {
    ref_vec = model.node(*g0).position - x1;
  } else if (orientation.has_value()) {
    ref_vec = orientation_vector_in_basic(model, *orientation, cid.value_or(CoordId{0}),
                                          x1, ctx);
  }

  if (length < kFrameTolerance) {
    if (!allow_coincident) {
      throw SolverError(std::format(
          "{}: coincident grid points are not supported", ctx));
    }
    if (ref_vec.norm() < kFrameTolerance) {
      throw SolverError(std::format(
          "{}: coincident grids require an explicit orientation", ctx));
    }
    axis = coincident_orientation_is_x ? ref_vec : choose_reference_axis(Vec3{1.0, 0.0, 0.0});
    length = 1.0;
    if (coincident_orientation_is_x)
      ref_vec = choose_reference_axis(normalized_or_throw(axis, ctx));
  }

  const Vec3 ex = normalized_or_throw(axis, ctx);

  if (length >= kFrameTolerance) {
    if (ref_vec.norm() < kFrameTolerance)
      ref_vec = choose_reference_axis(ex);
  } else {
    ref_vec = choose_reference_axis(ex);
  }

  Vec3 ey_candidate = ref_vec - ex * ex.dot(ref_vec);
  if (ey_candidate.norm() < kFrameTolerance) {
    ref_vec = choose_reference_axis(ex);
    ey_candidate = ref_vec - ex * ex.dot(ref_vec);
  }
  const Vec3 ey = normalized_or_throw(ey_candidate, ctx);
  const Vec3 ez = normalized_or_throw(ex.cross(ey), ctx);
  const Vec3 ey_fixed = normalized_or_throw(ez.cross(ex), ctx);

  LineFrame frame;
  frame.length = length;
  frame.rotation << ex.x, ey_fixed.x, ez.x, ex.y, ey_fixed.y, ez.y, ex.z,
      ey_fixed.z, ez.z;
  frame.lambda.setZero();
  for (int node = 0; node < 2; ++node) {
    const int offset = 6 * node;
    frame.lambda.block<3, 3>(offset, offset) = frame.rotation.transpose();
    frame.lambda.block<3, 3>(offset + 3, offset + 3) =
        frame.rotation.transpose();
  }
  return frame;
}

[[nodiscard]] HermiteData hermite_shape(const double x, const double L) {
  const double s = x / L;
  HermiteData h;
  h.N[0] = 1.0 - 3.0 * s * s + 2.0 * s * s * s;
  h.N[1] = L * (s - 2.0 * s * s + s * s * s);
  h.N[2] = 3.0 * s * s - 2.0 * s * s * s;
  h.N[3] = L * (-s * s + s * s * s);

  h.dN_dx[0] = (-6.0 * s + 6.0 * s * s) / L;
  h.dN_dx[1] = 1.0 - 4.0 * s + 3.0 * s * s;
  h.dN_dx[2] = (6.0 * s - 6.0 * s * s) / L;
  h.dN_dx[3] = -2.0 * s + 3.0 * s * s;

  h.d2N_dx2[0] = (-6.0 + 12.0 * s) / (L * L);
  h.d2N_dx2[1] = (-4.0 + 6.0 * s) / L;
  h.d2N_dx2[2] = (6.0 - 12.0 * s) / (L * L);
  h.d2N_dx2[3] = (-2.0 + 6.0 * s) / L;
  return h;
}

[[nodiscard]] std::array<double, 2> linear_shape(const double x,
                                                 const double L) {
  return {1.0 - x / L, x / L};
}

[[nodiscard]] LineSection section_from_bar_property(const PBar &prop,
                                                    const Mat1 &mat) {
  return {
      .E = mat.E,
      .G = (mat.G != 0.0) ? mat.G : mat.E / (2.0 * (1.0 + mat.nu)),
      .rho = mat.rho,
      .alpha = mat.A,
      .ref_temp = mat.ref_temp,
      .A = prop.A,
      .I1 = prop.I1,
      .I2 = prop.I2,
      .I12 = 0.0,
      .J = prop.J,
      .nsm = prop.nsm,
  };
}

[[nodiscard]] LineSection section_from_barl_property(const PBarL &prop,
                                                     const Mat1 &mat) {
  return {
      .E = mat.E,
      .G = (mat.G != 0.0) ? mat.G : mat.E / (2.0 * (1.0 + mat.nu)),
      .rho = mat.rho,
      .alpha = mat.A,
      .ref_temp = mat.ref_temp,
      .A = prop.A,
      .I1 = prop.I1,
      .I2 = prop.I2,
      .I12 = 0.0,
      .J = prop.J,
      .nsm = prop.nsm,
  };
}

[[nodiscard]] LineSection section_from_beam_property(const PBeam &prop,
                                                     const Mat1 &mat) {
  return {
      .E = mat.E,
      .G = (mat.G != 0.0) ? mat.G : mat.E / (2.0 * (1.0 + mat.nu)),
      .rho = mat.rho,
      .alpha = mat.A,
      .ref_temp = mat.ref_temp,
      .A = prop.A,
      .I1 = prop.I1,
      .I2 = prop.I2,
      .I12 = prop.I12,
      .J = prop.J,
      .nsm = prop.nsm,
  };
}

[[nodiscard]] LineSection resolve_line_section(const Model &model,
                                               const PropertyId pid,
                                               const ElementType type,
                                               const ElementId eid) {
  const Property &prop = model.property(pid);
  if (type == ElementType::CBAR) {
    if (const auto *p = std::get_if<PBar>(&prop))
      return section_from_bar_property(*p, model.material(p->mid));
    if (const auto *p = std::get_if<PBarL>(&prop))
      return section_from_barl_property(*p, model.material(p->mid));
    throw SolverError(std::format(
        "CBAR {}: property {} must be PBAR or PBARL", eid.value, pid.value));
  }

  if (const auto *p = std::get_if<PBeam>(&prop))
    return section_from_beam_property(*p, model.material(p->mid));
  throw SolverError(std::format("CBEAM {}: property {} must be PBEAM",
                                eid.value, pid.value));
}

// cppcheck-suppress constParameterReference -- Eigen matrices are mutated via operator()
void add_linear_2x2(LocalKe &K, const int i0, const int i1,
                    const Eigen::Matrix2d &sub) {
  K(i0, i0) += sub(0, 0);
  K(i0, i1) += sub(0, 1);
  K(i1, i0) += sub(1, 0);
  K(i1, i1) += sub(1, 1);
}

// cppcheck-suppress constParameterReference -- Eigen matrices are mutated via operator()
void add_plane_4x4(LocalKe &K, const std::array<int, 4> &idx,
                   const Eigen::Matrix4d &sub) {
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      K(idx[i], idx[j]) += sub(i, j);
}

[[nodiscard]] LocalKe build_beam_local_stiffness(const LineSection &section,
                                                 const double L) {
  LocalKe K = LocalKe::Zero(12, 12);

  const Eigen::Matrix2d axial =
      section.E * section.A / L *
      (Eigen::Matrix2d() << 1.0, -1.0, -1.0, 1.0).finished();
  add_linear_2x2(K, 0, 6, axial);

  const Eigen::Matrix2d torsion =
      section.G * section.J / L *
      (Eigen::Matrix2d() << 1.0, -1.0, -1.0, 1.0).finished();
  add_linear_2x2(K, 3, 9, torsion);

  const double gauss_point = std::sqrt(3.0 / 5.0);
  const std::array<double, 3> gauss_pts{-gauss_point, 0.0, gauss_point};
  static constexpr std::array<double, 3> gauss_wts{5.0 / 9.0, 8.0 / 9.0,
                                                   5.0 / 9.0};

  Eigen::Matrix4d kv = Eigen::Matrix4d::Zero();
  Eigen::Matrix4d kw = Eigen::Matrix4d::Zero();
  for (int igp = 0; igp < 3; ++igp) {
    const double x = 0.5 * L * (gauss_pts[igp] + 1.0);
    const double jac = 0.5 * L * gauss_wts[igp];
    const HermiteData h = hermite_shape(x, L);

    Eigen::Matrix<double, 1, 4> bv;
    bv << h.d2N_dx2[0], h.d2N_dx2[1], h.d2N_dx2[2], h.d2N_dx2[3];
    kv += section.E * section.I2 * (bv.transpose() * bv) * jac;

    Eigen::Matrix<double, 1, 4> bw;
    bw << h.d2N_dx2[0], -h.d2N_dx2[1], h.d2N_dx2[2], -h.d2N_dx2[3];
    kw += section.E * section.I1 * (bw.transpose() * bw) * jac;
  }

  add_plane_4x4(K, {1, 5, 7, 11}, kv);
  add_plane_4x4(K, {2, 4, 8, 10}, kw);
  return K;
}

[[nodiscard]] LocalKe build_beam_local_mass(const LineSection &section,
                                            const double L) {
  LocalKe M = LocalKe::Zero(12, 12);
  const double mu = section.rho * section.A + section.nsm;
  if (mu == 0.0 && section.rho == 0.0)
    return M;

  static constexpr std::array<double, 4> gauss_pts{
      -0.8611363115940526, -0.3399810435848563, 0.3399810435848563,
      0.8611363115940526};
  static constexpr std::array<double, 4> gauss_wts{
      0.3478548451374538, 0.6521451548625461, 0.6521451548625461,
      0.3478548451374538};

  Eigen::Matrix2d maxial = Eigen::Matrix2d::Zero();
  Eigen::Matrix2d mtorsion = Eigen::Matrix2d::Zero();
  Eigen::Matrix4d mv = Eigen::Matrix4d::Zero();
  Eigen::Matrix4d mw = Eigen::Matrix4d::Zero();
  const double torsional_mu = section.rho * (section.I1 + section.I2);

  for (int gp = 0; gp < 4; ++gp) {
    const double x = 0.5 * L * (gauss_pts[gp] + 1.0);
    const double jac = 0.5 * L * gauss_wts[gp];
    const auto nlin = linear_shape(x, L);
    const HermiteData h = hermite_shape(x, L);

    Eigen::Matrix<double, 2, 1> n_ax;
    n_ax << nlin[0], nlin[1];
    maxial += mu * (n_ax * n_ax.transpose()) * jac;
    mtorsion += torsional_mu * (n_ax * n_ax.transpose()) * jac;

    Eigen::Matrix<double, 4, 1> nv;
    nv << h.N[0], h.N[1], h.N[2], h.N[3];
    mv += mu * (nv * nv.transpose()) * jac;

    Eigen::Matrix<double, 4, 1> nw;
    nw << h.N[0], -h.N[1], h.N[2], -h.N[3];
    mw += mu * (nw * nw.transpose()) * jac;
  }

  add_linear_2x2(M, 0, 6, maxial);
  add_linear_2x2(M, 3, 9, mtorsion);
  add_plane_4x4(M, {1, 5, 7, 11}, mv);
  add_plane_4x4(M, {2, 4, 8, 10}, mw);
  return M;
}

[[nodiscard]] double scalar_stiffness_from_property(const Model &model,
                                                    const PropertyId pid,
                                                    const ElementType type,
                                                    const ElementId eid) {
  const Property &prop = model.property(pid);
  if (const auto *pelas = std::get_if<PElas>(&prop))
    return pelas->k;
  throw SolverError(std::format(
      "{} {}: property {} must be PELAS",
      (type == ElementType::CELAS1) ? "CELAS1" : "CELAS2", eid.value,
      pid.value));
}

[[nodiscard]] double scalar_mass_from_property(const Model &model,
                                               const PropertyId pid,
                                               const ElementType type,
                                               const ElementId eid) {
  const Property &prop = model.property(pid);
  if (const auto *pmass = std::get_if<PMass>(&prop))
    return pmass->mass;
  throw SolverError(std::format(
      "{} {}: property {} must be PMASS",
      (type == ElementType::CMASS1) ? "CMASS1" : "CMASS2", eid.value,
      pid.value));
}

[[nodiscard]] const PBush &lookup_pbush(const Model &model, const PropertyId pid,
                                        const ElementId eid) {
  const Property &prop = model.property(pid);
  if (!std::holds_alternative<PBush>(prop))
    throw SolverError(std::format("CBUSH {}: property {} is not PBUSH",
                                  eid.value, pid.value));
  return std::get<PBush>(prop);
}

[[nodiscard]] LocalFe zero_force(const int ndof) {
  return LocalFe::Zero(ndof);
}

[[nodiscard]] std::vector<EqIndex> line_global_dofs(const std::array<NodeId, 2> &nodes,
                                                    const DofMap &dof_map) {
  std::vector<EqIndex> result;
  result.reserve(12);
  for (const NodeId nid : nodes) {
    const auto &blk = dof_map.block(nid);
    for (int d = 0; d < 6; ++d)
      result.push_back(blk.eq[d]);
  }
  return result;
}

[[nodiscard]] std::array<double, 4> plane_shape_values(const std::string &type,
                                                       const HermiteData &h) {
  if (type == "FY" || type == "FZ") {
    if (type == "FY")
      return h.N;
    return {h.N[0], -h.N[1], h.N[2], -h.N[3]};
  }
  if (type == "MY")
    return {-h.dN_dx[0], h.dN_dx[1], -h.dN_dx[2], h.dN_dx[3]};
  return {h.dN_dx[0], h.dN_dx[1], h.dN_dx[2], h.dN_dx[3]};
}

// cppcheck-suppress constParameterReference -- Eigen vectors are mutated via operator()
void add_plane_load(LocalFe &fe, const std::array<int, 4> &indices,
                    const std::array<double, 4> &shape, const double value,
                    const double weight) {
  for (int i = 0; i < 4; ++i)
    fe(indices[i]) += shape[i] * value * weight;
}

[[nodiscard]] std::pair<double, double>
load_interval(const Pload1Load &load, const double L) {
  double x1 = load.x1;
  double x2 = load.x2.value_or((uppercase_copy(load.scale_type) == "FR") ? 1.0 : L);
  if (uppercase_copy(load.scale_type) == "FR") {
    x1 *= L;
    x2 *= L;
  } else if (uppercase_copy(load.scale_type) != "LE") {
    throw SolverError(std::format(
        "PLOAD1 on element {}: scale type '{}' is not supported",
        load.element.value, load.scale_type));
  }
  if (x2 < x1)
    std::swap(x1, x2);
  if (x1 < -1e-12 || x2 > L + 1e-12) {
    throw SolverError(std::format(
        "PLOAD1 on element {}: load stations [{:.6g}, {:.6g}] lie outside the "
        "element span [0, {:.6g}]",
        load.element.value, x1, x2, L));
  }
  x1 = std::clamp(x1, 0.0, L);
  x2 = std::clamp(x2, 0.0, L);
  return {x1, x2};
}

} // namespace

CBarBeamElement::CBarBeamElement(ElementType type, ElementId eid, PropertyId pid,
                                 std::array<NodeId, 2> node_ids,
                                 const Model &model,
                                 std::optional<Vec3> orientation,
                                 std::optional<NodeId> g0)
    : type_(type), eid_(eid), pid_(pid), nodes_(node_ids), model_(model),
      orientation_(std::move(orientation)), g0_(g0) {}

LocalKe CBarBeamElement::stiffness_matrix() const {
  const LineSection section = resolve_line_section(model_, pid_, type_, eid_);
  const LineFrame frame = build_line_frame(nodes_, model_, orientation_, g0_,
                                           std::nullopt,
                                           std::format("{} {}", (type_ == ElementType::CBAR) ? "CBAR" : "CBEAM",
                                                       eid_.value),
                                           false, false);
  const LocalKe k_local = build_beam_local_stiffness(section, frame.length);
  return frame.lambda.transpose() * k_local * frame.lambda;
}

LocalKe CBarBeamElement::mass_matrix() const {
  const LineSection section = resolve_line_section(model_, pid_, type_, eid_);
  const LineFrame frame = build_line_frame(nodes_, model_, orientation_, g0_,
                                           std::nullopt,
                                           std::format("{} {}", (type_ == ElementType::CBAR) ? "CBAR" : "CBEAM",
                                                       eid_.value),
                                           false, false);
  const LocalKe m_local = build_beam_local_mass(section, frame.length);
  return frame.lambda.transpose() * m_local * frame.lambda;
}

LocalFe CBarBeamElement::thermal_load(std::span<const double> temperatures,
                                      double t_ref) const {
  const LineSection section = resolve_line_section(model_, pid_, type_, eid_);
  const LineFrame frame = build_line_frame(nodes_, model_, orientation_, g0_,
                                           std::nullopt,
                                           std::format("{} {}", (type_ == ElementType::CBAR) ? "CBAR" : "CBEAM",
                                                       eid_.value),
                                           false, false);
  if (section.alpha == 0.0)
    return zero_force(NUM_DOFS);

  const double temp_avg = 0.5 * (temperatures[0] + temperatures[1]);
  const double dT = temp_avg - t_ref;
  LocalFe fe_local = LocalFe::Zero(NUM_DOFS);
  const double axial_force = section.E * section.A * section.alpha * dT;
  fe_local(0) = -axial_force;
  fe_local(6) = axial_force;
  return frame.lambda.transpose() * fe_local;
}

std::vector<EqIndex>
CBarBeamElement::global_dof_indices(const DofMap &dof_map) const {
  return line_global_dofs(nodes_, dof_map);
}

CBushElement::CBushElement(ElementId eid, PropertyId pid,
                           std::array<NodeId, 2> node_ids, const Model &model,
                           std::optional<Vec3> orientation,
                           std::optional<NodeId> g0,
                           std::optional<CoordId> cid)
    : eid_(eid), pid_(pid), nodes_(node_ids), model_(model),
      orientation_(std::move(orientation)), g0_(g0), cid_(cid) {}

LocalKe CBushElement::stiffness_matrix() const {
  const PBush &prop = lookup_pbush(model_, pid_, eid_);
  const Vec3 x1 = model_.node(nodes_[0]).position;
  const Vec3 x2 = model_.node(nodes_[1]).position;
  const bool coincident = (x2 - x1).norm() < kFrameTolerance;
  const LineFrame frame = build_line_frame(
      nodes_, model_, orientation_, g0_, cid_,
      std::format("CBUSH {}", eid_.value), true, true);

  LocalKe k_local = LocalKe::Zero(NUM_DOFS, NUM_DOFS);
  for (int comp = 0; comp < 6; ++comp) {
    const double k = prop.k[comp];
    k_local(comp, comp) += k;
    k_local(comp, comp + 6) -= k;
    k_local(comp + 6, comp) -= k;
    k_local(comp + 6, comp + 6) += k;
  }

  if (coincident && !orientation_.has_value() && !g0_.has_value()) {
    throw SolverError(std::format(
        "CBUSH {}: coincident grids require an explicit orientation",
        eid_.value));
  }

  return frame.lambda.transpose() * k_local * frame.lambda;
}

LocalKe CBushElement::mass_matrix() const {
  return LocalKe::Zero(NUM_DOFS, NUM_DOFS);
}

LocalFe CBushElement::thermal_load(std::span<const double> /*temperatures*/,
                                   double /*t_ref*/) const {
  return zero_force(NUM_DOFS);
}

std::vector<EqIndex>
CBushElement::global_dof_indices(const DofMap &dof_map) const {
  return line_global_dofs(nodes_, dof_map);
}

ScalarSpringElement::ScalarSpringElement(ElementType type, ElementId eid,
                                         PropertyId pid,
                                         std::vector<NodeId> nodes,
                                         std::array<int, 2> components,
                                         double value, const Model &model)
    : type_(type), eid_(eid), pid_(pid), nodes_(std::move(nodes)),
      components_(components), value_(value), model_(model) {}

double ScalarSpringElement::stiffness_value() const {
  if (type_ == ElementType::CELAS1)
    return scalar_stiffness_from_property(model_, pid_, type_, eid_);
  return value_;
}

LocalKe ScalarSpringElement::stiffness_matrix() const {
  const double k = stiffness_value();
  if (nodes_.size() == 1) {
    LocalKe K(1, 1);
    K(0, 0) = k;
    return K;
  }
  LocalKe K(2, 2);
  K << k, -k, -k, k;
  return K;
}

LocalKe ScalarSpringElement::mass_matrix() const {
  return LocalKe::Zero(num_dofs(), num_dofs());
}

LocalFe ScalarSpringElement::thermal_load(std::span<const double> /*temperatures*/,
                                          double /*t_ref*/) const {
  return zero_force(num_dofs());
}

std::vector<EqIndex>
ScalarSpringElement::global_dof_indices(const DofMap &dof_map) const {
  std::vector<EqIndex> result;
  result.reserve(nodes_.size());
  for (size_t i = 0; i < nodes_.size(); ++i)
    result.push_back(dof_map.eq_index(nodes_[i], components_[i] - 1));
  return result;
}

ScalarMassElement::ScalarMassElement(ElementType type, ElementId eid,
                                     PropertyId pid, std::vector<NodeId> nodes,
                                     std::array<int, 2> components,
                                     double value, const Model &model)
    : type_(type), eid_(eid), pid_(pid), nodes_(std::move(nodes)),
      components_(components), value_(value), model_(model) {}

double ScalarMassElement::mass_value() const {
  if (type_ == ElementType::CMASS1)
    return scalar_mass_from_property(model_, pid_, type_, eid_);
  return value_;
}

LocalKe ScalarMassElement::stiffness_matrix() const {
  return LocalKe::Zero(num_dofs(), num_dofs());
}

LocalKe ScalarMassElement::mass_matrix() const {
  const double m = mass_value();
  if (nodes_.size() == 1) {
    LocalKe M(1, 1);
    M(0, 0) = m;
    return M;
  }
  LocalKe M(2, 2);
  M << m, -m, -m, m;
  return M;
}

LocalFe ScalarMassElement::thermal_load(std::span<const double> /*temperatures*/,
                                        double /*t_ref*/) const {
  return zero_force(num_dofs());
}

std::vector<EqIndex>
ScalarMassElement::global_dof_indices(const DofMap &dof_map) const {
  std::vector<EqIndex> result;
  result.reserve(nodes_.size());
  for (size_t i = 0; i < nodes_.size(); ++i)
    result.push_back(dof_map.eq_index(nodes_[i], components_[i] - 1));
  return result;
}

LocalFe compute_pload1_equivalent_load(const ElementData &elem,
                                       const Model &model,
                                       const Pload1Load &load) {
  if (elem.type != ElementType::CBAR && elem.type != ElementType::CBEAM) {
    throw SolverError(std::format(
        "PLOAD1 on element {} is only supported for CBAR and CBEAM",
        elem.id.value));
  }

  const std::array<NodeId, 2> nodes{elem.nodes[0], elem.nodes[1]};
  const LineFrame frame = build_line_frame(
      nodes, model, elem.orientation, elem.g0, std::nullopt,
      std::format("PLOAD1 element {}", elem.id.value), false, false);
  const auto [xa, xb] = load_interval(load, frame.length);
  if (std::abs(xb - xa) < 1e-12)
    return LocalFe::Zero(12);

  const std::string load_type = uppercase_copy(load.load_type);
  const double p2 = load.p2.value_or(load.p1);
  LocalFe fe_local = LocalFe::Zero(12);

  static constexpr std::array<double, 4> gauss_pts{
      -0.8611363115940526, -0.3399810435848563, 0.3399810435848563,
      0.8611363115940526};
  static constexpr std::array<double, 4> gauss_wts{
      0.3478548451374538, 0.6521451548625461, 0.6521451548625461,
      0.3478548451374538};

  for (int gp = 0; gp < 4; ++gp) {
    const double x = 0.5 * (xb - xa) * (gauss_pts[gp] + 1.0) + xa;
    const double jac = 0.5 * (xb - xa) * gauss_wts[gp];
    const double xi = (xb > xa) ? (x - xa) / (xb - xa) : 0.0;
    const double q = load.p1 + (p2 - load.p1) * xi;
    const auto nlin = linear_shape(x, frame.length);
    const HermiteData h = hermite_shape(x, frame.length);

    if (load_type == "FX") {
      fe_local(0) += nlin[0] * q * jac;
      fe_local(6) += nlin[1] * q * jac;
    } else if (load_type == "MX") {
      fe_local(3) += nlin[0] * q * jac;
      fe_local(9) += nlin[1] * q * jac;
    } else if (load_type == "FY" || load_type == "MZ") {
      const auto shape = plane_shape_values(load_type, h);
      add_plane_load(fe_local, {1, 5, 7, 11}, shape, q, jac);
    } else if (load_type == "FZ" || load_type == "MY") {
      const auto shape = plane_shape_values(load_type, h);
      add_plane_load(fe_local, {2, 4, 8, 10}, shape, q, jac);
    } else {
      throw SolverError(std::format(
          "PLOAD1 on element {}: load type '{}' is not supported",
          elem.id.value, load.load_type));
    }
  }

  return frame.lambda.transpose() * fe_local;
}

} // namespace vibestran
