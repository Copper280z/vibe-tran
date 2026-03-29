// src/solver/linear_static.cpp
// Linear static analysis pipeline.
// K * u = F_ext + F_thermal

#include "solver/linear_static.hpp"
#include "core/coord_sys.hpp"
#include "core/factor_ratio_check.hpp"
#include "core/logger.hpp"
#include "core/mpc_handler.hpp"
#include "elements/cquad4.hpp"
#include "elements/ctria3.hpp"
#include "elements/element_factory.hpp"
#include "elements/line_elements.hpp"
#include "elements/rbe_constraints.hpp"
#include "elements/solid_elements.hpp"
#include "assembly_parallel.hpp"
#include "solver/analysis_support.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numbers>
#include <set>
#include <spdlog/spdlog.h>

namespace vibestran {

namespace {

[[nodiscard]] MaterialId property_material_id(const Property &prop) {
  if (const auto *pshell = std::get_if<PShell>(&prop))
    return pshell->mid1;
  if (const auto *psolid = std::get_if<PSolid>(&prop))
    return psolid->mid;
  if (const auto *pbar = std::get_if<PBar>(&prop))
    return pbar->mid;
  if (const auto *pbarl = std::get_if<PBarL>(&prop))
    return pbarl->mid;
  if (const auto *pbeam = std::get_if<PBeam>(&prop))
    return pbeam->mid;
  return MaterialId{0};
}

[[nodiscard]] double component_value(const Vec3 &v, const int component) {
  switch (component) {
  case 1:
    return v.x;
  case 2:
    return v.y;
  case 3:
    return v.z;
  default:
    return 0.0;
  }
}

[[nodiscard]] double wtmass_scale(const Model &model) {
  auto wt_it = model.params.find("WTMASS");
  if (wt_it == model.params.end())
    return 1.0;
  try {
    return std::stod(wt_it->second);
  } catch (...) {
    return 1.0;
  }
}

[[nodiscard]] Vec3 load_direction_in_basic(const Model &model, const CoordId cid,
                                           const Vec3 &direction,
                                           const Vec3 &position) {
  if (cid.value == 0)
    return direction;
  const auto cs_it = model.coord_systems.find(cid);
  if (cs_it == model.coord_systems.end()) {
    throw SolverError(std::format(
        "Coordinate system {} not found for inertial load direction",
        cid.value));
  }
  return apply_rotation(rotation_matrix(cs_it->second, position), direction);
}

[[nodiscard]] std::unordered_map<NodeId, double>
build_nodal_temperature_map(const Model &model, const SubCase &sc) {
  std::unordered_map<NodeId, double> nodal_temps;
  int temp_set = sc.temp_load_set;
  if (temp_set == 0)
    temp_set = sc.load_set.value;

  for (const Load *lp : model.loads_for_set(LoadSetId(temp_set))) {
    if (const TempLoad *tl = std::get_if<TempLoad>(lp))
      nodal_temps[tl->node] = tl->temperature;
  }

  const auto tempd_it = model.tempd.find(temp_set);
  if (tempd_it != model.tempd.end()) {
    const double t_default = tempd_it->second;
    for (const auto &[nid, _] : model.nodes) {
      if (!nodal_temps.contains(nid))
        nodal_temps[nid] = t_default;
    }
  }
  return nodal_temps;
}

[[nodiscard]] double
average_temperature(const std::vector<NodeId> &nodes,
                    const std::unordered_map<NodeId, double> &nodal_temps,
                    const double default_temp) {
  if (nodes.empty())
    return default_temp;
  double sum = 0.0;
  for (NodeId nid : nodes) {
    const auto it = nodal_temps.find(nid);
    sum += (it != nodal_temps.end()) ? it->second : default_temp;
  }
  return sum / static_cast<double>(nodes.size());
}

[[nodiscard]] Eigen::Matrix<double, 6, 6>
solid_constitutive_matrix(const Mat1 &mat) {
  const double lam =
      mat.E * mat.nu / ((1.0 + mat.nu) * (1.0 - 2.0 * mat.nu));
  const double mu = mat.E / (2.0 * (1.0 + mat.nu));
  Eigen::Matrix<double, 6, 6> D = Eigen::Matrix<double, 6, 6>::Zero();
  D(0, 0) = lam + 2.0 * mu;
  D(0, 1) = lam;
  D(0, 2) = lam;
  D(1, 0) = lam;
  D(1, 1) = lam + 2.0 * mu;
  D(1, 2) = lam;
  D(2, 0) = lam;
  D(2, 1) = lam;
  D(2, 2) = lam + 2.0 * mu;
  D(3, 3) = mu;
  D(4, 4) = mu;
  D(5, 5) = mu;
  return D;
}

[[nodiscard]] double solid_von_mises(const Eigen::Matrix<double, 6, 1> &sigma) {
  const double s1 = sigma(0);
  const double s2 = sigma(1);
  const double s3 = sigma(2);
  const double t12 = sigma(3);
  const double t23 = sigma(4);
  const double t31 = sigma(5);
  return std::sqrt(0.5 * ((s1 - s2) * (s1 - s2) + (s2 - s3) * (s2 - s3) +
                          (s3 - s1) * (s3 - s1) +
                          6.0 * (t12 * t12 + t23 * t23 + t31 * t31)));
}

[[nodiscard]] SolidStressPoint
make_solid_stress_point(const NodeId node,
                        const Eigen::Matrix<double, 6, 1> &sigma) {
  SolidStressPoint point;
  point.node = node;
  point.sx = sigma(0);
  point.sy = sigma(1);
  point.sz = sigma(2);
  point.sxy = sigma(3);
  point.syz = sigma(4);
  point.szx = sigma(5);
  point.von_mises = solid_von_mises(sigma);
  return point;
}

[[nodiscard]] PlateStressPoint
make_plate_stress_point(const NodeId node,
                        const Eigen::Vector3d &membrane_stress,
                        const Eigen::Vector3d &bending_moment) {
  PlateStressPoint point;
  point.node = node;
  point.sx = membrane_stress(0);
  point.sy = membrane_stress(1);
  point.sxy = membrane_stress(2);
  point.mx = bending_moment(0);
  point.my = bending_moment(1);
  point.mxy = bending_moment(2);
  point.von_mises =
      std::sqrt(point.sx * point.sx - point.sx * point.sy +
                point.sy * point.sy + 3.0 * point.sxy * point.sxy);
  return point;
}

[[nodiscard]] Eigen::Matrix<double, 6, 1>
solid_sigma_from_B(const Eigen::Matrix<double, 6, 6> &D,
                   const Eigen::MatrixXd &B, const Eigen::VectorXd &ue,
                   const double alpha, const double temperature,
                   const double reference_temperature) {
  Eigen::Matrix<double, 6, 1> eps_th;
  eps_th << alpha * (temperature - reference_temperature),
      alpha * (temperature - reference_temperature),
      alpha * (temperature - reference_temperature), 0.0, 0.0, 0.0;
  return D * (B * ue - eps_th);
}

} // namespace

LinearStaticSolver::LinearStaticSolver(std::unique_ptr<SolverBackend> backend)
    : backend_(std::move(backend)) {}

SolverResults LinearStaticSolver::solve(const Model &model) {
  model.validate();

  SolverResults results;
  for (const auto &sc : model.analysis.subcases) {
    results.subcases.push_back(solve_subcase(model, sc));
  }
  return results;
}

SubCaseResults LinearStaticSolver::solve_subcase(const Model &model,
                                                 const SubCase &sc) {
  using Clock = std::chrono::steady_clock;
  using Ms = std::chrono::duration<double, std::milli>;
  const auto t0 = Clock::now();

  // 1. Build DOF map and apply SPC boundary conditions
  DofMap dof_map = build_dof_map(model, sc);
  const auto t1 = Clock::now();
  spdlog::debug("[subcase {}] build_dof_map: {:.3f} ms  ({} free DOFs)",
                sc.id, Ms(t1 - t0).count(), dof_map.num_free_dofs());

  // 2. Build MPC handler (CD-frame SPCs + explicit MPCs + RBE2/RBE3)
  MpcHandler mpc_handler;
  build_mpc_system(model, sc, dof_map, mpc_handler);
  const int n = mpc_handler.num_reduced();
  const auto t2 = Clock::now();
  spdlog::debug("[subcase {}] build_mpc_system: {:.3f} ms  ({} reduced DOFs)",
                sc.id, Ms(t2 - t1).count(), n);

  // 3. Assemble global K and F using pre-MPC dof_map
  SparseMatrixBuilder K_builder(n);
  std::vector<double> F(static_cast<size_t>(n), 0.0);

  assemble(model, sc, mpc_handler, K_builder, F);
  const auto t3 = Clock::now();
  spdlog::debug("[subcase {}] assemble K: {:.3f} ms", sc.id, Ms(t3 - t2).count());

  apply_point_loads(model, sc, mpc_handler, F);
  const auto t4 = Clock::now();
  spdlog::debug("[subcase {}] apply_point_loads: {:.3f} ms", sc.id, Ms(t4 - t3).count());

  apply_pressure_loads(model, sc, mpc_handler, F);
  const auto t4b = Clock::now();
  spdlog::debug("[subcase {}] apply_pressure_loads: {:.3f} ms", sc.id,
                Ms(t4b - t4).count());

  apply_inertial_loads(model, sc, mpc_handler, F);
  const auto t4c = Clock::now();
  spdlog::debug("[subcase {}] apply_inertial_loads: {:.3f} ms", sc.id,
                Ms(t4c - t4b).count());

  apply_thermal_loads(model, sc, mpc_handler, K_builder, F);
  const auto t5 = Clock::now();
  spdlog::debug("[subcase {}] apply_thermal_loads: {:.3f} ms", sc.id,
                Ms(t5 - t4c).count());

  // 4. Build CSR
  auto csr = K_builder.build_csr();
  const auto t5b = Clock::now();
  spdlog::debug("[subcase {}] build_csr: {:.3f} ms  ({} nnz)", sc.id, Ms(t5b - t5).count(), csr.nnz);

  const SparseMatrixBuilder::CsrData* solve_csr = &csr;
  SparseMatrixBuilder::CsrData expanded_csr;
  if (backend_->requires_full_symmetric_csr()) {
    expanded_csr = csr.expanded_symmetric();
    solve_csr = &expanded_csr;
  }

  const FactorRatioCheckPolicy factor_ratio_policy =
      build_factor_ratio_check_policy(
          model, std::format("[subcase {}]", sc.id), "stiffness matrix");
  std::vector<double> u_reduced =
      backend_->solve(*solve_csr, F, &factor_ratio_policy);
  const auto t6 = Clock::now();
  spdlog::debug("[subcase {}] linear solve: {:.3f} ms", sc.id, Ms(t6 - t5b).count());

  // Log iterative solver convergence info when available (PCG backends).
  {
    int iters = backend_->last_iteration_count();
    if (iters >= 0)
      spdlog::debug("[subcase {}] PCG: {} iterations, estimated residual = {:.3e}",
                    sc.id, iters, backend_->last_estimated_error());
  }

  // Compute the true relative residual r = K*u - F in the reduced system.
  // For direct solvers this should be near machine epsilon; for PCG backends
  // it reflects the iterative convergence quality.
  {
    const std::vector<double> Ku = csr.multiply(u_reduced);
    double r_norm_sq = 0.0;
    double f_norm_sq = 0.0;
    for (int row = 0; row < n; ++row) {
      const double ri = Ku[static_cast<size_t>(row)] - F[static_cast<size_t>(row)];
      r_norm_sq += ri * ri;
      f_norm_sq += F[static_cast<size_t>(row)] * F[static_cast<size_t>(row)];
    }
    const double rel_res = (f_norm_sq > 1e-300)
        ? std::sqrt(r_norm_sq / f_norm_sq)
        : std::sqrt(r_norm_sq);
    spdlog::info("[subcase {}] relative residual ||K*u - F|| / ||F|| = {:.3e}",
                 sc.id, rel_res);
  }

  // 5. Recover full displacement vector (free + dep DOFs)
  int n_full = mpc_handler.full_dof_map().num_free_dofs();
  std::vector<double> u_free(static_cast<size_t>(n_full), 0.0);
  mpc_handler.recover_dependent_dofs(u_free, u_reduced);

  // 6. Recover results (displacements in CD frame + element stresses)
  SubCaseResults result = recover_results(model, sc,
                                         mpc_handler.full_dof_map(),
                                         u_free);
  const auto t7 = Clock::now();
  spdlog::debug("[subcase {}] recover_results: {:.3f} ms", sc.id, Ms(t7 - t6).count());
  spdlog::debug("[subcase {}] total: {:.3f} ms", sc.id, Ms(t7 - t0).count());

  return result;
}

DofMap LinearStaticSolver::build_dof_map(const Model &model,
                                         const SubCase &sc) {
  return build_analysis_dof_map(model, sc);
}

void LinearStaticSolver::build_mpc_system(const Model &model,
                                           const SubCase &sc,
                                           DofMap &dof_map,
                                           MpcHandler &mpc_handler) {
  build_analysis_mpc_system(model, sc, dof_map, mpc_handler);
}

void LinearStaticSolver::assemble(const Model &model, const SubCase & /*sc*/,
                                  const MpcHandler &mpc_handler,
                                  SparseMatrixBuilder &K_builder,
                                  std::vector<double> & /*F*/) {
  K_builder.reserve_triplets(detail::estimate_triplet_capacity(model));
  detail::assemble_element_matrix(
      model, mpc_handler, K_builder,
      [](const ElementBase &elem) { return elem.stiffness_matrix(); });
}

void LinearStaticSolver::apply_point_loads(const Model &model,
                                           const SubCase &sc,
                                           const MpcHandler &mpc_handler,
                                           std::vector<double> &F) {
  const DofMap &dof_map = mpc_handler.full_dof_map();
  for (const Load *lp : model.loads_for_set(sc.load_set)) {
    std::visit(
        [&](const auto &load) {
          using T = std::decay_t<decltype(load)>;

          if constexpr (std::is_same_v<T, ForceLoad>) {
            // Rotate force from CID to basic if CID ≠ 0
            Vec3 force{load.scale * load.direction.x,
                       load.scale * load.direction.y,
                       load.scale * load.direction.z};
            if (load.cid.value != 0) {
              auto cs_it = model.coord_systems.find(load.cid);
              if (cs_it != model.coord_systems.end()) {
                // Force in local CID frame → basic frame
                const Vec3& node_pos = model.node(load.node).position;
                Mat3 T3 = rotation_matrix(cs_it->second, node_pos);
                force = apply_rotation(T3, force);
              }
            }

            std::array<EqIndex, 6> full_eqs;
            dof_map.global_indices(load.node, full_eqs);
            std::vector<EqIndex> gdofs(full_eqs.begin(), full_eqs.end());
            double fe[6] = {force.x, force.y, force.z, 0, 0, 0};
            mpc_handler.apply_to_force(gdofs,
                std::span<const double>(fe, 6), F);

          } else if constexpr (std::is_same_v<T, MomentLoad>) {
            Vec3 moment{load.scale * load.direction.x,
                        load.scale * load.direction.y,
                        load.scale * load.direction.z};
            if (load.cid.value != 0) {
              auto cs_it = model.coord_systems.find(load.cid);
              if (cs_it != model.coord_systems.end()) {
                const Vec3& node_pos = model.node(load.node).position;
                Mat3 T3 = rotation_matrix(cs_it->second, node_pos);
                moment = apply_rotation(T3, moment);
              }
            }
            std::array<EqIndex, 6> full_eqs;
            dof_map.global_indices(load.node, full_eqs);
            std::vector<EqIndex> gdofs(full_eqs.begin(), full_eqs.end());
            double fe[6] = {0, 0, 0, moment.x, moment.y, moment.z};
            mpc_handler.apply_to_force(gdofs,
                std::span<const double>(fe, 6), F);
          }
        },
        *lp);
  }
}

void LinearStaticSolver::apply_inertial_loads(const Model &model,
                                              const SubCase &sc,
                                              const MpcHandler &mpc_handler,
                                              std::vector<double> &F) {
  const double wtmass = wtmass_scale(model);
  std::unordered_map<NodeId, Vec3> nodal_accels;
  bool has_inertial_load = false;

  auto add_accel_to_node = [&](NodeId nid, const Vec3 &accel) {
    nodal_accels[nid] = nodal_accels[nid] + accel;
  };

  for (const Load *lp : model.loads_for_set(sc.load_set)) {
    std::visit(
        [&](const auto &load) {
          using T = std::decay_t<decltype(load)>;
          if constexpr (std::is_same_v<T, GravLoad>) {
            has_inertial_load = true;
            for (const auto &[nid, gp] : model.nodes) {
              const Vec3 accel =
                  load_direction_in_basic(model, load.cid,
                                          load.direction * load.scale,
                                          gp.position);
              add_accel_to_node(nid, accel);
            }
          } else if constexpr (std::is_same_v<T, Accel1Load>) {
            has_inertial_load = true;
            for (NodeId nid : load.nodes) {
              const Vec3 accel = load_direction_in_basic(
                  model, load.cid, load.direction * load.scale,
                  model.node(nid).position);
              add_accel_to_node(nid, accel);
            }
          } else if constexpr (std::is_same_v<T, AccelLoad>) {
            throw SolverError(std::format(
                "ACCEL load set {} is parsed but not implemented",
                load.sid.value));
          }
        },
        *lp);
  }

  if (!has_inertial_load)
    return;

  const DofMap &dof_map = mpc_handler.full_dof_map();
  for (const auto &elem_data : model.elements) {
    auto elem = make_element(elem_data, model);
    LocalKe mass = elem->mass_matrix();
    if (mass.rows() == 0)
      continue;
    mass *= wtmass;
    if (mass.cwiseAbs().maxCoeff() < 1e-30)
      continue;

    LocalFe accel = LocalFe::Zero(elem->num_dofs());
    switch (elem_data.type) {
    case ElementType::CQUAD4:
    case ElementType::CTRIA3:
    case ElementType::CBAR:
    case ElementType::CBEAM:
    case ElementType::CBUSH:
      for (size_t i = 0; i < elem_data.nodes.size(); ++i) {
        const Vec3 a = nodal_accels[elem_data.nodes[i]];
        accel(6 * static_cast<int>(i) + 0) = a.x;
        accel(6 * static_cast<int>(i) + 1) = a.y;
        accel(6 * static_cast<int>(i) + 2) = a.z;
      }
      break;
    case ElementType::CHEXA8:
    case ElementType::CHEXA20:
    case ElementType::CTETRA4:
    case ElementType::CTETRA10:
    case ElementType::CPENTA6:
      for (size_t i = 0; i < elem_data.nodes.size(); ++i) {
        const Vec3 a = nodal_accels[elem_data.nodes[i]];
        accel(3 * static_cast<int>(i) + 0) = a.x;
        accel(3 * static_cast<int>(i) + 1) = a.y;
        accel(3 * static_cast<int>(i) + 2) = a.z;
      }
      break;
    case ElementType::CELAS1:
    case ElementType::CELAS2:
      break;
    case ElementType::CMASS1:
    case ElementType::CMASS2:
      for (size_t i = 0; i < elem_data.nodes.size(); ++i) {
        const auto it = nodal_accels.find(elem_data.nodes[i]);
        if (it == nodal_accels.end())
          continue;
        accel(static_cast<int>(i)) =
            component_value(it->second, elem_data.components[i]);
      }
      break;
    }

    const LocalFe fe = mass * accel;
    if (fe.cwiseAbs().maxCoeff() < 1e-30)
      continue;
    const auto gdofs = elem->global_dof_indices(dof_map);
    std::vector<double> fe_vec(fe.data(), fe.data() + fe.size());
    mpc_handler.apply_to_force(gdofs, fe_vec, F);
  }
}

void LinearStaticSolver::apply_thermal_loads(
    const Model &model, const SubCase &sc,
    const MpcHandler &mpc_handler,
    SparseMatrixBuilder & /*K_builder*/, std::vector<double> &F) {
  // Build nodal temperature map from TEMP cards and/or TEMPD defaults.
  // TEMPERATURE(LOAD) selects the temperature set; if not specified, fall back
  // to the structural load set for backward compatibility.
  std::unordered_map<NodeId, double> nodal_temps;

  int temp_set = sc.temp_load_set;
  if (temp_set == 0) temp_set = sc.load_set.value; // backward compat

  // Individual TEMP cards for this set
  for (const Load *lp : model.loads_for_set(LoadSetId(temp_set))) {
    if (const TempLoad *tl = std::get_if<TempLoad>(lp))
      nodal_temps[tl->node] = tl->temperature;
  }

  // TEMPD (default temperature for all nodes not covered by individual TEMP cards)
  auto tempd_it = model.tempd.find(temp_set);
  if (tempd_it != model.tempd.end()) {
    double T_default = tempd_it->second;
    for (const auto& [nid, _] : model.nodes) {
      if (nodal_temps.find(nid) == nodal_temps.end())
        nodal_temps[nid] = T_default;
    }
  }

  if (nodal_temps.empty())
    return;

  const DofMap &dof_map = mpc_handler.full_dof_map();
  for (const auto &elem_data : model.elements) {
    auto elem = make_element(elem_data, model);
    auto node_ids = elem->node_ids();
    const int nn = static_cast<int>(node_ids.size());

    // Reference temperature from the element's material (MAT1 TREF field)
    MaterialId mid{0};
    if (elem_data.type != ElementType::CELAS2 &&
        elem_data.type != ElementType::CMASS2) {
      const auto &prop = model.property(elem_data.pid);
      mid = property_material_id(prop);
    }
    double elem_t_ref = (mid.value != 0) ? model.material(mid).ref_temp : 0.0;

    std::vector<double> temps(static_cast<size_t>(nn));
    for (int i = 0; i < nn; ++i) {
      auto it = nodal_temps.find(node_ids[i]);
      temps[static_cast<size_t>(i)] =
          (it != nodal_temps.end()) ? it->second : elem_t_ref;
    }

    LocalFe fe = elem->thermal_load(temps, elem_t_ref);
    auto gdofs = elem->global_dof_indices(dof_map);
    std::vector<double> fe_vec(fe.data(), fe.data() + fe.size());

    mpc_handler.apply_to_force(gdofs, fe_vec, F);
  }
}

SubCaseResults
LinearStaticSolver::recover_results(const Model &model, const SubCase &sc,
                                    const DofMap &dof_map,
                                    const std::vector<double> &u_free) {
  SubCaseResults res;
  res.id = sc.id;
  res.label = sc.label;

  // ── Recover displacements ─────────────────────────────────────────────────
  std::vector<NodeId> sorted_nodes;
  sorted_nodes.reserve(model.nodes.size());
  for (const auto &[nid, _] : model.nodes)
    sorted_nodes.push_back(nid);
  std::sort(sorted_nodes.begin(), sorted_nodes.end());

  for (NodeId nid : sorted_nodes) {
    NodeDisplacement nd;
    nd.node = nid;
    for (int d = 0; d < 6; ++d) {
      EqIndex eq = dof_map.eq_index(nid, d);
      nd.d[d] = (eq != CONSTRAINED_DOF && eq < static_cast<int>(u_free.size()))
                    ? u_free[static_cast<size_t>(eq)]
                    : 0.0;
    }
    // Rotate displacements to CD frame for output
    const GridPoint &gp = model.node(nid);
    if (gp.cd.value != 0) {
      auto cs_it = model.coord_systems.find(gp.cd);
      if (cs_it != model.coord_systems.end()) {
        Mat3 T3 = rotation_matrix(cs_it->second, gp.position);
        // v_cd = T3^T * v_basic
        // T3^T[i][j] = T3[j][i]
        Vec3 u_basic{nd.d[0], nd.d[1], nd.d[2]};
        // v_cd[i] = sum_j T3[j][i] * v_basic[j]
        Vec3 u_cd{
            T3(0,0)*u_basic.x + T3(1,0)*u_basic.y + T3(2,0)*u_basic.z,
            T3(0,1)*u_basic.x + T3(1,1)*u_basic.y + T3(2,1)*u_basic.z,
            T3(0,2)*u_basic.x + T3(1,2)*u_basic.y + T3(2,2)*u_basic.z,
        };
        nd.d[0] = u_cd.x; nd.d[1] = u_cd.y; nd.d[2] = u_cd.z;

        Vec3 rot_basic{nd.d[3], nd.d[4], nd.d[5]};
        Vec3 rot_cd{
            T3(0,0)*rot_basic.x + T3(1,0)*rot_basic.y + T3(2,0)*rot_basic.z,
            T3(0,1)*rot_basic.x + T3(1,1)*rot_basic.y + T3(2,1)*rot_basic.z,
            T3(0,2)*rot_basic.x + T3(1,2)*rot_basic.y + T3(2,2)*rot_basic.z,
        };
        nd.d[3] = rot_cd.x; nd.d[4] = rot_cd.y; nd.d[5] = rot_cd.z;
      }
    }
    res.displacements.push_back(nd);
  }

  // ── Build nodal temperature map for thermal stress correction ────────────
  const std::unordered_map<NodeId, double> nodal_temps_rec =
      build_nodal_temperature_map(model, sc);

  if (sc.has_any_stress_output()) {
    std::set<std::string> unsupported_types;
    for (const auto &elem_data : model.elements) {
      switch (elem_data.type) {
      case ElementType::CBUSH:
        unsupported_types.insert("CBUSH");
        break;
      case ElementType::CELAS1:
        unsupported_types.insert("CELAS1");
        break;
      case ElementType::CELAS2:
        unsupported_types.insert("CELAS2");
        break;
      case ElementType::CMASS1:
        unsupported_types.insert("CMASS1");
        break;
      case ElementType::CMASS2:
        unsupported_types.insert("CMASS2");
        break;
      default:
        break;
      }
    }

    if (!unsupported_types.empty()) {
      std::string families;
      for (const auto &name : unsupported_types) {
        if (!families.empty())
          families += ", ";
        families += name;
      }
      spdlog::warn(
          "[subcase {}] stress recovery is not implemented for element "
          "families [{}]; those results were skipped",
          sc.id, families);
    }
  }

  // ── Recover element stresses ──────────────────────────────────────────────
  for (const auto &elem_data : model.elements) {
    auto elem = make_element(elem_data, model);
    auto gdofs = elem->global_dof_indices(dof_map);

    const int nd_ = static_cast<int>(gdofs.size());
    Eigen::VectorXd ue = Eigen::VectorXd::Zero(nd_);
    for (int i = 0; i < nd_; ++i) {
      EqIndex eq = gdofs[i];
      if (eq != CONSTRAINED_DOF && eq < static_cast<int>(u_free.size()))
        ue(i) = u_free[static_cast<size_t>(eq)];
    }

    if (elem_data.type == ElementType::CBAR ||
        elem_data.type == ElementType::CBEAM) {
      const MaterialId mid = property_material_id(model.property(elem_data.pid));
      const Mat1 &mat_ = model.material(mid);
      const double avg_temp =
          average_temperature(elem_data.nodes, nodal_temps_rec, mat_.ref_temp);

      std::array<double, CBarBeamElement::NUM_DOFS> line_u{};
      for (int i = 0; i < CBarBeamElement::NUM_DOFS; ++i)
        line_u[static_cast<size_t>(i)] = ue(i);

      CBarBeamElement line_elem(
          elem_data.type, elem_data.id, elem_data.pid,
          {elem_data.nodes[0], elem_data.nodes[1]}, model, elem_data.orientation,
          elem_data.g0);
      const auto recovery =
          line_elem.recover_stress(line_u, avg_temp, mat_.ref_temp);

      LineStress ls;
      ls.eid = elem_data.id;
      ls.etype = elem_data.type;
      ls.end_a.node = elem_data.nodes[0];
      ls.end_a.s = recovery.end_a.s;
      ls.end_a.axial = recovery.end_a.axial;
      ls.end_a.smax = recovery.end_a.smax;
      ls.end_a.smin = recovery.end_a.smin;
      ls.end_b.node = elem_data.nodes[1];
      ls.end_b.s = recovery.end_b.s;
      ls.end_b.axial = recovery.end_b.axial;
      ls.end_b.smax = recovery.end_b.smax;
      ls.end_b.smin = recovery.end_b.smin;
      res.line_stresses.push_back(ls);
    } else if (elem_data.type == ElementType::CQUAD4 ||
               elem_data.type == ElementType::CTRIA3) {
      PlateStress ps;
      ps.eid   = elem_data.id;
      ps.etype = elem_data.type;

      if (elem_data.type == ElementType::CQUAD4) {
        const auto &pshell_ = std::get<PShell>(model.property(elem_data.pid));
        const Mat1 &mat_ = model.material(pshell_.mid1);
        std::array<NodeId, 4> nodes{
            elem_data.nodes[0], elem_data.nodes[1], elem_data.nodes[2],
            elem_data.nodes[3]};
        const double T_avg4 =
            average_temperature(elem_data.nodes, nodal_temps_rec, mat_.ref_temp);
        const auto response = CQuad4::recover_centroid_response(
            elem_data.id, elem_data.pid, nodes, model,
            std::span<const double>(ue.data(), static_cast<size_t>(ue.size())),
            T_avg4, mat_.ref_temp);
        const PlateStressPoint centroid = make_plate_stress_point(
            NodeId{0}, response.membrane_stress, response.bending_moment);
        ps.sx = centroid.sx;
        ps.sy = centroid.sy;
        ps.sxy = centroid.sxy;
        ps.mx = centroid.mx;
        ps.my = centroid.my;
        ps.mxy = centroid.mxy;
        ps.von_mises = centroid.von_mises;

        static constexpr std::array<std::pair<double, double>, 4>
            kQuadCornerCoords{{{-1.0, -1.0}, {1.0, -1.0},
                               {1.0, 1.0}, {-1.0, 1.0}}};
        for (int n = 0; n < 4; ++n) {
          const NodeId nid = elem_data.nodes[static_cast<size_t>(n)];
          const auto temp_it = nodal_temps_rec.find(nid);
          const double point_temp =
              (temp_it != nodal_temps_rec.end()) ? temp_it->second : mat_.ref_temp;
          const auto point = CQuad4::recover_response(
              elem_data.id, elem_data.pid, nodes, model,
              std::span<const double>(ue.data(), static_cast<size_t>(ue.size())),
              kQuadCornerCoords[static_cast<size_t>(n)].first,
              kQuadCornerCoords[static_cast<size_t>(n)].second, point_temp,
              mat_.ref_temp);
          ps.nodal.push_back(make_plate_stress_point(
              nid, point.membrane_stress, point.bending_moment));
        }
      } else {
        auto node_c = [&]() -> std::array<Vec3, 3> {
          std::array<Vec3, 3> c;
          for (int i = 0; i < 3; ++i)
            c[i] = model.node(elem_data.nodes[i]).position;
          return c;
        }();
        // Build local shell frame — same fix as CQUAD4: using global X,Y
        // fails for elements not in the XY plane (e.g. XZ-plane elements
        // have constant Y → A2 = 0 → NaN).
        Vec3 v12_t = node_c[1] - node_c[0];
        Vec3 v13_t = node_c[2] - node_c[0];
        Vec3 e3_t  = v12_t.cross(v13_t).normalized();
        Vec3 e1_t  = v12_t.normalized();
        Vec3 e2_t  = e3_t.cross(e1_t);
        std::array<double, 3> xl_t{}, yl_t{};
        for (int n = 0; n < 3; ++n) {
          xl_t[n] = node_c[n].dot(e1_t);
          yl_t[n] = node_c[n].dot(e2_t);
        }
        double x1 = xl_t[0], y1 = yl_t[0];
        double x2 = xl_t[1], y2 = yl_t[1];
        double x3 = xl_t[2], y3 = yl_t[2];
        double A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
        double b1 = y2 - y3, b2 = y3 - y1, b3 = y1 - y2;
        double d1 = x3 - x2, d2 = x1 - x3, d3 = x2 - x1;
        Eigen::MatrixXd Bm(3, 6);
        Bm.setZero();
        Bm(0, 0) = b1; Bm(0, 2) = b2; Bm(0, 4) = b3;
        Bm(1, 1) = d1; Bm(1, 3) = d2; Bm(1, 5) = d3;
        Bm(2, 0) = d1; Bm(2, 1) = b1; Bm(2, 2) = d2;
        Bm(2, 3) = b2; Bm(2, 4) = d3; Bm(2, 5) = b3;
        Bm /= A2;
        const auto &pshell_ = std::get<PShell>(model.property(elem_data.pid));
        const Mat1 &mat_ = model.material(pshell_.mid1);
        double E_ = mat_.E, nu_ = mat_.nu, c_ = E_ / (1 - nu_ * nu_);
        Eigen::Matrix3d Dm_;
        Dm_ << c_, c_ * nu_, 0, c_ * nu_, c_, 0, 0, 0, c_ * (1 - nu_) / 2;
        Eigen::VectorXd u_mem(6);
        for (int n = 0; n < 3; ++n) {
          Vec3 u_glob(ue(6 * n), ue(6 * n + 1), ue(6 * n + 2));
          u_mem(2 * n)     = u_glob.dot(e1_t);
          u_mem(2 * n + 1) = u_glob.dot(e2_t);
        }
        const double T_avg3 =
            average_temperature(elem_data.nodes, nodal_temps_rec, mat_.ref_temp);
        double dT3 = T_avg3 - mat_.ref_temp;
        double alpha3 = mat_.A;
        Eigen::Vector3d eps_th3{alpha3 * dT3, alpha3 * dT3, 0.0};

        Eigen::Vector3d sigma = Dm_ * (Bm * u_mem - eps_th3);
        const PlateStressPoint centroid = make_plate_stress_point(
            NodeId{0}, sigma, Eigen::Vector3d::Zero());
        ps.sx = centroid.sx;
        ps.sy = centroid.sy;
        ps.sxy = centroid.sxy;
        ps.mx = 0.0;
        ps.my = 0.0;
        ps.mxy = 0.0;
        ps.von_mises = centroid.von_mises;
        for (NodeId nid : elem_data.nodes)
          ps.nodal.push_back(
              make_plate_stress_point(nid, sigma, Eigen::Vector3d::Zero()));
      }
      res.plate_stresses.push_back(ps);
    } else if (elem_data.type == ElementType::CHEXA8 ||
               elem_data.type == ElementType::CTETRA4 ||
               elem_data.type == ElementType::CTETRA10 ||
               elem_data.type == ElementType::CPENTA6) {
      SolidStress ss;
      ss.eid   = elem_data.id;
      ss.etype = elem_data.type;

      const auto &psol_ = std::get<PSolid>(model.property(elem_data.pid));
      const Mat1 &mat_ = model.material(psol_.mid);
      const Eigen::Matrix<double, 6, 6> D_ = solid_constitutive_matrix(mat_);

      Eigen::Matrix<double, 6, 1> sigma;
      sigma.setZero();
      auto eval_ctetra4 = [&]() {
        auto nc = [&]() -> std::array<Vec3, 4> {
          std::array<Vec3, 4> c;
          for (int i = 0; i < 4; ++i)
            c[i] = model.node(elem_data.nodes[static_cast<size_t>(i)]).position;
          return c;
        }();
        const double x1 = nc[0].x, y1 = nc[0].y, z1 = nc[0].z;
        const double x2 = nc[1].x, y2 = nc[1].y, z2 = nc[1].z;
        const double x3 = nc[2].x, y3 = nc[2].y, z3 = nc[2].z;
        const double x4 = nc[3].x, y4 = nc[3].y, z4 = nc[3].z;
        Eigen::Matrix4d A4;
        A4 << 1, x1, y1, z1, 1, x2, y2, z2, 1, x3, y3, z3, 1, x4, y4, z4;
        const double V6 = A4.determinant();
        Eigen::Matrix4d cofA = Eigen::Matrix4d::Zero();
        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 4; ++j) {
            Eigen::Matrix3d m3;
            int ri = 0;
            for (int r = 0; r < 4; ++r) {
              if (r == j)
                continue;
              int ci = 0;
              for (int c = 0; c < 4; ++c) {
                if (c == i)
                  continue;
                m3(ri, ci++) = A4(r, c);
              }
              ++ri;
            }
            cofA(i, j) = std::pow(-1.0, i + j) * m3.determinant();
          }
        }
        Eigen::MatrixXd B(6, 12);
        B.setZero();
        for (int n = 0; n < 4; ++n) {
          const double bx = cofA(1, n) / V6;
          const double by = cofA(2, n) / V6;
          const double bz = cofA(3, n) / V6;
          const int c0 = 3 * n;
          B(0, c0) = bx;
          B(1, c0 + 1) = by;
          B(2, c0 + 2) = bz;
          B(3, c0) = by;
          B(3, c0 + 1) = bx;
          B(4, c0 + 1) = bz;
          B(4, c0 + 2) = by;
          B(5, c0) = bz;
          B(5, c0 + 2) = bx;
        }
        const double temp =
            average_temperature(elem_data.nodes, nodal_temps_rec, mat_.ref_temp);
        return solid_sigma_from_B(D_, B, ue, mat_.A, temp, mat_.ref_temp);
      };

      struct Tetra10ShapeData {
        std::array<double, 10> N{};
        std::array<double, 10> dNdL1{};
        std::array<double, 10> dNdL2{};
        std::array<double, 10> dNdL3{};
      };
      const auto tet10_shape = [](const double L1, const double L2,
                                  const double L3) {
        const double L4 = 1.0 - L1 - L2 - L3;
        Tetra10ShapeData s;
        s.N[0] = L1 * (2 * L1 - 1);
        s.N[1] = L2 * (2 * L2 - 1);
        s.N[2] = L3 * (2 * L3 - 1);
        s.N[3] = L4 * (2 * L4 - 1);
        s.N[4] = 4 * L1 * L2;
        s.N[5] = 4 * L2 * L3;
        s.N[6] = 4 * L1 * L3;
        s.N[7] = 4 * L1 * L4;
        s.N[8] = 4 * L2 * L4;
        s.N[9] = 4 * L3 * L4;
        s.dNdL1[0] = 4 * L1 - 1;
        s.dNdL1[1] = 0.0;
        s.dNdL1[2] = 0.0;
        s.dNdL1[3] = -(4 * L4 - 1);
        s.dNdL1[4] = 4 * L2;
        s.dNdL1[5] = 0.0;
        s.dNdL1[6] = 4 * L3;
        s.dNdL1[7] = 4 * (L4 - L1);
        s.dNdL1[8] = -4 * L2;
        s.dNdL1[9] = -4 * L3;
        s.dNdL2[0] = 0.0;
        s.dNdL2[1] = 4 * L2 - 1;
        s.dNdL2[2] = 0.0;
        s.dNdL2[3] = -(4 * L4 - 1);
        s.dNdL2[4] = 4 * L1;
        s.dNdL2[5] = 4 * L3;
        s.dNdL2[6] = 0.0;
        s.dNdL2[7] = -4 * L1;
        s.dNdL2[8] = 4 * (L4 - L2);
        s.dNdL2[9] = -4 * L3;
        s.dNdL3[0] = 0.0;
        s.dNdL3[1] = 0.0;
        s.dNdL3[2] = 4 * L3 - 1;
        s.dNdL3[3] = -(4 * L4 - 1);
        s.dNdL3[4] = 0.0;
        s.dNdL3[5] = 4 * L2;
        s.dNdL3[6] = 4 * L1;
        s.dNdL3[7] = -4 * L1;
        s.dNdL3[8] = -4 * L2;
        s.dNdL3[9] = 4 * (L4 - L3);
        return s;
      };
      auto eval_ctetra10 = [&](const double L1, const double L2,
                               const double L3) {
        auto nc = [&]() -> std::array<Vec3, 10> {
          std::array<Vec3, 10> c;
          for (int i = 0; i < 10; ++i)
            c[i] = model.node(elem_data.nodes[static_cast<size_t>(i)]).position;
          return c;
        }();
        std::array<double, 10> temp{};
        for (int i = 0; i < 10; ++i) {
          const auto it =
              nodal_temps_rec.find(elem_data.nodes[static_cast<size_t>(i)]);
          temp[static_cast<size_t>(i)] =
              (it != nodal_temps_rec.end()) ? it->second : mat_.ref_temp;
        }
        const auto sd = tet10_shape(L1, L2, L3);
        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 10; ++n) {
          J(0, 0) += sd.dNdL1[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].x;
          J(0, 1) += sd.dNdL1[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].y;
          J(0, 2) += sd.dNdL1[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].z;
          J(1, 0) += sd.dNdL2[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].x;
          J(1, 1) += sd.dNdL2[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].y;
          J(1, 2) += sd.dNdL2[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].z;
          J(2, 0) += sd.dNdL3[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].x;
          J(2, 1) += sd.dNdL3[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].y;
          J(2, 2) += sd.dNdL3[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].z;
        }
        const Eigen::Matrix3d Jinv = J.inverse();
        Eigen::MatrixXd B(6, 30);
        B.setZero();
        for (int n = 0; n < 10; ++n) {
          const double dnx =
              Jinv(0, 0) * sd.dNdL1[static_cast<size_t>(n)] +
              Jinv(0, 1) * sd.dNdL2[static_cast<size_t>(n)] +
              Jinv(0, 2) * sd.dNdL3[static_cast<size_t>(n)];
          const double dny =
              Jinv(1, 0) * sd.dNdL1[static_cast<size_t>(n)] +
              Jinv(1, 1) * sd.dNdL2[static_cast<size_t>(n)] +
              Jinv(1, 2) * sd.dNdL3[static_cast<size_t>(n)];
          const double dnz =
              Jinv(2, 0) * sd.dNdL1[static_cast<size_t>(n)] +
              Jinv(2, 1) * sd.dNdL2[static_cast<size_t>(n)] +
              Jinv(2, 2) * sd.dNdL3[static_cast<size_t>(n)];
          const int c0 = 3 * n;
          B(0, c0) = dnx;
          B(1, c0 + 1) = dny;
          B(2, c0 + 2) = dnz;
          B(3, c0) = dny;
          B(3, c0 + 1) = dnx;
          B(4, c0 + 1) = dnz;
          B(4, c0 + 2) = dny;
          B(5, c0) = dnz;
          B(5, c0 + 2) = dnx;
        }
        double point_temp = 0.0;
        for (int n = 0; n < 10; ++n)
          point_temp += sd.N[static_cast<size_t>(n)] * temp[static_cast<size_t>(n)];
        return solid_sigma_from_B(D_, B, ue, mat_.A, point_temp, mat_.ref_temp);
      };

      auto eval_cpenta6 = [&](const double L1, const double L2,
                              const double zeta) {
        auto nc = [&]() -> std::array<Vec3, 6> {
          std::array<Vec3, 6> c;
          for (int i = 0; i < 6; ++i)
            c[i] = model.node(elem_data.nodes[static_cast<size_t>(i)]).position;
          return c;
        }();
        std::array<double, 6> temp{};
        for (int i = 0; i < 6; ++i) {
          const auto it =
              nodal_temps_rec.find(elem_data.nodes[static_cast<size_t>(i)]);
          temp[static_cast<size_t>(i)] =
              (it != nodal_temps_rec.end()) ? it->second : mat_.ref_temp;
        }
        const auto sd = CPenta6::shape_functions(L1, L2, zeta);
        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 6; ++n) {
          J(0, 0) += sd.dNdL1[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].x;
          J(0, 1) += sd.dNdL1[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].y;
          J(0, 2) += sd.dNdL1[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].z;
          J(1, 0) += sd.dNdL2[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].x;
          J(1, 1) += sd.dNdL2[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].y;
          J(1, 2) += sd.dNdL2[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].z;
          J(2, 0) += sd.dNdzeta[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].x;
          J(2, 1) += sd.dNdzeta[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].y;
          J(2, 2) += sd.dNdzeta[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].z;
        }
        const Eigen::Matrix3d Jinv = J.inverse();
        Eigen::MatrixXd B(6, 18);
        B.setZero();
        for (int n = 0; n < 6; ++n) {
          const double dnx =
              Jinv(0, 0) * sd.dNdL1[static_cast<size_t>(n)] +
              Jinv(0, 1) * sd.dNdL2[static_cast<size_t>(n)] +
              Jinv(0, 2) * sd.dNdzeta[static_cast<size_t>(n)];
          const double dny =
              Jinv(1, 0) * sd.dNdL1[static_cast<size_t>(n)] +
              Jinv(1, 1) * sd.dNdL2[static_cast<size_t>(n)] +
              Jinv(1, 2) * sd.dNdzeta[static_cast<size_t>(n)];
          const double dnz =
              Jinv(2, 0) * sd.dNdL1[static_cast<size_t>(n)] +
              Jinv(2, 1) * sd.dNdL2[static_cast<size_t>(n)] +
              Jinv(2, 2) * sd.dNdzeta[static_cast<size_t>(n)];
          const int c0 = 3 * n;
          B(0, c0) = dnx;
          B(1, c0 + 1) = dny;
          B(2, c0 + 2) = dnz;
          B(3, c0) = dny;
          B(3, c0 + 1) = dnx;
          B(4, c0 + 1) = dnz;
          B(4, c0 + 2) = dny;
          B(5, c0) = dnz;
          B(5, c0 + 2) = dnx;
        }
        double point_temp = 0.0;
        for (int n = 0; n < 6; ++n)
          point_temp += sd.N[static_cast<size_t>(n)] * temp[static_cast<size_t>(n)];
        return solid_sigma_from_B(D_, B, ue, mat_.A, point_temp, mat_.ref_temp);
      };

      auto eval_chexa8 = [&](const double xi, const double eta,
                             const double zeta) {
        auto nc = [&]() -> std::array<Vec3, 8> {
          std::array<Vec3, 8> c;
          for (int i = 0; i < 8; ++i)
            c[i] = model.node(elem_data.nodes[static_cast<size_t>(i)]).position;
          return c;
        }();
        std::array<double, 8> temp{};
        for (int i = 0; i < 8; ++i) {
          const auto it =
              nodal_temps_rec.find(elem_data.nodes[static_cast<size_t>(i)]);
          temp[static_cast<size_t>(i)] =
              (it != nodal_temps_rec.end()) ? it->second : mat_.ref_temp;
        }
        const auto sd = CHexa8::shape_functions(xi, eta, zeta);
        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 8; ++n) {
          J(0, 0) += sd.dNdxi[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].x;
          J(0, 1) += sd.dNdxi[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].y;
          J(0, 2) += sd.dNdxi[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].z;
          J(1, 0) += sd.dNdeta[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].x;
          J(1, 1) += sd.dNdeta[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].y;
          J(1, 2) += sd.dNdeta[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].z;
          J(2, 0) += sd.dNdzeta[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].x;
          J(2, 1) += sd.dNdzeta[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].y;
          J(2, 2) += sd.dNdzeta[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].z;
        }
        const Eigen::Matrix3d Jinv = J.inverse();
        Eigen::MatrixXd B(6, 24);
        B.setZero();
        for (int n = 0; n < 8; ++n) {
          const double dnx =
              Jinv(0, 0) * sd.dNdxi[static_cast<size_t>(n)] +
              Jinv(0, 1) * sd.dNdeta[static_cast<size_t>(n)] +
              Jinv(0, 2) * sd.dNdzeta[static_cast<size_t>(n)];
          const double dny =
              Jinv(1, 0) * sd.dNdxi[static_cast<size_t>(n)] +
              Jinv(1, 1) * sd.dNdeta[static_cast<size_t>(n)] +
              Jinv(1, 2) * sd.dNdzeta[static_cast<size_t>(n)];
          const double dnz =
              Jinv(2, 0) * sd.dNdxi[static_cast<size_t>(n)] +
              Jinv(2, 1) * sd.dNdeta[static_cast<size_t>(n)] +
              Jinv(2, 2) * sd.dNdzeta[static_cast<size_t>(n)];
          const int c0 = 3 * n;
          B(0, c0) = dnx;
          B(1, c0 + 1) = dny;
          B(2, c0 + 2) = dnz;
          B(3, c0) = dny;
          B(3, c0 + 1) = dnx;
          B(4, c0 + 1) = dnz;
          B(4, c0 + 2) = dny;
          B(5, c0) = dnz;
          B(5, c0 + 2) = dnx;
        }
        double point_temp = 0.0;
        for (int n = 0; n < 8; ++n)
          point_temp += sd.N[static_cast<size_t>(n)] * temp[static_cast<size_t>(n)];
        return solid_sigma_from_B(D_, B, ue, mat_.A, point_temp, mat_.ref_temp);
      };

      if (elem_data.type == ElementType::CTETRA4) {
        sigma = eval_ctetra4();
        for (NodeId nid : elem_data.nodes)
          ss.nodal.push_back(make_solid_stress_point(nid, sigma));
      } else if (elem_data.type == ElementType::CTETRA10) {
        sigma = eval_ctetra10(0.25, 0.25, 0.25);
        static constexpr std::array<std::array<double, 3>, 10> kTet10Coords{{
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0},
            {0.0, 0.0, 0.0},
            {0.5, 0.5, 0.0},
            {0.0, 0.5, 0.5},
            {0.5, 0.0, 0.5},
            {0.5, 0.0, 0.0},
            {0.0, 0.5, 0.0},
            {0.0, 0.0, 0.5},
        }};
        for (int n = 0; n < 10; ++n) {
          const auto point =
              eval_ctetra10(kTet10Coords[static_cast<size_t>(n)][0],
                            kTet10Coords[static_cast<size_t>(n)][1],
                            kTet10Coords[static_cast<size_t>(n)][2]);
          ss.nodal.push_back(
              make_solid_stress_point(elem_data.nodes[static_cast<size_t>(n)],
                                      point));
        }
      } else if (elem_data.type == ElementType::CPENTA6) {
        sigma = eval_cpenta6(1.0 / 3.0, 1.0 / 3.0, 0.0);
        static constexpr std::array<std::array<double, 3>, 6> kPentaCoords{{
            {1.0, 0.0, -1.0},
            {0.0, 1.0, -1.0},
            {0.0, 0.0, -1.0},
            {1.0, 0.0, 1.0},
            {0.0, 1.0, 1.0},
            {0.0, 0.0, 1.0},
        }};
        for (int n = 0; n < 6; ++n) {
          const auto point =
              eval_cpenta6(kPentaCoords[static_cast<size_t>(n)][0],
                           kPentaCoords[static_cast<size_t>(n)][1],
                           kPentaCoords[static_cast<size_t>(n)][2]);
          ss.nodal.push_back(
              make_solid_stress_point(elem_data.nodes[static_cast<size_t>(n)],
                                      point));
        }
      } else {
        sigma = eval_chexa8(0.0, 0.0, 0.0);
        static constexpr std::array<std::array<double, 3>, 8> kHexCoords{{
            {-1.0, -1.0, -1.0},
            {1.0, -1.0, -1.0},
            {1.0, 1.0, -1.0},
            {-1.0, 1.0, -1.0},
            {-1.0, -1.0, 1.0},
            {1.0, -1.0, 1.0},
            {1.0, 1.0, 1.0},
            {-1.0, 1.0, 1.0},
        }};
        for (int n = 0; n < 8; ++n) {
          const auto point =
              eval_chexa8(kHexCoords[static_cast<size_t>(n)][0],
                          kHexCoords[static_cast<size_t>(n)][1],
                          kHexCoords[static_cast<size_t>(n)][2]);
          ss.nodal.push_back(
              make_solid_stress_point(elem_data.nodes[static_cast<size_t>(n)],
                                      point));
        }
      }

      ss.sx = sigma(0);
      ss.sy = sigma(1);
      ss.sz = sigma(2);
      ss.sxy = sigma(3);
      ss.syz = sigma(4);
      ss.szx = sigma(5);
      ss.von_mises = solid_von_mises(sigma);
      res.solid_stresses.push_back(ss);
    }
  }

  return res;
}

} // namespace vibestran
