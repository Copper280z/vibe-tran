// src/solver/linear_static.cpp
// Linear static analysis pipeline.
// K * u = F_ext + F_thermal

#include "solver/linear_static.hpp"
#include "core/coord_sys.hpp"
#include "core/logger.hpp"
#include "core/mpc_handler.hpp"
#include "elements/cquad4.hpp"
#include "elements/ctria3.hpp"
#include "elements/element_factory.hpp"
#include "elements/rbe_constraints.hpp"
#include "elements/solid_elements.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numbers>
#include <spdlog/spdlog.h>

namespace vibestran {

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

  apply_thermal_loads(model, sc, mpc_handler, K_builder, F);
  const auto t5 = Clock::now();
  spdlog::debug("[subcase {}] apply_thermal_loads: {:.3f} ms", sc.id, Ms(t5 - t4).count());

  // 4. Solve
  auto csr = K_builder.build_csr();
  const auto t5b = Clock::now();
  spdlog::debug("[subcase {}] build_csr: {:.3f} ms  ({} nnz)", sc.id, Ms(t5b - t5).count(), csr.nnz);

  std::vector<double> u_reduced = backend_->solve(csr, F);
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
    double r_norm_sq = 0.0;
    double f_norm_sq = 0.0;
    const auto& rp = csr.row_ptr;
    const auto& ci = csr.col_ind;
    const auto& cv = csr.values;
    for (int row = 0; row < n; ++row) {
      double ku = 0.0;
      for (int idx = rp[static_cast<size_t>(row)];
           idx < rp[static_cast<size_t>(row + 1)]; ++idx)
        ku += cv[static_cast<size_t>(idx)] *
              u_reduced[static_cast<size_t>(ci[static_cast<size_t>(idx)])];
      const double ri = ku - F[static_cast<size_t>(row)];
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
  DofMap dmap;
  dmap.build(model.nodes, 6);

  // Apply SPCs
  // For nodes with CD≠0 (non-basic displacement coordinate system),
  // translational SPC DOFs refer to CD-frame directions, not basic axes.
  // Skip those here; they are converted to MPCs in build_mpc_system().
  {
    std::vector<std::pair<NodeId, int>> spc_constraints;
    for (const Spc *spc : model.spcs_for_set(sc.spc_set)) {
      auto node_it = model.nodes.find(spc->node);
      bool has_cd = (node_it != model.nodes.end() &&
                     node_it->second.cd.value != 0);
      for (int d = 0; d < 6; ++d) {
        if (!spc->dofs.has(d + 1))
          continue;
        // Skip translational DOFs for CD≠0 nodes — handled as MPCs
        if (has_cd && d < 3)
          continue;
        spc_constraints.emplace_back(spc->node, d);
      }
    }
    dmap.constrain_batch(spc_constraints);
  }

  // Constrain rotational DOFs on solid-element-only nodes
  std::unordered_map<NodeId, bool> node_has_shell;
  for (const auto &nid_gp : model.nodes)
    node_has_shell[nid_gp.first] = false;

  for (const auto &elem : model.elements) {
    bool is_shell =
        (elem.type == ElementType::CQUAD4 || elem.type == ElementType::CTRIA3);
    if (is_shell)
      for (NodeId nid : elem.nodes)
        node_has_shell[nid] = true;
  }

  {
    std::vector<std::pair<NodeId, int>> rot_constraints;
    rot_constraints.reserve(node_has_shell.size() * 3);
    for (const auto &[nid, has_shell] : node_has_shell)
      if (!has_shell)
        for (int d = 3; d < 6; ++d)
          rot_constraints.emplace_back(nid, d);
    dmap.constrain_batch(rot_constraints);
  }

  return dmap;
}

void LinearStaticSolver::build_mpc_system(const Model &model,
                                           const SubCase &sc,
                                           DofMap &dof_map,
                                           MpcHandler &mpc_handler) {
  // Collect MPCs from all sources into one vector
  std::vector<Mpc> all_mpcs;

  // 1. CD-frame SPCs: for nodes with CD≠basic, translational SPC DOFs refer
  //    to CD-frame directions.  Each CD-frame constraint becomes either:
  //    (a) a direct basic SPC (when the constraint aligns with a basic axis), or
  //    (b) an MPC: T3[:,d]^T · [u1,u2,u3] = 0.
  //
  //    When a node has multiple CD-frame SPCs, we use Gaussian elimination
  //    with column pivoting to produce independent MPCs with unique dependent
  //    DOFs, avoiding conflicts in the MPC handler.
  {
    // Pre-build node→SPC lookup
    std::unordered_map<NodeId, DofSet> node_spc_dofs;
    for (const Spc *spc : model.spcs_for_set(sc.spc_set)) {
      if (spc->value != 0.0) continue;
      node_spc_dofs[spc->node].mask |= spc->dofs.mask;
    }

    std::vector<std::pair<NodeId, int>> direct_constraints;

    for (const auto &[nid, gp] : model.nodes) {
      if (gp.cd == CoordId{0})
        continue;
      auto spc_it = node_spc_dofs.find(nid);
      if (spc_it == node_spc_dofs.end())
        continue;

      auto cs_it = model.coord_systems.find(gp.cd);
      if (cs_it == model.coord_systems.end())
        continue;

      const CoordSys &cs = cs_it->second;
      Mat3 T3 = rotation_matrix(cs, gp.position);
      DofSet dofs = spc_it->second;

      // Collect constrained translational CD-frame DOFs
      int cd_dofs[3];
      int n_trans = 0;
      for (int d = 0; d < 3; ++d)
        if (dofs.has(d + 1))
          cd_dofs[n_trans++] = d;

      if (n_trans == 3) {
        // All translational DOFs constrained → constrain all basic
        for (int j = 0; j < 3; ++j)
          direct_constraints.emplace_back(nid, j);
      } else if (n_trans > 0) {
        // Build constraint matrix A[i][j] = T3(j, cd_dofs[i])
        // Each row is a CD direction expressed in basic coordinates.
        // Gaussian elimination with column pivoting produces independent
        // equations with unique dominant basic DOFs.
        double A[3][3] = {};
        int col_perm[3] = {0, 1, 2};

        for (int i = 0; i < n_trans; ++i)
          for (int j = 0; j < 3; ++j)
            A[i][j] = T3(j, cd_dofs[i]);

        for (int row = 0; row < n_trans; ++row) {
          // Partial column pivoting: find largest |A[row][col]| for col >= row
          int best_col = row;
          double best_val = std::abs(A[row][row]);
          for (int col = row + 1; col < 3; ++col) {
            if (std::abs(A[row][col]) > best_val) {
              best_val = std::abs(A[row][col]);
              best_col = col;
            }
          }
          if (best_col != row) {
            for (int i = 0; i < n_trans; ++i)
              std::swap(A[i][row], A[i][best_col]);
            std::swap(col_perm[row], col_perm[best_col]);
          }
          // Eliminate rows below the pivot
          if (std::abs(A[row][row]) < 1e-14) continue;
          for (int i = row + 1; i < n_trans; ++i) {
            double factor = A[i][row] / A[row][row];
            for (int j = row; j < 3; ++j)
              A[i][j] -= factor * A[row][j];
          }
        }

        // Generate either direct SPCs or MPCs from the triangular system
        for (int row = 0; row < n_trans; ++row) {
          if (std::abs(A[row][row]) < 1e-14) continue;

          // Count non-zero entries in this row (from diagonal onward)
          int nnz = 0;
          for (int col = row; col < 3; ++col)
            if (std::abs(A[row][col]) > 1e-14) nnz++;

          if (nnz == 1) {
            // Single non-zero → direct basic SPC (constraint aligns with axis)
            direct_constraints.emplace_back(nid, col_perm[row]);
          } else {
            // Multiple terms → MPC
            Mpc mpc;
            mpc.sid = MpcSetId{0};
            for (int col = row; col < 3; ++col) {
              if (std::abs(A[row][col]) > 1e-14)
                mpc.terms.push_back({nid, col_perm[col] + 1, A[row][col]});
            }
            if (!mpc.terms.empty())
              all_mpcs.push_back(std::move(mpc));
          }
        }
      }

      // Rotational DOFs: solid-only nodes already have all rotations
      // constrained in build_dof_map, so CD-frame rotation SPCs are
      // automatically satisfied.  For future shell support, rotational
      // CD-frame SPCs would need similar treatment here.
    }

    // Apply any direct basic constraints identified above
    if (!direct_constraints.empty())
      dof_map.constrain_batch(direct_constraints);
  }

  // 2. RBE2/RBE3
  for (const auto &rbe2 : model.rbe2s)
    expand_rbe2(rbe2, model, all_mpcs);
  for (const auto &rbe3 : model.rbe3s)
    expand_rbe3(rbe3, model, all_mpcs);

  // 3. Explicit MPCs from active MPC set
  if (sc.mpc_set.value != 0) {
    for (const Mpc *mpc : model.mpcs_for_set(sc.mpc_set))
      all_mpcs.push_back(*mpc);
  }

  std::vector<const Mpc*> mpc_ptrs;
  mpc_ptrs.reserve(all_mpcs.size());
  for (const auto& m : all_mpcs)
    mpc_ptrs.push_back(&m);

  mpc_handler.build(mpc_ptrs, dof_map);
}

void LinearStaticSolver::assemble(const Model &model, const SubCase & /*sc*/,
                                  const MpcHandler &mpc_handler,
                                  SparseMatrixBuilder &K_builder,
                                  std::vector<double> & /*F*/) {
  const DofMap &dof_map = mpc_handler.full_dof_map();
  for (const auto &elem_data : model.elements) {
    auto elem = make_element(elem_data, model);
    LocalKe Ke = elem->stiffness_matrix();
    auto gdofs = elem->global_dof_indices(dof_map);

    const int nd = static_cast<int>(gdofs.size());
    std::vector<double> ke_row(static_cast<size_t>(nd * nd));
    for (int r = 0; r < nd; ++r)
      for (int c = 0; c < nd; ++c)
        ke_row[static_cast<size_t>(r * nd + c)] = Ke(r, c);

    mpc_handler.apply_to_stiffness(gdofs, ke_row, K_builder);
  }
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
    const auto& prop = model.property(elem_data.pid);
    MaterialId mid{0};
    if (std::holds_alternative<PShell>(prop))
      mid = std::get<PShell>(prop).mid1;
    else if (std::holds_alternative<PSolid>(prop))
      mid = std::get<PSolid>(prop).mid;
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
  std::unordered_map<NodeId, double> nodal_temps_rec;
  int temp_set_rec = sc.temp_load_set;
  if (temp_set_rec == 0) temp_set_rec = sc.load_set.value;
  for (const Load *lp : model.loads_for_set(LoadSetId(temp_set_rec))) {
    if (const TempLoad *tl = std::get_if<TempLoad>(lp))
      nodal_temps_rec[tl->node] = tl->temperature;
  }
  auto tempd_rec_it = model.tempd.find(temp_set_rec);
  if (tempd_rec_it != model.tempd.end()) {
    double T_default = tempd_rec_it->second;
    for (const auto& [nid, _] : model.nodes) {
      if (nodal_temps_rec.find(nid) == nodal_temps_rec.end())
        nodal_temps_rec[nid] = T_default;
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

    if (elem_data.type == ElementType::CQUAD4 ||
        elem_data.type == ElementType::CTRIA3) {
      PlateStress ps;
      ps.eid   = elem_data.id;
      ps.etype = elem_data.type;

      if (elem_data.type == ElementType::CQUAD4) {
        auto node_c = [&]() -> std::array<Vec3, 4> {
          std::array<Vec3, 4> c;
          for (int i = 0; i < 4; ++i)
            c[i] = model.node(elem_data.nodes[i]).position;
          return c;
        }();
        auto sd = CQuad4::shape_functions(0.0, 0.0);
        Eigen::Matrix2d J = Eigen::Matrix2d::Zero();
        for (int n = 0; n < 4; ++n) {
          J(0, 0) += sd.dNdxi[n] * node_c[n].x;
          J(0, 1) += sd.dNdxi[n] * node_c[n].y;
          J(1, 0) += sd.dNdeta[n] * node_c[n].x;
          J(1, 1) += sd.dNdeta[n] * node_c[n].y;
        }
        Eigen::Matrix2d Jinv = J.inverse();
        Eigen::MatrixXd dNdx(2, 4);
        for (int n = 0; n < 4; ++n) {
          dNdx(0, n) = Jinv(0, 0) * sd.dNdxi[n] + Jinv(0, 1) * sd.dNdeta[n];
          dNdx(1, n) = Jinv(1, 0) * sd.dNdxi[n] + Jinv(1, 1) * sd.dNdeta[n];
        }
        Eigen::MatrixXd Bm(3, 8);
        Bm.setZero();
        for (int n = 0; n < 4; ++n) {
          Bm(0, 2 * n) = dNdx(0, n);
          Bm(1, 2 * n + 1) = dNdx(1, n);
          Bm(2, 2 * n) = dNdx(1, n);
          Bm(2, 2 * n + 1) = dNdx(0, n);
        }
        Eigen::VectorXd u_mem(8);
        for (int n = 0; n < 4; ++n) {
          u_mem(2 * n) = ue(6 * n);
          u_mem(2 * n + 1) = ue(6 * n + 1);
        }
        const auto &pshell_ = std::get<PShell>(model.property(elem_data.pid));
        const Mat1 &mat_ = model.material(pshell_.mid1);
        double E_ = mat_.E, nu_ = mat_.nu;
        double c_ = E_ / (1 - nu_ * nu_);
        Eigen::Matrix3d Dm_;
        Dm_ << c_, c_ * nu_, 0, c_ * nu_, c_, 0, 0, 0, c_ * (1 - nu_) / 2;

        double T_avg4 = 0.0;
        for (int n = 0; n < 4; ++n) {
          auto it = nodal_temps_rec.find(elem_data.nodes[n]);
          T_avg4 += (it != nodal_temps_rec.end()) ? it->second : mat_.ref_temp;
        }
        T_avg4 /= 4.0;
        double dT4 = T_avg4 - mat_.ref_temp;
        double alpha4 = mat_.A;
        Eigen::Vector3d eps_th4{alpha4 * dT4, alpha4 * dT4, 0.0};

        Eigen::Vector3d sigma = Dm_ * (Bm * u_mem - eps_th4);
        ps.sx = sigma(0); ps.sy = sigma(1); ps.sxy = sigma(2);
        ps.mx = 0; ps.my = 0; ps.mxy = 0;
        ps.von_mises = std::sqrt(ps.sx * ps.sx - ps.sx * ps.sy + ps.sy * ps.sy +
                                 3 * ps.sxy * ps.sxy);
      } else {
        auto node_c = [&]() -> std::array<Vec3, 3> {
          std::array<Vec3, 3> c;
          for (int i = 0; i < 3; ++i)
            c[i] = model.node(elem_data.nodes[i]).position;
          return c;
        }();
        double x1 = node_c[0].x, y1 = node_c[0].y;
        double x2 = node_c[1].x, y2 = node_c[1].y;
        double x3 = node_c[2].x, y3 = node_c[2].y;
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
          u_mem(2 * n) = ue(6 * n);
          u_mem(2 * n + 1) = ue(6 * n + 1);
        }
        double T_avg3 = 0.0;
        for (int n = 0; n < 3; ++n) {
          auto it = nodal_temps_rec.find(elem_data.nodes[n]);
          T_avg3 += (it != nodal_temps_rec.end()) ? it->second : mat_.ref_temp;
        }
        T_avg3 /= 3.0;
        double dT3 = T_avg3 - mat_.ref_temp;
        double alpha3 = mat_.A;
        Eigen::Vector3d eps_th3{alpha3 * dT3, alpha3 * dT3, 0.0};

        Eigen::Vector3d sigma = Dm_ * (Bm * u_mem - eps_th3);
        ps.sx = sigma(0); ps.sy = sigma(1); ps.sxy = sigma(2);
        ps.mx = 0; ps.my = 0; ps.mxy = 0;
        ps.von_mises = std::sqrt(ps.sx * ps.sx - ps.sx * ps.sy + ps.sy * ps.sy +
                                 3 * ps.sxy * ps.sxy);
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
      Eigen::Matrix<double, 6, 6> D_ = [&]() {
        double lam = mat_.E * mat_.nu / ((1 + mat_.nu) * (1 - 2 * mat_.nu));
        double mu_ = mat_.E / (2 * (1 + mat_.nu));
        Eigen::Matrix<double, 6, 6> D;
        D.setZero();
        D(0, 0) = lam + 2 * mu_; D(0, 1) = lam; D(0, 2) = lam;
        D(1, 0) = lam; D(1, 1) = lam + 2 * mu_; D(1, 2) = lam;
        D(2, 0) = lam; D(2, 1) = lam; D(2, 2) = lam + 2 * mu_;
        D(3, 3) = mu_; D(4, 4) = mu_; D(5, 5) = mu_;
        return D;
      }();

      Eigen::Matrix<double, 6, 1> sigma;
      sigma.setZero();

      if (elem_data.type == ElementType::CTETRA10) {
        auto nc10 = [&]() -> std::array<Vec3,10> {
          std::array<Vec3,10> arr;
          for (int i = 0; i < 10; ++i)
            arr[i] = model.node(elem_data.nodes[i]).position;
          return arr;
        }();
        double L1=0.25, L2=0.25, L3=0.25;
        double L4 = 1.0 - L1 - L2 - L3;
        std::array<double,10> dNdL1, dNdL2, dNdL3;
        dNdL1[0] = 4*L1 - 1; dNdL1[1] = 0; dNdL1[2] = 0; dNdL1[3] = -(4*L4-1);
        dNdL1[4] = 4*L2; dNdL1[5] = 0; dNdL1[6] = 4*L3; dNdL1[7] = 4*(L4-L1); dNdL1[8] = -4*L2; dNdL1[9] = -4*L3;
        dNdL2[0] = 0; dNdL2[1] = 4*L2-1; dNdL2[2] = 0; dNdL2[3] = -(4*L4-1);
        dNdL2[4] = 4*L1; dNdL2[5] = 4*L3; dNdL2[6] = 0; dNdL2[7] = -4*L1; dNdL2[8] = 4*(L4-L2); dNdL2[9] = -4*L3;
        dNdL3[0] = 0; dNdL3[1] = 0; dNdL3[2] = 4*L3-1; dNdL3[3] = -(4*L4-1);
        dNdL3[4] = 0; dNdL3[5] = 4*L2; dNdL3[6] = 4*L1; dNdL3[7] = -4*L1; dNdL3[8] = -4*L2; dNdL3[9] = 4*(L4-L3);

        Eigen::Matrix3d J10 = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 10; ++n) {
          J10(0,0)+=dNdL1[n]*nc10[n].x; J10(0,1)+=dNdL1[n]*nc10[n].y; J10(0,2)+=dNdL1[n]*nc10[n].z;
          J10(1,0)+=dNdL2[n]*nc10[n].x; J10(1,1)+=dNdL2[n]*nc10[n].y; J10(1,2)+=dNdL2[n]*nc10[n].z;
          J10(2,0)+=dNdL3[n]*nc10[n].x; J10(2,1)+=dNdL3[n]*nc10[n].y; J10(2,2)+=dNdL3[n]*nc10[n].z;
        }
        Eigen::Matrix3d Jinv10 = J10.inverse();
        Eigen::MatrixXd B10(6, 30); B10.setZero();
        for (int n = 0; n < 10; ++n) {
          double dnx = Jinv10(0,0)*dNdL1[n]+Jinv10(0,1)*dNdL2[n]+Jinv10(0,2)*dNdL3[n];
          double dny = Jinv10(1,0)*dNdL1[n]+Jinv10(1,1)*dNdL2[n]+Jinv10(1,2)*dNdL3[n];
          double dnz = Jinv10(2,0)*dNdL1[n]+Jinv10(2,1)*dNdL2[n]+Jinv10(2,2)*dNdL3[n];
          int c0=3*n;
          B10(0,c0)=dnx; B10(1,c0+1)=dny; B10(2,c0+2)=dnz;
          B10(3,c0)=dny; B10(3,c0+1)=dnx;
          B10(4,c0+1)=dnz; B10(4,c0+2)=dny;
          B10(5,c0)=dnz; B10(5,c0+2)=dnx;
        }
        double T10=0;
        for (int i=0; i<10; ++i) {
          auto it=nodal_temps_rec.find(elem_data.nodes[i]);
          T10 += (it!=nodal_temps_rec.end()) ? it->second : mat_.ref_temp;
        }
        T10 /= 10.0;
        double dT10 = T10 - mat_.ref_temp;
        Eigen::Matrix<double,6,1> eps_th10;
        eps_th10 << mat_.A*dT10, mat_.A*dT10, mat_.A*dT10, 0, 0, 0;
        sigma = D_ * (B10 * ue - eps_th10);
      } else if (elem_data.type == ElementType::CTETRA4) {
        auto nc = [&]() -> std::array<Vec3, 4> {
          std::array<Vec3, 4> c;
          for (int i = 0; i < 4; ++i)
            c[i] = model.node(elem_data.nodes[i]).position;
          return c;
        }();
        double x1 = nc[0].x, y1 = nc[0].y, z1 = nc[0].z;
        double x2 = nc[1].x, y2 = nc[1].y, z2 = nc[1].z;
        double x3 = nc[2].x, y3 = nc[2].y, z3 = nc[2].z;
        double x4 = nc[3].x, y4 = nc[3].y, z4 = nc[3].z;
        Eigen::Matrix4d A4;
        A4 << 1, x1, y1, z1, 1, x2, y2, z2, 1, x3, y3, z3, 1, x4, y4, z4;
        double V6 = A4.determinant();
        Eigen::Matrix4d cofA = Eigen::Matrix4d::Zero();
        for (int i = 0; i < 4; ++i)
          for (int j = 0; j < 4; ++j) {
            Eigen::Matrix3d m3;
            int ri = 0;
            for (int r = 0; r < 4; ++r) {
              if (r == j) continue;
              int ci_ = 0;
              for (int cc = 0; cc < 4; ++cc) {
                if (cc == i) continue;
                m3(ri, ci_++) = A4(r, cc);
              }
              ri++;
            }
            cofA(i, j) = std::pow(-1.0, i + j) * m3.determinant();
          }
        Eigen::MatrixXd B(6, 12);
        B.setZero();
        for (int n = 0; n < 4; ++n) {
          double bx = cofA(1, n) / V6, by = cofA(2, n) / V6,
                 bz = cofA(3, n) / V6;
          int c0 = 3 * n;
          B(0, c0) = bx; B(1, c0+1) = by; B(2, c0+2) = bz;
          B(3, c0) = by; B(3, c0+1) = bx;
          B(4, c0+1) = bz; B(4, c0+2) = by;
          B(5, c0) = bz; B(5, c0+2) = bx;
        }
        double T_avg_tet = 0.0;
        for (int i = 0; i < 4; ++i) {
          auto it = nodal_temps_rec.find(elem_data.nodes[i]);
          T_avg_tet += (it != nodal_temps_rec.end()) ? it->second : mat_.ref_temp;
        }
        T_avg_tet /= 4.0;
        double dT_tet = T_avg_tet - mat_.ref_temp;
        Eigen::Matrix<double, 6, 1> eps_th_tet;
        eps_th_tet << mat_.A*dT_tet, mat_.A*dT_tet, mat_.A*dT_tet, 0, 0, 0;
        sigma = D_ * (B * ue - eps_th_tet);
      } else if (elem_data.type == ElementType::CPENTA6) {
        // Centroid stress recovery: L1=1/3, L2=1/3, zeta=0
        auto sd = CPenta6::shape_functions(1.0/3.0, 1.0/3.0, 0.0);
        auto nc = [&]() -> std::array<Vec3, 6> {
          std::array<Vec3, 6> c;
          for (int i = 0; i < 6; ++i)
            c[i] = model.node(elem_data.nodes[i]).position;
          return c;
        }();
        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 6; ++n) {
          J(0,0) += sd.dNdL1[n]*nc[n].x; J(0,1) += sd.dNdL1[n]*nc[n].y; J(0,2) += sd.dNdL1[n]*nc[n].z;
          J(1,0) += sd.dNdL2[n]*nc[n].x; J(1,1) += sd.dNdL2[n]*nc[n].y; J(1,2) += sd.dNdL2[n]*nc[n].z;
          J(2,0) += sd.dNdzeta[n]*nc[n].x; J(2,1) += sd.dNdzeta[n]*nc[n].y; J(2,2) += sd.dNdzeta[n]*nc[n].z;
        }
        Eigen::Matrix3d Jinv = J.inverse();
        Eigen::MatrixXd B(6, 18); B.setZero();
        for (int n = 0; n < 6; ++n) {
          double dnx = Jinv(0,0)*sd.dNdL1[n]+Jinv(0,1)*sd.dNdL2[n]+Jinv(0,2)*sd.dNdzeta[n];
          double dny = Jinv(1,0)*sd.dNdL1[n]+Jinv(1,1)*sd.dNdL2[n]+Jinv(1,2)*sd.dNdzeta[n];
          double dnz = Jinv(2,0)*sd.dNdL1[n]+Jinv(2,1)*sd.dNdL2[n]+Jinv(2,2)*sd.dNdzeta[n];
          int c0 = 3*n;
          B(0,c0)=dnx; B(1,c0+1)=dny; B(2,c0+2)=dnz;
          B(3,c0)=dny; B(3,c0+1)=dnx;
          B(4,c0+1)=dnz; B(4,c0+2)=dny;
          B(5,c0)=dnz; B(5,c0+2)=dnx;
        }
        double T_avg_penta = 0.0;
        for (int i = 0; i < 6; ++i) {
          auto it = nodal_temps_rec.find(elem_data.nodes[i]);
          T_avg_penta += (it != nodal_temps_rec.end()) ? it->second : mat_.ref_temp;
        }
        T_avg_penta /= 6.0;
        double dT_penta = T_avg_penta - mat_.ref_temp;
        Eigen::Matrix<double,6,1> eps_th_penta;
        eps_th_penta << mat_.A*dT_penta, mat_.A*dT_penta, mat_.A*dT_penta, 0, 0, 0;
        sigma = D_ * (B * ue - eps_th_penta);
      } else {
        auto sd = CHexa8::shape_functions(0, 0, 0);
        auto nc = [&]() -> std::array<Vec3, 8> {
          std::array<Vec3, 8> c;
          for (int i = 0; i < 8; ++i)
            c[i] = model.node(elem_data.nodes[i]).position;
          return c;
        }();
        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 8; ++n) {
          J(0, 0) += sd.dNdxi[n] * nc[n].x;
          J(0, 1) += sd.dNdxi[n] * nc[n].y;
          J(0, 2) += sd.dNdxi[n] * nc[n].z;
          J(1, 0) += sd.dNdeta[n] * nc[n].x;
          J(1, 1) += sd.dNdeta[n] * nc[n].y;
          J(1, 2) += sd.dNdeta[n] * nc[n].z;
          J(2, 0) += sd.dNdzeta[n] * nc[n].x;
          J(2, 1) += sd.dNdzeta[n] * nc[n].y;
          J(2, 2) += sd.dNdzeta[n] * nc[n].z;
        }
        Eigen::Matrix3d Jinv = J.inverse();
        Eigen::MatrixXd dNdx(3, 8);
        for (int n = 0; n < 8; ++n) {
          dNdx(0, n) = Jinv(0,0)*sd.dNdxi[n] + Jinv(0,1)*sd.dNdeta[n] + Jinv(0,2)*sd.dNdzeta[n];
          dNdx(1, n) = Jinv(1,0)*sd.dNdxi[n] + Jinv(1,1)*sd.dNdeta[n] + Jinv(1,2)*sd.dNdzeta[n];
          dNdx(2, n) = Jinv(2,0)*sd.dNdxi[n] + Jinv(2,1)*sd.dNdeta[n] + Jinv(2,2)*sd.dNdzeta[n];
        }
        Eigen::MatrixXd B(6, 24);
        B.setZero();
        for (int n = 0; n < 8; ++n) {
          double dnx = dNdx(0, n), dny = dNdx(1, n), dnz = dNdx(2, n);
          int c0 = 3 * n;
          B(0, c0) = dnx; B(1, c0+1) = dny; B(2, c0+2) = dnz;
          B(3, c0) = dny; B(3, c0+1) = dnx;
          B(4, c0+1) = dnz; B(4, c0+2) = dny;
          B(5, c0) = dnz; B(5, c0+2) = dnx;
        }
        double T_avg_hex = 0.0;
        for (int i = 0; i < 8; ++i) {
          auto it = nodal_temps_rec.find(elem_data.nodes[i]);
          T_avg_hex += (it != nodal_temps_rec.end()) ? it->second : mat_.ref_temp;
        }
        T_avg_hex /= 8.0;
        double dT_hex = T_avg_hex - mat_.ref_temp;
        Eigen::Matrix<double, 6, 1> eps_th_hex;
        eps_th_hex << mat_.A*dT_hex, mat_.A*dT_hex, mat_.A*dT_hex, 0, 0, 0;
        sigma = D_ * (B * ue - eps_th_hex);
      }

      ss.sx = sigma(0); ss.sy = sigma(1); ss.sz = sigma(2);
      ss.sxy = sigma(3); ss.syz = sigma(4); ss.szx = sigma(5);
      double s1 = ss.sx, s2 = ss.sy, s3 = ss.sz,
             t12 = ss.sxy, t23 = ss.syz, t31 = ss.szx;
      ss.von_mises =
          std::sqrt(0.5 * ((s1-s2)*(s1-s2) + (s2-s3)*(s2-s3) +
                           (s3-s1)*(s3-s1) +
                           6*(t12*t12 + t23*t23 + t31*t31)));
      res.solid_stresses.push_back(ss);
    }
  }

  return res;
}

} // namespace vibestran
