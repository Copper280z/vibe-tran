// src/solver/linear_static.cpp
// Linear static analysis pipeline.
// K * u = F_ext + F_thermal

#include "solver/linear_static.hpp"
#include "elements/cquad4.hpp"
#include "elements/ctria3.hpp"
#include "elements/element_factory.hpp"
#include "elements/solid_elements.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <format>
#include <numbers>

namespace nastran {

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
  // 1. Build DOF map and apply boundary conditions
  DofMap dof_map = build_dof_map(model, sc);
  const int n = dof_map.num_free_dofs();

  // 2. Assemble global K and F
  SparseMatrixBuilder K_builder(n);
  std::vector<double> F(static_cast<size_t>(n), 0.0);

  assemble(model, sc, dof_map, K_builder, F);
  apply_point_loads(model, sc, dof_map, F);
  apply_thermal_loads(model, sc, dof_map, K_builder, F);

  // 3. Solve
  auto csr = K_builder.build_csr();
  std::vector<double> u_free = backend_->solve(csr, F);

  // 4. Recover results
  return recover_results(model, sc, dof_map, u_free);
}

DofMap LinearStaticSolver::build_dof_map(const Model &model,
                                         const SubCase &sc) {
  // Determine default DOF count per node:
  // If any shell element exists → 6 DOF/node
  // Solid-only → 3 DOF/node (but to be safe, use 6 always and just constrain
  // extra) For simplicity: always use 6 DOF/node; solid elements only use
  // T1-T3.
  DofMap dmap;
  dmap.build(model.nodes, 6);

  // Apply SPCs
  for (const Spc *spc : model.spcs_for_set(sc.spc_set)) {
    for (int d = 0; d < 6; ++d) {
      if (spc->dofs.has(d + 1))
        dmap.constrain(spc->node, d);
    }
  }

  // For solid-element-only nodes: constrain rotational DOFs (3-5)
  // We do this by checking which nodes are only connected to solid elements.
  // Build node → element type mapping
  std::unordered_map<NodeId, bool> node_has_shell;
  for (const auto &nid_gp : model.nodes)
    node_has_shell[nid_gp.first] = false;

  for (const auto &elem : model.elements) {
    bool is_shell =
        (elem.type == ElementType::CQUAD4 || elem.type == ElementType::CTRIA3);
    // CTETRA10 nodes are solid-only (no rotational DOFs needed)
    if (is_shell)
      for (NodeId nid : elem.nodes)
        node_has_shell[nid] = true;
  }

  // Constrain rotational DOFs on nodes that have NO shell element
  for (const auto &[nid, has_shell] : node_has_shell) {
    if (!has_shell) {
      for (int d = 3; d < 6; ++d)
        dmap.constrain(nid, d);
    }
  }

  return dmap;
}

void LinearStaticSolver::assemble(const Model &model, const SubCase & /*sc*/,
                                  const DofMap &dof_map,
                                  SparseMatrixBuilder &K_builder,
                                  std::vector<double> & /*F*/) {
  for (const auto &elem_data : model.elements) {
    auto elem = make_element(elem_data, model);
    LocalKe Ke = elem->stiffness_matrix();
    auto gdofs = elem->global_dof_indices(dof_map);

    // Convert to int32 span for add_element_stiffness
    std::vector<int32_t> gdofs32(gdofs.begin(), gdofs.end());

    // Eigen stores column-major; we need row-major for our assembly
    // Transpose: Ke.data() is col-major, so we re-extract row-major
    const int nd = static_cast<int>(gdofs.size());
    std::vector<double> ke_row(static_cast<size_t>(nd * nd));
    for (int r = 0; r < nd; ++r)
      for (int c = 0; c < nd; ++c)
        ke_row[static_cast<size_t>(r * nd + c)] = Ke(r, c);

    K_builder.add_element_stiffness(gdofs32, ke_row);
  }
}

void LinearStaticSolver::apply_point_loads(const Model &model,
                                           const SubCase &sc,
                                           const DofMap &dof_map,
                                           std::vector<double> &F) {
  for (const Load *lp : model.loads_for_set(sc.load_set)) {
    std::visit(
        [&](const auto &load) {
          using T = std::decay_t<decltype(load)>;

          if constexpr (std::is_same_v<T, ForceLoad>) {
            // Effective force = scale * direction (direction has magnitude in
            // Nastran)
            double fx = load.scale * load.direction.x;
            double fy = load.scale * load.direction.y;
            double fz = load.scale * load.direction.z;

            auto add_f = [&](int dof_0based, double val) {
              EqIndex eq = dof_map.eq_index(load.node, dof_0based);
              if (eq != CONSTRAINED_DOF && eq < static_cast<int>(F.size()))
                F[static_cast<size_t>(eq)] += val;
            };
            add_f(0, fx);
            add_f(1, fy);
            add_f(2, fz);
          } else if constexpr (std::is_same_v<T, MomentLoad>) {
            double mx = load.scale * load.direction.x;
            double my = load.scale * load.direction.y;
            double mz = load.scale * load.direction.z;

            auto add_f = [&](int dof_0based, double val) {
              EqIndex eq = dof_map.eq_index(load.node, dof_0based);
              if (eq != CONSTRAINED_DOF && eq < static_cast<int>(F.size()))
                F[static_cast<size_t>(eq)] += val;
            };
            add_f(3, mx);
            add_f(4, my);
            add_f(5, mz);
          }
          // TempLoad handled separately
        },
        *lp);
  }
}

void LinearStaticSolver::apply_thermal_loads(
    const Model &model, const SubCase &sc, const DofMap &dof_map,
    SparseMatrixBuilder & /*K_builder*/, std::vector<double> &F) {
  // Build nodal temperature map for this load set
  std::unordered_map<NodeId, double> nodal_temps;

  // Default: use t_ref as initial temperature (no thermal strain if T = T_ref)
  double t_ref = sc.t_ref;

  for (const Load *lp : model.loads_for_set(sc.load_set)) {
    if (const TempLoad *tl = std::get_if<TempLoad>(lp))
      nodal_temps[tl->node] = tl->temperature;
  }

  if (nodal_temps.empty())
    return; // No thermal loads

  // For each element, gather nodal temperatures and compute thermal load vector
  for (const auto &elem_data : model.elements) {
    auto elem = make_element(elem_data, model);
    auto node_ids = elem->node_ids();
    const int nn = static_cast<int>(node_ids.size());

    std::vector<double> temps(static_cast<size_t>(nn));
    for (int i = 0; i < nn; ++i) {
      auto it = nodal_temps.find(node_ids[i]);
      // If node has no explicit temp, use t_ref (ΔT = 0 → no load)
      temps[static_cast<size_t>(i)] =
          (it != nodal_temps.end()) ? it->second : t_ref;
    }

    LocalFe fe = elem->thermal_load(temps, t_ref);
    auto gdofs = elem->global_dof_indices(dof_map);
    std::vector<int32_t> gdofs32(gdofs.begin(), gdofs.end());
    std::vector<double> fe_vec(fe.data(), fe.data() + fe.size());

    // Apply to F (subtract because K*u = F_ext - F_th convention in some texts,
    // but here we use K*u = F_ext + F_th where F_th is already -Bᵀ*D*ε_th
    // Our element thermal_load computes +Bᵀ*D*ε_th as the equivalent nodal
    // force)
    for (size_t i = 0; i < gdofs32.size(); ++i) {
      if (gdofs32[i] == CONSTRAINED_DOF)
        continue;
      if (gdofs32[i] < static_cast<int32_t>(F.size()))
        F[static_cast<size_t>(gdofs32[i])] += fe_vec[i];
    }
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
  // Sort nodes for deterministic output
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
    res.displacements.push_back(nd);
  }

  // ── Build nodal temperature map for thermal stress correction ────────────
  // σ_mech = D * (B*u - ε_th).  Without subtracting ε_th the recovered stress
  // would be D * ε_total, which is wrong for any case with thermal loads.
  std::unordered_map<NodeId, double> nodal_temps_rec;
  for (const Load *lp : model.loads_for_set(sc.load_set)) {
    if (const TempLoad *tl = std::get_if<TempLoad>(lp))
      nodal_temps_rec[tl->node] = tl->temperature;
  }

  // ── Recover element stresses ──────────────────────────────────────────────
  for (const auto &elem_data : model.elements) {
    auto elem = make_element(elem_data, model);
    auto gdofs = elem->global_dof_indices(dof_map);

    // Build element displacement vector
    const int nd_ = static_cast<int>(gdofs.size());
    Eigen::VectorXd ue = Eigen::VectorXd::Zero(nd_);
    for (int i = 0; i < nd_; ++i) {
      EqIndex eq = gdofs[i];
      if (eq != CONSTRAINED_DOF && eq < static_cast<int>(u_free.size()))
        ue(i) = u_free[static_cast<size_t>(eq)];
    }

    // Compute stress at centroid
    if (elem_data.type == ElementType::CQUAD4 ||
        elem_data.type == ElementType::CTRIA3) {
      // Plate stress recovery: σ = D * B * u (at centroid)
      PlateStress ps;
      ps.eid = elem_data.id;

      // For CQUAD4: evaluate B at xi=0, eta=0 (centroid)
      if (elem_data.type == ElementType::CQUAD4) {
        // Re-compute B at centroid
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
        // Extract membrane displacements [u1,v1,u2,v2,...]
        Eigen::VectorXd u_mem(8);
        for (int n = 0; n < 4; ++n) {
          u_mem(2 * n) = ue(6 * n);
          u_mem(2 * n + 1) = ue(6 * n + 1);
        }

        // Get D matrix
        const auto &pshell_ = std::get<PShell>(model.property(elem_data.pid));
        const Mat1 &mat_ = model.material(pshell_.mid1);
        double E_ = mat_.E, nu_ = mat_.nu;
        double c_ = E_ / (1 - nu_ * nu_);
        Eigen::Matrix3d Dm_;
        Dm_ << c_, c_ * nu_, 0, c_ * nu_, c_, 0, 0, 0, c_ * (1 - nu_) / 2;

        Eigen::Vector3d sigma = Dm_ * Bm * u_mem;
        ps.sx = sigma(0);
        ps.sy = sigma(1);
        ps.sxy = sigma(2);
        ps.mx = 0;
        ps.my = 0;
        ps.mxy = 0; // bending moment recovery TBD
        ps.von_mises = std::sqrt(ps.sx * ps.sx - ps.sx * ps.sy + ps.sy * ps.sy +
                                 3 * ps.sxy * ps.sxy);
      } else {
        // CTRIA3 centroid stress
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
        Bm(0, 0) = b1;
        Bm(0, 2) = b2;
        Bm(0, 4) = b3;
        Bm(1, 1) = d1;
        Bm(1, 3) = d2;
        Bm(1, 5) = d3;
        Bm(2, 0) = d1;
        Bm(2, 1) = b1;
        Bm(2, 2) = d2;
        Bm(2, 3) = b2;
        Bm(2, 4) = d3;
        Bm(2, 5) = b3;
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
        Eigen::Vector3d sigma = Dm_ * Bm * u_mem;
        ps.sx = sigma(0);
        ps.sy = sigma(1);
        ps.sxy = sigma(2);
        ps.mx = 0;
        ps.my = 0;
        ps.mxy = 0;
        ps.von_mises = std::sqrt(ps.sx * ps.sx - ps.sx * ps.sy + ps.sy * ps.sy +
                                 3 * ps.sxy * ps.sxy);
      }
      res.plate_stresses.push_back(ps);
    } else if (elem_data.type == ElementType::CHEXA8 ||
               elem_data.type == ElementType::CTETRA4 ||
               elem_data.type == ElementType::CTETRA10) {
      SolidStress ss;
      ss.eid = elem_data.id;

      const auto &psol_ = std::get<PSolid>(model.property(elem_data.pid));
      const Mat1 &mat_ = model.material(psol_.mid);
      Eigen::Matrix<double, 6, 6> D_ = [&]() {
        double lam = mat_.E * mat_.nu / ((1 + mat_.nu) * (1 - 2 * mat_.nu));
        double mu_ = mat_.E / (2 * (1 + mat_.nu));
        Eigen::Matrix<double, 6, 6> D;
        D.setZero();
        D(0, 0) = lam + 2 * mu_;
        D(0, 1) = lam;
        D(0, 2) = lam;
        D(1, 0) = lam;
        D(1, 1) = lam + 2 * mu_;
        D(1, 2) = lam;
        D(2, 0) = lam;
        D(2, 1) = lam;
        D(2, 2) = lam + 2 * mu_;
        D(3, 3) = mu_;
        D(4, 4) = mu_;
        D(5, 5) = mu_;
        return D;
      }();

      Eigen::Matrix<double, 6, 1> sigma;
      sigma.setZero();

      if (elem_data.type == ElementType::CTETRA10) {
        // 10-node tet: evaluate B at centroid (L1=L2=L3=L4=0.25)
        // using quadratic shape function derivatives
        // Build B at centroid via CTetra10's shape function logic
        auto nc10 = [&]() -> std::array<Vec3,10> {
          std::array<Vec3,10> arr;
          for (int i = 0; i < 10; ++i)
            arr[i] = model.node(elem_data.nodes[i]).position;
          return arr;
        }();
        // Shape function derivatives at centroid (L1=L2=L3=0.25, L4=0.25)
        // dNdL1 etc. evaluated at centroid
        double L1=0.25, L2=0.25, L3=0.25;
        double L4 = 1.0 - L1 - L2 - L3;
        // dN/dL1 [10]
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
          T10 += (it!=nodal_temps_rec.end()) ? it->second : sc.t_ref;
        }
        T10 /= 10.0;
        double dT10 = T10 - sc.t_ref;
        Eigen::Matrix<double,6,1> eps_th10;
        eps_th10 << mat_.A*dT10, mat_.A*dT10, mat_.A*dT10, 0, 0, 0;
        sigma = D_ * (B10 * ue - eps_th10);
      } else if (elem_data.type == ElementType::CTETRA4) {
        // Constant strain tet: compute B at any point (use node coords)
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
              if (r == j)
                continue;
              int ci_ = 0;
              for (int cc = 0; cc < 4; ++cc) {
                if (cc == i)
                  continue;
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
        // Thermal correction: σ = D*(B*u − ε_th).  ε_th = α*dT*[1,1,1,0,0,0].
        double T_avg_tet = 0.0;
        for (int i = 0; i < 4; ++i) {
          auto it = nodal_temps_rec.find(elem_data.nodes[i]);
          T_avg_tet += (it != nodal_temps_rec.end()) ? it->second : sc.t_ref;
        }
        T_avg_tet /= 4.0;
        double dT_tet = T_avg_tet - sc.t_ref;
        double alpha_tet = mat_.A;
        Eigen::Matrix<double, 6, 1> eps_th_tet;
        eps_th_tet << alpha_tet*dT_tet, alpha_tet*dT_tet, alpha_tet*dT_tet, 0, 0, 0;
        sigma = D_ * (B * ue - eps_th_tet);
      } else {
        // CHEXA8: evaluate B at centroid (0,0,0)
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
          dNdx(0, n) = Jinv(0, 0) * sd.dNdxi[n] + Jinv(0, 1) * sd.dNdeta[n] +
                       Jinv(0, 2) * sd.dNdzeta[n];
          dNdx(1, n) = Jinv(1, 0) * sd.dNdxi[n] + Jinv(1, 1) * sd.dNdeta[n] +
                       Jinv(1, 2) * sd.dNdzeta[n];
          dNdx(2, n) = Jinv(2, 0) * sd.dNdxi[n] + Jinv(2, 1) * sd.dNdeta[n] +
                       Jinv(2, 2) * sd.dNdzeta[n];
        }
        Eigen::MatrixXd B(6, 24);
        B.setZero();
        for (int n = 0; n < 8; ++n) {
          double dnx = dNdx(0, n), dny = dNdx(1, n), dnz = dNdx(2, n);
          int c0 = 3 * n;
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
        // Thermal correction: σ = D*(B*u − ε_th).
        double T_avg_hex = 0.0;
        for (int i = 0; i < 8; ++i) {
          auto it = nodal_temps_rec.find(elem_data.nodes[i]);
          T_avg_hex += (it != nodal_temps_rec.end()) ? it->second : sc.t_ref;
        }
        T_avg_hex /= 8.0;
        double dT_hex = T_avg_hex - sc.t_ref;
        double alpha_hex = mat_.A;
        Eigen::Matrix<double, 6, 1> eps_th_hex;
        eps_th_hex << alpha_hex*dT_hex, alpha_hex*dT_hex, alpha_hex*dT_hex, 0, 0, 0;
        sigma = D_ * (B * ue - eps_th_hex);
      }

      ss.sx = sigma(0);
      ss.sy = sigma(1);
      ss.sz = sigma(2);
      ss.sxy = sigma(3);
      ss.syz = sigma(4);
      ss.szx = sigma(5);
      // Von Mises
      double s1 = ss.sx, s2 = ss.sy, s3 = ss.sz, t12 = ss.sxy, t23 = ss.syz,
             t31 = ss.szx;
      ss.von_mises =
          std::sqrt(0.5 * ((s1 - s2) * (s1 - s2) + (s2 - s3) * (s2 - s3) +
                           (s3 - s1) * (s3 - s1) +
                           6 * (t12 * t12 + t23 * t23 + t31 * t31)));
      res.solid_stresses.push_back(ss);
    }
  }

  return res;
}

} // namespace nastran
