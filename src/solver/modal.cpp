// src/solver/modal.cpp
// Modal analysis pipeline (SOL 103).

#include "solver/modal.hpp"
#include "core/coord_sys.hpp"
#include "core/mpc_handler.hpp"
#include "elements/element_factory.hpp"
#include "elements/rbe_constraints.hpp"

#include <Eigen/Sparse>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <numbers>

namespace vibetran {

ModalSolver::ModalSolver(std::unique_ptr<EigensolverBackend> backend)
    : backend_(std::move(backend)) {}

ModalSolverResults ModalSolver::solve(const Model& model) {
    model.validate();
    ModalSolverResults results;
    for (const auto& sc : model.analysis.subcases)
        results.subcases.push_back(solve_subcase(model, sc));
    return results;
}

ModalSubCaseResults ModalSolver::solve_subcase(const Model& model,
                                               const SubCase& sc) {
    using Clock = std::chrono::steady_clock;
    using Ms    = std::chrono::duration<double, std::milli>;
    const auto t0 = Clock::now();

    // ── Find EIGRL ────────────────────────────────────────────────────────────
    auto eigrl_it = model.eigrls.find(sc.eigrl_id);
    if (eigrl_it == model.eigrls.end())
        throw SolverError(std::format(
            "[subcase {}] METHOD {} references unknown EIGRL",
            sc.id, sc.eigrl_id));
    const EigRL& eigrl = eigrl_it->second;

    // ── WTMASS ────────────────────────────────────────────────────────────────
    double wtmass = 1.0;
    auto wt_it = model.params.find("WTMASS");
    if (wt_it != model.params.end()) {
        try { wtmass = std::stod(wt_it->second); }
        catch (...) {}
    }

    // ── 1. DOF map ────────────────────────────────────────────────────────────
    DofMap dof_map = build_dof_map(model, sc);
    const auto t1 = Clock::now();
    std::clog << std::format("[modal sc {}] build_dof_map: {:.3f} ms  ({} free DOFs)\n",
                             sc.id, Ms(t1 - t0).count(), dof_map.num_free_dofs());

    // ── 2. MPC system ─────────────────────────────────────────────────────────
    MpcHandler mpc_handler;
    build_mpc_system(model, sc, dof_map, mpc_handler);
    const int n = mpc_handler.num_reduced();
    const auto t2 = Clock::now();
    std::clog << std::format("[modal sc {}] build_mpc_system: {:.3f} ms  ({} reduced DOFs)\n",
                             sc.id, Ms(t2 - t1).count(), n);

    // ── 3. Assemble K and M ───────────────────────────────────────────────────
    SparseMatrixBuilder K_builder(n);
    SparseMatrixBuilder M_builder(n);

    assemble_stiffness(model, mpc_handler, K_builder);
    assemble_mass(model, mpc_handler, M_builder, wtmass);
    const auto t3 = Clock::now();
    std::clog << std::format("[modal sc {}] assemble K+M: {:.3f} ms\n",
                             sc.id, Ms(t3 - t2).count());

    // ── 4. Convert to Eigen::SparseMatrix ─────────────────────────────────────
    auto to_eigen = [](const SparseMatrixBuilder::CsrData& csr)
        -> Eigen::SparseMatrix<double>
    {
        using Triplet = Eigen::Triplet<double>;
        std::vector<Triplet> triplets;
        triplets.reserve(static_cast<size_t>(csr.nnz));
        for (int row = 0; row < csr.n; ++row)
            for (int idx = csr.row_ptr[row]; idx < csr.row_ptr[row + 1]; ++idx)
                triplets.emplace_back(row, csr.col_ind[idx], csr.values[idx]);
        Eigen::SparseMatrix<double> mat(csr.n, csr.n);
        mat.setFromTriplets(triplets.begin(), triplets.end());
        return mat;
    };

    auto K_csr = K_builder.build_csr();
    auto M_csr = M_builder.build_csr();
    Eigen::SparseMatrix<double> K_eigen = to_eigen(K_csr);
    Eigen::SparseMatrix<double> M_eigen = to_eigen(M_csr);
    const auto t4 = Clock::now();
    std::clog << std::format("[modal sc {}] build sparse matrices: {:.3f} ms  "
                             "(K nnz={}, M nnz={})\n",
                             sc.id, Ms(t4 - t3).count(),
                             K_csr.nnz, M_csr.nnz);

    // ── 5. Eigensolver ────────────────────────────────────────────────────────
    // σ = (2π·V1)² if V1 > 0, else -1.0 (shifts K+M, making it positive def.)
    const double sigma = (eigrl.v1 > 0.0)
        ? std::pow(2.0 * std::numbers::pi * eigrl.v1, 2.0)
        : -1.0;

    std::clog << std::format("[modal sc {}] launching {} (nd={}, sigma={:.6g})\n",
                             sc.id, backend_->name(), eigrl.nd, sigma);

    std::vector<EigenPair> pairs = backend_->solve(K_eigen, M_eigen,
                                                    eigrl.nd, sigma);
    const auto t5 = Clock::now();
    std::clog << std::format("[modal sc {}] eigensolver: {:.3f} ms  ({} modes)\n",
                             sc.id, Ms(t5 - t4).count(), pairs.size());

    // ── 6. Build ModalSubCaseResults ──────────────────────────────────────────
    ModalSubCaseResults msc;
    msc.id          = sc.id;
    msc.label       = sc.label;
    msc.eigvec_print = sc.eigvec_print;
    msc.eigvec_plot  = sc.eigvec_plot;

    for (int i = 0; i < static_cast<int>(pairs.size()); ++i) {
        const EigenPair& ep = pairs[i];
        ModeResult mr;
        mr.mode_number      = i + 1;
        mr.eigenvalue       = ep.eigenvalue;
        mr.radians_per_sec  = std::sqrt(std::max(ep.eigenvalue, 0.0));
        mr.cycles_per_sec   = mr.radians_per_sec / (2.0 * std::numbers::pi);

        // Compute generalised mass: φᵀ M φ (should be ≈1 for mass-normalised)
        mr.gen_mass = ep.eigenvector.transpose() * M_eigen * ep.eigenvector;

        // Normalise if needed (Spectra mass-normalises, but apply NORM=MAX)
        Eigen::VectorXd phi = ep.eigenvector;
        if (eigrl.norm == EigRL::Norm::Max) {
            double max_abs = phi.cwiseAbs().maxCoeff();
            if (max_abs > 1e-30) phi /= max_abs;
        } else {
            // Mass normalisation: φ / sqrt(φᵀMφ)
            if (mr.gen_mass > 1e-30)
                phi /= std::sqrt(mr.gen_mass);
            mr.gen_mass = 1.0;
        }

        mr.shape = recover_mode_shape(model, mpc_handler, phi);
        msc.modes.push_back(std::move(mr));
    }

    const auto t6 = Clock::now();
    std::clog << std::format("[modal sc {}] total: {:.3f} ms\n",
                             sc.id, Ms(t6 - t0).count());

    return msc;
}

// ── DOF map (identical to LinearStaticSolver::build_dof_map) ─────────────────

DofMap ModalSolver::build_dof_map(const Model& model, const SubCase& sc) {
    DofMap dmap;
    dmap.build(model.nodes, 6);

    // Apply SPCs (skip translational DOFs for CD≠0 nodes — handled as MPCs)
    {
        std::vector<std::pair<NodeId, int>> spc_constraints;
        for (const Spc* spc : model.spcs_for_set(sc.spc_set)) {
            auto node_it = model.nodes.find(spc->node);
            bool has_cd  = (node_it != model.nodes.end() &&
                            node_it->second.cd.value != 0);
            for (int d = 0; d < 6; ++d) {
                if (!spc->dofs.has(d + 1)) continue;
                if (has_cd && d < 3) continue; // handled as MPC
                spc_constraints.emplace_back(spc->node, d);
            }
        }
        dmap.constrain_batch(spc_constraints);
    }

    // Constrain rotational DOFs on solid-element-only nodes
    std::unordered_map<NodeId, bool> node_has_shell;
    for (const auto& [nid, _] : model.nodes)
        node_has_shell[nid] = false;

    for (const auto& elem : model.elements) {
        bool is_shell = (elem.type == ElementType::CQUAD4 ||
                         elem.type == ElementType::CTRIA3);
        if (is_shell)
            for (NodeId nid : elem.nodes)
                node_has_shell[nid] = true;
    }

    {
        std::vector<std::pair<NodeId, int>> rot_constraints;
        rot_constraints.reserve(node_has_shell.size() * 3);
        for (const auto& [nid, has_shell] : node_has_shell)
            if (!has_shell)
                for (int d = 3; d < 6; ++d)
                    rot_constraints.emplace_back(nid, d);
        dmap.constrain_batch(rot_constraints);
    }

    return dmap;
}

// ── MPC system (identical to LinearStaticSolver::build_mpc_system) ────────────

void ModalSolver::build_mpc_system(const Model& model, const SubCase& sc,
                                   DofMap& dof_map, MpcHandler& mpc_handler) {
    std::vector<Mpc> all_mpcs;

    // CD-frame SPCs
    {
        std::unordered_map<NodeId, DofSet> node_spc_dofs;
        for (const Spc* spc : model.spcs_for_set(sc.spc_set)) {
            if (spc->value != 0.0) continue;
            node_spc_dofs[spc->node].mask |= spc->dofs.mask;
        }

        std::vector<std::pair<NodeId, int>> direct_constraints;

        for (const auto& [nid, gp] : model.nodes) {
            if (gp.cd == CoordId{0}) continue;
            auto spc_it = node_spc_dofs.find(nid);
            if (spc_it == node_spc_dofs.end()) continue;

            auto cs_it = model.coord_systems.find(gp.cd);
            if (cs_it == model.coord_systems.end()) continue;

            const CoordSys& cs = cs_it->second;
            Mat3 T3 = rotation_matrix(cs, gp.position);
            DofSet dofs = spc_it->second;

            int cd_dofs[3], n_trans = 0;
            for (int d = 0; d < 3; ++d)
                if (dofs.has(d + 1)) cd_dofs[n_trans++] = d;

            if (n_trans == 3) {
                for (int j = 0; j < 3; ++j)
                    direct_constraints.emplace_back(nid, j);
            } else if (n_trans > 0) {
                double A[3][3] = {};
                int col_perm[3] = {0, 1, 2};
                for (int i = 0; i < n_trans; ++i)
                    for (int j = 0; j < 3; ++j)
                        A[i][j] = T3(j, cd_dofs[i]);

                for (int row = 0; row < n_trans; ++row) {
                    int best_col = row;
                    double best_val = std::abs(A[row][row]);
                    for (int col = row + 1; col < 3; ++col)
                        if (std::abs(A[row][col]) > best_val) {
                            best_val = std::abs(A[row][col]);
                            best_col = col;
                        }
                    if (best_col != row) {
                        for (int i = 0; i < n_trans; ++i)
                            std::swap(A[i][row], A[i][best_col]);
                        std::swap(col_perm[row], col_perm[best_col]);
                    }
                    if (std::abs(A[row][row]) < 1e-14) continue;
                    for (int i = row + 1; i < n_trans; ++i) {
                        double factor = A[i][row] / A[row][row];
                        for (int j = row; j < 3; ++j)
                            A[i][j] -= factor * A[row][j];
                    }
                }

                for (int row = 0; row < n_trans; ++row) {
                    if (std::abs(A[row][row]) < 1e-14) continue;
                    int nnz = 0;
                    for (int col = row; col < 3; ++col)
                        if (std::abs(A[row][col]) > 1e-14) nnz++;
                    if (nnz == 1) {
                        direct_constraints.emplace_back(nid, col_perm[row]);
                    } else {
                        Mpc mpc;
                        mpc.sid = MpcSetId{0};
                        for (int col = row; col < 3; ++col)
                            if (std::abs(A[row][col]) > 1e-14)
                                mpc.terms.push_back({nid, col_perm[col] + 1,
                                                     A[row][col]});
                        if (!mpc.terms.empty())
                            all_mpcs.push_back(std::move(mpc));
                    }
                }
            }
        }
        if (!direct_constraints.empty())
            dof_map.constrain_batch(direct_constraints);
    }

    // RBE2 / RBE3
    for (const auto& rbe2 : model.rbe2s) expand_rbe2(rbe2, model, all_mpcs);
    for (const auto& rbe3 : model.rbe3s) expand_rbe3(rbe3, model, all_mpcs);

    // Explicit MPCs
    if (sc.mpc_set.value != 0)
        for (const Mpc* mpc : model.mpcs_for_set(sc.mpc_set))
            all_mpcs.push_back(*mpc);

    std::vector<const Mpc*> mpc_ptrs;
    mpc_ptrs.reserve(all_mpcs.size());
    for (const auto& m : all_mpcs)
        mpc_ptrs.push_back(&m);

    mpc_handler.build(mpc_ptrs, dof_map);
}

// ── Assemble stiffness ────────────────────────────────────────────────────────

void ModalSolver::assemble_stiffness(const Model& model,
                                     const MpcHandler& mpc_handler,
                                     SparseMatrixBuilder& K_builder) {
    const DofMap& dof_map = mpc_handler.full_dof_map();
    for (const auto& elem_data : model.elements) {
        auto elem  = make_element(elem_data, model);
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

// ── Assemble mass ─────────────────────────────────────────────────────────────

void ModalSolver::assemble_mass(const Model& model,
                                const MpcHandler& mpc_handler,
                                SparseMatrixBuilder& M_builder,
                                double wtmass) {
    const DofMap& dof_map = mpc_handler.full_dof_map();
    for (const auto& elem_data : model.elements) {
        auto elem  = make_element(elem_data, model);
        LocalKe Me = elem->mass_matrix();
        if (wtmass != 1.0) Me *= wtmass;
        auto gdofs = elem->global_dof_indices(dof_map);

        const int nd = static_cast<int>(gdofs.size());
        std::vector<double> me_row(static_cast<size_t>(nd * nd));
        for (int r = 0; r < nd; ++r)
            for (int c = 0; c < nd; ++c)
                me_row[static_cast<size_t>(r * nd + c)] = Me(r, c);

        mpc_handler.apply_to_stiffness(gdofs, me_row, M_builder);
    }
}

// ── Mode shape recovery ───────────────────────────────────────────────────────

std::vector<NodeDisplacement>
ModalSolver::recover_mode_shape(const Model& model,
                                const MpcHandler& mpc_handler,
                                const Eigen::VectorXd& phi_reduced) const {
    // Expand reduced → full (pre-MPC) DOFs
    const int n_full = mpc_handler.full_dof_map().num_free_dofs();
    std::vector<double> phi_full(static_cast<size_t>(n_full), 0.0);

    std::vector<double> phi_vec(static_cast<size_t>(phi_reduced.size()));
    for (int i = 0; i < phi_reduced.size(); ++i)
        phi_vec[static_cast<size_t>(i)] = phi_reduced(i);

    mpc_handler.recover_dependent_dofs(phi_full, phi_vec);

    const DofMap& dof_map = mpc_handler.full_dof_map();

    // Collect node IDs sorted for consistent output ordering
    std::vector<NodeId> sorted_nodes;
    sorted_nodes.reserve(model.nodes.size());
    for (const auto& [nid, _] : model.nodes)
        sorted_nodes.push_back(nid);
    std::sort(sorted_nodes.begin(), sorted_nodes.end());

    std::vector<NodeDisplacement> shape;
    shape.reserve(sorted_nodes.size());

    for (NodeId nid : sorted_nodes) {
        NodeDisplacement nd;
        nd.node = nid;
        for (int d = 0; d < 6; ++d) {
            EqIndex eq = dof_map.eq_index(nid, d);
            nd.d[d] = (eq != CONSTRAINED_DOF &&
                       eq < static_cast<int>(phi_full.size()))
                      ? phi_full[static_cast<size_t>(eq)]
                      : 0.0;
        }
        shape.push_back(nd);
    }
    return shape;
}

} // namespace vibetran
