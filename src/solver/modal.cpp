// src/solver/modal.cpp
// Modal analysis pipeline (SOL 103).

#include "solver/modal.hpp"
#include "core/coord_sys.hpp"
#include "core/factor_ratio_check.hpp"
#include "core/logger.hpp"
#include "core/mpc_handler.hpp"
#include "elements/element_factory.hpp"
#include "elements/rbe_constraints.hpp"
#include "assembly_parallel.hpp"
#include "solver/analysis_support.hpp"

#include <Eigen/Sparse>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <limits>
#include <numbers>
#include <spdlog/spdlog.h>

namespace vibestran {

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
    spdlog::debug("[modal sc {}] build_dof_map: {:.3f} ms  ({} free DOFs)",
                  sc.id, Ms(t1 - t0).count(), dof_map.num_free_dofs());

    // ── 2. MPC system ─────────────────────────────────────────────────────────
    MpcHandler mpc_handler;
    build_mpc_system(model, sc, dof_map, mpc_handler);
    const int n = mpc_handler.num_reduced();
    const auto t2 = Clock::now();
    spdlog::debug("[modal sc {}] build_mpc_system: {:.3f} ms  ({} reduced DOFs)",
                  sc.id, Ms(t2 - t1).count(), n);

    // ── 3. Assemble K and M ───────────────────────────────────────────────────
    SparseMatrixBuilder K_builder(n);
    SparseMatrixBuilder M_builder(n);
    const size_t matrix_triplet_capacity = detail::estimate_triplet_capacity(model);
    K_builder.reserve_triplets(matrix_triplet_capacity);
    M_builder.reserve_triplets(matrix_triplet_capacity);

    assemble_stiffness(model, mpc_handler, K_builder);
    assemble_mass(model, mpc_handler, M_builder, wtmass);
    const auto t3 = Clock::now();
    spdlog::debug("[modal sc {}] assemble K+M: {:.3f} ms", sc.id, Ms(t3 - t2).count());

    // ── 4. Convert to Eigen::SparseMatrix ─────────────────────────────────────
    auto to_eigen = [](const SparseMatrixBuilder::CsrData& csr)
        -> Eigen::SparseMatrix<double>
    {
        using ESMR = Eigen::SparseMatrix<double, Eigen::RowMajor>;
        using ESM = Eigen::SparseMatrix<double>;

        Eigen::Map<const ESMR> mat_row(csr.n, csr.n, csr.nnz,
                                       csr.row_ptr.data(),
                                       csr.col_ind.data(),
                                       csr.values.data());
        ESM mat(mat_row);
        mat.makeCompressed();
        return mat;
    };

    auto K_csr = K_builder.build_csr();
    auto M_csr = M_builder.build_csr();

    Eigen::SparseMatrix<double> K_eigen = to_eigen(K_csr);
    Eigen::SparseMatrix<double> M_eigen = to_eigen(M_csr);
    const auto t4 = Clock::now();
    spdlog::debug("[modal sc {}] build sparse matrices: {:.3f} ms  (K nnz={}, M nnz={})",
                  sc.id, Ms(t4 - t3).count(), K_csr.nnz, M_csr.nnz);

    // ── 5. Eigensolver ────────────────────────────────────────────────────────
    // σ = (2π·V1)² if V1 > 0, else -1.0 (shifts K+M, making it positive def.)
    const double sigma = (eigrl.v1 > 0.0)
        ? std::pow(2.0 * std::numbers::pi * eigrl.v1, 2.0)
        : -1.0;

    spdlog::debug("[modal sc {}] launching {} (nd={}, sigma={:.6g})",
                  sc.id, backend_->name(), eigrl.nd, sigma);

    const FactorRatioCheckPolicy factor_ratio_policy =
        build_factor_ratio_check_policy(
            model, std::format("[modal sc {}]", sc.id), "shift matrix");
    std::vector<EigenPair> pairs = backend_->solve(
        K_eigen, M_eigen, eigrl.nd, sigma, &factor_ratio_policy);
    const auto t5 = Clock::now();
    spdlog::debug("[modal sc {}] eigensolver: {:.3f} ms  ({} modes)",
                  sc.id, Ms(t5 - t4).count(), pairs.size());

    // Log per-mode residual ||K*φ - λM*φ|| / ||K*φ||.
    // Near-zero rigid-body modes (λ ≈ 0) produce ||K*φ|| ≈ 0; the residual is
    // reported as the absolute norm ||K*φ - λM*φ|| for those modes instead.
    for (int i = 0; i < static_cast<int>(pairs.size()); ++i) {
        const Eigen::VectorXd Kphi =
            K_eigen.selfadjointView<Eigen::Lower>() * pairs[i].eigenvector;
        const Eigen::VectorXd Mphi =
            M_eigen.selfadjointView<Eigen::Lower>() * pairs[i].eigenvector;
        const Eigen::VectorXd res  = Kphi - pairs[i].eigenvalue * Mphi;
        const double kphi_norm = Kphi.norm();
        const double rel_res   = (kphi_norm > 1e-300)
            ? res.norm() / kphi_norm : res.norm();
        const double freq_hz =
            std::sqrt(std::max(pairs[i].eigenvalue, 0.0)) /
            (2.0 * std::numbers::pi);
        spdlog::info("[modal sc {}] mode {} ({:.4g} Hz): "
                     "residual ||K*φ - λM*φ|| / ||K*φ|| = {:.3e}",
                     sc.id, i + 1, freq_hz, rel_res);
    }

    // ── 6. Build ModalSubCaseResults ──────────────────────────────────────────
    ModalSubCaseResults msc;
    msc.id          = sc.id;
    msc.label       = sc.label;
    // In SOL 103, DISPLACEMENT is an alias for EIGENVECTOR.
    // Map both disp_print and disp_plot → eigvec_print (F06 text output).
    // Map disp_plot → eigvec_plot (OP2 binary output).
    msc.eigvec_print = sc.eigvec_print || sc.disp_print || sc.disp_plot;
    msc.eigvec_plot  = sc.eigvec_plot  || sc.disp_plot;

    for (int i = 0; i < static_cast<int>(pairs.size()); ++i) {
        const EigenPair& ep = pairs[i];
        ModeResult mr;
        mr.mode_number      = i + 1;
        mr.eigenvalue       = ep.eigenvalue;
        mr.radians_per_sec  = std::sqrt(std::max(ep.eigenvalue, 0.0));
        mr.cycles_per_sec   = mr.radians_per_sec / (2.0 * std::numbers::pi);

        // Compute generalised mass: φᵀ M φ (should be ≈1 for mass-normalised)
        mr.gen_mass = ep.eigenvector.dot(
            M_eigen.selfadjointView<Eigen::Lower>() * ep.eigenvector);

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
    spdlog::debug("[modal sc {}] total: {:.3f} ms", sc.id, Ms(t6 - t0).count());

    return msc;
}

// ── DOF map (identical to LinearStaticSolver::build_dof_map) ─────────────────

DofMap ModalSolver::build_dof_map(const Model& model, const SubCase& sc) {
    return build_analysis_dof_map(model, sc);
}

// ── MPC system (identical to LinearStaticSolver::build_mpc_system) ────────────

void ModalSolver::build_mpc_system(const Model& model, const SubCase& sc,
                                   DofMap& dof_map, MpcHandler& mpc_handler) {
    build_analysis_mpc_system(model, sc, dof_map, mpc_handler);
}

// ── Assemble stiffness ────────────────────────────────────────────────────────

void ModalSolver::assemble_stiffness(const Model& model,
                                     const MpcHandler& mpc_handler,
                                     SparseMatrixBuilder& K_builder) {
    detail::assemble_element_matrix(
        model, mpc_handler, K_builder,
        [](const ElementBase& elem) { return elem.stiffness_matrix(); });
}

// ── Assemble mass ─────────────────────────────────────────────────────────────

void ModalSolver::assemble_mass(const Model& model,
                                const MpcHandler& mpc_handler,
                                SparseMatrixBuilder& M_builder,
                                double wtmass) {
    detail::assemble_element_matrix(
        model, mpc_handler, M_builder,
        [wtmass](const ElementBase& elem) {
            LocalKe mass = elem.mass_matrix();
            if (wtmass != 1.0)
                mass *= wtmass;
            return mass;
        });
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

} // namespace vibestran
