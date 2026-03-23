// src/core/mpc_handler.cpp
// Master-slave elimination for multi-point constraints.

#include "core/mpc_handler.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

namespace vibestran {

void MpcHandler::build(std::span<const Mpc* const> mpcs, DofMap& dof_map) {
    // Save pre-MPC dof_map
    full_dof_map_ = dof_map;
    n_full_ = dof_map.num_free_dofs();

    if (mpcs.empty()) {
        // No MPCs: trivial identity mapping
        index_map_.resize(static_cast<size_t>(n_full_));
        for (int i = 0; i < n_full_; ++i)
            index_map_[static_cast<size_t>(i)] = static_cast<EqIndex>(i);
        n_reduced_ = n_full_;
        return;
    }

    // Build eliminations using pre-MPC eq indices
    std::vector<std::pair<NodeId, int>> dep_dofs_to_constrain;

    for (const Mpc* mpc : mpcs) {
        if (mpc->terms.empty())
            continue;

        // Choose term with largest |coeff| as dependent
        size_t dep_idx = 0;
        double max_abs = 0.0;
        for (size_t i = 0; i < mpc->terms.size(); ++i) {
            double a = std::abs(mpc->terms[i].coeff);
            if (a > max_abs) { max_abs = a; dep_idx = i; }
        }

        const MpcTerm& dep_term = mpc->terms[dep_idx];
        EqIndex dep_eq = dof_map.eq_index(dep_term.node, dep_term.dof - 1);

        if (dep_eq == CONSTRAINED_DOF) {
            std::cerr << std::format(
                "MPC set {}: dependent DOF (node {}, dof {}) is already SPC-constrained; "
                "skipping MPC\n",
                mpc->sid.value, dep_term.node.value, dep_term.dof);
            continue;
        }

        // Build elimination with pre-MPC eq indices
        MpcElimination elim;
        elim.dep = dep_eq;
        double a_dep = dep_term.coeff;
        for (size_t i = 0; i < mpc->terms.size(); ++i) {
            if (i == dep_idx) continue;
            const MpcTerm& t = mpc->terms[i];
            EqIndex eq = dof_map.eq_index(t.node, t.dof - 1);
            if (eq == CONSTRAINED_DOF)
                continue;
            elim.terms.emplace_back(eq, -t.coeff / a_dep);
        }
        eliminations_.push_back(std::move(elim));
        dep_dofs_to_constrain.emplace_back(dep_term.node, dep_term.dof - 1);
    }

    if (eliminations_.empty()) {
        // All MPCs skipped
        index_map_.resize(static_cast<size_t>(n_full_));
        for (int i = 0; i < n_full_; ++i)
            index_map_[static_cast<size_t>(i)] = static_cast<EqIndex>(i);
        n_reduced_ = n_full_;
        return;
    }

    // Cycle detection via topological sort on the dep-dep dependency graph
    {
        std::unordered_set<EqIndex> dep_set;
        for (const auto& e : eliminations_)
            dep_set.insert(e.dep);

        bool any_chain = false;
        for (const auto& e : eliminations_)
            for (const auto& [ind_eq, c] : e.terms)
                if (dep_set.count(ind_eq)) { any_chain = true; break; }

        if (any_chain) {
            std::unordered_map<EqIndex, int> in_degree;
            std::unordered_map<EqIndex, std::vector<EqIndex>> fwd;
            for (const auto& e : eliminations_)
                in_degree[e.dep] = 0;
            for (const auto& e : eliminations_)
                for (const auto& [ind_eq, c] : e.terms)
                    if (dep_set.count(ind_eq)) {
                        fwd[ind_eq].push_back(e.dep);
                        in_degree[e.dep]++;
                    }
            std::vector<EqIndex> queue;
            for (auto& [eq, deg] : in_degree)
                if (deg == 0) queue.push_back(eq);
            int processed = 0;
            while (!queue.empty()) {
                EqIndex cur = queue.back(); queue.pop_back();
                ++processed;
                for (EqIndex next : fwd[cur])
                    if (--in_degree[next] == 0)
                        queue.push_back(next); // cppcheck-suppress useStlAlgorithm
            }
            if (processed < static_cast<int>(in_degree.size()))
                throw SolverError("MPC dependency graph contains a cycle; "
                                  "check MPC equations for circular references");
        }
    }

    // Constrain dep DOFs in the main dof_map
    dof_map.constrain_batch(dep_dofs_to_constrain);

    // Build index_map_: pre-MPC eq → post-MPC eq
    std::unordered_set<EqIndex> dep_set;
    for (const auto& e : eliminations_)
        dep_set.insert(e.dep);

    std::vector<EqIndex> sorted_deps(dep_set.begin(), dep_set.end());
    std::sort(sorted_deps.begin(), sorted_deps.end());

    index_map_.resize(static_cast<size_t>(n_full_), CONSTRAINED_DOF);
    int removed = 0;
    size_t dp = 0;
    for (int i = 0; i < n_full_; ++i) {
        // Advance dep pointer
        while (dp < sorted_deps.size() && sorted_deps[dp] < static_cast<EqIndex>(i))
            ++dp;
        if (dp < sorted_deps.size() && sorted_deps[dp] == static_cast<EqIndex>(i)) {
            index_map_[static_cast<size_t>(i)] = CONSTRAINED_DOF;
            ++removed;
        } else {
            index_map_[static_cast<size_t>(i)] = static_cast<EqIndex>(i - removed);
        }
    }
    n_reduced_ = n_full_ - removed;

    // Update elim.terms from pre-MPC to post-MPC (reduced) eq indices
    for (auto& elim : eliminations_) {
        for (auto& [ind_eq, c] : elim.terms) {
            if (ind_eq >= 0 && ind_eq < n_full_)
                ind_eq = index_map_[static_cast<size_t>(ind_eq)];
        }
        // Remove terms whose independent DOF is itself a dep (now CONSTRAINED_DOF)
        // This happens in chained MPCs — handled via DAG ordering above.
        elim.terms.erase(
            std::remove_if(elim.terms.begin(), elim.terms.end(),
                           [](const auto& p) { return p.first == CONSTRAINED_DOF; }),
            elim.terms.end());
    }

    // Build O(1) lookup from dep eq → elimination index
    dep_to_elim_.clear();
    dep_to_elim_.reserve(eliminations_.size());
    for (int i = 0; i < static_cast<int>(eliminations_.size()); ++i)
        dep_to_elim_[eliminations_[i].dep] = i;
}

EqIndex MpcHandler::reduced_index(EqIndex full) const {
    if (full < 0 || full >= static_cast<EqIndex>(index_map_.size()))
        return CONSTRAINED_DOF;
    return index_map_[static_cast<size_t>(full)];
}

std::vector<std::pair<EqIndex, double>>
MpcHandler::t_column(EqIndex full_eq) const {
    if (full_eq == CONSTRAINED_DOF)
        return {};
    // Check if it's a dep DOF (O(1) lookup)
    auto it = dep_to_elim_.find(full_eq);
    if (it != dep_to_elim_.end())
        return eliminations_[it->second].terms;
    // Free non-dep DOF
    EqIndex r = reduced_index(full_eq);
    if (r == CONSTRAINED_DOF)
        return {};
    return {{r, 1.0}};
}

// cppcheck-suppress unusedFunction -- called from linear_static.cpp
void MpcHandler::apply_to_stiffness(std::span<const EqIndex> gdofs_full,
                                     std::span<const double> ke,
                                     SparseMatrixBuilder& K_builder) const {
    int ndof = static_cast<int>(gdofs_full.size());

    // Fast path: if no element DOF is a dependent MPC DOF, use direct assembly
    // with simple index remapping (avoids dense T^T*Ke*T transformation).
    {
        bool any_dep = false;
        if (has_constraints()) {
            for (int i = 0; i < ndof && !any_dep; ++i) {
                EqIndex full = gdofs_full[i];
                if (full != CONSTRAINED_DOF && dep_to_elim_.count(full))
                    any_dep = true;
            }
        }
        if (!any_dep) {
            std::vector<int32_t> gd32(static_cast<size_t>(ndof));
            for (int i = 0; i < ndof; ++i) {
                EqIndex full = gdofs_full[i];
                gd32[static_cast<size_t>(i)] =
                    (full == CONSTRAINED_DOF || full < 0 ||
                     full >= static_cast<EqIndex>(index_map_.size()))
                        ? CONSTRAINED_DOF
                        : index_map_[static_cast<size_t>(full)];
            }
            K_builder.add_element_stiffness(
                gd32, std::vector<double>(ke.begin(), ke.end()));
            return;
        }
    }

    // Build compact T: collect unique active reduced column indices.
    // For each element DOF i, t_column(gdofs_full[i]) yields (reduced_eq, coeff) pairs.
    // active_cols = union of all reduced_eq values across all element DOFs.
    std::vector<EqIndex> active_cols;
    active_cols.reserve(static_cast<size_t>(ndof) * 2);

    // Store column data per element row: T_cols[i] = list of (col_in_active, coeff)
    using ColCoeff = std::vector<std::pair<int, double>>;
    std::vector<ColCoeff> row_entries(static_cast<size_t>(ndof));

    for (int i = 0; i < ndof; ++i) {
        auto tc = t_column(gdofs_full[i]);
        for (const auto& [r, c] : tc) {
            if (r == CONSTRAINED_DOF || r < 0) continue;
            // Find or add to active_cols
            int col_idx = -1;
            for (int j = 0; j < static_cast<int>(active_cols.size()); ++j)
                if (active_cols[j] == r) { col_idx = j; break; }
            if (col_idx < 0) {
                col_idx = static_cast<int>(active_cols.size());
                active_cols.push_back(r);
            }
            row_entries[static_cast<size_t>(i)].emplace_back(col_idx, c);
        }
    }

    int na = static_cast<int>(active_cols.size());
    if (na == 0) return; // all DOFs constrained

    // Build dense T_compact: ndof × na
    std::vector<double> T(static_cast<size_t>(ndof * na), 0.0);
    for (int i = 0; i < ndof; ++i)
        for (const auto& [ci, c] : row_entries[static_cast<size_t>(i)])
            T[static_cast<size_t>(i * na + ci)] += c;

    // tmp = Ke * T_compact  (ndof × na)
    std::vector<double> tmp(static_cast<size_t>(ndof * na), 0.0);
    for (int i = 0; i < ndof; ++i)
        for (int j = 0; j < na; ++j)
            for (int k = 0; k < ndof; ++k)
                tmp[static_cast<size_t>(i * na + j)] +=
                    ke[static_cast<size_t>(i * ndof + k)] *
                    T[static_cast<size_t>(k * na + j)];

    // k_red = T_compact^T * tmp  (na × na)
    std::vector<double> k_red(static_cast<size_t>(na * na), 0.0);
    for (int i = 0; i < na; ++i)
        for (int j = 0; j < na; ++j)
            for (int k = 0; k < ndof; ++k)
                k_red[static_cast<size_t>(i * na + j)] +=
                    T[static_cast<size_t>(k * na + i)] *
                    tmp[static_cast<size_t>(k * na + j)];

    // Scatter k_red into K_builder using active_cols as global indices
    std::vector<int32_t> rdofs(static_cast<size_t>(na));
    for (int i = 0; i < na; ++i)
        rdofs[static_cast<size_t>(i)] = static_cast<int32_t>(active_cols[i]);

    K_builder.add_element_stiffness(rdofs, k_red);
}

// cppcheck-suppress unusedFunction -- called from linear_static.cpp
void MpcHandler::apply_to_force(std::span<const EqIndex> gdofs_full,
                                 std::span<const double> fe,
                                 std::vector<double>& F) const {
    // Fast path: if no element DOF is dependent, use direct index mapping
    bool any_dep = false;
    if (has_constraints()) {
        for (size_t i = 0; i < gdofs_full.size() && !any_dep; ++i) {
            EqIndex full = gdofs_full[i];
            if (full != CONSTRAINED_DOF && dep_to_elim_.count(full))
                any_dep = true;
        }
    }
    if (!any_dep) {
        for (size_t i = 0; i < gdofs_full.size(); ++i) {
            EqIndex full = gdofs_full[i];
            if (full == CONSTRAINED_DOF || full < 0 ||
                full >= static_cast<EqIndex>(index_map_.size()))
                continue;
            EqIndex r = index_map_[static_cast<size_t>(full)];
            if (r >= 0 && r < static_cast<EqIndex>(F.size()))
                F[static_cast<size_t>(r)] += fe[i];
        }
        return;
    }
    for (size_t i = 0; i < gdofs_full.size(); ++i) {
        for (const auto& [r, c] : t_column(gdofs_full[i])) {
            if (r >= 0 && r < static_cast<EqIndex>(F.size()))
                F[static_cast<size_t>(r)] += c * fe[i];
        }
    }
}

void MpcHandler::recover_dependent_dofs(std::vector<double>& u_free_full,
                                         const std::vector<double>& u_reduced) const {
    // Fill free (non-dep) entries from u_reduced
    for (size_t i = 0; i < index_map_.size(); ++i) {
        EqIndex r = index_map_[i];
        if (r != CONSTRAINED_DOF && r < static_cast<EqIndex>(u_reduced.size()))
            u_free_full[i] = u_reduced[static_cast<size_t>(r)];
    }
    // Compute dep DOF values: u_dep = sum_j c_j * u_reduced[j]
    for (const auto& elim : eliminations_) {
        double val = 0.0;
        for (const auto& [r, c] : elim.terms) {
            if (r >= 0 && r < static_cast<EqIndex>(u_reduced.size()))
                val += c * u_reduced[static_cast<size_t>(r)];
        }
        if (elim.dep >= 0 && elim.dep < static_cast<EqIndex>(u_free_full.size()))
            u_free_full[static_cast<size_t>(elim.dep)] = val;
    }
}

} // namespace vibestran
