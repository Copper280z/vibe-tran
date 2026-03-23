#pragma once
// include/core/mpc_handler.hpp
// Master-slave elimination for multi-point constraints (MPCs).
//
// An MPC equation takes the form:
//   sum_i coeff_i * u[node_i, dof_i] = 0
//
// The dependent DOF (largest |coeff|) is eliminated.  The transformation
// T maps the reduced DOF vector to the full (post-SPC, pre-MPC) vector:
//   u_full = T * u_reduced
//
// The reduced stiffness is K_red = T^T * K_full * T, and similarly for F.
//
// Workflow:
//   1. Build dof_map with SPCs only.
//   2. Call mpc_handler.build(mpcs, dof_map) — saves pre-MPC dof_map
//      internally, then constrains dep DOFs in dof_map.
//   3. For assembly, use mpc_handler.full_dof_map() (pre-MPC) to get gdofs.
//   4. Pass pre-MPC gdofs and ke/fe to apply_to_stiffness/apply_to_force.
//   5. K_builder is sized for n_reduced = dof_map.num_free_dofs().
//   6. After solve, call recover_dependent_dofs() to recover dep DOF values.

#include "core/dof_map.hpp"
#include "core/model.hpp"
#include "core/sparse_matrix.hpp"
#include <span>
#include <vector>

namespace vibestran {

/// Describes how one dependent DOF is expressed in terms of independent DOFs.
struct MpcElimination {
    EqIndex dep;   ///< pre-MPC eq index of the dependent DOF
    /// u_dep = sum_i coeff_i * u[ind_reduced_eq_i]
    /// ind_reduced_eq_i are POST-MPC reduced eq indices
    std::vector<std::pair<EqIndex, double>> terms;
};

class MpcHandler {
public:
    /// Build the elimination list from MPC equations.
    /// Saves a copy of dof_map (pre-MPC), then constrains dep DOFs in dof_map.
    /// After return, dof_map is post-SPC+post-MPC (the reduced system).
    /// @param mpcs     MPC equations to apply.
    /// @param dof_map  Modified in-place: dep DOFs are constrained.
    void build(std::span<const Mpc* const> mpcs, DofMap& dof_map);

    /// The dof_map BEFORE MPC elimination.  Use this to compute gdofs
    /// for assembly (so that dep DOF entries are still valid eq indices,
    /// not CONSTRAINED_DOF).
    [[nodiscard]] const DofMap& full_dof_map() const noexcept {
        return full_dof_map_;
    }

    /// Transform element stiffness contribution via T^T Ke T.
    /// @param gdofs_full  Pre-MPC eq indices from full_dof_map() (size ndof).
    /// @param ke          Element stiffness row-major (ndof × ndof).
    /// @param K_builder   Sized for the reduced (post-MPC) system.
    void apply_to_stiffness(std::span<const EqIndex> gdofs_full,
                            std::span<const double> ke,
                            SparseMatrixBuilder& K_builder) const;

    /// Transform element force contribution.
    /// @param gdofs_full  Pre-MPC eq indices from full_dof_map().
    /// @param fe          Element force vector (size ndof).
    /// @param F           Post-MPC global force vector.
    void apply_to_force(std::span<const EqIndex> gdofs_full,
                        std::span<const double> fe,
                        std::vector<double>& F) const;

    /// After solving the reduced system, recover dep DOF values.
    /// @param u_free_full  Output vector sized for pre-MPC free DOFs.
    ///                     On return, all entries (free and dep) are filled.
    /// @param u_reduced    The solution of the reduced (post-MPC) system.
    void recover_dependent_dofs(std::vector<double>& u_free_full,
                                const std::vector<double>& u_reduced) const;

    /// True if any MPCs were registered.
    [[nodiscard]] bool has_constraints() const noexcept {
        return !eliminations_.empty();
    }

    /// Number of free DOFs in the reduced (post-MPC) system.
    [[nodiscard]] int num_reduced() const noexcept { return n_reduced_; }

    /// Map pre-MPC eq index → post-MPC eq index.
    /// Returns CONSTRAINED_DOF for dep or SPC-constrained DOFs.
    [[nodiscard]] EqIndex reduced_index(EqIndex full) const;

private:
    DofMap full_dof_map_;  ///< pre-MPC dof_map (post-SPC only)

    // Maps pre-MPC free eq index → post-MPC reduced eq index
    // (CONSTRAINED_DOF for dep DOFs)
    std::vector<EqIndex> index_map_;
    int n_full_{0};    ///< n_free_dofs before MPC elimination
    int n_reduced_{0}; ///< n_free_dofs after MPC elimination

    std::vector<MpcElimination> eliminations_;

    /// O(1) lookup: pre-MPC eq index → index in eliminations_ (-1 if not dep)
    std::unordered_map<EqIndex, int> dep_to_elim_;

    /// Compute the T-column for a given pre-MPC eq index:
    /// Returns {(reduced_eq, coeff)} pairs.
    /// For free non-dep DOFs: one entry (reduced_eq, 1.0).
    /// For dep DOFs: one entry per independent term.
    std::vector<std::pair<EqIndex, double>>
    t_column(EqIndex full_eq) const;
};

} // namespace vibestran
