#pragma once
// include/solver/modal.hpp
// Modal analysis pipeline (SOL 103): K φ = λ M φ
//   1. Build DOF map (SPCs applied)
//   2. Build MPC system (RBE2/RBE3, explicit MPCs, CD-frame SPCs)
//   3. Assemble global stiffness K and consistent mass M
//   4. Apply MPC transform: K_red = Tᵀ K T, M_red = Tᵀ M T
//   5. Solve: K_red φ = λ M_red φ  via EigensolverBackend
//   6. Normalise mode shapes (mass or max norm)
//   7. Recover full mode shapes and build ModalSolverResults

#include "core/dof_map.hpp"
#include "core/mpc_handler.hpp"
#include "core/model.hpp"
#include "core/sparse_matrix.hpp"
#include "io/results.hpp"
#include "solver/eigensolver_backend.hpp"
#include <memory>

namespace vibetran {

class ModalSolver {
public:
    explicit ModalSolver(std::unique_ptr<EigensolverBackend> backend);

    /// Solve all modal subcases in the model's analysis case.
    [[nodiscard]] ModalSolverResults solve(const Model& model);

private:
    std::unique_ptr<EigensolverBackend> backend_;

    ModalSubCaseResults solve_subcase(const Model& model, const SubCase& sc);

    DofMap build_dof_map(const Model& model, const SubCase& sc);

    void build_mpc_system(const Model& model, const SubCase& sc,
                          DofMap& dof_map, MpcHandler& mpc_handler);

    /// Assemble global stiffness K into K_builder (reduced system).
    void assemble_stiffness(const Model& model,
                            const MpcHandler& mpc_handler,
                            SparseMatrixBuilder& K_builder);

    /// Assemble global consistent mass M into M_builder (reduced system).
    /// Applies WTMASS scaling if present in model.params.
    void assemble_mass(const Model& model,
                       const MpcHandler& mpc_handler,
                       SparseMatrixBuilder& M_builder,
                       double wtmass);

    /// Expand a reduced eigenvector into per-node displacements.
    std::vector<NodeDisplacement>
    recover_mode_shape(const Model& model,
                       const MpcHandler& mpc_handler,
                       const Eigen::VectorXd& phi_reduced) const;
};

} // namespace vibetran
