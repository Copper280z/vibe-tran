#pragma once
// include/solver/linear_static.hpp
// Orchestrates the linear static analysis pipeline:
//   1. Build DOF map (number free DOFs, apply SPCs)
//   2. Build MPC system (RBE2/RBE3, explicit MPCs, CD-frame SPCs)
//   3. Assemble global stiffness matrix K (MPC-transformed)
//   4. Assemble global force vector F (point loads + thermal loads)
//   5. Solve K_red * u_red = F_red
//   6. Recover full displacements (free + dep DOFs)
//   7. Recover element stresses
//   8. Return SubCaseResults

#include "core/dof_map.hpp"
#include "core/mpc_handler.hpp"
#include "core/model.hpp"
#include "core/sparse_matrix.hpp"
#include "io/results.hpp"
#include "solver/solver_backend.hpp"
#include <memory>

namespace vibestran {

class LinearStaticSolver {
public:
    explicit LinearStaticSolver(std::unique_ptr<SolverBackend> backend);

    /// Solve all subcases in the model's analysis case.
    [[nodiscard]] SolverResults solve(const Model& model);

private:
    std::unique_ptr<SolverBackend> backend_;

    SubCaseResults solve_subcase(const Model& model, const SubCase& sc);

    /// Build DofMap with SPC constraints only (no MPC elimination yet).
    DofMap build_dof_map(const Model& model, const SubCase& sc);

    /// Build MpcHandler: CD-frame SPCs, RBE2/RBE3, explicit MPCs.
    /// Modifies dof_map: constrains dependent DOFs.
    void build_mpc_system(const Model& model, const SubCase& sc,
                          DofMap& dof_map, MpcHandler& mpc_handler);

    /// Assemble K and F using the pre-MPC dof_map (from mpc_handler.full_dof_map()).
    void assemble(const Model& model,
                  const SubCase& sc,
                  const MpcHandler& mpc_handler,
                  SparseMatrixBuilder& K_builder,
                  std::vector<double>& F);

    /// Apply point loads (FORCE, MOMENT) to F via MpcHandler
    void apply_point_loads(const Model& model,
                           const SubCase& sc,
                           const MpcHandler& mpc_handler,
                           std::vector<double>& F);

    /// Apply thermal loads to F via MpcHandler
    void apply_thermal_loads(const Model& model,
                             const SubCase& sc,
                             const MpcHandler& mpc_handler,
                             SparseMatrixBuilder& K_builder,
                             std::vector<double>& F);

    /// Recover stresses and displacements from full (pre-MPC) dof_map + u_free
    SubCaseResults recover_results(const Model& model,
                                   const SubCase& sc,
                                   const DofMap& dof_map,
                                   const std::vector<double>& u_free);
};

} // namespace vibestran
