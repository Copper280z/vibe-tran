#pragma once
// include/elements/rbe_constraints.hpp
// Expand RBE2 and RBE3 elements into MPC equations.

#include "core/model.hpp"
#include <vector>

namespace vibestran {

/// Expand an RBE2 element into MPC equations.
/// For each dependent node gm and each constrained DOF d in rbe.cm, generates
/// rigid-body MPCs coupling gm to the independent node gn.
/// Rotation DOFs that are already constrained (solid-only nodes) are skipped
/// by not generating MPCs for them; the solver will silently skip them.
/// @param rbe    The RBE2 definition.
/// @param model  Used to look up node positions.
/// @param out    Appended with the generated MPC equations.
void expand_rbe2(const Rbe2& rbe, const Model& model, std::vector<Mpc>& out);

/// Expand an RBE3 element into MPC equations.
/// Generates one MPC per constrained DOF in rbe.refc, expressing the reference
/// node motion as a weighted average of independent node motions.
/// @param rbe    The RBE3 definition.
/// @param model  Used to look up node positions.
/// @param out    Appended with the generated MPC equations.
void expand_rbe3(const Rbe3& rbe, const Model& model, std::vector<Mpc>& out);

} // namespace vibestran
