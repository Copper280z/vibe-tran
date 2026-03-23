#pragma once
// include/core/dof_map.hpp
// Maps (NodeId, local_dof 0-5) to a global equation (row/column) index.
// Also tracks which DOFs are free vs. constrained.
//
// GPU readiness: The internal data is a contiguous flat array of int32
// offsets, suitable for transfer to device memory.

#include "core/types.hpp"
#include "model.hpp"
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace vibestran {

/// Index into the global (assembled) equation system
using EqIndex = int32_t;
static constexpr EqIndex CONSTRAINED_DOF = -1;

/// Per-node DOF offset block: 6 equation indices (or CONSTRAINED_DOF)
struct NodeDofBlock {
  std::array<EqIndex, 6> eq{}; // eq[0]=T1 .. eq[5]=R3
  NodeDofBlock() { eq.fill(CONSTRAINED_DOF); }
};

class DofMap {
public:
  DofMap() = default;

  /// Build from the model: assign equation indices to all free DOFs.
  /// dofs_per_node is 6 for general 3-D; shell elements use all 6,
  /// solid elements only 3 (T1-T3) — the map always reserves 6 slots
  /// but only activates the ones the element type needs.
  void build(const std::unordered_map<NodeId, GridPoint> &nodes,
             int default_dofs_per_node = 6);

  /// Mark a DOF as constrained (equation index → CONSTRAINED_DOF)
  void constrain(NodeId node, int local_dof_0based);

  /// Constrain multiple DOFs in one pass.  Equivalent to calling constrain()
  /// for each entry but O(M log M + N·log M) instead of O(M·N).
  void constrain_batch(std::span<const std::pair<NodeId, int>> dofs);

  /// Retrieve equation index for (node, local_dof 0-based).
  /// Returns CONSTRAINED_DOF if fixed.
  [[nodiscard]] EqIndex eq_index(NodeId node, int local_dof_0based) const;

  /// True if the given node+dof is free
  [[nodiscard]] bool is_free(NodeId node, int local_dof_0based) const {
    return eq_index(node, local_dof_0based) != CONSTRAINED_DOF;
  }

  /// Number of free (active) DOFs = size of the global stiffness matrix
  [[nodiscard]] int num_free_dofs() const noexcept { return num_free_; }

  /// Number of total DOFs (free + constrained)
  [[nodiscard]] int num_total_dofs() const noexcept { return num_total_; }

  /// All 6 equation indices for a node
  [[nodiscard]] const NodeDofBlock &block(NodeId node) const;

  /// Fill a small array with the 6 global eq indices for a node.
  /// Useful for element assembly loops.
  void global_indices(NodeId node, std::span<EqIndex, 6> out) const;

  /// For a given node, fill the local-to-global map for a subset of DOFs.
  /// local_dofs is a list of 0-based local dof indices (e.g. {0,1,2} for
  /// solids).
  std::vector<EqIndex>
  global_indices_subset(NodeId node, std::span<const int> local_dofs) const;

private:
  std::unordered_map<NodeId, NodeDofBlock> blocks_;
  int num_free_{0};
  int num_total_{0};
};

} // namespace vibestran
