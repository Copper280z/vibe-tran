// src/solver/analysis_support.cpp

#include "solver/analysis_support.hpp"

#include "core/coord_sys.hpp"
#include "elements/rbe_constraints.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <unordered_map>

namespace vibestran {

namespace {

using ActiveMask = std::array<bool, 6>;

[[nodiscard]] ActiveMask empty_mask() {
  ActiveMask mask{};
  mask.fill(false);
  return mask;
}

void mark_component(ActiveMask &mask, const int component) {
  if (component >= 1 && component <= 6)
    mask[static_cast<size_t>(component - 1)] = true;
}

void mark_all(ActiveMask &mask) { mask.fill(true); }

void mark_basic_translations(ActiveMask &mask) {
  mask[0] = true;
  mask[1] = true;
  mask[2] = true;
}

void mark_element_dofs(const ElementData &elem,
                       std::unordered_map<NodeId, ActiveMask> &node_masks) {
  switch (elem.type) {
  case ElementType::CQUAD4:
  case ElementType::CTRIA3:
  case ElementType::CBAR:
  case ElementType::CBEAM:
  case ElementType::CBUSH:
    for (NodeId nid : elem.nodes)
      mark_all(node_masks[nid]);
    break;
  case ElementType::CHEXA8:
  case ElementType::CHEXA20:
  case ElementType::CTETRA4:
  case ElementType::CTETRA10:
  case ElementType::CPENTA6:
    for (NodeId nid : elem.nodes)
      mark_basic_translations(node_masks[nid]);
    break;
  case ElementType::CELAS1:
  case ElementType::CELAS2:
  case ElementType::CMASS1:
  case ElementType::CMASS2:
    if (!elem.nodes.empty())
      mark_component(node_masks[elem.nodes[0]], elem.components[0]);
    if (elem.nodes.size() > 1)
      mark_component(node_masks[elem.nodes[1]], elem.components[1]);
    break;
  }
}

void mark_rigid_element_dofs(const Model &model,
                             std::unordered_map<NodeId, ActiveMask> &node_masks) {
  for (const auto &rbe2 : model.rbe2s) {
    for (int dof = 1; dof <= 6; ++dof) {
      if (!rbe2.cm.has(dof))
        continue;
      mark_component(node_masks[rbe2.gn], dof);
      for (NodeId nid : rbe2.gm)
        mark_component(node_masks[nid], dof);
    }
  }

  for (const auto &rbe3 : model.rbe3s) {
    for (int dof = 1; dof <= 6; ++dof)
      if (rbe3.refc.has(dof))
        mark_component(node_masks[rbe3.ref_node], dof);
    for (const auto &group : rbe3.weight_groups) {
      for (int dof = 1; dof <= 6; ++dof) {
        if (!group.component.has(dof))
          continue;
        for (NodeId nid : group.nodes)
          mark_component(node_masks[nid], dof);
      }
    }
  }
}

void mark_explicit_mpc_dofs(const Model &model, const SubCase &sc,
                            std::unordered_map<NodeId, ActiveMask> &node_masks) {
  if (sc.mpc_set.value == 0)
    return;
  for (const Mpc *mpc : model.mpcs_for_set(sc.mpc_set)) {
    for (const auto &term : mpc->terms)
      mark_component(node_masks[term.node], term.dof);
  }
}

void reduce_3x3_constraints(NodeId nid, const Mat3 &T3,
                            const std::array<int, 3> &cd_dofs,
                            const int count, const int base_offset,
                            std::vector<Mpc> &all_mpcs,
                            std::vector<std::pair<NodeId, int>> &direct_constraints) {
  if (count == 0)
    return;
  if (count == 3) {
    for (int j = 0; j < 3; ++j)
      direct_constraints.emplace_back(nid, base_offset + j);
    return;
  }

  double A[3][3] = {};
  int col_perm[3] = {0, 1, 2};
  for (int i = 0; i < count; ++i)
    for (int j = 0; j < 3; ++j)
      A[i][j] = T3(j, cd_dofs[i]);

  for (int row = 0; row < count; ++row) {
    int best_col = row;
    double best_val = std::abs(A[row][row]);
    for (int col = row + 1; col < 3; ++col) {
      if (std::abs(A[row][col]) > best_val) {
        best_val = std::abs(A[row][col]);
        best_col = col;
      }
    }
    if (best_col != row) {
      for (int i = 0; i < count; ++i)
        std::swap(A[i][row], A[i][best_col]);
      std::swap(col_perm[row], col_perm[best_col]);
    }
    if (std::abs(A[row][row]) < 1e-14)
      continue;
    for (int i = row + 1; i < count; ++i) {
      const double factor = A[i][row] / A[row][row];
      for (int j = row; j < 3; ++j)
        A[i][j] -= factor * A[row][j];
    }
  }

  for (int row = 0; row < count; ++row) {
    if (std::abs(A[row][row]) < 1e-14)
      continue;

    int nnz = 0;
    for (int col = row; col < 3; ++col)
      if (std::abs(A[row][col]) > 1e-14)
        ++nnz;

    if (nnz == 1) {
      direct_constraints.emplace_back(nid, base_offset + col_perm[row]);
      continue;
    }

    Mpc mpc;
    mpc.sid = MpcSetId{0};
    for (int col = row; col < 3; ++col) {
      if (std::abs(A[row][col]) > 1e-14)
        mpc.terms.push_back({nid, base_offset + col_perm[col] + 1, A[row][col]});
    }
    if (!mpc.terms.empty())
      all_mpcs.push_back(std::move(mpc));
  }
}

} // namespace

DofMap build_analysis_dof_map(const Model &model, const SubCase &sc) {
  DofMap dmap;
  dmap.build(model.nodes, 6);

  std::unordered_map<NodeId, ActiveMask> node_masks;
  for (const auto &[nid, _] : model.nodes)
    node_masks.emplace(nid, empty_mask());

  for (const auto &elem : model.elements)
    mark_element_dofs(elem, node_masks);
  mark_rigid_element_dofs(model, node_masks);
  mark_explicit_mpc_dofs(model, sc, node_masks);

  std::vector<std::pair<NodeId, int>> inactive_constraints;
  for (const auto &[nid, mask] : node_masks) {
    for (int d = 0; d < 6; ++d) {
      if (!mask[static_cast<size_t>(d)])
        inactive_constraints.emplace_back(nid, d);
    }
  }
  dmap.constrain_batch(inactive_constraints);

  std::vector<std::pair<NodeId, int>> spc_constraints;
  for (const Spc *spc : model.spcs_for_set(sc.spc_set)) {
    auto node_it = model.nodes.find(spc->node);
    const bool has_cd =
        (node_it != model.nodes.end() && node_it->second.cd.value != 0);
    for (int d = 0; d < 6; ++d) {
      if (!spc->dofs.has(d + 1))
        continue;
      if (has_cd)
        continue;
      spc_constraints.emplace_back(spc->node, d);
    }
  }
  dmap.constrain_batch(spc_constraints);

  return dmap;
}

// cppcheck-suppress unusedFunction -- referenced from linear_static.cpp and modal.cpp
void build_analysis_mpc_system(const Model &model, const SubCase &sc,
                               DofMap &dof_map, MpcHandler &mpc_handler) {
  std::vector<Mpc> all_mpcs;

  {
    std::unordered_map<NodeId, DofSet> node_spc_dofs;
    for (const Spc *spc : model.spcs_for_set(sc.spc_set)) {
      if (spc->value != 0.0)
        continue;
      node_spc_dofs[spc->node].mask |= spc->dofs.mask;
    }

    std::vector<std::pair<NodeId, int>> direct_constraints;
    for (const auto &[nid, gp] : model.nodes) {
      if (gp.cd == CoordId{0})
        continue;
      const auto spc_it = node_spc_dofs.find(nid);
      if (spc_it == node_spc_dofs.end())
        continue;
      const auto cs_it = model.coord_systems.find(gp.cd);
      if (cs_it == model.coord_systems.end())
        continue;

      const Mat3 T3 = rotation_matrix(cs_it->second, gp.position);
      const DofSet dofs = spc_it->second;

      std::array<int, 3> trans{};
      int n_trans = 0;
      for (int d = 0; d < 3; ++d) {
        if (dofs.has(d + 1) && dof_map.is_free(nid, d))
          trans[n_trans++] = d;
      }
      reduce_3x3_constraints(nid, T3, trans, n_trans, 0, all_mpcs,
                             direct_constraints);

      std::array<int, 3> rot{};
      int n_rot = 0;
      for (int d = 0; d < 3; ++d) {
        if (dofs.has(d + 4) && dof_map.is_free(nid, d + 3))
          rot[n_rot++] = d;
      }
      reduce_3x3_constraints(nid, T3, rot, n_rot, 3, all_mpcs,
                             direct_constraints);
    }

    if (!direct_constraints.empty())
      dof_map.constrain_batch(direct_constraints);
  }

  for (const auto &rbe2 : model.rbe2s)
    expand_rbe2(rbe2, model, all_mpcs);
  for (const auto &rbe3 : model.rbe3s)
    expand_rbe3(rbe3, model, all_mpcs);
  if (sc.mpc_set.value != 0) {
    const auto explicit_mpcs = model.mpcs_for_set(sc.mpc_set);
    std::transform(explicit_mpcs.begin(), explicit_mpcs.end(),
                   std::back_inserter(all_mpcs),
                   [](const Mpc *mpc) { return *mpc; });
  }

  std::vector<const Mpc *> mpc_ptrs;
  mpc_ptrs.reserve(all_mpcs.size());
  std::transform(all_mpcs.begin(), all_mpcs.end(), std::back_inserter(mpc_ptrs),
                 [](const Mpc &mpc) { return &mpc; });
  mpc_handler.build(mpc_ptrs, dof_map);
}

} // namespace vibestran
