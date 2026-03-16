// src/core/dof_map.cpp
#include "core/dof_map.hpp"
#include <stdexcept>
#include <format>
#include <algorithm>
#include <vector>

namespace nastran {

void DofMap::build(const std::unordered_map<NodeId, GridPoint>& nodes,
                   int default_dofs_per_node) {
    blocks_.clear();
    num_free_  = 0;
    num_total_ = 0;

    // Assign equation indices in node-ID order for determinism
    // (sort node IDs first)
    std::vector<NodeId> sorted_ids;
    sorted_ids.reserve(nodes.size());
    for (const auto& [nid, _] : nodes)
        sorted_ids.push_back(nid);
    std::sort(sorted_ids.begin(), sorted_ids.end());

    for (NodeId nid : sorted_ids) {
        NodeDofBlock& blk = blocks_[nid];
        for (int d = 0; d < default_dofs_per_node; ++d) {
            blk.eq[d] = num_free_++;
        }
        // remaining slots (if dofs_per_node < 6) stay CONSTRAINED_DOF
        num_total_ += default_dofs_per_node;
    }
    num_total_ = num_free_; // before constraints
}

void DofMap::constrain(NodeId node, int local_dof_0based) {
    auto it = blocks_.find(node);
    if (it == blocks_.end())
        throw std::runtime_error(std::format("constrain: node {} not in DofMap", node.value));

    EqIndex& eq = it->second.eq[local_dof_0based];
    if (eq == CONSTRAINED_DOF) return; // already constrained

    // Remove from free DOFs: shift all higher indices down by 1
    EqIndex removed = eq;
    eq = CONSTRAINED_DOF;

    for (auto& [_, blk] : blocks_) {
        for (auto& e : blk.eq) {
            if (e > removed) --e;
        }
    }
    --num_free_;
}

void DofMap::constrain_batch(std::span<const std::pair<NodeId, int>> dofs) {
    // Step 1: collect eq indices of all DOFs that will be removed.
    std::vector<EqIndex> removed;
    removed.reserve(dofs.size());
    for (const auto &[nid, d] : dofs) {
        auto it = blocks_.find(nid);
        if (it == blocks_.end())
            throw std::runtime_error(
                std::format("constrain_batch: node {} not in DofMap", nid.value));
        EqIndex eq = it->second.eq[d];
        if (eq != CONSTRAINED_DOF)
            removed.push_back(eq);
    }

    if (removed.empty()) return;

    // Step 2: sort so we can binary-search the shift for each live index.
    std::sort(removed.begin(), removed.end());
    // Remove duplicates (same DOF listed twice).
    removed.erase(std::unique(removed.begin(), removed.end()), removed.end());

    // Step 3: single pass — shift every live eq index down by the number of
    // removed indices that are strictly less than it.
    for (auto &[_, blk] : blocks_) {
        for (auto &e : blk.eq) {
            if (e == CONSTRAINED_DOF) continue;
            // Count removed entries < e
            auto cnt = static_cast<EqIndex>(
                std::lower_bound(removed.begin(), removed.end(), e) - removed.begin());
            e -= cnt;
        }
    }

    // Step 4: mark the targeted slots as constrained.
    for (const auto &[nid, d] : dofs) {
        EqIndex &eq = blocks_.at(nid).eq[d];
        // eq may have already been shifted in step 3, but we just set it to
        // CONSTRAINED_DOF regardless.
        eq = CONSTRAINED_DOF;
    }

    num_free_ -= static_cast<int>(removed.size());
}

EqIndex DofMap::eq_index(NodeId node, int local_dof_0based) const {
    auto it = blocks_.find(node);
    if (it == blocks_.end())
        return CONSTRAINED_DOF;
    return it->second.eq[local_dof_0based];
}

const NodeDofBlock& DofMap::block(NodeId node) const {
    auto it = blocks_.find(node);
    if (it == blocks_.end())
        throw std::runtime_error(std::format("DofMap: node {} not found", node.value));
    return it->second;
}

void DofMap::global_indices(NodeId node, std::span<EqIndex, 6> out) const {
    const auto& blk = block(node);
    for (int i = 0; i < 6; ++i) out[i] = blk.eq[i];
}

std::vector<EqIndex> DofMap::global_indices_subset(
    NodeId node, std::span<const int> local_dofs) const
{
    const auto& blk = block(node);
    std::vector<EqIndex> result;
    result.reserve(local_dofs.size());
    std::transform(local_dofs.begin(), local_dofs.end(), std::back_inserter(result),
                   [&](int d) { return blk.eq[d]; });
    return result;
}

} // namespace nastran
