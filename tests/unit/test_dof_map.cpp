// tests/unit/test_dof_map.cpp
// Tests: DofMap assignment, constraint application, index correctness.
// These are fundamental — if DOF indexing is wrong, every analysis is wrong.

#include <gtest/gtest.h>
#include "core/dof_map.hpp"
#include "core/model.hpp"
#include "solver/analysis_support.hpp"
#include <set>

using namespace vibestran;

// Helper: build a simple 2-node model
static std::unordered_map<NodeId, GridPoint> two_nodes() {
    std::unordered_map<NodeId, GridPoint> nodes;
    nodes[NodeId{1}] = GridPoint{NodeId{1}, CoordId{0}, Vec3{0,0,0}, CoordId{0}};
    nodes[NodeId{2}] = GridPoint{NodeId{2}, CoordId{0}, Vec3{1,0,0}, CoordId{0}};
    return nodes;
}

TEST(DofMap, TwoNodesTwelveDOFBeforeConstraint) {
    auto nodes = two_nodes();
    DofMap dm;
    dm.build(nodes, 6);
    EXPECT_EQ(dm.num_free_dofs(), 12);
    EXPECT_EQ(dm.num_total_dofs(), 12);
}

TEST(DofMap, AllIndicesAreUnique) {
    auto nodes = two_nodes();
    DofMap dm;
    dm.build(nodes, 6);

    std::set<EqIndex> seen;
    for (int n : {1,2}) {
        for (int d = 0; d < 6; ++d) {
            EqIndex eq = dm.eq_index(NodeId{n}, d);
            EXPECT_NE(eq, CONSTRAINED_DOF);
            EXPECT_TRUE(seen.insert(eq).second) << "Duplicate eq index " << eq;
        }
    }
    EXPECT_EQ(static_cast<int>(seen.size()), 12);
}

TEST(DofMap, ConstrainReducesFreeCount) {
    auto nodes = two_nodes();
    DofMap dm;
    dm.build(nodes, 6);

    dm.constrain(NodeId{1}, 0); // fix T1 of node 1
    EXPECT_EQ(dm.num_free_dofs(), 11);
    EXPECT_EQ(dm.eq_index(NodeId{1}, 0), CONSTRAINED_DOF);
}

TEST(DofMap, ConstraintShiftsHigherIndices) {
    // After constraining one DOF, indices above it should shift down by 1
    auto nodes = two_nodes();
    DofMap dm;
    dm.build(nodes, 6);

    // Record index of node2/T1 before constraint
    EqIndex before = dm.eq_index(NodeId{2}, 0);
    // Constrain node1/T1 (the first DOF in sorted order)
    dm.constrain(NodeId{1}, 0);
    EqIndex after = dm.eq_index(NodeId{2}, 0);

    // If before > 0 (i.e., the constrained dof was before node2/T1), after = before - 1
    // We know node IDs are sorted: node1 < node2, so node1 DOFs come first
    EXPECT_EQ(after, before - 1);
}

TEST(DofMap, MultipleConstraints) {
    auto nodes = two_nodes();
    DofMap dm;
    dm.build(nodes, 6);

    // Fix all DOFs on node 1 (fully clamped)
    for (int d = 0; d < 6; ++d) dm.constrain(NodeId{1}, d);
    EXPECT_EQ(dm.num_free_dofs(), 6);
    for (int d = 0; d < 6; ++d)
        EXPECT_EQ(dm.eq_index(NodeId{1}, d), CONSTRAINED_DOF);
}

TEST(DofMap, UnknownNodeReturnsConstrained) {
    auto nodes = two_nodes();
    DofMap dm;
    dm.build(nodes, 6);
    EXPECT_EQ(dm.eq_index(NodeId{999}, 0), CONSTRAINED_DOF);
}

TEST(DofMap, GlobalIndicesSubset) {
    auto nodes = two_nodes();
    DofMap dm;
    dm.build(nodes, 6);

    static constexpr int solid_dofs[3] = {0,1,2};
    auto indices = dm.global_indices_subset(NodeId{1}, solid_dofs);
    EXPECT_EQ(indices.size(), 3u);
    for (auto eq : indices) EXPECT_NE(eq, CONSTRAINED_DOF);
}

// Tests for is_free (only used in tests)
TEST(DofMap, IsFreeTrueForUnconstrainedDof) {
    auto nodes = two_nodes();
    DofMap dm;
    dm.build(nodes, 6);

    // All DOFs should be free before any constraint
    for (int d = 0; d < 6; ++d) {
        EXPECT_TRUE(dm.is_free(NodeId{1}, d));
        EXPECT_TRUE(dm.is_free(NodeId{2}, d));
    }
}

TEST(DofMap, IsFreeFalseAfterConstrain) {
    auto nodes = two_nodes();
    DofMap dm;
    dm.build(nodes, 6);

    dm.constrain(NodeId{1}, 2); // fix T3 of node 1
    EXPECT_FALSE(dm.is_free(NodeId{1}, 2));
    // Other DOFs on same node should still be free
    EXPECT_TRUE(dm.is_free(NodeId{1}, 0));
    EXPECT_TRUE(dm.is_free(NodeId{2}, 2));
}

TEST(DofMap, IsFreeFalseForUnknownNode) {
    auto nodes = two_nodes();
    DofMap dm;
    dm.build(nodes, 6);

    // Unknown nodes are reported as not-free (CONSTRAINED_DOF)
    EXPECT_FALSE(dm.is_free(NodeId{999}, 0));
}

// Tests for global_indices (only used in tests)
TEST(DofMap, GlobalIndicesMatchEqIndex) {
    auto nodes = two_nodes();
    DofMap dm;
    dm.build(nodes, 6);

    std::array<EqIndex, 6> out{};
    dm.global_indices(NodeId{1}, out);

    for (int d = 0; d < 6; ++d)
        EXPECT_EQ(out[d], dm.eq_index(NodeId{1}, d));
}

// Helper: build a 4-node model
static std::unordered_map<NodeId, GridPoint> four_nodes() {
    std::unordered_map<NodeId, GridPoint> nodes;
    for (int i = 1; i <= 4; ++i)
        nodes[NodeId{i}] = GridPoint{NodeId{i}, CoordId{0},
                                     Vec3{static_cast<double>(i), 0, 0}, CoordId{0}};
    return nodes;
}

// constrain_batch must produce the same final numbering as the equivalent
// sequence of constrain() calls.
TEST(DofMap, ConstrainBatchMatchesSerialConstrain) {
    auto nodes = four_nodes();

    // Serial reference
    DofMap serial;
    serial.build(nodes, 6);
    serial.constrain(NodeId{1}, 3);
    serial.constrain(NodeId{1}, 4);
    serial.constrain(NodeId{1}, 5);
    serial.constrain(NodeId{2}, 3);
    serial.constrain(NodeId{2}, 4);
    serial.constrain(NodeId{2}, 5);
    serial.constrain(NodeId{3}, 0);
    serial.constrain(NodeId{4}, 2);

    // Batch
    DofMap batch;
    batch.build(nodes, 6);
    std::vector<std::pair<NodeId,int>> dofs = {
        {NodeId{1},3},{NodeId{1},4},{NodeId{1},5},
        {NodeId{2},3},{NodeId{2},4},{NodeId{2},5},
        {NodeId{3},0},{NodeId{4},2}
    };
    batch.constrain_batch(dofs);

    EXPECT_EQ(batch.num_free_dofs(), serial.num_free_dofs());
    for (int n = 1; n <= 4; ++n)
        for (int d = 0; d < 6; ++d)
            EXPECT_EQ(batch.eq_index(NodeId{n}, d), serial.eq_index(NodeId{n}, d))
                << "mismatch at node " << n << " dof " << d;
}

// Duplicate entries in the batch list must not double-count or corrupt indices.
TEST(DofMap, ConstrainBatchIgnoresDuplicates) {
    auto nodes = two_nodes();
    DofMap dm;
    dm.build(nodes, 6);

    // Constraining the same DOF twice should be idempotent.
    std::vector<std::pair<NodeId,int>> dofs = {
        {NodeId{1},0},{NodeId{1},0},{NodeId{1},1}
    };
    dm.constrain_batch(dofs);

    EXPECT_EQ(dm.num_free_dofs(), 10); // 12 - 2 unique constraints
    EXPECT_EQ(dm.eq_index(NodeId{1}, 0), CONSTRAINED_DOF);
    EXPECT_EQ(dm.eq_index(NodeId{1}, 1), CONSTRAINED_DOF);
    EXPECT_NE(dm.eq_index(NodeId{1}, 2), CONSTRAINED_DOF);
}

TEST(DofMap, GlobalIndicesAfterConstraint) {
    auto nodes = two_nodes();
    DofMap dm;
    dm.build(nodes, 6);
    dm.constrain(NodeId{1}, 0);

    std::array<EqIndex, 6> out{};
    dm.global_indices(NodeId{1}, out);

    EXPECT_EQ(out[0], CONSTRAINED_DOF);
    for (int d = 1; d < 6; ++d)
        EXPECT_NE(out[d], CONSTRAINED_DOF);
}

static Model make_analysis_support_model() {
    Model m;
    auto add_node = [&](int id, double x, double y, double z) {
        m.nodes[NodeId{id}] = GridPoint{NodeId{id}, CoordId{0}, Vec3{x, y, z}, CoordId{0}};
    };

    add_node(1, 0.0, 0.0, 0.0);
    add_node(2, 1.0, 0.0, 0.0);
    add_node(3, 0.0, 1.0, 0.0);
    add_node(4, 1.0, 1.0, 0.0);
    add_node(5, 0.0, 0.0, 1.0);
    add_node(6, 1.0, 0.0, 1.0);
    add_node(7, 0.0, 1.0, 1.0);
    add_node(8, 1.0, 1.0, 1.0);

    return m;
}

TEST(AnalysisSupport, ShellSolidSharedNodeKeepsRotationsActive) {
    Model m = make_analysis_support_model();

    ElementData shell;
    shell.id = ElementId{1};
    shell.type = ElementType::CQUAD4;
    shell.nodes = {NodeId{1}, NodeId{2}, NodeId{4}, NodeId{3}};
    m.elements.push_back(shell);

    ElementData solid;
    solid.id = ElementId{2};
    solid.type = ElementType::CTETRA4;
    solid.nodes = {NodeId{1}, NodeId{5}, NodeId{6}, NodeId{7}};
    m.elements.push_back(solid);

    const DofMap dmap = build_analysis_dof_map(m, SubCase{});
    EXPECT_TRUE(dmap.is_free(NodeId{1}, 3));
    EXPECT_TRUE(dmap.is_free(NodeId{1}, 4));
    EXPECT_TRUE(dmap.is_free(NodeId{1}, 5));
    EXPECT_FALSE(dmap.is_free(NodeId{5}, 3));
}

TEST(AnalysisSupport, BeamShellSharedNodeKeepsBeamRotationsActive) {
    Model m = make_analysis_support_model();

    ElementData beam;
    beam.id = ElementId{1};
    beam.type = ElementType::CBAR;
    beam.nodes = {NodeId{1}, NodeId{2}};
    m.elements.push_back(beam);

    ElementData shell;
    shell.id = ElementId{2};
    shell.type = ElementType::CQUAD4;
    shell.nodes = {NodeId{1}, NodeId{3}, NodeId{4}, NodeId{2}};
    m.elements.push_back(shell);

    const DofMap dmap = build_analysis_dof_map(m, SubCase{});
    EXPECT_TRUE(dmap.is_free(NodeId{1}, 3));
    EXPECT_TRUE(dmap.is_free(NodeId{1}, 4));
    EXPECT_TRUE(dmap.is_free(NodeId{1}, 5));
}

TEST(AnalysisSupport, BeamSolidSharedNodeKeepsBeamRotationsActive) {
    Model m = make_analysis_support_model();

    ElementData beam;
    beam.id = ElementId{1};
    beam.type = ElementType::CBAR;
    beam.nodes = {NodeId{1}, NodeId{2}};
    m.elements.push_back(beam);

    ElementData solid;
    solid.id = ElementId{2};
    solid.type = ElementType::CTETRA4;
    solid.nodes = {NodeId{1}, NodeId{5}, NodeId{6}, NodeId{7}};
    m.elements.push_back(solid);

    const DofMap dmap = build_analysis_dof_map(m, SubCase{});
    EXPECT_TRUE(dmap.is_free(NodeId{1}, 3));
    EXPECT_TRUE(dmap.is_free(NodeId{1}, 4));
    EXPECT_TRUE(dmap.is_free(NodeId{1}, 5));
    EXPECT_FALSE(dmap.is_free(NodeId{5}, 3));
}
