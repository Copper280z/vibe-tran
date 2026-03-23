// tests/unit/test_mpc_handler.cpp
// Unit tests for MpcHandler: master-slave elimination.

#include "core/dof_map.hpp"
#include "core/mpc_handler.hpp"
#include "core/model.hpp"
#include "io/bdf_parser.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace vibestran;

// ── Helper: build a minimal model with n nodes (all shell) ───────────────────
static Model make_model_n_nodes(int n) {
    Model m;
    for (int i = 1; i <= n; ++i) {
        GridPoint gp;
        gp.id = NodeId{i};
        gp.position = Vec3{static_cast<double>(i), 0, 0};
        m.nodes[gp.id] = gp;
    }
    return m;
}

// Build a DofMap for n nodes, all 6 DOFs free, no SPCs
static DofMap make_full_dof_map(const Model& model) {
    DofMap dm;
    dm.build(model.nodes, 6);
    return dm;
}

// ── Test 1: Simple pair constraint u1 + u2 = 0 ───────────────────────────────
// Two nodes, only DOF 1 (T1) active. MPC: u_node1_T1 + u_node2_T1 = 0.
// After elimination: 1 free DOF instead of 2.
TEST(MpcHandler, SimplePairConstraint) {
    Model model = make_model_n_nodes(2);
    DofMap dof_map = make_full_dof_map(model);

    // Constrain all except T1 on both nodes (simulate solid-only, minimal test)
    {
        std::vector<std::pair<NodeId, int>> rot_cons;
        for (int n = 1; n <= 2; ++n)
            for (int d = 1; d < 6; ++d)  // constrain DOFs 1-5 (leave DOF 0 = T1 free)
                rot_cons.emplace_back(NodeId{n}, d);
        dof_map.constrain_batch(rot_cons);
    }
    // Now 2 free DOFs: node1 T1 (eq 0) and node2 T1 (eq 1)
    ASSERT_EQ(dof_map.num_free_dofs(), 2);

    // MPC: 1*u[node1,T1] + 1*u[node2,T1] = 0
    Mpc mpc;
    mpc.sid = MpcSetId{1};
    mpc.terms = {
        {NodeId{1}, 1, 1.0},
        {NodeId{2}, 1, 1.0},
    };

    MpcHandler handler;
    std::vector<const Mpc*> mpc_ptrs = {&mpc};
    handler.build(mpc_ptrs, dof_map);

    // After elimination: 1 free DOF
    EXPECT_EQ(handler.num_reduced(), 1);
    // One elimination recorded
    EXPECT_TRUE(handler.has_constraints());
}

// ── Test 2: Rigid translation constraint u_dep - u_ind = 0 ───────────────────
// Two spring bars in series via MPC. Both nodes free in T1.
// MPC: u[node1,T1] - u[node2,T1] = 0 → both nodes move together.
// Apply force at node2 T1 = 1000 N. Result: u = F / (2*k) (two springs in parallel).
// Actually: with MPC u1 = u2, the spring between (fixed end and node1) plus spring
// between (node1 and node2) with no direct connection acts differently.
// Simpler: verify K transformation gives correct result.
TEST(MpcHandler, RigidTranslationConstraint) {
    // Two nodes: node 1 (T1 constrained by SPC → eq = CONSTRAINED_DOF after SPC)
    // Actually, let's do it differently: use a spring-like test.
    // 3 nodes, 1 DOF each (T1 only).
    // Node 1: fixed by SPC.
    // Node 2: free.
    // Node 3: free, MPC: u3 = u2 (rigid connection).
    // Spring 1-2 with k=1, spring 2-3 with k=1.
    // Apply F=1 at node3 → should give u2=u3=2/3? Actually with MPC u3=u2:
    // K_red = K transformed. Let me just verify the handler reduces DOF count.

    Model model = make_model_n_nodes(3);
    DofMap dof_map = make_full_dof_map(model);

    // Constrain all except T1
    {
        std::vector<std::pair<NodeId, int>> c;
        for (int n = 1; n <= 3; ++n)
            for (int d = 1; d < 6; ++d)
                c.emplace_back(NodeId{n}, d);
        dof_map.constrain_batch(c);
    }
    // SPC: fix node1 T1
    dof_map.constrain(NodeId{1}, 0);
    // Now 2 free DOFs: node2 T1 (eq 0), node3 T1 (eq 1)
    ASSERT_EQ(dof_map.num_free_dofs(), 2);

    // MPC: u[node3,T1] - u[node2,T1] = 0
    Mpc mpc;
    mpc.sid = MpcSetId{1};
    mpc.terms = {
        {NodeId{3}, 1, 1.0},
        {NodeId{2}, 1, -1.0},
    };

    MpcHandler handler;
    std::vector<const Mpc*> ptrs = {&mpc};
    handler.build(ptrs, dof_map);

    // After MPC elimination: 1 reduced DOF
    EXPECT_EQ(handler.num_reduced(), 1);

    // Verify that the pre-MPC eq for node2 is in the handler
    EqIndex eq2 = handler.full_dof_map().eq_index(NodeId{2}, 0);
    EqIndex eq3 = handler.full_dof_map().eq_index(NodeId{3}, 0);
    EXPECT_NE(eq2, CONSTRAINED_DOF);
    EXPECT_NE(eq3, CONSTRAINED_DOF);
    // One of eq2, eq3 should map to reduced (the independent one)
    EqIndex r2 = handler.reduced_index(eq2);
    EqIndex r3 = handler.reduced_index(eq3);
    // One should be 0 (independent), one CONSTRAINED_DOF (dependent)
    EXPECT_TRUE((r2 == 0 && r3 == CONSTRAINED_DOF) ||
                (r3 == 0 && r2 == CONSTRAINED_DOF));
}

// ── Test 3: Multi-term constraint u1 + 2*u2 + 3*u3 = 0 ──────────────────────
TEST(MpcHandler, MultiTermConstraint) {
    Model model = make_model_n_nodes(3);
    DofMap dof_map = make_full_dof_map(model);

    // Keep only T1 for each node
    {
        std::vector<std::pair<NodeId, int>> c;
        for (int n = 1; n <= 3; ++n)
            for (int d = 1; d < 6; ++d)
                c.emplace_back(NodeId{n}, d);
        dof_map.constrain_batch(c);
    }
    ASSERT_EQ(dof_map.num_free_dofs(), 3);

    // MPC: 1*u1 + 2*u2 + 3*u3 = 0
    // Largest |coeff| is 3 → node3 is dependent
    Mpc mpc;
    mpc.sid = MpcSetId{1};
    mpc.terms = {
        {NodeId{1}, 1, 1.0},
        {NodeId{2}, 1, 2.0},
        {NodeId{3}, 1, 3.0},
    };

    MpcHandler handler;
    std::vector<const Mpc*> ptrs = {&mpc};
    handler.build(ptrs, dof_map);

    EXPECT_EQ(handler.num_reduced(), 2);
    EXPECT_TRUE(handler.has_constraints());

    // Verify elimination: u3 = -(1/3)*u1 - (2/3)*u2
    // The dependent DOF (node3 T1) should have CONSTRAINED_DOF in reduced map
    EqIndex eq3 = handler.full_dof_map().eq_index(NodeId{3}, 0);
    EXPECT_EQ(handler.reduced_index(eq3), CONSTRAINED_DOF);
}

// ── Test 4: Recover dependent DOFs after mock solve ──────────────────────────
TEST(MpcHandler, RecoverDependentDofs) {
    // MPC: u1 + u2 = 0 → u2 = -u1
    Model model = make_model_n_nodes(2);
    DofMap dof_map = make_full_dof_map(model);
    {
        std::vector<std::pair<NodeId, int>> c;
        for (int n = 1; n <= 2; ++n)
            for (int d = 1; d < 6; ++d)
                c.emplace_back(NodeId{n}, d);
        dof_map.constrain_batch(c);
    }

    Mpc mpc;
    mpc.sid = MpcSetId{1};
    mpc.terms = {
        {NodeId{1}, 1, 1.0},
        {NodeId{2}, 1, 1.0},
    };

    MpcHandler handler;
    std::vector<const Mpc*> ptrs = {&mpc};
    handler.build(ptrs, dof_map);
    ASSERT_EQ(handler.num_reduced(), 1);

    // Mock solution: the independent DOF has value 5.0
    std::vector<double> u_reduced = {5.0};
    int n_full = handler.full_dof_map().num_free_dofs();
    std::vector<double> u_full(static_cast<size_t>(n_full), 0.0);
    handler.recover_dependent_dofs(u_full, u_reduced);

    // Verify: one DOF = 5.0 (independent), other = -5.0 (dependent)
    EXPECT_NEAR(std::abs(u_full[0]) + std::abs(u_full[1]), 10.0, 1e-10);
    EXPECT_NEAR(u_full[0] + u_full[1], 0.0, 1e-10); // MPC satisfied
}

// ── Test 5: SPC and MPC on different DOFs ─────────────────────────────────────
TEST(MpcHandler, SpcAndMpcNoOverlap) {
    // 3 nodes. Node1 T1 SPC'd. Node2 T1 and Node3 T1 connected by MPC.
    // Total free before MPC: 2. After MPC: 1.
    Model model = make_model_n_nodes(3);
    DofMap dof_map = make_full_dof_map(model);

    {
        std::vector<std::pair<NodeId, int>> c;
        for (int n = 1; n <= 3; ++n)
            for (int d = 1; d < 6; ++d)
                c.emplace_back(NodeId{n}, d);
        dof_map.constrain_batch(c);
    }
    dof_map.constrain(NodeId{1}, 0); // SPC on node1 T1
    ASSERT_EQ(dof_map.num_free_dofs(), 2); // node2 T1 and node3 T1

    Mpc mpc;
    mpc.sid = MpcSetId{1};
    mpc.terms = {
        {NodeId{2}, 1, 1.0},
        {NodeId{3}, 1, -1.0},
    };

    MpcHandler handler;
    std::vector<const Mpc*> ptrs = {&mpc};
    handler.build(ptrs, dof_map);

    EXPECT_EQ(handler.num_reduced(), 1);
}

// ── Test 6: Cycle detection ────────────────────────────────────────────────────
TEST(MpcHandler, CycleDetection) {
    // MPC 1: u1 = u2
    // MPC 2: u2 = u1  (circular!)
    // → both u1 and u2 would be dependent on each other.
    Model model = make_model_n_nodes(2);
    DofMap dof_map = make_full_dof_map(model);

    {
        std::vector<std::pair<NodeId, int>> c;
        for (int n = 1; n <= 2; ++n)
            for (int d = 1; d < 6; ++d)
                c.emplace_back(NodeId{n}, d);
        dof_map.constrain_batch(c);
    }

    Mpc mpc1, mpc2;
    mpc1.sid = MpcSetId{1};
    mpc1.terms = {{NodeId{1}, 1, 1.0}, {NodeId{2}, 1, -1.0}};

    mpc2.sid = MpcSetId{1};
    mpc2.terms = {{NodeId{2}, 1, 1.0}, {NodeId{1}, 1, -1.0}};

    MpcHandler handler;
    std::vector<const Mpc*> ptrs = {&mpc1, &mpc2};
    // Two MPCs both trying to constrain their respective dep DOFs creates a cycle
    // (node1 dep on node2, node2 dep on node1)
    EXPECT_THROW(handler.build(ptrs, dof_map), SolverError);
}

// ── Test 7: BdfParser MPC multi-line card ────────────────────────────────────
TEST(MpcHandler, BdfParser_MPC_MultiLine) {
    const std::string bdf = R"(
BEGIN BULK
MPC,    1,  101, 1, 1.0,  102, 1, -0.5,  103, 1,
+,                                          -0.5
ENDDATA
)";
    Model model = BdfParser::parse_string(bdf);
    ASSERT_EQ(model.mpcs.size(), 1u);
    const Mpc& mpc = model.mpcs[0];
    EXPECT_EQ(mpc.sid.value, 1);
    // Terms: (101,1,1.0), (102,1,-0.5), (103,1,-0.5)
    ASSERT_EQ(mpc.terms.size(), 3u);
    EXPECT_EQ(mpc.terms[0].node.value, 101);
    EXPECT_NEAR(mpc.terms[0].coeff, 1.0, 1e-12);
    EXPECT_EQ(mpc.terms[1].node.value, 102);
    EXPECT_NEAR(mpc.terms[1].coeff, -0.5, 1e-12);
    EXPECT_EQ(mpc.terms[2].node.value, 103);
    EXPECT_NEAR(mpc.terms[2].coeff, -0.5, 1e-12);
}

// ── Test 8: BdfParser MPCADD merges sets ─────────────────────────────────────
TEST(MpcHandler, BdfParser_MPCADD) {
    const std::string bdf = R"(
BEGIN BULK
MPC,   10,  1, 1, 1.0,  2, 1, -1.0
MPC,   20,  3, 1, 1.0,  4, 1, -1.0
MPCADD, 99,  10,  20
ENDDATA
)";
    Model model = BdfParser::parse_string(bdf);
    // After MPCADD, set 99 should contain 2 MPCs (merged from sets 10 and 20)
    auto v = model.mpcs_for_set(MpcSetId{99});
    EXPECT_EQ(v.size(), 2u);
}
