// tests/unit/test_quality_checks.cpp
// Unit tests for the input deck quality checker.
// Tests prove mathematical correctness of geometry algorithms, mode dispatch,
// and topology/physical checks.

#include <gtest/gtest.h>
#include "core/quality_checks.hpp"
#include "core/exceptions.hpp"
#include "core/model.hpp"
#include <cmath>
#include <numbers>

using namespace vibestran;

// ── Model-building helpers ────────────────────────────────────────────────────

namespace {

/// Add a node at given position
void add_node(Model& m, int id, double x, double y, double z) {
    GridPoint gp;
    gp.id = NodeId{id};
    gp.position = Vec3{x, y, z};
    m.nodes[NodeId{id}] = gp;
}

/// Add a MAT1 material
void add_mat(Model& m, int mid, double E = 2e11, double nu = 0.3, double rho = 7800.0) {
    Mat1 mat;
    mat.id  = MaterialId{mid};
    mat.E   = E;
    mat.nu  = nu;
    mat.rho = rho;
    m.materials[MaterialId{mid}] = mat;
}

/// Add a PSHELL property
void add_pshell(Model& m, int pid, int mid, double t = 0.01) {
    PShell ps;
    ps.pid  = PropertyId{pid};
    ps.mid1 = MaterialId{mid};
    ps.t    = t;
    m.properties[PropertyId{pid}] = ps;
}

/// Add a PSolid property
void add_psolid(Model& m, int pid, int mid) {
    PSolid ps;
    ps.pid = PropertyId{pid};
    ps.mid = MaterialId{mid};
    m.properties[PropertyId{pid}] = ps;
}

/// Add a CQUAD4 element
void add_cquad4(Model& m, int eid, int pid, int n1, int n2, int n3, int n4) {
    ElementData ed;
    ed.id   = ElementId{eid};
    ed.type = ElementType::CQUAD4;
    ed.pid  = PropertyId{pid};
    ed.nodes = {NodeId{n1}, NodeId{n2}, NodeId{n3}, NodeId{n4}};
    m.elements.push_back(ed);
}

/// Add a CTRIA3 element
void add_ctria3(Model& m, int eid, int pid, int n1, int n2, int n3) {
    ElementData ed;
    ed.id   = ElementId{eid};
    ed.type = ElementType::CTRIA3;
    ed.pid  = PropertyId{pid};
    ed.nodes = {NodeId{n1}, NodeId{n2}, NodeId{n3}};
    m.elements.push_back(ed);
}

/// Add a CHEXA8 element
void add_chexa8(Model& m, int eid, int pid,
                int n1, int n2, int n3, int n4, int n5, int n6, int n7, int n8) {
    ElementData ed;
    ed.id   = ElementId{eid};
    ed.type = ElementType::CHEXA8;
    ed.pid  = PropertyId{pid};
    ed.nodes = {NodeId{n1},NodeId{n2},NodeId{n3},NodeId{n4},
                NodeId{n5},NodeId{n6},NodeId{n7},NodeId{n8}};
    m.elements.push_back(ed);
}

/// Add a CTETRA4 element
void add_ctetra4(Model& m, int eid, int pid, int n1, int n2, int n3, int n4) {
    ElementData ed;
    ed.id   = ElementId{eid};
    ed.type = ElementType::CTETRA4;
    ed.pid  = PropertyId{pid};
    ed.nodes = {NodeId{n1}, NodeId{n2}, NodeId{n3}, NodeId{n4}};
    m.elements.push_back(ed);
}

/// Add a CBUSH element
void add_cbush(Model& m, int eid, int pid, int n1, int n2) {
    ElementData ed;
    ed.id   = ElementId{eid};
    ed.type = ElementType::CBUSH;
    ed.pid  = PropertyId{pid};
    ed.nodes = {NodeId{n1}, NodeId{n2}};
    m.elements.push_back(ed);
}

/// Add a CELAS2 element (inline stiffness, no property needed)
void add_celas2(Model& m, int eid, int n1, int n2) {
    ElementData ed;
    ed.id         = ElementId{eid};
    ed.type       = ElementType::CELAS2;
    ed.nodes      = {NodeId{n1}, NodeId{n2}};
    ed.components = {1, 1};
    ed.value      = 1e6;
    m.elements.push_back(ed);
}

/// Add a SPC constraining all DOFs on a node
void add_spc_all(Model& m, int sid, int nid) {
    Spc spc;
    spc.sid  = SpcSetId{sid};
    spc.node = NodeId{nid};
    spc.dofs = DofSet::all();
    m.spcs.push_back(spc);
}

/// Minimal viable SOL 101 model: 4-node flat quad with constraints and load
Model make_minimal_sol101_model() {
    Model m;
    m.analysis.sol = SolutionType::LinearStatic;
    SubCase sc;
    sc.id       = 1;
    sc.load_set = LoadSetId{1};
    sc.spc_set  = SpcSetId{1};
    m.analysis.subcases.push_back(sc);

    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 1.0, 0.0, 0.0);
    add_node(m, 3, 1.0, 1.0, 0.0);
    add_node(m, 4, 0.0, 1.0, 0.0);
    add_mat(m, 1);
    add_pshell(m, 1, 1);
    add_cquad4(m, 1, 1, 1, 2, 3, 4);
    add_spc_all(m, 1, 1);
    add_spc_all(m, 1, 2);

    ForceLoad fl;
    fl.sid       = LoadSetId{1};
    fl.node      = NodeId{3};
    fl.scale     = 1000.0;
    fl.direction = Vec3{0.0, 0.0, 1.0};
    m.loads.push_back(fl);

    return m;
}

} // namespace

// ── QualityThresholdsParsing ──────────────────────────────────────────────────

TEST(QualityThresholdsParsing, DefaultsAreCorrect) {
    Model m;
    QualityThresholds t = build_thresholds(m);
    EXPECT_EQ(t.mode, CheckMode::Strict);
    EXPECT_DOUBLE_EQ(t.max_aspect_ratio, 20.0);
    EXPECT_DOUBLE_EQ(t.min_jacobian_ratio, 0.0);
    EXPECT_DOUBLE_EQ(t.max_warp_angle, 30.0);
    EXPECT_DOUBLE_EQ(t.max_taper_ratio, 0.5);
    EXPECT_DOUBLE_EQ(t.min_interior_angle, 10.0);
    EXPECT_DOUBLE_EQ(t.max_interior_angle, 170.0);
    EXPECT_DOUBLE_EQ(t.dup_node_tol, 1e-8);
    EXPECT_TRUE(t.auto_merge_nodes);
    EXPECT_DOUBLE_EQ(t.maxratio, 1e7);
    EXPECT_EQ(t.bailout, 0);
}

TEST(QualityThresholdsParsing, LenientModeActivated) {
    Model m;
    m.params["CHECKMODE"] = "LENIENT";
    QualityThresholds t = build_thresholds(m);
    EXPECT_EQ(t.mode, CheckMode::Lenient);
    // In lenient mode with no explicit BAILOUT, default to -1
    EXPECT_EQ(t.bailout, -1);
}

TEST(QualityThresholdsParsing, ParamsOverrideDefaults) {
    Model m;
    m.params["ASPECT"]   = "10.0";
    m.params["WARP"]     = "15.0";
    m.params["JACMIN"]   = "0.2";
    m.params["TAPER"]    = "0.3";
    m.params["MINANGLE"] = "5.0";
    m.params["MAXANGLE"] = "160.0";
    m.params["DUPNODTOL"]= "1e-6";
    m.params["AUTOMERGE"]= "0";
    m.params["MAXRATIO"] = "1e5";
    m.params["BAILOUT"]  = "-1";

    QualityThresholds t = build_thresholds(m);
    EXPECT_DOUBLE_EQ(t.max_aspect_ratio, 10.0);
    EXPECT_DOUBLE_EQ(t.max_warp_angle, 15.0);
    EXPECT_DOUBLE_EQ(t.min_jacobian_ratio, 0.2);
    EXPECT_DOUBLE_EQ(t.max_taper_ratio, 0.3);
    EXPECT_DOUBLE_EQ(t.min_interior_angle, 5.0);
    EXPECT_DOUBLE_EQ(t.max_interior_angle, 160.0);
    EXPECT_NEAR(t.dup_node_tol, 1e-6, 1e-20);
    EXPECT_FALSE(t.auto_merge_nodes);
    EXPECT_NEAR(t.maxratio, 1e5, 1.0);
    EXPECT_EQ(t.bailout, -1);
}

TEST(QualityThresholdsParsing, ExplicitBailoutInLenientMode) {
    Model m;
    m.params["CHECKMODE"] = "LENIENT";
    m.params["BAILOUT"]   = "0";
    QualityThresholds t = build_thresholds(m);
    EXPECT_EQ(t.bailout, 0);  // explicit overrides the lenient default of -1
}

// ── ElementQualityGeometry: CQUAD4 ───────────────────────────────────────────

TEST(ElementQualityGeometry, Cquad4UnitSquareAspect) {
    Model m;
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 1.0, 0.0, 0.0);
    add_node(m, 3, 1.0, 1.0, 0.0);
    add_node(m, 4, 0.0, 1.0, 0.0);
    add_mat(m, 1); add_pshell(m, 1, 1);
    add_cquad4(m, 1, 1, 1, 2, 3, 4);

    auto q = compute_element_quality(m.elements[0], m);
    ASSERT_TRUE(q.has_value());
    EXPECT_NEAR(q->aspect_ratio, 1.0, 1e-12);
    EXPECT_NEAR(q->warp_angle, 0.0, 1e-10);
    EXPECT_NEAR(q->taper_ratio, 0.0, 1e-12);
    EXPECT_NEAR(q->min_interior_angle, 90.0, 1e-10);
    EXPECT_NEAR(q->max_interior_angle, 90.0, 1e-10);
}

TEST(ElementQualityGeometry, Cquad4RectangleAspect) {
    Model m;
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 10.0, 0.0, 0.0);
    add_node(m, 3, 10.0, 1.0, 0.0);
    add_node(m, 4, 0.0,  1.0, 0.0);
    add_mat(m, 1); add_pshell(m, 1, 1);
    add_cquad4(m, 1, 1, 1, 2, 3, 4);

    auto q = compute_element_quality(m.elements[0], m);
    ASSERT_TRUE(q.has_value());
    EXPECT_NEAR(q->aspect_ratio, 10.0, 1e-10);
    EXPECT_NEAR(q->warp_angle, 0.0, 1e-10);
    EXPECT_NEAR(q->min_interior_angle, 90.0, 1e-10);
}

TEST(ElementQualityGeometry, Cquad4TwistedWarpAngle) {
    // Lift node 3 in Z to create a non-planar quad
    Model m;
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 1.0, 0.0, 0.0);
    add_node(m, 3, 1.0, 1.0, 0.5);  // lifted
    add_node(m, 4, 0.0, 1.0, 0.0);
    add_mat(m, 1); add_pshell(m, 1, 1);
    add_cquad4(m, 1, 1, 1, 2, 3, 4);

    auto q = compute_element_quality(m.elements[0], m);
    ASSERT_TRUE(q.has_value());
    EXPECT_GT(q->warp_angle, 0.0);
    EXPECT_LT(q->warp_angle, 90.0);
}

TEST(ElementQualityGeometry, Cquad4TaperRatioSymmetric) {
    // Rhombus: equal triangular areas → taper = 0
    Model m;
    add_node(m, 1, 0.0,  0.0, 0.0);
    add_node(m, 2, 2.0,  1.0, 0.0);
    add_node(m, 3, 0.0,  2.0, 0.0);
    add_node(m, 4, -2.0, 1.0, 0.0);
    add_mat(m, 1); add_pshell(m, 1, 1);
    add_cquad4(m, 1, 1, 1, 2, 3, 4);

    auto q = compute_element_quality(m.elements[0], m);
    ASSERT_TRUE(q.has_value());
    EXPECT_NEAR(q->taper_ratio, 0.0, 1e-10);
}

TEST(ElementQualityGeometry, Cquad4TaperRatioSkewed) {
    // Trapezoidal quad: unequal areas → taper > 0
    Model m;
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 4.0, 0.0, 0.0);
    add_node(m, 3, 3.0, 1.0, 0.0);
    add_node(m, 4, 1.0, 1.0, 0.0);
    add_mat(m, 1); add_pshell(m, 1, 1);
    add_cquad4(m, 1, 1, 1, 2, 3, 4);

    auto q = compute_element_quality(m.elements[0], m);
    ASSERT_TRUE(q.has_value());
    EXPECT_GT(q->taper_ratio, 0.0);
    EXPECT_LE(q->taper_ratio, 1.0);
}

// ── ElementQualityGeometry: CTRIA3 ───────────────────────────────────────────

TEST(ElementQualityGeometry, Ctria3EquilateralAngles) {
    // Equilateral triangle: all angles = 60°
    Model m;
    double h = std::sqrt(3.0) / 2.0;
    add_node(m, 1, 0.0,   0.0, 0.0);
    add_node(m, 2, 1.0,   0.0, 0.0);
    add_node(m, 3, 0.5,   h,   0.0);
    add_mat(m, 1); add_pshell(m, 1, 1);
    add_ctria3(m, 1, 1, 1, 2, 3);

    auto q = compute_element_quality(m.elements[0], m);
    ASSERT_TRUE(q.has_value());
    EXPECT_NEAR(q->aspect_ratio, 1.0, 1e-10);
    EXPECT_NEAR(q->min_interior_angle, 60.0, 1e-8);
    EXPECT_NEAR(q->max_interior_angle, 60.0, 1e-8);
    EXPECT_NEAR(q->warp_angle, 0.0, 1e-12);
    EXPECT_NEAR(q->taper_ratio, 0.0, 1e-12);
}

TEST(ElementQualityGeometry, Ctria3RightTriangleAngles) {
    // 3-4-5 right triangle: angles = 90°, ~53.13°, ~36.87°
    Model m;
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 3.0, 0.0, 0.0);
    add_node(m, 3, 0.0, 4.0, 0.0);
    add_mat(m, 1); add_pshell(m, 1, 1);
    add_ctria3(m, 1, 1, 1, 2, 3);

    auto q = compute_element_quality(m.elements[0], m);
    ASSERT_TRUE(q.has_value());
    EXPECT_NEAR(q->max_interior_angle, 90.0, 1e-8);
    EXPECT_NEAR(q->min_interior_angle,
                std::atan(3.0 / 4.0) * (180.0 / std::numbers::pi), 1e-8);
    EXPECT_NEAR(q->aspect_ratio, 5.0 / 3.0, 1e-10);
}

TEST(ElementQualityGeometry, Ctria3SkinnyShouldHaveSmallMinAngle) {
    // Very thin isoceles triangle: base=100, height=0.5 → apex angle ≈ 0.57°,
    // base angles ≈ 89.7°. Aspect ratio = max_edge/min_edge = 100/50.0025 ≈ 2.0.
    Model m;
    add_node(m, 1, 0.0, 0.0,   0.0);
    add_node(m, 2, 100.0, 0.0, 0.0);
    add_node(m, 3, 50.0,  0.5, 0.0);
    add_mat(m, 1); add_pshell(m, 1, 1);
    add_ctria3(m, 1, 1, 1, 2, 3);

    auto q = compute_element_quality(m.elements[0], m);
    ASSERT_TRUE(q.has_value());
    EXPECT_LT(q->min_interior_angle, 5.0);
    // Aspect ratio is max_edge/min_edge: edges are ~50, ~50, 100 → ≈ 2.0
    EXPECT_NEAR(q->aspect_ratio, 2.0, 0.01);
}

// ── ElementQualityGeometry: CHEXA8 ───────────────────────────────────────────

TEST(ElementQualityGeometry, Chexa8UnitCube) {
    Model m;
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 1.0, 0.0, 0.0);
    add_node(m, 3, 1.0, 1.0, 0.0);
    add_node(m, 4, 0.0, 1.0, 0.0);
    add_node(m, 5, 0.0, 0.0, 1.0);
    add_node(m, 6, 1.0, 0.0, 1.0);
    add_node(m, 7, 1.0, 1.0, 1.0);
    add_node(m, 8, 0.0, 1.0, 1.0);
    add_mat(m, 1); add_psolid(m, 1, 1);
    add_chexa8(m, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8);

    auto q = compute_element_quality(m.elements[0], m);
    ASSERT_TRUE(q.has_value());
    EXPECT_NEAR(q->aspect_ratio, 1.0, 1e-10);
    // All Gauss-point Jacobians should be equal and positive for a unit cube → ratio = 1.0
    EXPECT_NEAR(q->min_jacobian_ratio, 1.0, 1e-10);
}

TEST(ElementQualityGeometry, Chexa8ElongatedAspect) {
    Model m;
    add_node(m, 1, 0.0,  0.0, 0.0);
    add_node(m, 2, 10.0, 0.0, 0.0);
    add_node(m, 3, 10.0, 1.0, 0.0);
    add_node(m, 4, 0.0,  1.0, 0.0);
    add_node(m, 5, 0.0,  0.0, 1.0);
    add_node(m, 6, 10.0, 0.0, 1.0);
    add_node(m, 7, 10.0, 1.0, 1.0);
    add_node(m, 8, 0.0,  1.0, 1.0);
    add_mat(m, 1); add_psolid(m, 1, 1);
    add_chexa8(m, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8);

    auto q = compute_element_quality(m.elements[0], m);
    ASSERT_TRUE(q.has_value());
    EXPECT_NEAR(q->aspect_ratio, 10.0, 1e-10);
}

// ── ElementQualityGeometry: CTETRA4 ──────────────────────────────────────────

TEST(ElementQualityGeometry, Ctetra4StandardOrientationPositiveJacobian) {
    Model m;
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 1.0, 0.0, 0.0);
    add_node(m, 3, 0.0, 1.0, 0.0);
    add_node(m, 4, 0.0, 0.0, 1.0);
    add_mat(m, 1); add_psolid(m, 1, 1);
    add_ctetra4(m, 1, 1, 1, 2, 3, 4);

    auto q = compute_element_quality(m.elements[0], m);
    ASSERT_TRUE(q.has_value());
    EXPECT_GT(q->min_jacobian_ratio, 0.0);  // positive = valid orientation
}

TEST(ElementQualityGeometry, Ctetra4InvertedNegativeJacobian) {
    // Swap N3 and N4 to invert the element
    Model m;
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 1.0, 0.0, 0.0);
    add_node(m, 3, 0.0, 0.0, 1.0);  // swapped
    add_node(m, 4, 0.0, 1.0, 0.0);  // swapped
    add_mat(m, 1); add_psolid(m, 1, 1);
    add_ctetra4(m, 1, 1, 1, 2, 3, 4);

    auto q = compute_element_quality(m.elements[0], m);
    ASSERT_TRUE(q.has_value());
    EXPECT_LT(q->min_jacobian_ratio, 0.0);  // negative = inverted
}

// ── TopologyChecks ────────────────────────────────────────────────────────────

TEST(TopologyChecks, SingleCquad4HasFourFreeEdges) {
    Model m;
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 1.0, 0.0, 0.0);
    add_node(m, 3, 1.0, 1.0, 0.0);
    add_node(m, 4, 0.0, 1.0, 0.0);
    add_mat(m, 1); add_pshell(m, 1, 1);
    add_cquad4(m, 1, 1, 1, 2, 3, 4);

    auto topo = check_topology(m, 1e-8);
    ASSERT_EQ(topo.free_edge_elements.size(), 1u);
    EXPECT_EQ(topo.free_edge_elements[0].value, 1);
}

TEST(TopologyChecks, TwoAdjacentCquad4sShareOneEdge) {
    // Two quads sharing edge 2-3
    Model m;
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 1.0, 0.0, 0.0);
    add_node(m, 3, 1.0, 1.0, 0.0);
    add_node(m, 4, 0.0, 1.0, 0.0);
    add_node(m, 5, 2.0, 0.0, 0.0);
    add_node(m, 6, 2.0, 1.0, 0.0);
    add_mat(m, 1); add_pshell(m, 1, 1);
    add_cquad4(m, 1, 1, 1, 2, 3, 4);
    add_cquad4(m, 2, 1, 2, 5, 6, 3);  // shares edge 2-3

    auto topo = check_topology(m, 1e-8);
    // Both elements have at least one free edge (the boundary), but the shared edge is not free
    EXPECT_EQ(topo.free_edge_elements.size(), 2u);
    EXPECT_TRUE(topo.duplicate_node_pairs.empty());
}

TEST(TopologyChecks, OrphanedNode) {
    Model m;
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 1.0, 0.0, 0.0);
    add_node(m, 3, 1.0, 1.0, 0.0);
    add_node(m, 4, 0.0, 1.0, 0.0);
    add_node(m, 99, 5.0, 5.0, 5.0);  // orphaned
    add_mat(m, 1); add_pshell(m, 1, 1);
    add_cquad4(m, 1, 1, 1, 2, 3, 4);

    auto topo = check_topology(m, 1e-8);
    ASSERT_EQ(topo.orphaned_nodes.size(), 1u);
    EXPECT_EQ(topo.orphaned_nodes[0].value, 99);
}

TEST(TopologyChecks, OrphanedProperty) {
    Model m;
    add_mat(m, 1);
    add_pshell(m, 1, 1);
    add_pshell(m, 2, 1);  // orphaned — not used by any element
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 1.0, 0.0, 0.0);
    add_node(m, 3, 1.0, 1.0, 0.0);
    add_node(m, 4, 0.0, 1.0, 0.0);
    add_cquad4(m, 1, 1, 1, 2, 3, 4);

    auto topo = check_topology(m, 1e-8);
    ASSERT_EQ(topo.orphaned_properties.size(), 1u);
    EXPECT_EQ(topo.orphaned_properties[0].value, 2);
}

TEST(TopologyChecks, OrphanedMaterial) {
    Model m;
    add_mat(m, 1);
    add_mat(m, 2);  // orphaned — not referenced by any property
    add_pshell(m, 1, 1);
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 1.0, 0.0, 0.0);
    add_node(m, 3, 1.0, 1.0, 0.0);
    add_node(m, 4, 0.0, 1.0, 0.0);
    add_cquad4(m, 1, 1, 1, 2, 3, 4);

    auto topo = check_topology(m, 1e-8);
    ASSERT_EQ(topo.orphaned_materials.size(), 1u);
    EXPECT_EQ(topo.orphaned_materials[0].value, 2);
}

TEST(TopologyChecks, DuplicateNodesExact) {
    Model m;
    add_mat(m, 1); add_pshell(m, 1, 1);
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 1.0, 0.0, 0.0);
    add_node(m, 3, 1.0, 1.0, 0.0);
    add_node(m, 4, 0.0, 1.0, 0.0);
    add_node(m, 5, 0.0, 0.0, 0.0);  // exact duplicate of node 1
    add_cquad4(m, 1, 1, 1, 2, 3, 4);

    auto topo = check_topology(m, 1e-8);
    ASSERT_EQ(topo.duplicate_node_pairs.size(), 1u);
    auto [a, b] = topo.duplicate_node_pairs[0];
    EXPECT_TRUE((a.value == 1 && b.value == 5) || (a.value == 5 && b.value == 1));
}

TEST(TopologyChecks, DuplicateNodesWithinTolerance) {
    Model m;
    add_mat(m, 1); add_pshell(m, 1, 1);
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 1.0, 0.0, 0.0);
    add_node(m, 3, 1.0, 1.0, 0.0);
    add_node(m, 4, 0.0, 1.0, 0.0);
    // Node 5 is within 1e-9 of node 1 — within default tol=1e-8
    add_node(m, 5, 1e-10, 0.0, 0.0);
    add_cquad4(m, 1, 1, 1, 2, 3, 4);

    auto topo = check_topology(m, 1e-8);
    EXPECT_EQ(topo.duplicate_node_pairs.size(), 1u);
}

TEST(TopologyChecks, DuplicateNodesBeyondTolerance) {
    Model m;
    add_mat(m, 1); add_pshell(m, 1, 1);
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 1.0, 0.0, 0.0);
    add_node(m, 3, 1.0, 1.0, 0.0);
    add_node(m, 4, 0.0, 1.0, 0.0);
    add_node(m, 5, 1e-7, 0.0, 0.0);  // beyond 1e-8 tolerance
    add_cquad4(m, 1, 1, 1, 2, 3, 4);

    auto topo = check_topology(m, 1e-8);
    EXPECT_TRUE(topo.duplicate_node_pairs.empty());
}

TEST(TopologyChecks, ZeroLengthCbushNotFlaggedAsDuplicate) {
    // CBUSH with coincident nodes — must not appear in duplicate_node_pairs
    Model m;
    add_mat(m, 1); add_pshell(m, 1, 1);
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 1.0, 0.0, 0.0);
    add_node(m, 3, 1.0, 1.0, 0.0);
    add_node(m, 4, 0.0, 1.0, 0.0);
    add_node(m, 5, 0.0, 0.0, 0.0);  // coincident with node 1
    add_cquad4(m, 1, 1, 1, 2, 3, 4);

    // PBush property
    PBush pb; pb.pid = PropertyId{10}; pb.k.fill(1e6);
    m.properties[PropertyId{10}] = pb;
    add_cbush(m, 10, 10, 1, 5);  // connects coincident nodes

    auto topo = check_topology(m, 1e-8);
    EXPECT_TRUE(topo.duplicate_node_pairs.empty());
}

TEST(TopologyChecks, ZeroLengthCelas2NotFlaggedAsDuplicate) {
    // CELAS2 with coincident nodes — must not appear in duplicate_node_pairs
    Model m;
    add_mat(m, 1); add_pshell(m, 1, 1);
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 1.0, 0.0, 0.0);
    add_node(m, 3, 1.0, 1.0, 0.0);
    add_node(m, 4, 0.0, 1.0, 0.0);
    add_node(m, 5, 0.0, 0.0, 0.0);  // coincident with node 1
    add_cquad4(m, 1, 1, 1, 2, 3, 4);
    add_celas2(m, 10, 1, 5);  // CELAS2 connecting coincident nodes

    auto topo = check_topology(m, 1e-8);
    EXPECT_TRUE(topo.duplicate_node_pairs.empty());
}

TEST(TopologyChecks, DuplicateElements) {
    Model m;
    add_mat(m, 1); add_pshell(m, 1, 1);
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 1.0, 0.0, 0.0);
    add_node(m, 3, 1.0, 1.0, 0.0);
    add_node(m, 4, 0.0, 1.0, 0.0);
    add_cquad4(m, 1, 1, 1, 2, 3, 4);
    add_cquad4(m, 2, 1, 1, 2, 3, 4);  // same connectivity → duplicate

    auto topo = check_topology(m, 1e-8);
    ASSERT_EQ(topo.duplicate_element_pairs.size(), 1u);
    auto [e1, e2] = topo.duplicate_element_pairs[0];
    EXPECT_TRUE((e1.value == 1 && e2.value == 2) || (e1.value == 2 && e2.value == 1));
}

// ── PhysicalSanityChecks ──────────────────────────────────────────────────────

TEST(PhysicalSanityChecks, ZeroYoungsModulus) {
    Model m;
    add_mat(m, 1, 0.0, 0.3);  // E = 0

    auto r = check_physical(m);
    ASSERT_EQ(r.bad_E.size(), 1u);
    EXPECT_EQ(r.bad_E[0].value, 1);
}

TEST(PhysicalSanityChecks, NegativeYoungsModulus) {
    Model m;
    add_mat(m, 1, -1e9, 0.3);  // E < 0

    auto r = check_physical(m);
    ASSERT_EQ(r.bad_E.size(), 1u);
}

TEST(PhysicalSanityChecks, PoissonRatioAtHalf) {
    Model m;
    add_mat(m, 1, 2e11, 0.5);  // nu = 0.5 → invalid

    auto r = check_physical(m);
    ASSERT_EQ(r.bad_nu.size(), 1u);
    EXPECT_EQ(r.bad_nu[0].value, 1);
}

TEST(PhysicalSanityChecks, PoissonRatioAboveHalf) {
    Model m;
    add_mat(m, 1, 2e11, 0.6);  // nu = 0.6 → invalid

    auto r = check_physical(m);
    ASSERT_EQ(r.bad_nu.size(), 1u);
}

TEST(PhysicalSanityChecks, NegativeDensity) {
    Model m;
    add_mat(m, 1, 2e11, 0.3, -100.0);  // rho < 0

    auto r = check_physical(m);
    ASSERT_EQ(r.bad_rho.size(), 1u);
    EXPECT_EQ(r.bad_rho[0].value, 1);
}

TEST(PhysicalSanityChecks, ZeroThickness) {
    Model m;
    add_mat(m, 1);
    add_pshell(m, 1, 1, 0.0);  // t = 0

    auto r = check_physical(m);
    ASSERT_EQ(r.bad_thickness.size(), 1u);
    EXPECT_EQ(r.bad_thickness[0].value, 1);
}

TEST(PhysicalSanityChecks, NegativeThickness) {
    Model m;
    add_mat(m, 1);
    add_pshell(m, 1, 1, -0.01);  // t < 0

    auto r = check_physical(m);
    ASSERT_EQ(r.bad_thickness.size(), 1u);
}

TEST(PhysicalSanityChecks, ValidMaterialAndProperty) {
    Model m;
    add_mat(m, 1, 2e11, 0.3, 7800.0);  // all valid
    add_pshell(m, 1, 1, 0.01);          // valid thickness

    auto r = check_physical(m);
    EXPECT_TRUE(r.bad_E.empty());
    EXPECT_TRUE(r.bad_nu.empty());
    EXPECT_TRUE(r.bad_rho.empty());
    EXPECT_TRUE(r.bad_thickness.empty());
}

TEST(PhysicalSanityChecks, MissingLoadSet) {
    Model m;
    m.analysis.sol = SolutionType::LinearStatic;
    SubCase sc;
    sc.id = 1;
    sc.load_set = LoadSetId{99};  // no loads with this SID
    sc.spc_set  = SpcSetId{1};
    m.analysis.subcases.push_back(sc);

    add_node(m, 1, 0.0, 0.0, 0.0);
    add_spc_all(m, 1, 1);

    auto r = check_physical(m);
    ASSERT_EQ(r.subcases_no_load.size(), 1u);
    EXPECT_EQ(r.subcases_no_load[0], 1);
}

TEST(PhysicalSanityChecks, UnconstrainedModel) {
    Model m;
    m.analysis.sol = SolutionType::LinearStatic;
    SubCase sc;
    sc.id       = 1;
    sc.spc_set  = SpcSetId{0};  // no SPC set
    sc.load_set = LoadSetId{0};
    m.analysis.subcases.push_back(sc);

    auto r = check_physical(m);
    ASSERT_EQ(r.subcases_no_constraint.size(), 1u);
    EXPECT_EQ(r.subcases_no_constraint[0], 1);
}

// ── ModeDispatchStrict ────────────────────────────────────────────────────────

TEST(ModeDispatchStrict, AspectRatioViolationThrows) {
    Model m = make_minimal_sol101_model();
    // Add a 20:1 rectangle — will exceed default ASPECT threshold of 20
    add_node(m, 10, 0.0,  0.0, 0.0);
    add_node(m, 11, 21.0, 0.0, 0.0);
    add_node(m, 12, 21.0, 1.0, 0.0);
    add_node(m, 13, 0.0,  1.0, 0.0);
    add_cquad4(m, 2, 1, 10, 11, 12, 13);

    QualityThresholds t;  // defaults: strict, max_aspect = 20
    EXPECT_THROW(run_quality_checks(m, t), SolverError);
}

TEST(ModeDispatchStrict, OrphanedNodeThrows) {
    Model m = make_minimal_sol101_model();
    add_node(m, 99, 5.0, 5.0, 5.0);  // orphaned

    QualityThresholds t;
    EXPECT_THROW(run_quality_checks(m, t), SolverError);
}

TEST(ModeDispatchStrict, MissingLoadSetIsWarningNotThrow) {
    // SOL 101, SPC present, but load_set points to empty set → warning only.
    // Use a solid element to avoid the free-edge check (which applies to shells only).
    Model m;
    m.analysis.sol = SolutionType::LinearStatic;
    SubCase sc;
    sc.id       = 1;
    sc.load_set = LoadSetId{99};  // no loads with SID=99
    sc.spc_set  = SpcSetId{1};
    m.analysis.subcases.push_back(sc);

    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 1.0, 0.0, 0.0);
    add_node(m, 3, 1.0, 1.0, 0.0);
    add_node(m, 4, 0.0, 1.0, 0.0);
    add_node(m, 5, 0.0, 0.0, 1.0);
    add_node(m, 6, 1.0, 0.0, 1.0);
    add_node(m, 7, 1.0, 1.0, 1.0);
    add_node(m, 8, 0.0, 1.0, 1.0);
    add_mat(m, 1);
    add_psolid(m, 1, 1);
    add_chexa8(m, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8);
    add_spc_all(m, 1, 1);

    QualityThresholds t;
    // No throw — missing load is a warning in both modes
    EXPECT_NO_THROW(run_quality_checks(m, t));
}

TEST(ModeDispatchStrict, MissingConstraintSol101Throws) {
    Model m;
    m.analysis.sol = SolutionType::LinearStatic;
    SubCase sc;
    sc.id       = 1;
    sc.spc_set  = SpcSetId{0};
    sc.load_set = LoadSetId{0};
    m.analysis.subcases.push_back(sc);

    QualityThresholds t;
    EXPECT_THROW(run_quality_checks(m, t), SolverError);
}

TEST(ModeDispatchStrict, InvertedElementAlwaysThrows) {
    Model m = make_minimal_sol101_model();
    // Add inverted CTETRA4
    add_node(m, 10, 0.0, 0.0, 0.0);
    add_node(m, 11, 1.0, 0.0, 0.0);
    add_node(m, 12, 0.0, 0.0, 1.0);  // swapped → inversion
    add_node(m, 13, 0.0, 1.0, 0.0);
    add_psolid(m, 2, 1);
    add_ctetra4(m, 2, 2, 10, 11, 12, 13);

    QualityThresholds t;
    EXPECT_THROW(run_quality_checks(m, t), SolverError);
}

// ── ModeDispatchLenient ───────────────────────────────────────────────────────

TEST(ModeDispatchLenient, AspectRatioViolationWarnsNotThrows) {
    Model m = make_minimal_sol101_model();
    // 20:1 rectangle exceeds default aspect threshold
    add_node(m, 10, 0.0,  0.0, 0.0);
    add_node(m, 11, 21.0, 0.0, 0.0);
    add_node(m, 12, 21.0, 1.0, 0.0);
    add_node(m, 13, 0.0,  1.0, 0.0);
    add_cquad4(m, 2, 1, 10, 11, 12, 13);

    QualityThresholds t;
    t.mode = CheckMode::Lenient;
    EXPECT_NO_THROW(run_quality_checks(m, t));
}

TEST(ModeDispatchLenient, BadNuAlwaysThrowsEvenInLenient) {
    Model m;
    m.analysis.sol = SolutionType::LinearStatic;
    SubCase sc;
    sc.id = 1; sc.spc_set = SpcSetId{1}; sc.load_set = LoadSetId{0};
    m.analysis.subcases.push_back(sc);
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_spc_all(m, 1, 1);
    add_mat(m, 1, 2e11, 0.5);  // nu = 0.5 → always fatal

    QualityThresholds t;
    t.mode = CheckMode::Lenient;
    EXPECT_THROW(run_quality_checks(m, t), SolverError);
}

TEST(ModeDispatchLenient, BadEAlwaysThrowsEvenInLenient) {
    Model m;
    m.analysis.sol = SolutionType::LinearStatic;
    SubCase sc;
    sc.id = 1; sc.spc_set = SpcSetId{1}; sc.load_set = LoadSetId{0};
    m.analysis.subcases.push_back(sc);
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_spc_all(m, 1, 1);
    add_mat(m, 1, 0.0, 0.3);  // E = 0 → always fatal

    QualityThresholds t;
    t.mode = CheckMode::Lenient;
    EXPECT_THROW(run_quality_checks(m, t), SolverError);
}

TEST(ModeDispatchLenient, MissingConstraintSol101ThrowsEvenInLenient) {
    Model m;
    m.analysis.sol = SolutionType::LinearStatic;
    SubCase sc;
    sc.id = 1; sc.spc_set = SpcSetId{0}; sc.load_set = LoadSetId{0};
    m.analysis.subcases.push_back(sc);

    QualityThresholds t;
    t.mode = CheckMode::Lenient;
    EXPECT_THROW(run_quality_checks(m, t), SolverError);
}

TEST(ModeDispatchLenient, MissingConstraintSol103WarnsNotThrows) {
    Model m;
    m.analysis.sol = SolutionType::Modal;
    SubCase sc;
    sc.id = 1; sc.spc_set = SpcSetId{0}; sc.load_set = LoadSetId{0};
    m.analysis.subcases.push_back(sc);

    QualityThresholds t;
    t.mode = CheckMode::Lenient;
    EXPECT_NO_THROW(run_quality_checks(m, t));
}

TEST(ModeDispatchLenient, DuplicateNodesMergedWithAutoMerge) {
    Model m;
    m.analysis.sol = SolutionType::LinearStatic;
    SubCase sc; sc.id = 1; sc.spc_set = SpcSetId{1}; sc.load_set = LoadSetId{0};
    m.analysis.subcases.push_back(sc);

    add_mat(m, 1); add_pshell(m, 1, 1);
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 1.0, 0.0, 0.0);
    add_node(m, 3, 1.0, 1.0, 0.0);
    add_node(m, 4, 0.0, 1.0, 0.0);
    add_node(m, 5, 0.0, 0.0, 0.0);  // exact duplicate of node 1
    add_cquad4(m, 1, 1, 5, 2, 3, 4);  // references node 5
    add_spc_all(m, 1, 1);

    QualityThresholds t;
    t.mode = CheckMode::Lenient;
    t.auto_merge_nodes = true;

    EXPECT_NO_THROW(run_quality_checks(m, t));

    // Node 5 should be merged into node 1 (lower ID) and erased
    EXPECT_EQ(m.nodes.count(NodeId{5}), 0u);
    EXPECT_EQ(m.nodes.count(NodeId{1}), 1u);
    // Element's first node should now be node 1
    EXPECT_EQ(m.elements[0].nodes[0].value, 1);
}

TEST(ModeDispatchLenient, DuplicateNodesNotMergedWithAutoMergeDisabled) {
    Model m;
    m.analysis.sol = SolutionType::LinearStatic;
    SubCase sc; sc.id = 1; sc.spc_set = SpcSetId{1}; sc.load_set = LoadSetId{0};
    m.analysis.subcases.push_back(sc);

    add_mat(m, 1); add_pshell(m, 1, 1);
    add_node(m, 1, 0.0, 0.0, 0.0);
    add_node(m, 2, 1.0, 0.0, 0.0);
    add_node(m, 3, 1.0, 1.0, 0.0);
    add_node(m, 4, 0.0, 1.0, 0.0);
    add_node(m, 5, 0.0, 0.0, 0.0);  // duplicate of node 1
    add_cquad4(m, 1, 1, 5, 2, 3, 4);
    add_spc_all(m, 1, 1);

    QualityThresholds t;
    t.mode = CheckMode::Lenient;
    t.auto_merge_nodes = false;  // AUTOMERGE=0

    EXPECT_NO_THROW(run_quality_checks(m, t));

    // Node 5 must still be present (no merge occurred)
    EXPECT_EQ(m.nodes.count(NodeId{5}), 1u);
    EXPECT_EQ(m.elements[0].nodes[0].value, 5);  // unchanged
}

TEST(ModeDispatchLenient, InvertedElementAlwaysThrowsEvenInLenient) {
    Model m = make_minimal_sol101_model();
    add_node(m, 10, 0.0, 0.0, 0.0);
    add_node(m, 11, 1.0, 0.0, 0.0);
    add_node(m, 12, 0.0, 0.0, 1.0);  // inverted
    add_node(m, 13, 0.0, 1.0, 0.0);
    add_psolid(m, 2, 1);
    add_ctetra4(m, 2, 2, 10, 11, 12, 13);

    QualityThresholds t;
    t.mode = CheckMode::Lenient;
    EXPECT_THROW(run_quality_checks(m, t), SolverError);
}
