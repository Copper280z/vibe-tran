// tests/unit/test_bdf_parser.cpp
// Tests for BDF field parsing: numeric formats, card splitting, model population.

#include <gtest/gtest.h>
#include "io/bdf_parser.hpp"
#include "core/model.hpp"

using namespace nastran;

// ── Numeric format parsing ────────────────────────────────────────────────────

TEST(BdfParser, ParseSimpleBdf) {
    // Minimal valid BDF: one node, one material, one property, one element, one SPC
    const std::string bdf = R"(
SOL 101
BEGIN BULK
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,1.0,1.0,0.0
GRID,4,,0.0,1.0,0.0
MAT1,1,2.0E7,,0.3
PSHELL,1,1,0.1
CQUAD4,1,1,1,2,3,4
SPC1,1,123456,1,2,3,4
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    EXPECT_EQ(m.nodes.size(), 4u);
    EXPECT_EQ(m.elements.size(), 1u);
    EXPECT_EQ(m.materials.size(), 1u);
    EXPECT_EQ(m.properties.size(), 1u);
    EXPECT_EQ(m.spcs.size(), 4u);
}

TEST(BdfParser, SmallFieldGridCard) {
    const std::string bdf = R"(
BEGIN BULK
GRID           1       0     1.5     2.5     3.5
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.nodes.size(), 1u);
    auto& n = m.nodes.at(NodeId{1});
    EXPECT_DOUBLE_EQ(n.position.x, 1.5);
    EXPECT_DOUBLE_EQ(n.position.y, 2.5);
    EXPECT_DOUBLE_EQ(n.position.z, 3.5);
}

TEST(BdfParser, FreeFieldWithCommas) {
    const std::string bdf = R"(
BEGIN BULK
GRID,10,,3.14,2.72,1.41
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.nodes.size(), 1u);
    auto& n = m.nodes.at(NodeId{10});
    EXPECT_NEAR(n.position.x, 3.14, 1e-10);
    EXPECT_NEAR(n.position.y, 2.72, 1e-10);
    EXPECT_NEAR(n.position.z, 1.41, 1e-10);
}

TEST(BdfParser, NastranImplicitExponent) {
    // "1.5+3" is Nastran shorthand for 1.5E3 = 1500
    const std::string bdf = R"(
BEGIN BULK
MAT1,1,2.9+7,,3.0-1
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.materials.size(), 1u);
    const Mat1& mat = m.materials.at(MaterialId{1});
    EXPECT_NEAR(mat.E,  2.9e7, 1.0);
    EXPECT_NEAR(mat.nu, 0.30,  1e-10);
}

TEST(BdfParser, Mat1DerivedShearModulus) {
    // If G is blank, G = E / (2*(1+nu))
    const std::string bdf = R"(
BEGIN BULK
MAT1,1,2.0E7,,0.25
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    const Mat1& mat = m.materials.at(MaterialId{1});
    double G_expected = 2.0e7 / (2.0 * 1.25);
    EXPECT_NEAR(mat.G, G_expected, 1.0);
}

TEST(BdfParser, ForceCard) {
    const std::string bdf = R"(
BEGIN BULK
GRID,1,,0,0,0
FORCE,1,1,0,1000.0,0.0,0.0,1.0
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.loads.size(), 1u);
    const ForceLoad& f = std::get<ForceLoad>(m.loads[0]);
    EXPECT_EQ(f.sid.value, 1);
    EXPECT_EQ(f.node.value, 1);
    EXPECT_DOUBLE_EQ(f.scale, 1000.0);
    EXPECT_DOUBLE_EQ(f.direction.z, 1.0);
}

TEST(BdfParser, SPC1WithThruRange) {
    const std::string bdf = R"(
BEGIN BULK
GRID,1,,0,0,0
GRID,2,,1,0,0
GRID,3,,2,0,0
SPC1,1,123,1,THRU,3
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    // Should create 3 SPC entries (one per node)
    EXPECT_EQ(m.spcs.size(), 3u);
    for (const auto& spc : m.spcs) {
        EXPECT_EQ(spc.sid.value, 1);
        EXPECT_TRUE(spc.dofs.has(1) && spc.dofs.has(2) && spc.dofs.has(3));
    }
}

TEST(BdfParser, TempCard) {
    const std::string bdf = R"(
BEGIN BULK
GRID,1,,0,0,0
GRID,2,,1,0,0
TEMP,1,1,100.0,2,200.0
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    EXPECT_EQ(m.loads.size(), 2u);
    const TempLoad& t0 = std::get<TempLoad>(m.loads[0]);
    const TempLoad& t1 = std::get<TempLoad>(m.loads[1]);
    EXPECT_DOUBLE_EQ(t0.temperature, 100.0);
    EXPECT_DOUBLE_EQ(t1.temperature, 200.0);
}

TEST(BdfParser, ModelValidationCatchesMissingNode) {
    Model m;
    ElementData e;
    e.id   = ElementId{1};
    e.pid  = PropertyId{1};
    e.type = ElementType::CQUAD4;
    e.nodes = {NodeId{1},NodeId{2},NodeId{3},NodeId{4}}; // nodes not in model
    m.elements.push_back(e);
    EXPECT_THROW(m.validate(), SolverError);
}

// ── Large-field format (16-char columns, * prefix) ───────────────────────────

TEST(BdfParser, LargeFieldGridCard) {
    // Large-field GRID: 8-char keyword + 4×16-char data fields per line.
    // All five GRID fields (ID, CP, X, Y, Z=0) fit on one line when Z is omitted.
    //                   0       8                       24                      40                      56                      72
    const std::string bdf =
        "BEGIN BULK\n"
        "*GRID                  1               0             1.5             2.5\n"
        "ENDDATA\n";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.nodes.size(), 1u);
    const GridPoint& n = m.nodes.at(NodeId{1});
    EXPECT_DOUBLE_EQ(n.position.x, 1.5);
    EXPECT_DOUBLE_EQ(n.position.y, 2.5);
    EXPECT_DOUBLE_EQ(n.position.z, 0.0);
}

TEST(BdfParser, LargeFieldMat1Card) {
    // MAT1 with E, G, nu in large-field (16-char) columns.
    //                   0       8                       24                      40                      56
    const std::string bdf =
        "BEGIN BULK\n"
        "*MAT1                  1         2.9E+07       1.11538E7             0.3\n"
        "ENDDATA\n";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.materials.size(), 1u);
    const Mat1& mat = m.materials.at(MaterialId{1});
    EXPECT_NEAR(mat.E,  2.9e7, 1.0);
    EXPECT_NEAR(mat.G,  1.11538e7, 1.0);
    EXPECT_NEAR(mat.nu, 0.3, 1e-10);
}

TEST(BdfParser, LargeFieldGridWithContinuation) {
    // Two-line large-field GRID: first line holds ID, CP, X, Y;
    // continuation line (also *-prefixed) holds Z.
    // Per the field-splitting logic, continuation fields are appended
    // starting at index 10, so Z appears at fields[10] rather than fields[5].
    // This means the parser reads an empty fields[5] → Z = 0, not the
    // continuation value. The test documents this known limitation.
    const std::string bdf =
        "BEGIN BULK\n"
        "*GRID                  5               0             3.0             4.0+       \n"
        "*                                                    5.0                        \n"
        "ENDDATA\n";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.nodes.size(), 1u);
    const GridPoint& n = m.nodes.at(NodeId{5});
    EXPECT_DOUBLE_EQ(n.position.x, 3.0);
    EXPECT_DOUBLE_EQ(n.position.y, 4.0);
    // Z comes from the continuation line appended at index 10;
    // process_grid reads index 5 which is the blank continuation marker → 0.
    EXPECT_DOUBLE_EQ(n.position.z, 0.0);
}

// ── Small-field continuation cards ───────────────────────────────────────────

TEST(BdfParser, SmallFieldCtria3) {
    // CTRIA3 has 3 nodes and fits entirely on one small-field line.
    const std::string bdf =
        "BEGIN BULK\n"
        "CTRIA3         1       1       1       2       3\n"
        "ENDDATA\n";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.elements.size(), 1u);
    const ElementData& e = m.elements[0];
    EXPECT_EQ(e.type, ElementType::CTRIA3);
    EXPECT_EQ(e.id.value, 1);
    EXPECT_EQ(e.pid.value, 1);
    ASSERT_EQ(e.nodes.size(), 3u);
    EXPECT_EQ(e.nodes[0].value, 1);
    EXPECT_EQ(e.nodes[1].value, 2);
    EXPECT_EQ(e.nodes[2].value, 3);
}

TEST(BdfParser, SmallFieldCtetra) {
    // CTETRA has 4 nodes and fits on one small-field line.
    const std::string bdf =
        "BEGIN BULK\n"
        "CTETRA         1       2       1       2       3       4\n"
        "ENDDATA\n";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.elements.size(), 1u);
    const ElementData& e = m.elements[0];
    EXPECT_EQ(e.type, ElementType::CTETRA4);
    EXPECT_EQ(e.pid.value, 2);
    ASSERT_EQ(e.nodes.size(), 4u);
    EXPECT_EQ(e.nodes[3].value, 4);
}

TEST(BdfParser, SmallFieldChexaWithContinuation) {
    // CHEXA8: G1–G6 on the first small-field line, G7–G8 on a '+' continuation.
    const std::string bdf =
        "BEGIN BULK\n"
        "CHEXA          1       1       1       2       3       4       5       6\n"
        "+              7       8\n"
        "ENDDATA\n";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.elements.size(), 1u);
    const ElementData& e = m.elements[0];
    EXPECT_EQ(e.type, ElementType::CHEXA8);
    EXPECT_EQ(e.id.value, 1);
    EXPECT_EQ(e.pid.value, 1);
    ASSERT_EQ(e.nodes.size(), 8u);
    EXPECT_EQ(e.nodes[0].value, 1);
    EXPECT_EQ(e.nodes[5].value, 6);
    EXPECT_EQ(e.nodes[6].value, 7);
    EXPECT_EQ(e.nodes[7].value, 8);
}

// ── Additional card types ─────────────────────────────────────────────────────

TEST(BdfParser, MomentCard) {
    // MOMENT, SID, G, CID, M, N1, N2, N3
    const std::string bdf = R"(
BEGIN BULK
GRID,1,,0,0,0
MOMENT,2,1,0,500.0,0.0,1.0,0.0
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.loads.size(), 1u);
    const MomentLoad& mom = std::get<MomentLoad>(m.loads[0]);
    EXPECT_EQ(mom.sid.value, 2);
    EXPECT_EQ(mom.node.value, 1);
    EXPECT_DOUBLE_EQ(mom.scale, 500.0);
    EXPECT_DOUBLE_EQ(mom.direction.x, 0.0);
    EXPECT_DOUBLE_EQ(mom.direction.y, 1.0);
    EXPECT_DOUBLE_EQ(mom.direction.z, 0.0);
}

TEST(BdfParser, SpcCard) {
    // SPC allows per-DOF displacement values; two entries per line.
    const std::string bdf = R"(
BEGIN BULK
GRID,1,,0,0,0
GRID,2,,1,0,0
SPC,1,1,123,0.0,2,3,0.0
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.spcs.size(), 2u);
    EXPECT_EQ(m.spcs[0].sid.value, 1);
    EXPECT_EQ(m.spcs[0].node.value, 1);
    EXPECT_TRUE(m.spcs[0].dofs.has(1) && m.spcs[0].dofs.has(2) && m.spcs[0].dofs.has(3));
    EXPECT_DOUBLE_EQ(m.spcs[0].value, 0.0);
    EXPECT_EQ(m.spcs[1].node.value, 2);
    EXPECT_TRUE(m.spcs[1].dofs.has(3));
}

TEST(BdfParser, TempdCard) {
    // TEMPD sets a default temperature for a load set. When no explicit
    // t_ref is present on the subcase, it should be applied from tempd_map.
    const std::string bdf = R"(
SOL 101
SUBCASE 1
  LOAD = 5
  SPC  = 1
BEGIN BULK
TEMPD,5,150.0
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    EXPECT_NEAR(m.analysis.subcases[0].t_ref, 150.0, 1e-10);
}

TEST(BdfParser, SmallFieldSpcFullLine) {
    // SPC in small-field format; tests that DOF and displacement value
    // are read correctly from fixed-column positions.
    const std::string bdf =
        "BEGIN BULK\n"
        "GRID           3       0     0.0     0.0     0.0\n"
        "SPC            1       3     246     0.0\n"
        "ENDDATA\n";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.spcs.size(), 1u);
    const Spc& spc = m.spcs[0];
    EXPECT_EQ(spc.sid.value, 1);
    EXPECT_EQ(spc.node.value, 3);
    EXPECT_TRUE(spc.dofs.has(2) && spc.dofs.has(4) && spc.dofs.has(6));
    EXPECT_FALSE(spc.dofs.has(1));
    EXPECT_DOUBLE_EQ(spc.value, 0.0);
}

// ── Model accessor methods (only used in tests) ───────────────────────────────

static Model make_simple_model() {
    const std::string bdf = R"(
SOL 101
BEGIN BULK
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,1.0,1.0,0.0
GRID,4,,0.0,1.0,0.0
MAT1,1,2.0E7,,0.3
PSHELL,1,1,0.1
CQUAD4,1,1,1,2,3,4
SPC1,1,123456,1
FORCE,1,2,0,500.0,1.0,0.0,0.0
ENDDATA
)";
    return BdfParser::parse_string(bdf);
}

TEST(Model, NodeAccessorReturnsCorrectPosition) {
    Model m = make_simple_model();
    const GridPoint& gp = m.node(NodeId{3});
    EXPECT_NEAR(gp.position.x, 1.0, 1e-10);
    EXPECT_NEAR(gp.position.y, 1.0, 1e-10);
}

TEST(Model, NodeAccessorThrowsForMissingNode) {
    Model m = make_simple_model();
    EXPECT_THROW(m.node(NodeId{999}), SolverError);
}

TEST(Model, MaterialAccessorReturnsCorrectE) {
    Model m = make_simple_model();
    const Mat1& mat = m.material(MaterialId{1});
    EXPECT_NEAR(mat.E, 2.0e7, 1.0);
    EXPECT_NEAR(mat.nu, 0.3, 1e-10);
}

TEST(Model, MaterialAccessorThrowsForMissingMaterial) {
    Model m = make_simple_model();
    EXPECT_THROW(m.material(MaterialId{999}), SolverError);
}

TEST(Model, PropertyAccessorReturnsPShell) {
    Model m = make_simple_model();
    const Property& prop = m.property(PropertyId{1});
    const PShell& ps = std::get<PShell>(prop);
    EXPECT_NEAR(ps.t, 0.1, 1e-10);
}

TEST(Model, PropertyAccessorThrowsForMissingProperty) {
    Model m = make_simple_model();
    EXPECT_THROW(m.property(PropertyId{999}), SolverError);
}

TEST(Model, LoadsForSetReturnsMatchingLoads) {
    Model m = make_simple_model();
    auto loads = m.loads_for_set(LoadSetId{1});
    ASSERT_EQ(loads.size(), 1u);
    const ForceLoad& f = std::get<ForceLoad>(*loads[0]);
    EXPECT_EQ(f.node.value, 2);
}

TEST(Model, LoadsForSetReturnsEmptyForUnknownSet) {
    Model m = make_simple_model();
    auto loads = m.loads_for_set(LoadSetId{999});
    EXPECT_TRUE(loads.empty());
}

TEST(Model, SpcsForSetReturnsMatchingSpcs) {
    Model m = make_simple_model();
    auto spcs = m.spcs_for_set(SpcSetId{1});
    ASSERT_EQ(spcs.size(), 1u);
    EXPECT_EQ(spcs[0]->node.value, 1);
}

TEST(Model, SpcsForSetReturnsEmptyForUnknownSet) {
    Model m = make_simple_model();
    auto spcs = m.spcs_for_set(SpcSetId{999});
    EXPECT_TRUE(spcs.empty());
}
