// tests/unit/test_bdf_parser.cpp
// Tests for BDF field parsing: numeric formats, card splitting, model population.

#include <gtest/gtest.h>
#include "io/bdf_parser.hpp"
#include "core/logger.hpp"
#include "core/model.hpp"
#include <filesystem>
#include <fstream>
#include <numbers>
#include <sstream>
#include <spdlog/spdlog.h>

using namespace vibestran;

namespace {

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

} // namespace

// ── Numeric format parsing ────────────────────────────────────────────────────

TEST(BdfParser, ParseSimpleBdf) {
    // Minimal valid BDF: one node, one material, one property, one element, one SPC
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
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
    const auto& n = m.nodes.at(NodeId{1});
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
    const auto& n = m.nodes.at(NodeId{10});
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
GRID,1,,0.0,0.0,0.0
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

TEST(BdfParser, PloadCard) {
    const std::string bdf = R"(
BEGIN BULK
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,1.0,1.0,0.0
GRID,4,,0.0,1.0,0.0
PLOAD,7,-12.5,1,2,3,4
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.loads.size(), 1u);
    const PloadLoad& pl = std::get<PloadLoad>(m.loads[0]);
    EXPECT_EQ(pl.sid.value, 7);
    EXPECT_DOUBLE_EQ(pl.pressure, -12.5);
    ASSERT_EQ(pl.nodes.size(), 4u);
    EXPECT_EQ(pl.nodes[0].value, 1);
    EXPECT_EQ(pl.nodes[3].value, 4);
}

TEST(BdfParser, Pload1CardStub) {
    const std::string bdf = R"(
BEGIN BULK
CQUAD4,1,1,1,2,3,4
PLOAD1,9,1,FY,FR,0.0,25.0,1.0,50.0
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.loads.size(), 1u);
    const Pload1Load& pl = std::get<Pload1Load>(m.loads[0]);
    EXPECT_EQ(pl.sid.value, 9);
    EXPECT_EQ(pl.element.value, 1);
    EXPECT_EQ(pl.load_type, "FY");
    EXPECT_EQ(pl.scale_type, "FR");
    ASSERT_TRUE(pl.x2.has_value());
    ASSERT_TRUE(pl.p2.has_value());
    EXPECT_DOUBLE_EQ(*pl.x2, 1.0);
    EXPECT_DOUBLE_EQ(*pl.p2, 50.0);
}

TEST(BdfParser, ParsesLineBushAndScalarFamilies) {
    const std::string bdf = R"(
BEGIN BULK
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
GRID,4,,1.0,1.0,0.0
GRID,5,,0.0,0.0,1.0
GRID,6,,1.0,0.0,1.0
GRID,7,,0.0,1.0,1.0
GRID,8,,1.0,1.0,1.0
MAT1,1,2.0E7,,0.3
PBAR,10,1,2.5,0.1,0.2,0.3,0.4
PBUSH,20,K,1.0,2.0,3.0,4.0,5.0,6.0
PELAS,30,1000.0
PMASS,40,2.5
CBAR,100,10,1,2,0.0,1.0,0.0
CBUSH,200,20,3,4,0.0,1.0,0.0
CELAS1,300,30,5,1,6,1
CMASS1,400,40,7,2,8,2
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.properties.size(), 4u);
    ASSERT_EQ(m.elements.size(), 4u);

    const auto &pbar = std::get<PBar>(m.properties.at(PropertyId{10}));
    EXPECT_DOUBLE_EQ(pbar.A, 2.5);
    EXPECT_DOUBLE_EQ(pbar.J, 0.3);

    const auto &pbush = std::get<PBush>(m.properties.at(PropertyId{20}));
    EXPECT_DOUBLE_EQ(pbush.k[0], 1.0);
    EXPECT_DOUBLE_EQ(pbush.k[5], 6.0);

    const auto &cbush = m.elements[1];
    EXPECT_EQ(cbush.type, ElementType::CBUSH);
    ASSERT_TRUE(cbush.orientation.has_value());
    EXPECT_DOUBLE_EQ(cbush.orientation->y, 1.0);

    const auto &celas1 = m.elements[2];
    EXPECT_EQ(celas1.type, ElementType::CELAS1);
    EXPECT_EQ(celas1.components[0], 1);
    EXPECT_EQ(celas1.components[1], 1);

    const auto &cmass1 = m.elements[3];
    EXPECT_EQ(cmass1.type, ElementType::CMASS1);
    EXPECT_EQ(cmass1.components[0], 2);
    EXPECT_EQ(cmass1.components[1], 2);
}

TEST(BdfParser, ParsesPbarlAndAccel1ByRange) {
    const std::string bdf = R"(
BEGIN BULK
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,2.0,0.0,0.0
GRID,4,,3.0,0.0,0.0
GRID,5,,4.0,0.0,0.0
MAT1,1,2.0E7,,0.3
PBARL,11,1,MSCBMLO,TUBE,2.0,0.25,0.1
ACCEL1,9,0,3.0,1.0,0.0,0.0,1,THRU,5,BY,2
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    const auto &pbarl = std::get<PBarL>(m.properties.at(PropertyId{11}));
    EXPECT_EQ(pbarl.section_type, "TUBE");
    EXPECT_NEAR(pbarl.A, std::numbers::pi * (4.0 - 2.25) * 0.25, 1e-12);
    EXPECT_DOUBLE_EQ(pbarl.nsm, 0.1);

    ASSERT_EQ(m.loads.size(), 1u);
    const auto &accel1 = std::get<Accel1Load>(m.loads[0]);
    ASSERT_EQ(accel1.nodes.size(), 3u);
    EXPECT_EQ(accel1.nodes[0].value, 1);
    EXPECT_EQ(accel1.nodes[1].value, 3);
    EXPECT_EQ(accel1.nodes[2].value, 5);
}

TEST(BdfParser, ParsesCbeamG0AndGroundedScalarCards) {
    const std::string bdf = R"(
BEGIN BULK
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,9,,0.0,1.0,0.0
GRID,10,,0.0,0.0,0.0
MAT1,1,2.0E7,,0.3
PBEAM,12,1,1.5,0.2,0.3,0.0,0.4,0.1
CBEAM,101,12,1,2,9
CELAS2,301,55.0,10,2
CMASS2,401,3.5,10,2
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.elements.size(), 3u);
    const auto &cbeam = m.elements[0];
    EXPECT_EQ(cbeam.type, ElementType::CBEAM);
    ASSERT_TRUE(cbeam.g0.has_value());
    EXPECT_EQ(cbeam.g0->value, 9);

    const auto &celas2 = m.elements[1];
    EXPECT_EQ(celas2.type, ElementType::CELAS2);
    EXPECT_DOUBLE_EQ(celas2.value, 55.0);
    EXPECT_EQ(celas2.nodes.size(), 1u);
    EXPECT_EQ(celas2.components[0], 2);

    const auto &cmass2 = m.elements[2];
    EXPECT_EQ(cmass2.type, ElementType::CMASS2);
    EXPECT_DOUBLE_EQ(cmass2.value, 3.5);
    EXPECT_EQ(cmass2.nodes.size(), 1u);
}

TEST(BdfParser, Pload2CardExpandsThruRange) {
    const std::string bdf = R"(
BEGIN BULK
PLOAD2,3,-4.0,10,THRU,12,20
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.loads.size(), 4u);
    const Pload2Load& p0 = std::get<Pload2Load>(m.loads[0]);
    const Pload2Load& p1 = std::get<Pload2Load>(m.loads[1]);
    const Pload2Load& p2 = std::get<Pload2Load>(m.loads[2]);
    const Pload2Load& p3 = std::get<Pload2Load>(m.loads[3]);
    EXPECT_EQ(p0.element.value, 10);
    EXPECT_EQ(p1.element.value, 11);
    EXPECT_EQ(p2.element.value, 12);
    EXPECT_EQ(p3.element.value, 20);
    EXPECT_DOUBLE_EQ(p0.pressure, -4.0);
}

TEST(BdfParser, Pload4CardParsesVectorAndFaceIds) {
    const std::string bdf =
        "BEGIN BULK\n"
        "PLOAD4,5,17,8.0,9.0,10.0,11.0,101,104\n"
        "+,42,1.0,2.0,3.0\n"
        "ENDDATA\n";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.loads.size(), 1u);
    const Pload4Load& pl = std::get<Pload4Load>(m.loads[0]);
    EXPECT_EQ(pl.sid.value, 5);
    EXPECT_EQ(pl.element.value, 17);
    EXPECT_DOUBLE_EQ(pl.pressures[0], 8.0);
    EXPECT_DOUBLE_EQ(pl.pressures[3], 11.0);
    ASSERT_TRUE(pl.use_vector);
    EXPECT_EQ(pl.cid.value, 42);
    EXPECT_DOUBLE_EQ(pl.direction.x, 1.0);
    EXPECT_DOUBLE_EQ(pl.direction.y, 2.0);
    EXPECT_DOUBLE_EQ(pl.direction.z, 3.0);
    ASSERT_TRUE(pl.face_node1.has_value());
    ASSERT_TRUE(pl.face_node34.has_value());
    EXPECT_EQ(pl.face_node1->value, 101);
    EXPECT_EQ(pl.face_node34->value, 104);
}

TEST(BdfParser, SPC1WithThruRange) {
    const std::string bdf = R"(
BEGIN BULK
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,2.0,0.0,0.0
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
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
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
GRID,1,,0.0,0.0,0.0
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
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
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
CEND
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

TEST(BdfParser, ReportsUniqueUnsupportedCaseControlAndBulkCards) {
    const auto log_path =
        std::filesystem::temp_directory_path() / "vibestran_bdf_parser_unsupported_cards.log";
    init_logger(log_path);

    const std::string bdf = R"(
SOL 101
CEND
TITLE = FIRST TITLE
TITLE = SECOND TITLE
ECHO = NONE
SUBCASE 1
  LOAD = 1
  SPC  = 1
BEGIN BULK
GRID,1,,0.0,0.0,0.0
PBEAML,10,1
PBEAML,11,1
CELAS3,100,1,2
ENDDATA
)";

    Model m = BdfParser::parse_string(bdf);
    EXPECT_EQ(m.nodes.size(), 1u);

    spdlog::default_logger()->flush();
    const std::string log_text = read_text_file(log_path);

    EXPECT_NE(log_text.find("Unsupported BDF cards were ignored"), std::string::npos);
    EXPECT_NE(log_text.find("Case control: ECHO, TITLE"), std::string::npos);
    EXPECT_NE(log_text.find("Bulk data: CELAS3, PBEAML"), std::string::npos);
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
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
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

// ── Formulation selection parsing ─────────────────────────────────────────────

TEST(BdfParser, PsolidIsopEas) {
    // PSOLID field f[6] == "EAS" → SolidFormulation::EAS
    const std::string bdf = R"(
BEGIN BULK
PSOLID,1,1,0,SMEAR,NO,EAS
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.properties.size(), 1u);
    const PSolid& ps = std::get<PSolid>(m.properties.at(PropertyId{1}));
    EXPECT_EQ(ps.isop, SolidFormulation::EAS);
}

TEST(BdfParser, PsolidIsopSri) {
    // PSOLID field f[6] == "SRI" → SolidFormulation::SRI
    const std::string bdf = R"(
BEGIN BULK
PSOLID,1,1,0,SMEAR,NO,SRI
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.properties.size(), 1u);
    const PSolid& ps = std::get<PSolid>(m.properties.at(PropertyId{1}));
    EXPECT_EQ(ps.isop, SolidFormulation::SRI);
}

TEST(BdfParser, PsolidDefaultIsopEas) {
    // PSOLID without ISOP field → default SolidFormulation::EAS
    const std::string bdf = R"(
BEGIN BULK
PSOLID,1,1
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.properties.size(), 1u);
    const PSolid& ps = std::get<PSolid>(m.properties.at(PropertyId{1}));
    EXPECT_EQ(ps.isop, SolidFormulation::EAS);
}

TEST(BdfParser, Ctetra10Nodes) {
    // CTETRA with 10 node IDs → ElementType::CTETRA10
    const std::string bdf =
        "BEGIN BULK\n"
        "CTETRA         1       1       1       2       3       4       5       6\n"
        "+              7       8       9      10\n"
        "ENDDATA\n";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.elements.size(), 1u);
    const ElementData& e = m.elements[0];
    EXPECT_EQ(e.type, ElementType::CTETRA10);
    ASSERT_EQ(e.nodes.size(), 10u);
    EXPECT_EQ(e.nodes[0].value, 1);
    EXPECT_EQ(e.nodes[9].value, 10);
}

TEST(BdfParser, Ctetra4NodesStillCtetra4) {
    // CTETRA with 4 nodes → ElementType::CTETRA4 (existing behavior unchanged)
    const std::string bdf =
        "BEGIN BULK\n"
        "CTETRA         1       1       1       2       3       4\n"
        "ENDDATA\n";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.elements.size(), 1u);
    EXPECT_EQ(m.elements[0].type, ElementType::CTETRA4);
    ASSERT_EQ(m.elements[0].nodes.size(), 4u);
}

TEST(BdfParser, ParamShellformMindlin) {
    // PARAM,SHELLFORM,MINDLIN → all PShell::shell_form == ShellFormulation::MINDLIN
    const std::string bdf = R"(
BEGIN BULK
MAT1,1,2.0E7,,0.3
PSHELL,1,1,0.1
PSHELL,2,1,0.2
PARAM,SHELLFORM,MINDLIN
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.properties.size(), 2u);
    const PShell& ps1 = std::get<PShell>(m.properties.at(PropertyId{1}));
    const PShell& ps2 = std::get<PShell>(m.properties.at(PropertyId{2}));
    EXPECT_EQ(ps1.shell_form, ShellFormulation::MINDLIN);
    EXPECT_EQ(ps2.shell_form, ShellFormulation::MINDLIN);
    EXPECT_EQ(m.params.at("SHELLFORM"), "MINDLIN");
}

TEST(BdfParser, DefaultShellformMitc4) {
    // Without any PARAM → PShell::shell_form defaults to ShellFormulation::MITC4
    const std::string bdf = R"(
BEGIN BULK
MAT1,1,2.0E7,,0.3
PSHELL,1,1,0.1
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.properties.size(), 1u);
    const PShell& ps = std::get<PShell>(m.properties.at(PropertyId{1}));
    EXPECT_EQ(ps.shell_form, ShellFormulation::MITC4);
}

TEST(BdfParser, PshellContinuationAndShellOrientationFields) {
    const std::string bdf = R"(
BEGIN BULK
MAT1,1,2.0E7,,0.3
MAT1,2,4.0E7,,0.3
MAT1,3,6.0E7,,0.3
MAT1,4,8.0E7,,0.3
PSHELL,10,1,0.2,2,1.5,3,0.9,2.25
+, -0.12, 0.08, 4
CQUAD4,5,10,1,2,3,4,30.,0.015
CTRIA3,6,10,1,2,3,42,0.02
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);

    const PShell& ps = std::get<PShell>(m.properties.at(PropertyId{10}));
    EXPECT_EQ(ps.mid1, MaterialId{1});
    EXPECT_EQ(ps.mid2, MaterialId{2});
    EXPECT_EQ(ps.mid3, MaterialId{3});
    EXPECT_EQ(ps.mid4, MaterialId{4});
    EXPECT_DOUBLE_EQ(ps.t, 0.2);
    EXPECT_DOUBLE_EQ(ps.twelveI_t3, 1.5);
    EXPECT_DOUBLE_EQ(ps.tst, 0.9);
    EXPECT_DOUBLE_EQ(ps.nsm, 2.25);
    EXPECT_DOUBLE_EQ(ps.z1, -0.12);
    EXPECT_DOUBLE_EQ(ps.z2, 0.08);

    ASSERT_EQ(m.elements.size(), 2u);
    const ElementData& quad = m.elements[0];
    EXPECT_EQ(quad.type, ElementType::CQUAD4);
    ASSERT_TRUE(quad.theta.has_value());
    EXPECT_NEAR(*quad.theta, 30.0, 1e-12);
    ASSERT_FALSE(quad.mcid.has_value());
    EXPECT_DOUBLE_EQ(quad.zoffs, 0.015);

    const ElementData& tria = m.elements[1];
    EXPECT_EQ(tria.type, ElementType::CTRIA3);
    ASSERT_FALSE(tria.theta.has_value());
    ASSERT_TRUE(tria.mcid.has_value());
    EXPECT_EQ(*tria.mcid, CoordId{42});
    EXPECT_DOUBLE_EQ(tria.zoffs, 0.02);
}

// ── Fixed-width (small-field) format tests ──────────────────────────────────
// These test the more common BDF style seen in production, where cards use
// fixed 8-character columns and continuation lines have blank first fields.

TEST(BdfParser, BlankLabelContinuation_CHEXA) {
    // CHEXA with 8 nodes: first 6 on parent line, last 2 on blank-label continuation
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  SPC = 1
BEGIN BULK
GRID           1              0.      0.      0.
GRID           2              1.      0.      0.
GRID           3              1.      1.      0.
GRID           4              0.      1.      0.
GRID           5              0.      0.      1.
GRID           6              1.      0.      1.
GRID           7              1.      1.      1.
GRID           8              0.      1.      1.
MAT1           1 2.0E+07              .3
PSOLID         1       1
CHEXA          1       1       1       2       3       4       5       6
               7       8
SPC1           1  123456       1       2       3       4       5       6
               7       8
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.elements.size(), 1u);
    EXPECT_EQ(m.elements[0].type, ElementType::CHEXA8);
    ASSERT_EQ(m.elements[0].nodes.size(), 8u);
    EXPECT_EQ(m.elements[0].nodes[6].value, 7);
    EXPECT_EQ(m.elements[0].nodes[7].value, 8);
    // SPC1 with blank-label continuation should have all 8 nodes
    EXPECT_EQ(m.spcs.size(), 8u);
}

TEST(BdfParser, BlankLabelContinuation_SPC1_Long) {
    // SPC1 with multiple blank-label continuation lines (production-style)
    const std::string bdf = R"(
BEGIN BULK
GRID           1              0.      0.      0.
GRID           2              1.      0.      0.
GRID           3              2.      0.      0.
GRID           4              3.      0.      0.
GRID           5              4.      0.      0.
GRID           6              5.      0.      0.
GRID           7              6.      0.      0.
GRID           8              7.      0.      0.
GRID           9              8.      0.      0.
GRID          10              9.      0.      0.
GRID          11             10.      0.      0.
GRID          12             11.      0.      0.
GRID          13             12.      0.      0.
GRID          14             13.      0.      0.
GRID          15             14.      0.      0.
GRID          16             15.      0.      0.
GRID          17             16.      0.      0.
SPC1           1     123       1       2       3       4       5       6
               7       8       9      10      11      12      13      14
              15      16      17
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    EXPECT_EQ(m.spcs.size(), 17u);
}

TEST(BdfParser, FixedWidth_CPENTA6) {
    // CPENTA with 6 nodes in fixed-width format (no continuation needed)
    const std::string bdf = R"(
BEGIN BULK
GRID           1              0.      0.      0.
GRID           2              1.      0.      0.
GRID           3              .5  .86603      0.
GRID           4              0.      0.      1.
GRID           5              1.      0.      1.
GRID           6              .5  .86603      1.
MAT1           1 2.0E+07              .3
PSOLID         1       1
CPENTA         1       1       1       2       3       4       5       6
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.elements.size(), 1u);
    EXPECT_EQ(m.elements[0].type, ElementType::CPENTA6);
    ASSERT_EQ(m.elements[0].nodes.size(), 6u);
    for (int i = 0; i < 6; ++i)
        EXPECT_EQ(m.elements[0].nodes[i].value, i + 1);
}

TEST(BdfParser, FixedWidth_CORD2C_WithBlankContinuation) {
    // CORD2C spanning two lines with blank-label continuation (production-style)
    const std::string bdf = R"(
BEGIN BULK
CORD2C         2              0.      0.      0.      1.      0.      0.
              0.      1.      0.
GRID           1       2     .01      0.      0.       2
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.coord_systems.count(CoordId{2}), 1u);
    EXPECT_EQ(m.coord_systems.at(CoordId{2}).type, CoordType::Cylindrical);
    // Node 1: CP was 2 (cylindrical) but resolve_coordinates() transforms to
    // basic and clears CP to 0. CD should remain 2.
    ASSERT_EQ(m.nodes.count(NodeId{1}), 1u);
    EXPECT_EQ(m.nodes.at(NodeId{1}).cp.value, 0); // resolved to basic
    EXPECT_EQ(m.nodes.at(NodeId{1}).cd.value, 2);
    // CORD2C 2 has z-axis along basic X (A=origin, B=(1,0,0)).
    // For cylindrical (r=0.01, θ=0°, z=0), the r-direction at θ=0 is in the
    // local x-direction (defined by C=(0,1,0) projected onto the plane perp to
    // local z). So basic position = (z=0, r*cos0=0.01, 0) → (0, 0.01, 0).
    EXPECT_NEAR(m.nodes.at(NodeId{1}).position.x, 0.0, 1e-10);
    EXPECT_NEAR(m.nodes.at(NodeId{1}).position.y, 0.01, 1e-10);
    EXPECT_NEAR(m.nodes.at(NodeId{1}).position.z, 0.0, 1e-10);
}

TEST(BdfParser, TemperatureLoadCaseControl) {
    // TEMPERATURE(LOAD) = N should be parsed into SubCase.temp_load_set
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  SPC = 1
  TEMPERATURE(LOAD) = 3
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.analysis.subcases.size(), 1u);
    EXPECT_EQ(m.analysis.subcases[0].temp_load_set, 3);
}

TEST(BdfParser, TempLoadCaseControl) {
    // TEMP(LOAD) = N (short form) should also work
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  SPC = 1
  TEMP(LOAD) = 5
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.analysis.subcases.size(), 1u);
    EXPECT_EQ(m.analysis.subcases[0].temp_load_set, 5);
}

TEST(BdfParser, FixedWidth_GRID_NastranReals) {
    // Nastran-style real numbers in fixed-width GRID: no leading zero, trailing dot,
    // adjacent values without spaces (packed fields). Exact doublet.bdf column layout.
    //                1234567812345678123456781234567812345678123456781234567812345678
    const std::string bdf =
        "BEGIN BULK\n"
        "GRID           1       012.39486      0.1.968863\n"
        "ENDDATA\n";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.nodes.size(), 1u);
    const auto& n = m.nodes.at(NodeId{1});
    EXPECT_EQ(n.cp.value, 0);
    EXPECT_NEAR(n.position.x, 12.39486, 1e-4);
    EXPECT_NEAR(n.position.y, 0.0, 1e-12);
    EXPECT_NEAR(n.position.z, 1.968863, 1e-5);
}

TEST(BdfParser, FixedWidth_MAT1_WithAlpha) {
    // MAT1 with thermal expansion and TREF in fixed-width format
    const std::string bdf = R"(
BEGIN BULK
MAT1           1  81700.            .243    2.93.0000065     20.
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.materials.size(), 1u);
    const Mat1& mat = m.materials.at(MaterialId{1});
    EXPECT_NEAR(mat.E, 81700.0, 1e-6);
    EXPECT_NEAR(mat.nu, 0.243, 1e-6);
    EXPECT_NEAR(mat.rho, 2.93, 1e-6);
    EXPECT_NEAR(mat.A, 0.0000065, 1e-10);
    EXPECT_NEAR(mat.ref_temp, 20.0, 1e-6);
}

// ── Large-field KEYWORD* format (standard MSC Nastran) ──────────────────────

TEST(BdfParser, LargeFieldGridStar_WithContinuation) {
    // Standard MSC Nastran large-field format: keyword ends with '*' (e.g. GRID*)
    // and continuation starts with '*'.  The Mecway-generated test file uses this
    // convention.  Fields are 16 chars wide, so a two-line GRID* card fully
    // captures ID, CP, X1, X2, X3, CD.
    // Large-field layout (80-column): keyword(8) + 4×field(16) + cont(8)
    // cols 0-7:   GRID*
    // cols 8-23:  node ID (right-justified)
    // cols 24-39: CP
    // cols 40-55: X1
    // cols 56-71: X2  ← continuation marker occupies col 72
    // line 2: *-marker(8) + X3(16) + CD(16)
    const std::string bdf =
        "BEGIN BULK\n"
        "GRID*                  7               0        3.000000        4.000000*\n"
        "*               5.000000               0\n"
        "ENDDATA\n";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.nodes.size(), 1u);
    const GridPoint& n = m.nodes.at(NodeId{7});
    EXPECT_DOUBLE_EQ(n.position.x, 3.0);
    EXPECT_DOUBLE_EQ(n.position.y, 4.0);
    EXPECT_DOUBLE_EQ(n.position.z, 5.0);
}

TEST(BdfParser, GlobalCaseControlNoSubcase) {
    // Some solvers (e.g. Mecway) emit case control entries at the global level
    // without an explicit SUBCASE block.  These entries must be applied to the
    // default subcase so METHOD/SPC/DISPLACEMENT take effect.
    const std::string bdf = R"(
SOL 103
CEND
METHOD = 5
SPC = 2
DISPLACEMENT = ALL
BEGIN BULK
MAT1,1,2.0E7,,0.3,1.0
PSHELL,1,1,0.1
GRID,1,,0.,0.,0.
GRID,2,,1.,0.,0.
GRID,3,,1.,1.,0.
GRID,4,,0.,1.,0.
CQUAD4,1,1,1,2,3,4
SPC1,2,123456,1,2,3,4
EIGRL,5,,,4
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.analysis.subcases.size(), 1u);
    const SubCase& sc = m.analysis.subcases[0];
    // METHOD = 5 and SPC = 2 from global case control must be applied
    EXPECT_EQ(sc.eigrl_id, 5);
    EXPECT_EQ(sc.spc_set.value, 2);
    EXPECT_TRUE(sc.disp_print);
}

TEST(BdfParser, TEMPD_StoredInModel) {
    // TEMPD cards should populate model.tempd map
    const std::string bdf = R"(
BEGIN BULK
TEMPD          1     20.
TEMPD          2     60.
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.tempd.size(), 2u);
    EXPECT_NEAR(m.tempd.at(1), 20.0, 1e-12);
    EXPECT_NEAR(m.tempd.at(2), 60.0, 1e-12);
}
