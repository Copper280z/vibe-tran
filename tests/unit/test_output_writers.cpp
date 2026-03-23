// tests/unit/test_output_writers.cpp
// Tests for F06, CSV output writers and result helper functions.
//
// Coverage:
//   - compute_principal_2d: known analytical results
//   - compute_principal_3d: diagonal (axis-aligned) and off-diagonal tensors
//   - F06Writer: displacement table, CQUAD4 and CTRIA3 separate tables,
//                principal stress columns in plate table, output flag suppression
//   - CsvWriter: header format, column order, per-subcase row emission,
//                output flag suppression
//   - BdfParser: DISPLACEMENT=NONE / STRESS=NONE case-control flags
//   - BdfParser: PARAM,CSVOUT,YES

#include <gtest/gtest.h>
#include "io/results.hpp"
#include "io/bdf_parser.hpp"
#include "core/model.hpp"
#include <sstream>
#include <string>
#include <cmath>

using namespace vibestran;

// ── Helpers ───────────────────────────────────────────────────────────────────

static SubCaseResults make_sc_with_disps() {
    SubCaseResults sc;
    sc.id = 1;
    NodeDisplacement nd;
    nd.node = NodeId{5};
    nd.d = {1.0, 2.0, 3.0, 0.1, 0.2, 0.3};
    sc.displacements.push_back(nd);
    return sc;
}

static PlateStress make_quad4_stress(ElementId eid, double sx, double sy, double sxy) {
    PlateStress ps;
    ps.eid   = eid;
    ps.etype = ElementType::CQUAD4;
    ps.sx = sx; ps.sy = sy; ps.sxy = sxy;
    ps.von_mises = std::sqrt(sx*sx - sx*sy + sy*sy + 3.0*sxy*sxy);
    return ps;
}

static PlateStress make_tria3_stress(ElementId eid, double sx, double sy, double sxy) {
    PlateStress ps = make_quad4_stress(eid, sx, sy, sxy);
    ps.etype = ElementType::CTRIA3;
    return ps;
}

static Model make_empty_model_sc1() {
    Model m;
    m.analysis.sol = SolutionType::LinearStatic;
    SubCase sc;
    sc.id = 1;
    sc.disp_print   = true;
    sc.stress_print = true;
    m.analysis.subcases.push_back(sc);
    return m;
}

// ── compute_principal_2d ─────────────────────────────────────────────────────

TEST(PrincipalStress2D, UniaxialX) {
    // sx = 100, sy = 0, sxy = 0 → principal = {100, 0}, angle = 0°
    double major, minor, angle;
    compute_principal_2d(100.0, 0.0, 0.0, major, minor, angle);
    EXPECT_NEAR(major, 100.0, 1e-10);
    EXPECT_NEAR(minor,   0.0, 1e-10);
    EXPECT_NEAR(angle,   0.0, 1e-10);
}

TEST(PrincipalStress2D, PureShear) {
    // sx = 0, sy = 0, sxy = τ → principal = {τ, -τ}, angle = 45°
    double tau = 50.0;
    double major, minor, angle;
    compute_principal_2d(0.0, 0.0, tau, major, minor, angle);
    EXPECT_NEAR(major,  tau, 1e-10);
    EXPECT_NEAR(minor, -tau, 1e-10);
    EXPECT_NEAR(std::abs(angle), 45.0, 1e-6);
}

TEST(PrincipalStress2D, EqualBiaxial) {
    // sx = sy = p, sxy = 0 → both principals = p
    double p = 75.0;
    double major, minor, angle;
    compute_principal_2d(p, p, 0.0, major, minor, angle);
    EXPECT_NEAR(major, p, 1e-10);
    EXPECT_NEAR(minor, p, 1e-10);
}

TEST(PrincipalStress2D, GeneralTensor) {
    // sx=100, sy=50, sxy=30
    // avg = 75, diff = 25, R = sqrt(625+900) = sqrt(1525)
    double sx = 100.0, sy = 50.0, sxy = 30.0;
    double R = std::sqrt(25.0*25.0 + sxy*sxy);
    double major, minor, angle;
    compute_principal_2d(sx, sy, sxy, major, minor, angle);
    EXPECT_NEAR(major, 75.0 + R, 1e-8);
    EXPECT_NEAR(minor, 75.0 - R, 1e-8);
    // Check that the tensor is recovered: major and minor are eigenvalues
    EXPECT_GT(major, minor);
    // Verify trace and determinant are preserved
    EXPECT_NEAR(major + minor, sx + sy, 1e-8);
    EXPECT_NEAR(major * minor, sx*sy - sxy*sxy, 1e-6);
}

// ── compute_principal_3d ─────────────────────────────────────────────────────

TEST(PrincipalStress3D, DiagonalTensor) {
    // Diagonal stress tensor: eigenvalues are the diagonal elements
    double p[3];
    double v[3][3];
    compute_principal_3d(30.0, 20.0, 10.0, 0.0, 0.0, 0.0, p, v);
    EXPECT_NEAR(p[0], 30.0, 1e-8); // sorted descending
    EXPECT_NEAR(p[1], 20.0, 1e-8);
    EXPECT_NEAR(p[2], 10.0, 1e-8);
}

TEST(PrincipalStress3D, UniaxialStress) {
    // Only sx nonzero → one eigenvalue = sx, two = 0
    double p[3];
    double v[3][3];
    compute_principal_3d(100.0, 0.0, 0.0, 0.0, 0.0, 0.0, p, v);
    EXPECT_NEAR(p[0], 100.0, 1e-8);
    EXPECT_NEAR(p[1],   0.0, 1e-8);
    EXPECT_NEAR(p[2],   0.0, 1e-8);
}

TEST(PrincipalStress3D, TracePreserved) {
    // The trace (sum of eigenvalues) must equal sx+sy+sz
    double sx=80, sy=60, sz=40, sxy=15, syz=10, szx=5;
    double p[3];
    double v[3][3];
    compute_principal_3d(sx, sy, sz, sxy, syz, szx, p, v);
    EXPECT_NEAR(p[0]+p[1]+p[2], sx+sy+sz, 1e-6);
    // Must be sorted descending
    EXPECT_GE(p[0], p[1]);
    EXPECT_GE(p[1], p[2]);
}

TEST(PrincipalStress3D, EigenvectorsOrthonormal) {
    // Eigenvectors from Jacobi should be orthonormal
    double sx=80, sy=60, sz=40, sxy=15, syz=10, szx=5;
    double p[3];
    double v[3][3];
    compute_principal_3d(sx, sy, sz, sxy, syz, szx, p, v);
    for (int i = 0; i < 3; ++i) {
        double dot_ii = 0.0;
        for (int k = 0; k < 3; ++k) dot_ii += v[i][k]*v[i][k];
        EXPECT_NEAR(dot_ii, 1.0, 1e-10) << "row " << i << " not unit length";
        for (int j = i+1; j < 3; ++j) {
            double dot_ij = 0.0;
            for (int k = 0; k < 3; ++k) dot_ij += v[i][k]*v[j][k];
            EXPECT_NEAR(dot_ij, 0.0, 1e-10) << "rows " << i << "," << j << " not orthogonal";
        }
    }
}

// ── F06Writer ─────────────────────────────────────────────────────────────────

TEST(F06Writer, HeaderContainsNastranTitle) {
    SolverResults res;
    res.subcases.push_back({});
    res.subcases[0].id = 1;
    Model m = make_empty_model_sc1();
    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    std::string out = oss.str();
    EXPECT_NE(out.find("N A S T R A N"), std::string::npos);
    EXPECT_NE(out.find("S O L   1 0 1"), std::string::npos);
}

TEST(F06Writer, DisplacementTable) {
    SolverResults res;
    res.subcases.push_back(make_sc_with_disps());
    Model m = make_empty_model_sc1();
    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    std::string out = oss.str();
    EXPECT_NE(out.find("D I S P L A C E M E N T"), std::string::npos);
    // Node 5 should appear
    EXPECT_NE(out.find("5"), std::string::npos);
}

TEST(F06Writer, Quad4StressTablePresent) {
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.plate_stresses.push_back(make_quad4_stress(ElementId{1}, 100.0, 50.0, 30.0));
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    std::string out = oss.str();
    EXPECT_NE(out.find("C Q U A D 4"), std::string::npos);
    EXPECT_NE(out.find("MAJOR"), std::string::npos);
    EXPECT_NE(out.find("MINOR"), std::string::npos);
    EXPECT_NE(out.find("ANGLE"), std::string::npos);
    EXPECT_NE(out.find("VON MISES"), std::string::npos);
}

TEST(F06Writer, Tria3StressTablePresent) {
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.plate_stresses.push_back(make_tria3_stress(ElementId{2}, 80.0, 20.0, 10.0));
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    std::string out = oss.str();
    EXPECT_NE(out.find("C T R I A 3"), std::string::npos);
}

TEST(F06Writer, Quad4AndTria3TablesSeparate) {
    // When both element types are present, both tables should appear and be distinct
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.plate_stresses.push_back(make_quad4_stress(ElementId{1}, 100.0, 0.0, 0.0));
    sc.plate_stresses.push_back(make_tria3_stress(ElementId{2}, 50.0,  0.0, 0.0));
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    std::string out = oss.str();
    EXPECT_NE(out.find("C Q U A D 4"), std::string::npos);
    EXPECT_NE(out.find("C T R I A 3"), std::string::npos);
    // The two table headers should be at different positions
    auto pos_q = out.find("C Q U A D 4");
    auto pos_t = out.find("C T R I A 3");
    EXPECT_NE(pos_q, pos_t);
}

TEST(F06Writer, OnlyTria3NoQuad4Table) {
    // If only CTRIA3 stresses are present, no CQUAD4 table should be emitted
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.plate_stresses.push_back(make_tria3_stress(ElementId{3}, 20.0, 5.0, 2.0));
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    std::string out = oss.str();
    EXPECT_EQ(out.find("C Q U A D 4"), std::string::npos);
    EXPECT_NE(out.find("C T R I A 3"), std::string::npos);
}

TEST(F06Writer, OutputFlagSuppressesDisplacement) {
    SolverResults res;
    res.subcases.push_back(make_sc_with_disps());
    Model m = make_empty_model_sc1();
    m.analysis.subcases[0].disp_print = false;
    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    std::string out = oss.str();
    EXPECT_EQ(out.find("D I S P L A C E M E N T"), std::string::npos);
}

TEST(F06Writer, OutputFlagSuppressesStress) {
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.plate_stresses.push_back(make_quad4_stress(ElementId{1}, 100.0, 0.0, 0.0));
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    m.analysis.subcases[0].stress_print = false;
    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    std::string out = oss.str();
    EXPECT_EQ(out.find("C Q U A D 4"), std::string::npos);
}

TEST(F06Writer, Quad4PrincipalStressValues) {
    // Pure uniaxial: sx=100, sy=0, sxy=0 → major=100, minor=0, angle=0
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    PlateStress ps;
    ps.eid   = ElementId{7};
    ps.etype = ElementType::CQUAD4;
    ps.sx = 100.0; ps.sy = 0.0; ps.sxy = 0.0;
    ps.von_mises = 100.0;
    sc.plate_stresses.push_back(ps);
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    std::string out = oss.str();
    // major = 100, should appear in scientific notation with leading digits "1.000000"
    EXPECT_NE(out.find("1.000000e+02"), std::string::npos);
}

// ── CsvWriter ─────────────────────────────────────────────────────────────────

// Helper: write CSV and return the two file contents via ostringstreams
// by re-implementing the logic inline using temporary files.
// Since CsvWriter writes to filesystem paths, we use a tmpdir approach.
#include <filesystem>
#include <fstream>

static std::pair<std::string,std::string> run_csv_writer(
    const SolverResults& res, const Model& m)
{
    namespace fs = std::filesystem;
    auto tmp = fs::temp_directory_path() / "nastran_test_csv_XXXXXX";
    // create a unique stem name
    static int counter = 0;
    auto stem = fs::temp_directory_path() / ("nastran_test_csv_" + std::to_string(++counter));
    CsvWriter::write(res, m, stem);
    auto node_path = stem.string() + ".node.csv";
    auto elem_path = stem.string() + ".elem.csv";
    auto read_all = [](const std::string& p) {
        std::ifstream f(p);
        return std::string(std::istreambuf_iterator<char>(f), {});
    };
    std::string node_csv = read_all(node_path);
    std::string elem_csv = read_all(elem_path);
    fs::remove(node_path);
    fs::remove(elem_path);
    return {node_csv, elem_csv};
}

TEST(CsvWriter, NodeCsvHeaderStartsWithHash) {
    SolverResults res;
    res.subcases.push_back(make_sc_with_disps());
    Model m = make_empty_model_sc1();
    auto [node_csv, elem_csv] = run_csv_writer(res, m);
    EXPECT_EQ(node_csv[0], '#');
}

TEST(CsvWriter, ElemCsvHeaderStartsWithHash) {
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.plate_stresses.push_back(make_quad4_stress(ElementId{10}, 100.0, 0.0, 0.0));
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    auto [node_csv, elem_csv] = run_csv_writer(res, m);
    EXPECT_EQ(elem_csv[0], '#');
}

TEST(CsvWriter, NodeCsvContainsExpectedColumns) {
    SolverResults res;
    res.subcases.push_back(make_sc_with_disps());
    Model m = make_empty_model_sc1();
    auto [node_csv, elem_csv] = run_csv_writer(res, m);
    // Header should mention T1..R3
    EXPECT_NE(node_csv.find("T1"), std::string::npos);
    EXPECT_NE(node_csv.find("R3"), std::string::npos);
}

TEST(CsvWriter, NodeCsvDataRow) {
    // Node 5 with T1=1.0; verify node_id and subcase_id appear on data row
    SolverResults res;
    res.subcases.push_back(make_sc_with_disps());
    Model m = make_empty_model_sc1();
    auto [node_csv, elem_csv] = run_csv_writer(res, m);
    // Data row: "5, 1, ..."
    EXPECT_NE(node_csv.find("5, 1"), std::string::npos);
}

TEST(CsvWriter, ElemCsvContainsId) {
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.plate_stresses.push_back(make_quad4_stress(ElementId{42}, 100.0, 50.0, 30.0));
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    auto [node_csv, elem_csv] = run_csv_writer(res, m);
    EXPECT_NE(elem_csv.find("42"), std::string::npos);
    // No element type strings in CSV output
    EXPECT_EQ(elem_csv.find("CQUAD4"), std::string::npos);
}

TEST(CsvWriter, ElemCsvNoElementTypeString) {
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.plate_stresses.push_back(make_tria3_stress(ElementId{99}, 10.0, 5.0, 2.0));
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    auto [node_csv, elem_csv] = run_csv_writer(res, m);
    EXPECT_NE(elem_csv.find("99"), std::string::npos);
    EXPECT_EQ(elem_csv.find("CTRIA3"), std::string::npos);
}

TEST(CsvWriter, NodeCsvOutputFlagSuppressesDisplacement) {
    SolverResults res;
    res.subcases.push_back(make_sc_with_disps());
    Model m = make_empty_model_sc1();
    m.analysis.subcases[0].disp_print = false;
    auto [node_csv, elem_csv] = run_csv_writer(res, m);
    // Only the header line, no data rows (no node id "5")
    std::string without_header = node_csv.substr(node_csv.find('\n') + 1);
    EXPECT_TRUE(without_header.empty() || without_header.find_first_not_of("\n") == std::string::npos);
}

TEST(CsvWriter, ElemCsvOutputFlagSuppressesStress) {
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.plate_stresses.push_back(make_quad4_stress(ElementId{1}, 100.0, 0.0, 0.0));
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    m.analysis.subcases[0].stress_print = false;
    auto [node_csv, elem_csv] = run_csv_writer(res, m);
    std::string without_header = elem_csv.substr(elem_csv.find('\n') + 1);
    EXPECT_TRUE(without_header.empty() || without_header.find_first_not_of("\n") == std::string::npos);
}

// ── BdfParser: case-control output flags ─────────────────────────────────────

// ── BdfParser: case-control output flag parsing ───────────────────────────────

TEST(BdfParser, DefaultFlagsAllFalse) {
    // No output cards → all four flags default to false
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_FALSE(sc.disp_print);
    EXPECT_FALSE(sc.disp_plot);
    EXPECT_FALSE(sc.stress_print);
    EXPECT_FALSE(sc.stress_plot);
}

TEST(BdfParser, DisplacementAllSetsPrintOnly) {
    // No modifier → PRINT (F06/CSV) only; OP2 remains off
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  DISPLACEMENT = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_TRUE(sc.disp_print);
    EXPECT_FALSE(sc.disp_plot);
}

TEST(BdfParser, DisplacementPrintAllSetsPrintOnly) {
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  DISPLACEMENT(PRINT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_TRUE(sc.disp_print);
    EXPECT_FALSE(sc.disp_plot);
}

TEST(BdfParser, DisplacementPlotAllSetsPlotOnly) {
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  DISPLACEMENT(PLOT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_FALSE(sc.disp_print);
    EXPECT_TRUE(sc.disp_plot);
}

TEST(BdfParser, DisplacementPrintPlotAllSetsBoth) {
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  DISPLACEMENT(PRINT,PLOT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_TRUE(sc.disp_print);
    EXPECT_TRUE(sc.disp_plot);
}

TEST(BdfParser, DisplacementNoneClearsBoth) {
    // NONE clears both PRINT and PLOT
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  DISPLACEMENT(PRINT,PLOT) = ALL
  DISPLACEMENT = NONE
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_FALSE(sc.disp_print);
    EXPECT_FALSE(sc.disp_plot);
}

TEST(BdfParser, StressAllSetsPrintOnly) {
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  STRESS = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_TRUE(sc.stress_print);
    EXPECT_FALSE(sc.stress_plot);
}

TEST(BdfParser, StressPlotAllSetsPlotOnly) {
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  STRESS(PLOT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_FALSE(sc.stress_print);
    EXPECT_TRUE(sc.stress_plot);
}

TEST(BdfParser, StressPrintPlotAllSetsBoth) {
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  STRESS(PRINT,PLOT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_TRUE(sc.stress_print);
    EXPECT_TRUE(sc.stress_plot);
}

TEST(BdfParser, StressNoneClearsBoth) {
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  STRESS = NONE
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_FALSE(sc.stress_print);
    EXPECT_FALSE(sc.stress_plot);
}

TEST(BdfParser, MultipleSubcasesIndependentFlags) {
    // SC1: disp PRINT only; SC2: stress PLOT only
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  DISPLACEMENT(PRINT) = ALL
SUBCASE 2
  LOAD = 2
  SPC  = 1
  STRESS(PLOT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.analysis.subcases.size(), 2u);
    const auto& sc1 = m.analysis.subcases[0];
    const auto& sc2 = m.analysis.subcases[1];
    EXPECT_TRUE(sc1.disp_print);
    EXPECT_FALSE(sc1.disp_plot);
    EXPECT_FALSE(sc1.stress_print);
    EXPECT_FALSE(sc1.stress_plot);
    EXPECT_FALSE(sc2.disp_print);
    EXPECT_FALSE(sc2.disp_plot);
    EXPECT_FALSE(sc2.stress_print);
    EXPECT_TRUE(sc2.stress_plot);
}

// ── BdfParser: PARAM,CSVOUT,YES ───────────────────────────────────────────────

TEST(BdfParser, ParamCsvoutYes) {
    const std::string bdf = R"(
BEGIN BULK
PARAM,CSVOUT,YES
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    auto it = m.params.find("CSVOUT");
    ASSERT_NE(it, m.params.end());
    EXPECT_EQ(it->second, "YES");
}

TEST(BdfParser, ParamCsvoutAbsent) {
    const std::string bdf = R"(
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    EXPECT_EQ(m.params.find("CSVOUT"), m.params.end());
}

// ── BdfParser: SOL 103 case-control EIGENVECTOR / DISPLACEMENT flags ──────────

TEST(BdfParser, Sol103EigenvectorPrintSetsEigvecPrint) {
    const std::string bdf = R"(
SOL 103
CEND
SUBCASE 1
  METHOD = 1
  EIGENVECTOR(PRINT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_TRUE(sc.eigvec_print);
    EXPECT_FALSE(sc.eigvec_plot);
}

TEST(BdfParser, Sol103EigenvectorPlotSetsEigvecPlot) {
    const std::string bdf = R"(
SOL 103
CEND
SUBCASE 1
  METHOD = 1
  EIGENVECTOR(PLOT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_FALSE(sc.eigvec_print);
    EXPECT_TRUE(sc.eigvec_plot);
}

TEST(BdfParser, Sol103EigenvectorNoModifierSetsPrintOnly) {
    // EIGENVECTOR = ALL with no modifier is equivalent to PRINT
    const std::string bdf = R"(
SOL 103
CEND
SUBCASE 1
  METHOD = 1
  EIGENVECTOR = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_TRUE(sc.eigvec_print);
    EXPECT_FALSE(sc.eigvec_plot);
}

TEST(BdfParser, Sol103DisplacementPlotSetsDispPlot) {
    // DISPLACEMENT(PLOT)=ALL in SOL 103 sets disp_plot; the modal solver
    // maps disp_plot → eigvec_print (F06) and eigvec_plot (OP2).
    const std::string bdf = R"(
SOL 103
CEND
SUBCASE 1
  METHOD = 1
  DISPLACEMENT(PLOT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_FALSE(sc.disp_print);
    EXPECT_TRUE(sc.disp_plot);
    // eigvec flags remain false at parse time; mapping happens in ModalSolver
    EXPECT_FALSE(sc.eigvec_print);
    EXPECT_FALSE(sc.eigvec_plot);
}

TEST(BdfParser, Sol103DisplacementPrintSetsDispPrint) {
    const std::string bdf = R"(
SOL 103
CEND
SUBCASE 1
  METHOD = 1
  DISPLACEMENT(PRINT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_TRUE(sc.disp_print);
    EXPECT_FALSE(sc.disp_plot);
}

TEST(BdfParser, Sol103EigrlMethodParsed) {
    const std::string bdf = R"(
SOL 103
CEND
SUBCASE 1
  METHOD = 5
  EIGENVECTOR = ALL
BEGIN BULK
EIGRL,5,0.0,,10
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    EXPECT_EQ(m.analysis.subcases[0].eigrl_id, 5);
    ASSERT_TRUE(m.eigrls.count(5));
    EXPECT_EQ(m.eigrls.at(5).nd, 10);
    EXPECT_DOUBLE_EQ(m.eigrls.at(5).v1, 0.0);
}
