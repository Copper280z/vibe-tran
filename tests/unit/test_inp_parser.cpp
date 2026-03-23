// tests/unit/test_inp_parser.cpp
// Tests for CalculiX/Abaqus .inp file parser.

#include <gtest/gtest.h>
#include "io/inp_parser.hpp"
#include "io/bdf_parser.hpp"
#include "core/model.hpp"
#include "solver/linear_static.hpp"
#include "solver/solver_backend.hpp"

using namespace vibestran;

// ── Node parsing ─────────────────────────────────────────────────────────────

TEST(InpParser, NodeParsing) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
)";
    Model m = InpParser::parse_string(inp);
    ASSERT_EQ(m.nodes.size(), 4u);

    const auto& n1 = m.nodes.at(NodeId{1});
    EXPECT_DOUBLE_EQ(n1.position.x, 0.0);
    EXPECT_DOUBLE_EQ(n1.position.y, 0.0);
    EXPECT_DOUBLE_EQ(n1.position.z, 0.0);

    const auto& n3 = m.nodes.at(NodeId{3});
    EXPECT_DOUBLE_EQ(n3.position.x, 1.0);
    EXPECT_DOUBLE_EQ(n3.position.y, 1.0);
    EXPECT_DOUBLE_EQ(n3.position.z, 0.0);

    // All nodes should have cp=cd=0
    for (const auto& [id, gp] : m.nodes) {
        EXPECT_EQ(gp.cp.value, 0);
        EXPECT_EQ(gp.cd.value, 0);
    }
}

// ── Element parsing per type ─────────────────────────────────────────────────

TEST(InpParser, ElementC3D8) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
*ELEMENT, TYPE=C3D8
1, 1, 2, 3, 4, 5, 6, 7, 8
)";
    Model m = InpParser::parse_string(inp);
    ASSERT_EQ(m.elements.size(), 1u);
    EXPECT_EQ(m.elements[0].type, ElementType::CHEXA8);
    EXPECT_EQ(m.elements[0].id.value, 1);
    ASSERT_EQ(m.elements[0].nodes.size(), 8u);
    EXPECT_EQ(m.elements[0].nodes[0].value, 1);
    EXPECT_EQ(m.elements[0].nodes[7].value, 8);
}

TEST(InpParser, ElementC3D4) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.5, 0.866, 0.0
4, 0.5, 0.289, 0.816
*ELEMENT, TYPE=C3D4
1, 1, 2, 3, 4
)";
    Model m = InpParser::parse_string(inp);
    ASSERT_EQ(m.elements.size(), 1u);
    EXPECT_EQ(m.elements[0].type, ElementType::CTETRA4);
    ASSERT_EQ(m.elements[0].nodes.size(), 4u);
}

TEST(InpParser, ElementC3D10) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.5, 0.866, 0.0
4, 0.5, 0.289, 0.816
5, 0.5, 0.0, 0.0
6, 0.75, 0.433, 0.0
7, 0.25, 0.433, 0.0
8, 0.25, 0.1445, 0.408
9, 0.75, 0.1445, 0.408
10, 0.5, 0.5775, 0.408
*ELEMENT, TYPE=C3D10
1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
)";
    Model m = InpParser::parse_string(inp);
    ASSERT_EQ(m.elements.size(), 1u);
    EXPECT_EQ(m.elements[0].type, ElementType::CTETRA10);
    ASSERT_EQ(m.elements[0].nodes.size(), 10u);
}

TEST(InpParser, ElementC3D6) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.5, 0.866, 0.0
4, 0.0, 0.0, 1.0
5, 1.0, 0.0, 1.0
6, 0.5, 0.866, 1.0
*ELEMENT, TYPE=C3D6
1, 1, 2, 3, 4, 5, 6
)";
    Model m = InpParser::parse_string(inp);
    ASSERT_EQ(m.elements.size(), 1u);
    EXPECT_EQ(m.elements[0].type, ElementType::CPENTA6);
    ASSERT_EQ(m.elements[0].nodes.size(), 6u);
}

TEST(InpParser, ElementS4) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
*ELEMENT, TYPE=S4
1, 1, 2, 3, 4
)";
    Model m = InpParser::parse_string(inp);
    ASSERT_EQ(m.elements.size(), 1u);
    EXPECT_EQ(m.elements[0].type, ElementType::CQUAD4);
    ASSERT_EQ(m.elements[0].nodes.size(), 4u);
}

TEST(InpParser, ElementS3) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.5, 0.866, 0.0
*ELEMENT, TYPE=S3
1, 1, 2, 3
)";
    Model m = InpParser::parse_string(inp);
    ASSERT_EQ(m.elements.size(), 1u);
    EXPECT_EQ(m.elements[0].type, ElementType::CTRIA3);
    ASSERT_EQ(m.elements[0].nodes.size(), 3u);
}

TEST(InpParser, ElementC3D20Throws) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
*ELEMENT, TYPE=C3D20
1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
)";
    EXPECT_THROW(InpParser::parse_string(inp), ParseError);
}

// ── NSET explicit and GENERATE ───────────────────────────────────────────────

TEST(InpParser, NsetExplicit) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 2.0, 0.0, 0.0
*NSET, NSET=FIX
1, 2, 3
)";
    EXPECT_EQ(InpParser::parse_string(inp).nodes.size(), 3u);
    // Verify boundary using the set
    const std::string inp2 = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 2.0, 0.0, 0.0
*NSET, NSET=FIX
1, 2, 3
*BOUNDARY
FIX, 1, 3
)";
    Model m2 = InpParser::parse_string(inp2);
    EXPECT_EQ(m2.spcs.size(), 3u);
    for (const auto& spc : m2.spcs) {
        EXPECT_TRUE(spc.dofs.has(1) && spc.dofs.has(2) && spc.dofs.has(3));
    }
}

TEST(InpParser, NsetGenerate) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
3, 1.0, 0.0, 0.0
5, 2.0, 0.0, 0.0
7, 3.0, 0.0, 0.0
*NSET, NSET=ODDS, GENERATE
1, 7, 2
*BOUNDARY
ODDS, 1, 3
)";
    Model m = InpParser::parse_string(inp);
    // 4 nodes in set: 1, 3, 5, 7
    EXPECT_EQ(m.spcs.size(), 4u);
}

// ── ELSET explicit and GENERATE ──────────────────────────────────────────────

TEST(InpParser, ElsetExplicit) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.5, 0.866, 0.0
4, 0.5, 0.289, 0.816
*ELEMENT, TYPE=C3D4
1, 1, 2, 3, 4
2, 1, 2, 3, 4
*ELSET, ELSET=BODY
1, 2
*MATERIAL, NAME=STEEL
*ELASTIC
210000.0, 0.3
*SOLID SECTION, ELSET=BODY, MATERIAL=STEEL
)";
    Model m = InpParser::parse_string(inp);
    // Both elements should have a pid assigned
    EXPECT_NE(m.elements[0].pid.value, 0);
    EXPECT_EQ(m.elements[0].pid, m.elements[1].pid);
}

TEST(InpParser, ElsetGenerate) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.5, 0.866, 0.0
4, 0.5, 0.289, 0.816
*ELEMENT, TYPE=C3D4
1, 1, 2, 3, 4
2, 1, 2, 3, 4
3, 1, 2, 3, 4
*ELSET, ELSET=ALL, GENERATE
1, 3, 1
*MATERIAL, NAME=STEEL
*ELASTIC
210000.0, 0.3
*SOLID SECTION, ELSET=ALL, MATERIAL=STEEL
)";
    Model m = InpParser::parse_string(inp);
    for (const auto& e : m.elements) {
        EXPECT_NE(e.pid.value, 0);
    }
}

// ── Material properties ──────────────────────────────────────────────────────

TEST(InpParser, MaterialProperties) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.5, 0.866, 0.0
4, 0.5, 0.289, 0.816
*ELEMENT, TYPE=C3D4, ELSET=BODY
1, 1, 2, 3, 4
*MATERIAL, NAME=ALUMINUM
*ELASTIC
70000.0, 0.33
*DENSITY
2.7e-9
*EXPANSION
2.3e-5
*SOLID SECTION, ELSET=BODY, MATERIAL=ALUMINUM
)";
    Model m = InpParser::parse_string(inp);
    ASSERT_EQ(m.materials.size(), 1u);

    const auto& mat = m.materials.begin()->second;
    EXPECT_NEAR(mat.E, 70000.0, 1e-6);
    EXPECT_NEAR(mat.nu, 0.33, 1e-10);
    // G should be derived: E / (2*(1+nu))
    double G_expected = 70000.0 / (2.0 * 1.33);
    EXPECT_NEAR(mat.G, G_expected, 1e-3);
    EXPECT_NEAR(mat.rho, 2.7e-9, 1e-15);
    EXPECT_NEAR(mat.A, 2.3e-5, 1e-12);
}

// ── Solid section ────────────────────────────────────────────────────────────

TEST(InpParser, SolidSection) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
*ELEMENT, TYPE=C3D8, ELSET=BODY
1, 1, 2, 3, 4, 5, 6, 7, 8
*MATERIAL, NAME=STEEL
*ELASTIC
210000.0, 0.3
*SOLID SECTION, ELSET=BODY, MATERIAL=STEEL
)";
    Model m = InpParser::parse_string(inp);
    ASSERT_EQ(m.properties.size(), 1u);

    const auto& prop = m.properties.begin()->second;
    ASSERT_TRUE(std::holds_alternative<PSolid>(prop));
    const auto& ps = std::get<PSolid>(prop);
    EXPECT_EQ(ps.isop, SolidFormulation::EAS);

    // Element should reference this property
    EXPECT_EQ(m.elements[0].pid, ps.pid);
}

// ── Shell section ────────────────────────────────────────────────────────────

TEST(InpParser, ShellSection) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
*ELEMENT, TYPE=S4, ELSET=PLATE
1, 1, 2, 3, 4
*MATERIAL, NAME=STEEL
*ELASTIC
210000.0, 0.3
*SHELL SECTION, ELSET=PLATE, MATERIAL=STEEL
0.5
)";
    Model m = InpParser::parse_string(inp);
    ASSERT_EQ(m.properties.size(), 1u);

    const auto& prop = m.properties.begin()->second;
    ASSERT_TRUE(std::holds_alternative<PShell>(prop));
    const auto& ps = std::get<PShell>(prop);
    EXPECT_NEAR(ps.t, 0.5, 1e-10);

    // Element should reference this property
    EXPECT_EQ(m.elements[0].pid, ps.pid);
}

// ── Boundary conditions ──────────────────────────────────────────────────────

TEST(InpParser, BoundaryDofRange) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
*BOUNDARY
1, 1, 3
)";
    Model m = InpParser::parse_string(inp);
    ASSERT_EQ(m.spcs.size(), 1u);
    EXPECT_EQ(m.spcs[0].node.value, 1);
    EXPECT_TRUE(m.spcs[0].dofs.has(1));
    EXPECT_TRUE(m.spcs[0].dofs.has(2));
    EXPECT_TRUE(m.spcs[0].dofs.has(3));
    EXPECT_FALSE(m.spcs[0].dofs.has(4));
    EXPECT_DOUBLE_EQ(m.spcs[0].value, 0.0);
}

TEST(InpParser, BoundarySingleDof) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
*BOUNDARY
1, 2, 2
)";
    Model m = InpParser::parse_string(inp);
    ASSERT_EQ(m.spcs.size(), 1u);
    EXPECT_FALSE(m.spcs[0].dofs.has(1));
    EXPECT_TRUE(m.spcs[0].dofs.has(2));
    EXPECT_FALSE(m.spcs[0].dofs.has(3));
}

TEST(InpParser, BoundaryEnforcedDisplacement) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
*BOUNDARY
1, 1, 1, 0.5
)";
    Model m = InpParser::parse_string(inp);
    ASSERT_EQ(m.spcs.size(), 1u);
    EXPECT_DOUBLE_EQ(m.spcs[0].value, 0.5);
}

TEST(InpParser, BoundaryWithSetReference) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 2.0, 0.0, 0.0
*NSET, NSET=FIXED
1, 2, 3
*BOUNDARY
FIXED, 1, 6
)";
    Model m = InpParser::parse_string(inp);
    EXPECT_EQ(m.spcs.size(), 3u);
    for (const auto& spc : m.spcs) {
        for (int d = 1; d <= 6; ++d) {
            EXPECT_TRUE(spc.dofs.has(d));
        }
    }
}

// ── CLOAD ────────────────────────────────────────────────────────────────────

TEST(InpParser, CloadNodeId) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
*CLOAD
1, 3, -1000.0
)";
    Model m = InpParser::parse_string(inp);
    ASSERT_EQ(m.loads.size(), 1u);
    const auto& f = std::get<ForceLoad>(m.loads[0]);
    EXPECT_EQ(f.node.value, 1);
    EXPECT_DOUBLE_EQ(f.direction.x, 0.0);
    EXPECT_DOUBLE_EQ(f.direction.y, 0.0);
    EXPECT_DOUBLE_EQ(f.direction.z, -1000.0);
}

TEST(InpParser, CloadWithSet) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
*NSET, NSET=LOADED
1, 2
*CLOAD
LOADED, 1, 500.0
)";
    Model m = InpParser::parse_string(inp);
    ASSERT_EQ(m.loads.size(), 2u);
    for (const auto& load : m.loads) {
        const auto& f = std::get<ForceLoad>(load);
        EXPECT_DOUBLE_EQ(f.direction.x, 500.0);
    }
}

// ── Temperature ──────────────────────────────────────────────────────────────

TEST(InpParser, Temperature) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
*TEMPERATURE
1, 100.0
2, 200.0
)";
    Model m = InpParser::parse_string(inp);
    ASSERT_EQ(m.loads.size(), 2u);
    const auto& t0 = std::get<TempLoad>(m.loads[0]);
    const auto& t1 = std::get<TempLoad>(m.loads[1]);
    EXPECT_DOUBLE_EQ(t0.temperature, 100.0);
    EXPECT_DOUBLE_EQ(t1.temperature, 200.0);
}

// ── Step/subcase ─────────────────────────────────────────────────────────────

TEST(InpParser, StepSubcaseCreation) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
*STEP
*STATIC
*BOUNDARY
1, 1, 3
*CLOAD
1, 3, -100.0
*NODE FILE
U
*EL FILE
S
*NODE PRINT
U
*EL PRINT
S
*END STEP
)";
    Model m = InpParser::parse_string(inp);
    ASSERT_EQ(m.analysis.subcases.size(), 1u);
    const auto& sc = m.analysis.subcases[0];
    EXPECT_EQ(sc.id, 1);
    EXPECT_EQ(sc.load_set.value, 1);
    EXPECT_EQ(sc.spc_set.value, 1);
    EXPECT_TRUE(sc.disp_plot);
    EXPECT_TRUE(sc.disp_print);
    EXPECT_TRUE(sc.stress_plot);
    EXPECT_TRUE(sc.stress_print);
}

TEST(InpParser, DefaultSubcaseWhenNoStep) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
*BOUNDARY
1, 1, 3
*CLOAD
1, 3, -100.0
)";
    Model m = InpParser::parse_string(inp);
    ASSERT_EQ(m.analysis.subcases.size(), 1u);
    EXPECT_EQ(m.analysis.subcases[0].id, 1);
    EXPECT_EQ(m.analysis.subcases[0].load_set.value, 1);
    EXPECT_EQ(m.analysis.subcases[0].spc_set.value, 1);
}

// ── Comment skipping ─────────────────────────────────────────────────────────

TEST(InpParser, CommentSkipping) {
    const std::string inp = R"(
** This is a comment
*NODE
** Another comment in the middle
1, 0.0, 0.0, 0.0
** Comment between blocks
2, 1.0, 0.0, 0.0
)";
    Model m = InpParser::parse_string(inp);
    EXPECT_EQ(m.nodes.size(), 2u);
}

// ── Initial conditions ───────────────────────────────────────────────────────

TEST(InpParser, InitialConditionsTemperature) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
*INITIAL CONDITIONS, TYPE=TEMPERATURE
1, 20.0
)";
    Model m = InpParser::parse_string(inp);
    EXPECT_EQ(m.tempd.at(1), 20.0);
}

// ── Element with auto ELSET ─────────────────────────────────────────────────

TEST(InpParser, ElementAutoElset) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.5, 0.866, 0.0
4, 0.5, 0.289, 0.816
*ELEMENT, TYPE=C3D4, ELSET=BODY
1, 1, 2, 3, 4
2, 1, 2, 3, 4
*MATERIAL, NAME=STEEL
*ELASTIC
210000.0, 0.3
*SOLID SECTION, ELSET=BODY, MATERIAL=STEEL
)";
    Model m = InpParser::parse_string(inp);
    // Both elements should get pid from section
    EXPECT_NE(m.elements[0].pid.value, 0);
    EXPECT_NE(m.elements[1].pid.value, 0);
}

// ── Material reuse across sections ──────────────────────────────────────────

TEST(InpParser, MaterialReuse) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.5, 0.866, 0.0
4, 0.5, 0.289, 0.816
*ELEMENT, TYPE=C3D4, ELSET=GROUP1
1, 1, 2, 3, 4
*ELEMENT, TYPE=C3D4, ELSET=GROUP2
2, 1, 2, 3, 4
*MATERIAL, NAME=STEEL
*ELASTIC
210000.0, 0.3
*SOLID SECTION, ELSET=GROUP1, MATERIAL=STEEL
*SOLID SECTION, ELSET=GROUP2, MATERIAL=STEEL
)";
    Model m = InpParser::parse_string(inp);
    // Only one material should be created
    EXPECT_EQ(m.materials.size(), 1u);
    // But two properties
    EXPECT_EQ(m.properties.size(), 2u);
    // Both properties should reference the same material
    const auto& p1 = std::get<PSolid>(m.properties.at(m.elements[0].pid));
    const auto& p2 = std::get<PSolid>(m.properties.at(m.elements[1].pid));
    EXPECT_EQ(p1.mid, p2.mid);
}

// ── Reduced element types (R variants) ──────────────────────────────────────

TEST(InpParser, ReducedIntegrationVariants) {
    // C3D8R should map to CHEXA8, S4R to CQUAD4, S3R to CTRIA3
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
*ELEMENT, TYPE=C3D8R
1, 1, 2, 3, 4, 5, 6, 7, 8
*ELEMENT, TYPE=S4R
2, 1, 2, 3, 4
*ELEMENT, TYPE=S3R
3, 1, 2, 3
)";
    Model m = InpParser::parse_string(inp);
    ASSERT_EQ(m.elements.size(), 3u);
    EXPECT_EQ(m.elements[0].type, ElementType::CHEXA8);
    EXPECT_EQ(m.elements[1].type, ElementType::CQUAD4);
    EXPECT_EQ(m.elements[2].type, ElementType::CTRIA3);
}

// ── Full model parse — validates the complete model ─────────────────────────

TEST(InpParser, FullModelValidation) {
    const std::string inp = R"(
** Simple cantilever beam with one C3D8 element
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
**
*ELEMENT, TYPE=C3D8, ELSET=BODY
1, 1, 2, 3, 4, 5, 6, 7, 8
**
*MATERIAL, NAME=STEEL
*ELASTIC
210000.0, 0.3
*DENSITY
7.85e-9
**
*SOLID SECTION, ELSET=BODY, MATERIAL=STEEL
**
*NSET, NSET=FIX
1, 4, 5, 8
**
*STEP
*STATIC
**
*BOUNDARY
FIX, 1, 3
**
*CLOAD
2, 3, -100.0
3, 3, -100.0
6, 3, -100.0
7, 3, -100.0
**
*NODE FILE
U
*NODE PRINT
U
*EL FILE
S
*EL PRINT
S
**
*END STEP
)";
    Model m = InpParser::parse_string(inp);

    // Structural checks
    EXPECT_EQ(m.nodes.size(), 8u);
    EXPECT_EQ(m.elements.size(), 1u);
    EXPECT_EQ(m.materials.size(), 1u);
    EXPECT_EQ(m.properties.size(), 1u);
    EXPECT_EQ(m.spcs.size(), 4u);
    EXPECT_EQ(m.loads.size(), 4u);
    EXPECT_EQ(m.analysis.subcases.size(), 1u);

    // Model should pass validation
    EXPECT_NO_THROW(m.validate());
}

// ── BDF/INP equivalence — same problem solved both ways ─────────────────────

TEST(InpParser, BdfInpEquivalence) {
    // A single C3D8 element cantilever: 4 nodes fixed, 4 loaded in Z
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  DISPLACEMENT(PRINT,PLOT) = ALL
  STRESS(PRINT,PLOT) = ALL
BEGIN BULK
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,1.0,1.0,0.0
GRID,4,,0.0,1.0,0.0
GRID,5,,0.0,0.0,1.0
GRID,6,,1.0,0.0,1.0
GRID,7,,1.0,1.0,1.0
GRID,8,,0.0,1.0,1.0
MAT1,1,210000.0,,0.3
PSOLID,1,1
CHEXA,1,1,1,2,3,4,5,6
+,7,8
SPC1,1,123,1,4,5,8
FORCE,1,2,0,1.0,0.0,0.0,-100.0
FORCE,1,3,0,1.0,0.0,0.0,-100.0
FORCE,1,6,0,1.0,0.0,0.0,-100.0
FORCE,1,7,0,1.0,0.0,0.0,-100.0
ENDDATA
)";

    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
*ELEMENT, TYPE=C3D8, ELSET=BODY
1, 1, 2, 3, 4, 5, 6, 7, 8
*NSET, NSET=FIX
1, 4, 5, 8
*MATERIAL, NAME=STEEL
*ELASTIC
210000.0, 0.3
*SOLID SECTION, ELSET=BODY, MATERIAL=STEEL
*STEP
*STATIC
*BOUNDARY
FIX, 1, 3
*CLOAD
2, 3, -100.0
3, 3, -100.0
6, 3, -100.0
7, 3, -100.0
*NODE FILE
U
*NODE PRINT
U
*EL FILE
S
*EL PRINT
S
*END STEP
)";

    Model bdf_model = BdfParser::parse_string(bdf);
    Model inp_model = InpParser::parse_string(inp);

    // Both models should validate
    EXPECT_NO_THROW(bdf_model.validate());
    EXPECT_NO_THROW(inp_model.validate());

    // Solve both
    auto bdf_backend = std::make_unique<EigenSolverBackend>();
    LinearStaticSolver bdf_solver(std::move(bdf_backend));
    SolverResults bdf_results = bdf_solver.solve(bdf_model);

    auto inp_backend = std::make_unique<EigenSolverBackend>();
    LinearStaticSolver inp_solver(std::move(inp_backend));
    SolverResults inp_results = inp_solver.solve(inp_model);

    // Compare displacements
    ASSERT_EQ(bdf_results.subcases.size(), 1u);
    ASSERT_EQ(inp_results.subcases.size(), 1u);

    const auto& bdf_disp = bdf_results.subcases[0].displacements;
    const auto& inp_disp = inp_results.subcases[0].displacements;
    ASSERT_EQ(bdf_disp.size(), inp_disp.size());

    // Build lookup for BDF displacements by node ID
    std::unordered_map<int, std::array<double, 6>> bdf_disp_map;
    for (const auto& d : bdf_disp) {
        bdf_disp_map[d.node.value] = d.d;
    }

    for (const auto& d : inp_disp) {
        auto it = bdf_disp_map.find(d.node.value);
        ASSERT_NE(it, bdf_disp_map.end()) << "Node " << d.node.value << " missing in BDF results";
        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR(d.d[i], it->second[i], 1e-10)
                << "Node " << d.node.value << " DOF " << (i+1) << " mismatch";
        }
    }
}

// ── Boundary with omitted last DOF (same as first DOF) ──────────────────────

TEST(InpParser, BoundaryOmittedLastDof) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
*BOUNDARY
1, 3
)";
    Model m = InpParser::parse_string(inp);
    ASSERT_EQ(m.spcs.size(), 1u);
    EXPECT_FALSE(m.spcs[0].dofs.has(1));
    EXPECT_FALSE(m.spcs[0].dofs.has(2));
    EXPECT_TRUE(m.spcs[0].dofs.has(3));
}

// ── Multiple NSET blocks with same name append ──────────────────────────────

TEST(InpParser, NsetAppend) {
    const std::string inp = R"(
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 2.0, 0.0, 0.0
*NSET, NSET=ALL
1, 2
*NSET, NSET=ALL
3
*BOUNDARY
ALL, 1, 3
)";
    Model m = InpParser::parse_string(inp);
    // 3 nodes in set, each gets an SPC
    EXPECT_EQ(m.spcs.size(), 3u);
}
