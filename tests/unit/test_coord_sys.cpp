// tests/unit/test_coord_sys.cpp
// Unit tests for coordinate system transforms.

#include "core/coord_sys.hpp"
#include "io/bdf_parser.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

using namespace vibestran;

static constexpr double pi = std::numbers::pi;

// ── Helper: verify a Vec3 is approximately equal ─────────────────────────────
static void expect_near_vec3(const Vec3& actual, const Vec3& expected,
                              double tol = 1e-10) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

// ── Test 1: Basic (CoordId{0}) is identity ───────────────────────────────────
TEST(CoordSys, BasicIsIdentity) {
    // Basic CS: rectangular, origin at (0,0,0), aligned with global axes.
    CoordSys cs;
    cs.id   = CoordId{0};
    cs.type = CoordType::Rectangular;
    // axes = identity (default)
    Vec3 p{3.0, 4.0, 5.0};
    Vec3 result = to_basic(cs, p);
    expect_near_vec3(result, p);
}

// ── Test 2: CORD2R rotated 45° about Z ───────────────────────────────────────
TEST(CoordSys, CORD2R_Rotated45) {
    // CS rotated 45° about Z: origin at (0,0,0)
    // A=(0,0,0), B=(0,0,1), C=(cos45, sin45, 0) in basic
    CoordSys cs;
    cs.id   = CoordId{1};
    cs.type = CoordType::Rectangular;
    double c = std::cos(pi / 4), s = std::sin(pi / 4);
    build_axes(cs,
               Vec3{0, 0, 0},   // A = origin
               Vec3{0, 0, 1},   // B = +Z axis
               Vec3{c, s, 0}    // C = in XZ plane (rotated 45 about Z from X)
    );
    // In this CS, local X = (cos45, sin45, 0) in basic
    //                local Y = (-sin45, cos45, 0) in basic
    //                local Z = (0, 0, 1) in basic

    // Point (1, 0, 0) in local CS should map to (cos45, sin45, 0) in basic
    Vec3 p_local{1.0, 0.0, 0.0};
    Vec3 p_basic = to_basic(cs, p_local);
    expect_near_vec3(p_basic, Vec3{c, s, 0.0});

    // Point (0, 1, 0) in local CS → (-sin45, cos45, 0) in basic
    Vec3 p2 = to_basic(cs, Vec3{0.0, 1.0, 0.0});
    expect_near_vec3(p2, Vec3{-s, c, 0.0});
}

// ── Test 3: CORD2C unit cylinder ─────────────────────────────────────────────
TEST(CoordSys, CORD2C_UnitCylinder) {
    // Cylindrical CS with identity orientation (origin at (0,0,0), axes=identity)
    CoordSys cs;
    cs.id   = CoordId{2};
    cs.type = CoordType::Cylindrical;
    build_axes(cs, Vec3{0,0,0}, Vec3{0,0,1}, Vec3{1,0,0});
    // axes: eX = (1,0,0), eY = (0,1,0), eZ = (0,0,1) — identity

    // Node at (r=2, θ=π/4, z=1) → (√2, √2, 1) in basic
    Vec3 p_local{2.0, 45.0, 1.0}; // Nastran uses degrees
    Vec3 p_basic = to_basic(cs, p_local);
    double sq2 = std::sqrt(2.0);
    expect_near_vec3(p_basic, Vec3{sq2, sq2, 1.0}, 1e-10);
}

// ── Test 4: CORD2S unit sphere ────────────────────────────────────────────────
TEST(CoordSys, CORD2S_UnitSphere) {
    // Spherical CS with identity orientation
    CoordSys cs;
    cs.id   = CoordId{3};
    cs.type = CoordType::Spherical;
    build_axes(cs, Vec3{0,0,0}, Vec3{0,0,1}, Vec3{1,0,0});

    // Point (ρ=1, θ=90°, φ=90°) → (0, 1, 0) in basic
    // Nastran spherical: phi = polar angle from Z, theta = azimuth
    // p = (rho*sin(phi)*cos(theta), rho*sin(phi)*sin(theta), rho*cos(phi))
    //   = (1*sin(90)*cos(90), 1*sin(90)*sin(90), 1*cos(90))
    //   = (0, 1, 0)
    Vec3 p_local{1.0, 90.0, 90.0};
    Vec3 p_basic = to_basic(cs, p_local);
    expect_near_vec3(p_basic, Vec3{0.0, 1.0, 0.0}, 1e-10);
}

// ── Test 5: Cylindrical rotation matrix at θ=π/2 ─────────────────────────────
TEST(CoordSys, CORD2C_RotationMatrix) {
    // Cylindrical CS (identity orientation)
    CoordSys cs;
    cs.id   = CoordId{4};
    cs.type = CoordType::Cylindrical;
    build_axes(cs, Vec3{0,0,0}, Vec3{0,0,1}, Vec3{1,0,0});

    // At θ=π/2 (i.e., basic position (0, r, 0)):
    // e_r should point in +Y direction of basic: e_r = (0,1,0)
    // e_θ should point in -X direction of basic: e_θ = (-1,0,0)
    Vec3 basic_pos{0.0, 1.0, 0.0}; // r=1, θ=90°, z=0
    Mat3 T3 = rotation_matrix(cs, basic_pos);

    // T3[:,0] = e_r in basic = (0,1,0)
    EXPECT_NEAR(T3(0, 0), 0.0, 1e-10);  // e_r · e_X_basic
    EXPECT_NEAR(T3(1, 0), 1.0, 1e-10);  // e_r · e_Y_basic
    EXPECT_NEAR(T3(2, 0), 0.0, 1e-10);

    // T3[:,1] = e_θ in basic = (-1,0,0)
    EXPECT_NEAR(T3(0, 1), -1.0, 1e-10); // e_θ · e_X_basic
    EXPECT_NEAR(T3(1, 1), 0.0, 1e-10);
    EXPECT_NEAR(T3(2, 1), 0.0, 1e-10);

    // T3[:,2] = e_z in basic = (0,0,1)
    EXPECT_NEAR(T3(0, 2), 0.0, 1e-10);
    EXPECT_NEAR(T3(1, 2), 0.0, 1e-10);
    EXPECT_NEAR(T3(2, 2), 1.0, 1e-10);
}

// ── Test 6: Chained CORD2R ────────────────────────────────────────────────────
TEST(CoordSys, CORD2R_Chained) {
    // CS 1: origin (1,0,0), Z along +Z, rotated identity
    CoordSys cs1;
    cs1.id   = CoordId{1};
    cs1.type = CoordType::Rectangular;
    build_axes(cs1, Vec3{1,0,0}, Vec3{1,0,1}, Vec3{2,0,0});
    // Z = (0,0,1), X = (1,0,0), Y = (0,1,0) (standard, shifted origin)

    // CS 2 defined relative to CS 1:
    // origin = to_basic(cs1, (1,0,0)) = (2,0,0)
    // B = to_basic(cs1, (1,0,1)) = (2,0,1)
    // C = to_basic(cs1, (2,0,0)) = (3,0,0)
    CoordSys cs2;
    cs2.id   = CoordId{2};
    cs2.type = CoordType::Rectangular;
    // Resolve chained: transform cs2 defining points via cs1
    Vec3 a2_basic = to_basic(cs1, Vec3{1,0,0});
    Vec3 b2_basic = to_basic(cs1, Vec3{1,0,1});
    Vec3 c2_basic = to_basic(cs1, Vec3{2,0,0});
    build_axes(cs2, a2_basic, b2_basic, c2_basic);

    // A point (1,0,0) in CS 2 local = CS 1 local = basic+offset
    // to_basic(cs2, (1,0,0)) = cs2.origin + cs2.axes[0]*1 = (2,0,0) + (1,0,0) = (3,0,0)
    Vec3 p_cs2{1.0, 0.0, 0.0};
    Vec3 result = to_basic(cs2, p_cs2);
    expect_near_vec3(result, Vec3{3.0, 0.0, 0.0}, 1e-10);
}

// ── Test 7: BdfParser CORD2R ──────────────────────────────────────────────────
TEST(CoordSys, BdfParser_CORD2R) {
    const std::string bdf = R"(
BEGIN BULK
$ CORD2R, CID, RID, A1, A2, A3, B1, B2, B3, C1, C2, C3
CORD2R,  1,   0,  0.0, 0.0, 0.0,  0.0, 0.0, 1.0,  1.0, 0.0, 0.0
ENDDATA
)";
    Model model = BdfParser::parse_string(bdf);
    ASSERT_EQ(model.coord_systems.size(), 1u);
    auto it = model.coord_systems.find(CoordId{1});
    ASSERT_NE(it, model.coord_systems.end());
    const CoordSys& cs = it->second;
    EXPECT_EQ(cs.type, CoordType::Rectangular);
    // Origin A=(0,0,0), B=(0,0,1), C=(1,0,0)
    // Z = B-A normalized = (0,0,1)
    // temp_X = C-A normalized = (1,0,0)
    // Y = Z × X = (0,0,1) × (1,0,0) = (0,-0,...)
    // Actually: Y = ez × (C-A)/|(C-A)| = (0,0,1) × (1,0,0) = (0*0-1*0, 1*1-0*0, 0*0-0*1) = (0,1,0)
    // X = Y × Z = (0,1,0) × (0,0,1) = (1,0,0)
    EXPECT_NEAR(cs.origin.x, 0.0, 1e-12);
    EXPECT_NEAR(cs.origin.y, 0.0, 1e-12);
    EXPECT_NEAR(cs.origin.z, 0.0, 1e-12);
    EXPECT_NEAR(cs.axes[2].z, 1.0, 1e-12); // eZ = (0,0,1)
}

// ── Test 8: BdfParser CORD2C with non-trivial origin ─────────────────────────
TEST(CoordSys, BdfParser_CORD2C) {
    const std::string bdf = R"(
BEGIN BULK
CORD2C,  10,  0,  1.0, 0.0, 0.0,  1.0, 0.0, 1.0,  2.0, 0.0, 0.0
ENDDATA
)";
    Model model = BdfParser::parse_string(bdf);
    ASSERT_EQ(model.coord_systems.size(), 1u);
    auto it = model.coord_systems.find(CoordId{10});
    ASSERT_NE(it, model.coord_systems.end());
    const CoordSys& cs = it->second;
    EXPECT_EQ(cs.type, CoordType::Cylindrical);
    // Origin A=(1,0,0)
    EXPECT_NEAR(cs.origin.x, 1.0, 1e-12);
    EXPECT_NEAR(cs.origin.y, 0.0, 1e-12);
    EXPECT_NEAR(cs.origin.z, 0.0, 1e-12);
}

// ── Test 9: BdfParser CORD1R (defined by node IDs) ───────────────────────────
TEST(CoordSys, BdfParser_CORD1R) {
    const std::string bdf = R"(
BEGIN BULK
$ Define CS 5 by three nodes in basic Cartesian
GRID,  100,, 0.0, 0.0, 0.0
GRID,  101,, 0.0, 0.0, 1.0
GRID,  102,, 1.0, 0.0, 0.0
CORD1R,  5,  100,  101,  102
ENDDATA
)";
    Model model = BdfParser::parse_string(bdf);
    ASSERT_EQ(model.coord_systems.size(), 1u);
    auto it = model.coord_systems.find(CoordId{5});
    ASSERT_NE(it, model.coord_systems.end());
    const CoordSys& cs = it->second;
    EXPECT_EQ(cs.type, CoordType::Rectangular);
    // Origin = node 100 = (0,0,0)
    EXPECT_NEAR(cs.origin.x, 0.0, 1e-12);
    EXPECT_NEAR(cs.origin.z, 0.0, 1e-12);
    // Z axis = node101-node100 normalized = (0,0,1)
    EXPECT_NEAR(cs.axes[2].z, 1.0, 1e-10);
}

// ── Test 10: Chained CORD2R via BdfParser ────────────────────────────────────
TEST(CoordSys, BdfParser_ChainedCords) {
    const std::string bdf = R"(
BEGIN BULK
$ CS 1: identity, origin at (0,0,0)
CORD2R,  1,  0,  0.0, 0.0, 0.0,  0.0, 0.0, 1.0,  1.0, 0.0, 0.0
$ CS 2: relative to CS 1, origin at (1,0,0) in CS 1 → (1,0,0) in basic
CORD2R,  2,  1,  1.0, 0.0, 0.0,  1.0, 0.0, 1.0,  2.0, 0.0, 0.0
$ Node in CS 2 at (2,0,0): to_basic(cs2, (2,0,0)) = cs2.origin + cs2.axes[0]*2 = (1+2,0,0) = (3,0,0)
GRID,  1,  2,  2.0, 0.0, 0.0
ENDDATA
)";
    Model model = BdfParser::parse_string(bdf);
    ASSERT_EQ(model.coord_systems.size(), 2u);
    ASSERT_EQ(model.nodes.size(), 1u);

    // After resolve_coordinates(), the node position should be (3,0,0) in basic
    const GridPoint& gp = model.nodes.at(NodeId{1});
    EXPECT_EQ(gp.cp, CoordId{0}); // resolved to basic
    EXPECT_NEAR(gp.position.x, 3.0, 1e-10);
    EXPECT_NEAR(gp.position.y, 0.0, 1e-10);
    EXPECT_NEAR(gp.position.z, 0.0, 1e-10);
}
