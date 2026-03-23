// tests/unit/test_types.cpp
// Tests for fundamental types: Vec3, DofSet, CoordId.
// Validates mathematical correctness of the vector operations and
// DOF set encoding that underpins all element assembly.

#include <gtest/gtest.h>
#include "core/types.hpp"
#include <cmath>

using namespace vibestran;

// ── CoordId::basic (only used in tests) ──────────────────────────────────────

TEST(CoordId, BasicIsZero) {
    EXPECT_EQ(CoordId::basic().value, 0);
}

TEST(CoordId, BasicEquality) {
    EXPECT_EQ(CoordId::basic(), CoordId{0});
}

// ── DofSet (only used in tests) ───────────────────────────────────────────────

TEST(DofSet, FromNastranStringSingleDof) {
    DofSet ds = DofSet::from_nastran_string("1");
    EXPECT_TRUE(ds.has(DofComponent::T1));
    EXPECT_FALSE(ds.has(DofComponent::T2));
}

TEST(DofSet, FromNastranStringMultipleDofs) {
    DofSet ds = DofSet::from_nastran_string("123456");
    for (int d = 1; d <= 6; ++d)
        EXPECT_TRUE(ds.has(d)) << "DOF " << d << " should be set";
}

TEST(DofSet, FromNastranStringTranslationsOnly) {
    DofSet ds = DofSet::from_nastran_string("123");
    EXPECT_TRUE(ds.has(DofComponent::T1));
    EXPECT_TRUE(ds.has(DofComponent::T2));
    EXPECT_TRUE(ds.has(DofComponent::T3));
    EXPECT_FALSE(ds.has(DofComponent::R1));
    EXPECT_FALSE(ds.has(DofComponent::R2));
    EXPECT_FALSE(ds.has(DofComponent::R3));
}

TEST(DofSet, FromNastranStringInvalidCharThrows) {
    EXPECT_THROW(DofSet::from_nastran_string("7"), std::invalid_argument);
    EXPECT_THROW(DofSet::from_nastran_string("0"), std::invalid_argument);
}

TEST(DofSet, AllContainsAllSixDofs) {
    DofSet ds = DofSet::all();
    for (int d = 1; d <= 6; ++d)
        EXPECT_TRUE(ds.has(d)) << "DofSet::all() missing DOF " << d;
}

TEST(DofSet, NoneContainsNoDofs) {
    DofSet ds = DofSet::none();
    for (int d = 1; d <= 6; ++d)
        EXPECT_FALSE(ds.has(d)) << "DofSet::none() has unexpected DOF " << d;
}

TEST(DofSet, TranslationsContainsOnlyT1T2T3) {
    DofSet ds = DofSet::translations();
    EXPECT_TRUE(ds.has(DofComponent::T1));
    EXPECT_TRUE(ds.has(DofComponent::T2));
    EXPECT_TRUE(ds.has(DofComponent::T3));
    EXPECT_FALSE(ds.has(DofComponent::R1));
    EXPECT_FALSE(ds.has(DofComponent::R2));
    EXPECT_FALSE(ds.has(DofComponent::R3));
}

TEST(DofSet, AllAndNoneAreComplementary) {
    DofSet all = DofSet::all();
    DofSet none = DofSet::none();
    EXPECT_NE(all.mask, none.mask);
    EXPECT_EQ(all.mask, static_cast<uint8_t>(0x3F));
    EXPECT_EQ(none.mask, static_cast<uint8_t>(0));
}

// ── Vec3 (only used in tests) ─────────────────────────────────────────────────

TEST(Vec3, DotProductOrthogonalVectors) {
    Vec3 x{1, 0, 0};
    Vec3 y{0, 1, 0};
    EXPECT_DOUBLE_EQ(x.dot(y), 0.0);
}

TEST(Vec3, DotProductParallelVectors) {
    Vec3 a{2, 3, 4};
    EXPECT_DOUBLE_EQ(a.dot(a), 2*2 + 3*3 + 4*4);
}

TEST(Vec3, DotProductMixed) {
    Vec3 a{1, 2, 3};
    Vec3 b{4, 5, 6};
    EXPECT_DOUBLE_EQ(a.dot(b), 1*4 + 2*5 + 3*6); // 32
}

TEST(Vec3, CrossProductBasisVectors) {
    Vec3 x{1, 0, 0};
    Vec3 y{0, 1, 0};
    Vec3 z = x.cross(y);
    EXPECT_DOUBLE_EQ(z.x, 0.0);
    EXPECT_DOUBLE_EQ(z.y, 0.0);
    EXPECT_DOUBLE_EQ(z.z, 1.0);
}

TEST(Vec3, CrossProductAntiCommutative) {
    Vec3 a{1, 2, 3};
    Vec3 b{4, 5, 6};
    Vec3 axb = a.cross(b);
    Vec3 bxa = b.cross(a);
    EXPECT_DOUBLE_EQ(axb.x, -bxa.x);
    EXPECT_DOUBLE_EQ(axb.y, -bxa.y);
    EXPECT_DOUBLE_EQ(axb.z, -bxa.z);
}

TEST(Vec3, CrossProductParallelVectorsIsZero) {
    Vec3 a{1, 2, 3};
    Vec3 b{2, 4, 6}; // b = 2*a
    Vec3 c = a.cross(b);
    EXPECT_NEAR(c.x, 0.0, 1e-14);
    EXPECT_NEAR(c.y, 0.0, 1e-14);
    EXPECT_NEAR(c.z, 0.0, 1e-14);
}

TEST(Vec3, NormalizedUnitLength) {
    Vec3 v{3, 4, 0};
    Vec3 n = v.normalized();
    EXPECT_NEAR(n.norm(), 1.0, 1e-14);
    EXPECT_NEAR(n.x, 0.6, 1e-14);
    EXPECT_NEAR(n.y, 0.8, 1e-14);
    EXPECT_NEAR(n.z, 0.0, 1e-14);
}

TEST(Vec3, NormalizedArbitraryVector) {
    Vec3 v{1, 1, 1};
    Vec3 n = v.normalized();
    EXPECT_NEAR(n.norm(), 1.0, 1e-14);
    double expected = 1.0 / std::sqrt(3.0);
    EXPECT_NEAR(n.x, expected, 1e-14);
    EXPECT_NEAR(n.y, expected, 1e-14);
    EXPECT_NEAR(n.z, expected, 1e-14);
}

TEST(Vec3, NormalizedZeroVectorThrows) {
    Vec3 zero{0, 0, 0};
    // Cast to void to satisfy [[nodiscard]] while still testing the throw
    EXPECT_THROW((void)zero.normalized(), std::runtime_error);
}

TEST(Vec3, DotAndCrossConsistency) {
    // a × b should be perpendicular to both a and b
    Vec3 a{1, 2, 3};
    Vec3 b{4, -5, 6};
    Vec3 c = a.cross(b);
    EXPECT_NEAR(c.dot(a), 0.0, 1e-12);
    EXPECT_NEAR(c.dot(b), 0.0, 1e-12);
}
