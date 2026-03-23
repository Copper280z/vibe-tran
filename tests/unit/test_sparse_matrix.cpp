// tests/unit/test_sparse_matrix.cpp
// Tests for SparseMatrixBuilder: verifies that add() and triplets() accumulate
// entries correctly, and that build_csr() produces the expected structure.
// Also verifies the EigenSolverBackend::name() identification string.

#include <gtest/gtest.h>
#include "core/sparse_matrix.hpp"
#include "solver/solver_backend.hpp"

using namespace vibestran;

// ── SparseMatrixBuilder::add and triplets (only used in tests) ────────────────

TEST(SparseMatrixBuilder, AddSingleTriplet) {
    SparseMatrixBuilder builder(3);
    builder.add(0, 0, 5.0);
    const auto& t = builder.triplets();
    ASSERT_EQ(t.size(), 1u);
    EXPECT_EQ(t[0].row, 0);
    EXPECT_EQ(t[0].col, 0);
    EXPECT_DOUBLE_EQ(t[0].value, 5.0);
}

TEST(SparseMatrixBuilder, AddMultipleTriplets) {
    SparseMatrixBuilder builder(4);
    builder.add(0, 0, 1.0);
    builder.add(1, 1, 2.0);
    builder.add(2, 2, 3.0);
    builder.add(0, 2, -0.5);
    const auto& t = builder.triplets();
    EXPECT_EQ(t.size(), 4u);
}

TEST(SparseMatrixBuilder, TripletsAccumulateDuplicates) {
    // Duplicate entries in triplets are summed on build_csr(),
    // but triplets() should return them as-is (pre-finalization).
    SparseMatrixBuilder builder(2);
    builder.add(0, 0, 3.0);
    builder.add(0, 0, 4.0);
    const auto& t = builder.triplets();
    ASSERT_EQ(t.size(), 2u);
    // Both entries present; their sum (7.0) will appear in the assembled matrix
    double sum = t[0].value + t[1].value;
    EXPECT_DOUBLE_EQ(sum, 7.0);
}

TEST(SparseMatrixBuilder, BuildCsrFromAddEntriesCorrect) {
    // 2×2 diagonal matrix: K = diag(3, 5)
    SparseMatrixBuilder builder(2);
    builder.add(0, 0, 3.0);
    builder.add(1, 1, 5.0);
    auto csr = builder.build_csr();
    EXPECT_EQ(csr.n, 2);
    EXPECT_EQ(csr.nnz, 2);
    ASSERT_EQ(csr.row_ptr.size(), 3u);
    EXPECT_EQ(csr.row_ptr[0], 0);
    EXPECT_EQ(csr.row_ptr[1], 1);
    EXPECT_EQ(csr.row_ptr[2], 2);
}

TEST(SparseMatrixBuilder, SizeReturnsConstructorArg) {
    SparseMatrixBuilder builder(42);
    EXPECT_EQ(builder.size(), 42);
}

// ── EigenSolverBackend::name (only used in tests) ─────────────────────────────

TEST(EigenSolverBackend, NameIsNonEmpty) {
    EigenSolverBackend backend;
    EXPECT_FALSE(backend.name().empty());
}

TEST(EigenSolverBackend, NameIdentifiesEigenBackend) {
    EigenSolverBackend backend;
    // The name must identify the underlying factorization so log output is
    // meaningful.  When SuiteSparse is available the name is "SuiteSparse
    // CHOLMOD (CPU)"; otherwise "Eigen SimplicialLLT (CPU)".
    const auto n = backend.name();
    const bool is_cholmod = n.find("CHOLMOD") != std::string_view::npos;
    const bool is_eigen   = n.find("Eigen")   != std::string_view::npos;
    EXPECT_TRUE(is_cholmod || is_eigen) << "Unexpected backend name: " << n;
}
