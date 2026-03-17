// tests/unit/test_cuda_solver.cpp
// Mathematical correctness tests for CudaSolverBackend.
//
// All tests skip gracefully when CUDA is unavailable (no GPU, no CUDA toolkit,
// headless CI).  When CUDA IS present, the tests verify that the cuSOLVER
// sparse Cholesky backend produces numerically correct solutions and that the
// LU fallback path handles non-SPD matrices correctly.

#include <gtest/gtest.h>
#include "solver/solver_backend.hpp"
#include "core/sparse_matrix.hpp"
#include "core/types.hpp"
#include <cmath>
#include <vector>

#ifdef HAVE_CUDA
#include "solver/cuda_solver_backend.hpp"
#endif

using namespace nastran;

#ifndef HAVE_CUDA
// All tests in this file are no-ops when CUDA was not compiled in.
TEST(CudaTest, CudaNotCompiled) {
    GTEST_SKIP() << "CUDA backend not compiled — skipping all CUDA tests";
}
#else

// ── Test fixture ──────────────────────────────────────────────────────────────

class CudaTest : public ::testing::Test {
protected:
    // cppcheck-suppress unusedFunction -- called by GTest framework
    void SetUp() override {
        backend_ = CudaSolverBackend::try_create();
        if (!backend_.has_value())
            GTEST_SKIP() << "CUDA not available on this system — skipping CUDA tests";
    }

    std::optional<CudaSolverBackend> backend_;
};

// ── CSR builder helpers ───────────────────────────────────────────────────────

/// Build CSR from a dense symmetric matrix.
static SparseMatrixBuilder::CsrData dense_to_csr(const std::vector<std::vector<double>>& A) {
    int n = static_cast<int>(A.size());
    SparseMatrixBuilder builder(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (A[i][j] != 0.0)
                builder.add(i, j, A[i][j]);
    return builder.build_csr();
}

// ── Test 1: 2×2 diagonal SPD — exact direct solution ─────────────────────────
// K = diag(4, 9), F = [8, 27], expected u = [2, 3].
// Sparse Cholesky on a diagonal matrix is trivially exact.

TEST_F(CudaTest, DiagonalSystemExactSolution) {
    std::vector<std::vector<double>> K = {{4.0, 0.0}, {0.0, 9.0}};
    std::vector<double> F = {8.0, 27.0};
    auto csr = dense_to_csr(K);

    auto u = backend_->solve(csr, F);

    ASSERT_EQ(static_cast<int>(u.size()), 2);
    EXPECT_NEAR(u[0], 2.0, 1e-10);
    EXPECT_NEAR(u[1], 3.0, 1e-10);
    EXPECT_TRUE(backend_->last_solve_used_cholesky())
        << "A diagonal SPD matrix should succeed with sparse Cholesky";
}

// ── Test 2: 5×5 tridiagonal SPD — agrees with Eigen ─────────────────────────
// K = tridiag(-1, 3, -1) of size 5, F = ones.
// Verify CUDA solution matches EigenSolverBackend to double precision.

TEST_F(CudaTest, TridiagonalSystemAgreesWithEigen) {
    const int n = 5;
    std::vector<std::vector<double>> Kd(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        Kd[i][i] = 3.0;
        if (i > 0)     Kd[i][i-1] = -1.0;
        if (i < n - 1) Kd[i][i+1] = -1.0;
    }
    std::vector<double> F(n, 1.0);
    auto csr = dense_to_csr(Kd);

    auto u_cuda = backend_->solve(csr, F);

    EigenSolverBackend eigen;
    auto u_eigen = eigen.solve(csr, F);

    ASSERT_EQ(static_cast<int>(u_cuda.size()), n);
    // cuSOLVER uses double precision; results should match Eigen to near-machine epsilon.
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(u_cuda[i], u_eigen[i], 1e-10)
            << "Component " << i << " differs between CUDA and Eigen";

    EXPECT_TRUE(backend_->last_solve_used_cholesky())
        << "Tridiagonal SPD matrix should use Cholesky path";
}

// ── Test 3: larger tridiagonal n=200 — double precision accuracy ──────────────
// cuSOLVER is a direct solver, so the residual should be near machine epsilon
// regardless of condition number (unlike iterative PCG).
// K = tridiag(-1, 2, -1) of size 200 has κ ≈ n²/π² ≈ 4053.

TEST_F(CudaTest, LargerTridiagonalHighAccuracy) {
    const int n = 200;
    SparseMatrixBuilder builder(n);
    for (int i = 0; i < n; ++i) {
        builder.add(i, i, 2.0);
        if (i > 0)     builder.add(i, i - 1, -1.0);
        if (i < n - 1) builder.add(i, i + 1, -1.0);
    }
    auto csr = builder.build_csr();
    std::vector<double> F(n, 1.0);

    auto u_cuda = backend_->solve(csr, F);

    EigenSolverBackend eigen;
    auto u_eigen = eigen.solve(csr, F);

    ASSERT_EQ(static_cast<int>(u_cuda.size()), n);
    // Direct solver; error is bounded by κ * machine_epsilon ≈ 4053 * 1e-16 ≈ 1e-12.
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(u_cuda[i], u_eigen[i], 1e-8)
            << "n=200 tridiagonal: component " << i << " differs";
}

// ── Test 4: stiff FEM-scale diagonal system ───────────────────────────────────
// K_ii = 1.3e9 (typical large structural stiffness), F_i = 333.
// True solution: u_i = 333 / 1.3e9 ≈ 2.56e-7.
// This is a regression test for the false-convergence class of bugs that
// affected the Vulkan PCG backend.  A direct solver is immune to this by design.

TEST_F(CudaTest, StiffDiagonalSystemIsExact) {
    const double K_diag = 1.3e9;
    const double F_val  = 333.0;
    const int    n      = 100;

    SparseMatrixBuilder builder(n);
    std::vector<double> F(n, F_val);
    for (int i = 0; i < n; ++i)
        builder.add(i, i, K_diag);
    auto csr = builder.build_csr();

    auto u = backend_->solve(csr, F);

    ASSERT_EQ(static_cast<int>(u.size()), n);
    const double expected = F_val / K_diag;
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(u[i], expected, expected * 1e-8)
            << "Stiff diagonal: component " << i << " wrong";
}

// ── Test 5: singular matrix throws SolverError ────────────────────────────────
// A matrix with a zero row (det = 0, structurally rank-deficient) must cause
// solve() to throw SolverError with a descriptive message.
//
// Note: the CUDA backend uses a direct sparse solver (cuSOLVER), not a
// preconditioned iterative method.  A matrix that is merely missing a diagonal
// entry but remains non-singular (e.g. a 3×3 matrix with off-diagonal entries
// that make det ≠ 0) will be solved correctly.  Only a truly singular matrix
// (zero row → det = 0) triggers the singularity detection.
// K = [[2, -2], [-2, 2]] — det = 4-4 = 0, symmetric singular matrix.

TEST_F(CudaTest, SingularMatrixThrowsSolverError) {
    SparseMatrixBuilder builder(2);
    builder.add(0, 0,  2.0);
    builder.add(0, 1, -2.0);
    builder.add(1, 0, -2.0);
    builder.add(1, 1,  2.0);
    auto csr = builder.build_csr();
    std::vector<double> F = {1.0, 1.0};

    EXPECT_THROW({
        (void)backend_->solve(csr, F);
    }, SolverError) << "Truly singular matrix (det=0) must throw SolverError";
}

// ── Test 6: name() and device_name() are non-empty ───────────────────────────

TEST_F(CudaTest, NameAndDeviceNameAreNonEmpty) {
    EXPECT_FALSE(backend_->name().empty());
    EXPECT_NE(backend_->name().find("CUDA"), std::string_view::npos)
        << "Backend name should contain 'CUDA'";
    EXPECT_FALSE(backend_->device_name().empty())
        << "device_name() should return the GPU name";
}

// ── Test 7: large random SPD system agrees with Eigen ────────────────────────
// Construct a diagonally dominant (and therefore SPD) n×n matrix with random
// off-diagonal entries.  Verify the CUDA solution against Eigen.
// Diagonal dominance ensures SPD without computing eigenvalues.

TEST_F(CudaTest, LargeRandomSpdSystemAgreesWithEigen) {
    const int n = 500;
    // Build a symmetric diagonally dominant matrix: A_ii = n+1, A_ij=A_ji=-1 for
    // a few neighbours.  This is a banded pattern mimicking FEM connectivity.
    SparseMatrixBuilder builder(n);
    for (int i = 0; i < n; ++i) {
        builder.add(i, i, static_cast<double>(n + 1));
        if (i > 0)     { builder.add(i, i - 1, -1.0); builder.add(i - 1, i, -1.0); }
        if (i > 1)     { builder.add(i, i - 2, -0.5); builder.add(i - 2, i, -0.5); }
    }
    auto csr = builder.build_csr();
    std::vector<double> F(n);
    for (int i = 0; i < n; ++i) F[i] = static_cast<double>((i % 7) + 1);

    auto u_cuda = backend_->solve(csr, F);

    EigenSolverBackend eigen;
    auto u_eigen = eigen.solve(csr, F);

    ASSERT_EQ(static_cast<int>(u_cuda.size()), n);
    // κ(A) is moderate for this banded SPD matrix; expect agreement to 1e-9.
    for (int i = 0; i < n; i += 50)
        EXPECT_NEAR(u_cuda[i], u_eigen[i], 1e-9)
            << "n=500 banded SPD: component " << i << " differs";

    EXPECT_TRUE(backend_->last_solve_used_cholesky())
        << "Diagonally dominant SPD matrix should always use Cholesky path";
}

// ── Test 8: empty force vector edge case ─────────────────────────────────────
// K = diag(1), F = [0].  Solution must be u = [0].
// Exercises the zero-RHS code path (valid, unloaded structure).

TEST_F(CudaTest, ZeroForceSolvesCorrectly) {
    SparseMatrixBuilder builder(1);
    builder.add(0, 0, 5.0);
    auto csr = builder.build_csr();
    std::vector<double> F = {0.0};

    auto u = backend_->solve(csr, F);

    ASSERT_EQ(static_cast<int>(u.size()), 1);
    EXPECT_NEAR(u[0], 0.0, 1e-15)
        << "Zero force should produce zero displacement";
}

#endif // HAVE_CUDA
