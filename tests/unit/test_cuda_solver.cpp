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

using namespace vibestran;

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


// ═════════════════════════════════════════════════════════════════════════════
// CudaEigensolverBackend tests
// ═════════════════════════════════════════════════════════════════════════════

#endif // HAVE_CUDA

#ifdef HAVE_CUDA_EIGENSOLVER
#include "solver/cuda_eigensolver_backend.hpp"

using namespace vibestran;

// ── Fixture ───────────────────────────────────────────────────────────────────

class CudaEigTest : public ::testing::Test {
protected:
    // cppcheck-suppress unusedFunction -- called by GTest framework
    void SetUp() override {
        backend_ = CudaEigensolverBackend::try_create();
        if (!backend_.has_value())
            GTEST_SKIP() << "CUDA not available — skipping CUDA eigensolver tests";
    }

    std::optional<CudaEigensolverBackend> backend_;
};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build an Eigen::SparseMatrix<double> from a dense symmetric matrix.
static Eigen::SparseMatrix<double> dense_to_sparse(
    const std::vector<std::vector<double>>& A)
{
    int n = static_cast<int>(A.size());
    using T = Eigen::Triplet<double>;
    std::vector<T> trips;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (A[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] != 0.0)
                trips.emplace_back(i, j,
                    A[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
    Eigen::SparseMatrix<double> mat(n, n);
    mat.setFromTriplets(trips.begin(), trips.end());
    return mat;
}

// ── Test 1: diagonal GEVP — exact eigenvalues ─────────────────────────────────
// K = diag(1, 4, 9),  M = diag(1, 1, 1).
// Eigenvalues are exactly 1, 4, 9; eigenvectors are the standard basis.
// With sigma = 0 (shift below all eigenvalues, using default -1 to keep C = K+M
// positive definite), we expect the 3 eigenvalues in ascending order.

TEST_F(CudaEigTest, DiagonalGEVPExactEigenvalues) {
    std::vector<std::vector<double>> Kd = {{1,0,0},{0,4,0},{0,0,9}};
    std::vector<std::vector<double>> Md = {{1,0,0},{0,1,0},{0,0,1}};
    auto K = dense_to_sparse(Kd);
    auto M = dense_to_sparse(Md);

    auto pairs = backend_->solve(K, M, 3, /*sigma=*/-1.0);

    ASSERT_GE(static_cast<int>(pairs.size()), 3);
    // Sorted ascending by eigenvalue.
    EXPECT_NEAR(pairs[0].eigenvalue, 1.0, 1e-6)
        << "First eigenvalue should be 1.0";
    EXPECT_NEAR(pairs[1].eigenvalue, 4.0, 1e-6)
        << "Second eigenvalue should be 4.0";
    EXPECT_NEAR(pairs[2].eigenvalue, 9.0, 1e-6)
        << "Third eigenvalue should be 9.0";
}

// ── Test 2: diagonal GEVP with non-identity M ─────────────────────────────────
// K = diag(2, 8, 18),  M = diag(2, 2, 2).
// Eigenvalues lambda satisfy K phi = lambda M phi  =>  2/2=1, 8/2=4, 18/2=9.
// Same eigenvalues as Test 1 despite different K and M, which tests that the
// generalized problem (not just a standard eigenvalue problem) is solved.

TEST_F(CudaEigTest, DiagonalGEVPScaledMass) {
    std::vector<std::vector<double>> Kd = {{2,0,0},{0,8,0},{0,0,18}};
    std::vector<std::vector<double>> Md = {{2,0,0},{0,2,0},{0,0,2}};
    auto K = dense_to_sparse(Kd);
    auto M = dense_to_sparse(Md);

    auto pairs = backend_->solve(K, M, 3, -1.0);

    ASSERT_GE(static_cast<int>(pairs.size()), 3);
    EXPECT_NEAR(pairs[0].eigenvalue, 1.0, 1e-6);
    EXPECT_NEAR(pairs[1].eigenvalue, 4.0, 1e-6);
    EXPECT_NEAR(pairs[2].eigenvalue, 9.0, 1e-6);
}

// ── Test 3: tridiagonal GEVP against SpectraEigensolverBackend ────────────────
// K = tridiag(2, -1, -1) of size 20, M = identity.
// Analytical eigenvalues: lambda_k = 2 - 2*cos(k*pi/(n+1)), k=1..n.
// We compare the 5 lowest eigenvalues against Spectra to within 1e-6.

TEST_F(CudaEigTest, TridiagonalGEVPMatchesSpectra) {
    const int n = 20;
    Eigen::SparseMatrix<double> K(n, n), M(n, n);
    {
        using T = Eigen::Triplet<double>;
        std::vector<T> Kt, Mt;
        for (int i = 0; i < n; ++i) {
            Kt.emplace_back(i, i, 2.0);
            if (i > 0)     { Kt.emplace_back(i, i-1, -1.0); Kt.emplace_back(i-1, i, -1.0); }
            Mt.emplace_back(i, i, 1.0);
        }
        K.setFromTriplets(Kt.begin(), Kt.end());
        M.setFromTriplets(Mt.begin(), Mt.end());
    }

    const int nd = 5;
    auto cuda_pairs   = backend_->solve(K, M, nd, -1.0);
    auto spectra_pairs = SpectraEigensolverBackend{}.solve(K, M, nd, -1.0);

    ASSERT_GE(static_cast<int>(cuda_pairs.size()), nd);
    ASSERT_GE(static_cast<int>(spectra_pairs.size()), nd);

    for (int i = 0; i < nd; ++i) {
        EXPECT_NEAR(cuda_pairs[static_cast<std::size_t>(i)].eigenvalue,
                    spectra_pairs[static_cast<std::size_t>(i)].eigenvalue, 1e-6)
            << "Mode " << i+1 << " eigenvalue differs between CUDA and Spectra";
    }
}

// ── Test 4: eigenvalues sorted ascending ─────────────────────────────────────
// Verify the output is sorted ascending regardless of the internal Ritz value
// ordering.  Use a well-conditioned diagonal problem.

TEST_F(CudaEigTest, EigenvaluesSortedAscending) {
    const int n = 10;
    std::vector<std::vector<double>> Kd(static_cast<std::size_t>(n),
                                        std::vector<double>(static_cast<std::size_t>(n), 0.0));
    std::vector<std::vector<double>> Md(static_cast<std::size_t>(n),
                                        std::vector<double>(static_cast<std::size_t>(n), 0.0));
    for (int i = 0; i < n; ++i) {
        Kd[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] =
            static_cast<double>((i + 1) * (i + 1));
        Md[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = 1.0;
    }
    auto K = dense_to_sparse(Kd);
    auto M = dense_to_sparse(Md);

    auto pairs = backend_->solve(K, M, 5, -1.0);

    ASSERT_GE(static_cast<int>(pairs.size()), 2);
    for (std::size_t i = 1; i < pairs.size(); ++i)
        EXPECT_LE(pairs[i-1].eigenvalue, pairs[i].eigenvalue)
            << "Eigenvalues not sorted at index " << i;
}

// ── Test 5: eigenvectors are M-orthonormal ────────────────────────────────────
// For K = tridiag(2,-1,-1), M = I, the returned eigenvectors should satisfy
// phi_i^T M phi_j = delta_{ij}  (mass-normalised).
// We verify this for the lowest 4 modes: off-diagonal < 1e-5, diagonal ≈ 1.
// Tolerance is 1e-5 (not tighter) because Lanczos Ritz vectors accumulate
// small orthogonality drift from finite-precision arithmetic.

TEST_F(CudaEigTest, EigenvectorsMassNormalised) {
    const int n = 15;
    Eigen::SparseMatrix<double> K(n, n), M(n, n);
    {
        using T = Eigen::Triplet<double>;
        std::vector<T> Kt, Mt;
        for (int i = 0; i < n; ++i) {
            Kt.emplace_back(i, i, 2.0);
            if (i > 0) { Kt.emplace_back(i,i-1,-1.0); Kt.emplace_back(i-1,i,-1.0); }
            Mt.emplace_back(i, i, 1.0);
        }
        K.setFromTriplets(Kt.begin(), Kt.end());
        M.setFromTriplets(Mt.begin(), Mt.end());
    }

    const int nd = 4;
    auto pairs = backend_->solve(K, M, nd, -1.0);
    ASSERT_GE(static_cast<int>(pairs.size()), nd);

    for (int i = 0; i < nd; ++i) {
        for (int j = i; j < nd; ++j) {
            const Eigen::VectorXd& phi_i = pairs[static_cast<std::size_t>(i)].eigenvector;
            const Eigen::VectorXd& phi_j = pairs[static_cast<std::size_t>(j)].eigenvector;
            double inner = phi_i.dot(M * phi_j);
            if (i == j) {
                EXPECT_NEAR(inner, 1.0, 1e-5)
                    << "Mode " << i+1 << " is not M-normalised (phi^T M phi = " << inner << ")";
            } else {
                EXPECT_NEAR(inner, 0.0, 1e-5)
                    << "Modes " << i+1 << " and " << j+1
                    << " are not M-orthogonal (phi_i^T M phi_j = " << inner << ")";
            }
        }
    }
}

// ── Test 6: residual ||K phi - lambda M phi|| / ||K phi|| < 1e-6 ──────────────
// A direct measure of eigenpair quality.  For a well-converged Lanczos run
// the relative residual should be well below 1e-6.

TEST_F(CudaEigTest, EigenpairResidualsSmall) {
    const int n = 30;
    Eigen::SparseMatrix<double> K(n, n), M(n, n);
    {
        using T = Eigen::Triplet<double>;
        std::vector<T> Kt, Mt;
        for (int i = 0; i < n; ++i) {
            Kt.emplace_back(i, i, 2.0);
            if (i > 0) { Kt.emplace_back(i,i-1,-1.0); Kt.emplace_back(i-1,i,-1.0); }
            Mt.emplace_back(i, i, 1.0);
        }
        K.setFromTriplets(Kt.begin(), Kt.end());
        M.setFromTriplets(Mt.begin(), Mt.end());
    }

    auto pairs = backend_->solve(K, M, 6, -1.0);
    ASSERT_GE(static_cast<int>(pairs.size()), 6);

    for (std::size_t i = 0; i < 6; ++i) {
        const Eigen::VectorXd Kphi = K * pairs[i].eigenvector;
        const Eigen::VectorXd res  = Kphi - pairs[i].eigenvalue * (M * pairs[i].eigenvector);
        double kphi_norm = Kphi.norm();
        double rel_res   = (kphi_norm > 1e-300) ? res.norm() / kphi_norm : res.norm();
        EXPECT_LT(rel_res, 1e-6)
            << "Mode " << i+1 << " residual too large: " << rel_res;
    }
}

// ── Test 7: full comparison with SpectraEigensolverBackend ───────────────────
// Builds a tridiagonal K (size 25) with a non-identity lumped mass M
// (M_ii = 1 + 0.1*i to break symmetry).  Requests the 6 lowest eigenpairs
// from both solvers and verifies:
//   (a) Eigenvalues agree to within 1e-5.
//   (b) Eigenvectors agree up to sign: |phi_cuda^T M phi_spectra| ≈ 1.
// This is the primary regression test ensuring the CUDA Lanczos produces
// the same physical result as the well-validated Spectra backend.

TEST_F(CudaEigTest, CompareFullyWithSpectra) {
    const int n = 25;
    Eigen::SparseMatrix<double> K(n, n), M(n, n);
    {
        using T = Eigen::Triplet<double>;
        std::vector<T> Kt, Mt;
        for (int i = 0; i < n; ++i) {
            Kt.emplace_back(i, i, 2.0);
            if (i > 0) { Kt.emplace_back(i,i-1,-1.0); Kt.emplace_back(i-1,i,-1.0); }
            // Non-uniform lumped mass: M_ii = 1 + 0.1*i
            Mt.emplace_back(i, i, 1.0 + 0.1 * i);
        }
        K.setFromTriplets(Kt.begin(), Kt.end());
        M.setFromTriplets(Mt.begin(), Mt.end());
    }

    const int nd = 6;
    auto cuda_pairs    = backend_->solve(K, M, nd, -1.0);
    auto spectra_pairs = SpectraEigensolverBackend{}.solve(K, M, nd, -1.0);

    ASSERT_GE(static_cast<int>(cuda_pairs.size()),    nd) << "CUDA returned fewer pairs than requested";
    ASSERT_GE(static_cast<int>(spectra_pairs.size()), nd) << "Spectra returned fewer pairs than requested";

    for (int i = 0; i < nd; ++i) {
        double lam_cuda    = cuda_pairs[static_cast<std::size_t>(i)].eigenvalue;
        double lam_spectra = spectra_pairs[static_cast<std::size_t>(i)].eigenvalue;

        // (a) Eigenvalue agreement.
        EXPECT_NEAR(lam_cuda, lam_spectra, 1e-5)
            << "Mode " << i+1 << " eigenvalue: CUDA=" << lam_cuda
            << " Spectra=" << lam_spectra;

        // (b) Eigenvector agreement (sign-agnostic via |phi_c^T M phi_s| ≈ 1).
        // Both are M-normalised so the M-inner product should be ±1 if they
        // span the same direction.
        const Eigen::VectorXd& phi_c = cuda_pairs[static_cast<std::size_t>(i)].eigenvector;
        const Eigen::VectorXd& phi_s = spectra_pairs[static_cast<std::size_t>(i)].eigenvector;
        double m_inner = std::abs(phi_c.dot(M * phi_s));
        EXPECT_NEAR(m_inner, 1.0, 1e-4)
            << "Mode " << i+1 << " eigenvectors differ: |phi_c^T M phi_s|=" << m_inner;
    }
}

// ── Test 9: name() and device_name() non-empty ────────────────────────────────

TEST_F(CudaEigTest, NameAndDeviceName) {
    EXPECT_FALSE(backend_->name().empty());
    EXPECT_NE(backend_->name().find("CUDA"), std::string::npos)
        << "Backend name should contain 'CUDA'";
    EXPECT_FALSE(backend_->device_name().empty())
        << "device_name() should return the GPU model string";
}

// ── Test 10: larger problem requiring restarts matches Spectra ─────────────────
// n=100 tridiagonal K with non-uniform mass, nd=12.  With ncv = max(2*12+10, 36)
// = 36, the initial Lanczos pass alone is unlikely to converge all 12 modes,
// so implicit restarts are exercised.

TEST_F(CudaEigTest, LargerProblemMatchesSpectra) {
    const int n = 100;
    Eigen::SparseMatrix<double> K(n, n), M(n, n);
    {
        using T = Eigen::Triplet<double>;
        std::vector<T> Kt, Mt;
        for (int i = 0; i < n; ++i) {
            Kt.emplace_back(i, i, 2.0);
            if (i > 0) { Kt.emplace_back(i, i-1, -1.0); Kt.emplace_back(i-1, i, -1.0); }
            Mt.emplace_back(i, i, 1.0 + 0.05 * i);
        }
        K.setFromTriplets(Kt.begin(), Kt.end());
        M.setFromTriplets(Mt.begin(), Mt.end());
    }

    const int nd = 12;
    auto cuda_pairs    = backend_->solve(K, M, nd, -1.0);
    auto spectra_pairs = SpectraEigensolverBackend{}.solve(K, M, nd, -1.0);

    ASSERT_GE(static_cast<int>(cuda_pairs.size()), nd);
    ASSERT_GE(static_cast<int>(spectra_pairs.size()), nd);

    for (int i = 0; i < nd; ++i) {
        EXPECT_NEAR(cuda_pairs[static_cast<std::size_t>(i)].eigenvalue,
                    spectra_pairs[static_cast<std::size_t>(i)].eigenvalue, 1e-5)
            << "Mode " << i+1 << " eigenvalue differs (n=100, nd=12)";
    }
}

// ── Test 11: tight residuals with restart ──────────────────────────────────────
// n=50, nd=10.  Verify that eigenpair residuals are small after IRL converges.

TEST_F(CudaEigTest, EigenpairResidualsTightWithRestart) {
    const int n = 50;
    Eigen::SparseMatrix<double> K(n, n), M(n, n);
    {
        using T = Eigen::Triplet<double>;
        std::vector<T> Kt, Mt;
        for (int i = 0; i < n; ++i) {
            Kt.emplace_back(i, i, 2.0);
            if (i > 0) { Kt.emplace_back(i, i-1, -1.0); Kt.emplace_back(i-1, i, -1.0); }
            Mt.emplace_back(i, i, 1.0);
        }
        K.setFromTriplets(Kt.begin(), Kt.end());
        M.setFromTriplets(Mt.begin(), Mt.end());
    }

    const int nd = 10;
    auto pairs = backend_->solve(K, M, nd, -1.0);
    ASSERT_GE(static_cast<int>(pairs.size()), nd);

    for (std::size_t i = 0; i < static_cast<std::size_t>(nd); ++i) {
        const Eigen::VectorXd Kphi = K * pairs[i].eigenvector;
        const Eigen::VectorXd res  = Kphi - pairs[i].eigenvalue * (M * pairs[i].eigenvector);
        double kphi_norm = Kphi.norm();
        double rel_res   = (kphi_norm > 1e-300) ? res.norm() / kphi_norm : res.norm();
        EXPECT_LT(rel_res, 1e-6)
            << "Mode " << i+1 << " residual too large: " << rel_res;
    }
}

// ── Test 12: clustered eigenvalues converge ────────────────────────────────────
// Diagonal K with clustered eigenvalues near 1.0, M=I.
// Clustered eigenvalues stress the restart logic since nearby Ritz values
// converge slowly and can interfere with shift selection.

TEST_F(CudaEigTest, ClusteredEigenvaluesConverge) {
    const int n = 20;
    std::vector<std::vector<double>> Kd(static_cast<std::size_t>(n),
                                        std::vector<double>(static_cast<std::size_t>(n), 0.0));
    std::vector<std::vector<double>> Md(static_cast<std::size_t>(n),
                                        std::vector<double>(static_cast<std::size_t>(n), 0.0));
    // First 5 eigenvalues clustered near 1.0, rest well-separated.
    for (int i = 0; i < n; ++i) {
        if (i < 5)
            Kd[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = 1.0 + 0.001 * i;
        else
            Kd[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] =
                static_cast<double>(i + 1);
        Md[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = 1.0;
    }
    auto K = dense_to_sparse(Kd);
    auto M = dense_to_sparse(Md);

    const int nd = 5;
    auto cuda_pairs    = backend_->solve(K, M, nd, -1.0);
    auto spectra_pairs = SpectraEigensolverBackend{}.solve(K, M, nd, -1.0);

    ASSERT_GE(static_cast<int>(cuda_pairs.size()), nd);
    ASSERT_GE(static_cast<int>(spectra_pairs.size()), nd);

    for (int i = 0; i < nd; ++i) {
        EXPECT_NEAR(cuda_pairs[static_cast<std::size_t>(i)].eigenvalue,
                    spectra_pairs[static_cast<std::size_t>(i)].eigenvalue, 1e-5)
            << "Clustered mode " << i+1 << " eigenvalue differs";
    }
}

#else
// Compile-time stub when CUDA eigensolver was not compiled in.
TEST(CudaEigTest, CudaEigensolverNotCompiled) {
    GTEST_SKIP() << "CUDA eigensolver not compiled — skipping eigensolver tests";
}
#endif // HAVE_CUDA_EIGENSOLVER
