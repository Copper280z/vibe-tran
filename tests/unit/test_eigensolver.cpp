#include <gtest/gtest.h>

#include "solver/eigensolver_backend.hpp"

#include <Eigen/Sparse>

#include <cmath>
#include <numbers>
#include <string_view>
#include <vector>

namespace {

Eigen::SparseMatrix<double> make_tridiagonal_stiffness(int n) {
    Eigen::SparseMatrix<double> K(n, n);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<std::size_t>(3 * n - 2));
    for (int i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
        if (i > 0) {
            triplets.emplace_back(i, i - 1, -1.0);
            triplets.emplace_back(i - 1, i, -1.0);
        }
    }
    K.setFromTriplets(triplets.begin(), triplets.end());
    return K;
}

Eigen::SparseMatrix<double> make_diagonal_mass(int n, double start,
                                               double step) {
    Eigen::SparseMatrix<double> M(n, n);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        triplets.emplace_back(i, i, start + step * static_cast<double>(i));
    M.setFromTriplets(triplets.begin(), triplets.end());
    return M;
}

Eigen::SparseMatrix<double> lower_triangle_only(
    const Eigen::SparseMatrix<double>& mat) {
    Eigen::SparseMatrix<double> lower(mat.template triangularView<Eigen::Lower>());
    lower.makeCompressed();
    return lower;
}

} // namespace

TEST(SpectraEigensolverBackend, NameIdentifiesUnderlyingDirectSolver) {
    const auto name = vibestran::SpectraEigensolverBackend{}.name();
    const bool is_accel = name.find("Accelerate") != std::string_view::npos;
    const bool is_cholmod = name.find("CHOLMOD") != std::string_view::npos;
    const bool is_eigen = name.find("Eigen") != std::string_view::npos;
    EXPECT_TRUE(is_accel || is_cholmod || is_eigen)
        << "Unexpected backend name: " << name;
}

TEST(SpectraEigensolverBackend, TridiagonalAnalyticalEigenvaluesMatch) {
    constexpr int n = 20;
    constexpr int nd = 5;

    const Eigen::SparseMatrix<double> K = make_tridiagonal_stiffness(n);
    const Eigen::SparseMatrix<double> M = make_diagonal_mass(n, 1.0, 0.0);

    const auto pairs = vibestran::SpectraEigensolverBackend{}.solve(K, M, nd, -1.0);

    ASSERT_EQ(static_cast<int>(pairs.size()), nd);
    for (int mode = 0; mode < nd; ++mode) {
        const double k = static_cast<double>(mode + 1);
        const double expected =
            2.0 - 2.0 * std::cos(k * std::numbers::pi / static_cast<double>(n + 1));
        EXPECT_NEAR(pairs[static_cast<std::size_t>(mode)].eigenvalue, expected, 1e-10)
            << "Analytical eigenvalue mismatch for mode " << mode + 1;
    }
}

TEST(SpectraEigensolverBackend, LowerTriangularStorageMatchesFullStorage) {
    constexpr int n = 24;
    constexpr int nd = 6;

    const Eigen::SparseMatrix<double> K_full = make_tridiagonal_stiffness(n);
    const Eigen::SparseMatrix<double> M_full = make_diagonal_mass(n, 1.0, 0.05);
    const Eigen::SparseMatrix<double> K_lower = lower_triangle_only(K_full);
    const Eigen::SparseMatrix<double> M_lower = lower_triangle_only(M_full);

    vibestran::SpectraEigensolverBackend backend;
    const auto full_pairs = backend.solve(K_full, M_full, nd, -1.0);
    const auto lower_pairs = backend.solve(K_lower, M_lower, nd, -1.0);

    ASSERT_EQ(static_cast<int>(full_pairs.size()), nd);
    ASSERT_EQ(static_cast<int>(lower_pairs.size()), nd);

    for (int mode = 0; mode < nd; ++mode) {
        const auto& full = full_pairs[static_cast<std::size_t>(mode)];
        const auto& lower = lower_pairs[static_cast<std::size_t>(mode)];

        EXPECT_NEAR(full.eigenvalue, lower.eigenvalue, 1e-10)
            << "Eigenvalue mismatch for mode " << mode + 1;

        const double m_inner =
            std::abs(full.eigenvector.dot(M_full * lower.eigenvector));
        EXPECT_NEAR(m_inner, 1.0, 1e-8)
            << "Mass-normalised eigenvector mismatch for mode " << mode + 1;
    }
}
