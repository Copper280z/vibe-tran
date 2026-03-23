#pragma once
// include/solver/cuda_eigensolver_backend.hpp
// CUDA modal (eigensolver) backend using shift-and-invert Lanczos.
//
// Algorithm summary:
//   1. Form shift matrix C = K - sigma*M on CPU (Eigen sparse arithmetic).
//   2. Factorize C on GPU via cuDSS (tries SPD Cholesky, falls back to LU).
//   3. Run ncv Lanczos steps using the M-inner-product formulation:
//        A = C^{-1} M,  eigenvalues nu = 1/(lambda - sigma)
//      Each step:
//        w_k  = M * v_k             (cuSPARSE SpMV)
//        z_k  = C^{-1} * w_k       (cuDSS solve, reusing factorization)
//        alpha_k = w_k^T z_k        (cuBLAS dot  == <v_k, A v_k>_M)
//        r    = z_k - alpha*v_k - beta_{k-1}*v_{k-1}
//      Full re-orthogonalization via cuBLAS dgemv against all prior vectors.
//        beta_k = ||r||_M           (cuSPARSE SpMV for M*r, then cuBLAS dot)
//        v_{k+1} = r / beta_k
//   4. Build ncv×ncv symmetric tridiagonal T; solve its eigenvalue problem on
//      CPU with Eigen::SelfAdjointEigenSolver (tiny matrix).
//   5. Select nd Ritz pairs with largest |nu|, convert nu -> lambda, return
//      sorted ascending by lambda.
//
// Requirements: CUDA toolkit, cuDSS >= 0.7, cuBLAS, cuSPARSE.
//   Guarded by HAVE_CUDA_EIGENSOLVER (defined when all three libs are found).
//
// Use try_create() to construct — returns nullopt when no CUDA device is
// present so the caller can fall back gracefully.

#ifdef HAVE_CUDA_EIGENSOLVER

#include "solver/eigensolver_backend.hpp"
#include <memory>
#include <optional>

namespace vibetran {

// Opaque RAII context holding cuDSS, cuBLAS, and cuSPARSE handles.
// Defined in cuda_eigensolver_backend.cu to keep CUDA headers out of this file.
struct CudaEigenContext;

class CudaEigensolverBackend final : public EigensolverBackend {
public:
    ~CudaEigensolverBackend() override;
    CudaEigensolverBackend(CudaEigensolverBackend&&) noexcept;
    CudaEigensolverBackend& operator=(CudaEigensolverBackend&&) noexcept;
    CudaEigensolverBackend(const CudaEigensolverBackend&) = delete;
    CudaEigensolverBackend& operator=(const CudaEigensolverBackend&) = delete;

    /// Factory — returns nullopt when no CUDA device is available.
    [[nodiscard]] static std::optional<CudaEigensolverBackend>
    try_create() noexcept;

    /// Solve K φ = λ M φ using shift-and-invert Lanczos on the GPU.
    /// Returns up to nd EigenPairs sorted by ascending eigenvalue.
    /// Throws SolverError on unrecoverable GPU errors or failure to converge.
    [[nodiscard]] std::vector<EigenPair> solve(
        const Eigen::SparseMatrix<double>& K,
        const Eigen::SparseMatrix<double>& M,
        int nd, double sigma) override;

    [[nodiscard]] std::string name() const override;

    /// GPU device name reported by the CUDA runtime.
    [[nodiscard]] std::string_view device_name() const noexcept;

private:
    explicit CudaEigensolverBackend(std::unique_ptr<CudaEigenContext> ctx) noexcept;
    std::unique_ptr<CudaEigenContext> ctx_;
};

} // namespace vibetran

#endif // HAVE_CUDA_EIGENSOLVER
