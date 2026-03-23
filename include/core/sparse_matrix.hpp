#pragma once
// include/core/sparse_matrix.hpp
// Sparse symmetric matrix stored as a COO (triplet) builder that assembles
// into Compressed Sparse Row (CSR) format via Eigen.
//
// Design rationale for GPU readiness:
//   - The final CSR arrays (row_ptr, col_ind, values) are contiguous vectors
//     that can be transferred directly to device memory for cuSPARSE or
//     Vulkan SSBO usage without reformatting.
//   - Assembly uses addTriplet() which is cache-friendly for parallel element
//     assembly with atomic accumulation on GPU.

#include <cstdint>
#include <span>
#include <vector>

namespace vibestran {

/// A triplet (row, col, value) for sparse matrix assembly
struct Triplet {
  int32_t row;
  int32_t col;
  double value;
};

/// Builds a global stiffness matrix by accumulating element contributions.
/// After all elements are assembled, call finalize() to produce the CSR form.
class SparseMatrixBuilder {
public:
  explicit SparseMatrixBuilder(int size) : size_(size) {
    triplets_.reserve(size * 50); // rough estimate
  }

  /// Add a value to position (row, col) — duplicates are summed on finalize
  void add(int row, int col, double value) {
    triplets_.push_back({row, col, value});
  }

  /// Add a dense local stiffness block.
  /// global_dofs contains the global equation indices for each local DOF;
  /// CONSTRAINED_DOF entries are skipped automatically.
  /// ke must be (n x n) in row-major order where n = global_dofs.size()
  void add_element_stiffness(std::span<const int32_t> global_dofs,
                             std::span<const double> ke);

  /// Add an element load vector contribution to the global force vector.
  void add_element_force(std::span<const int32_t> global_dofs,
                         std::span<const double> fe, std::vector<double> &F);

  [[nodiscard]] int size() const noexcept { return size_; }

  /// Finalize: sort and deduplicate triplets in-place, then produce CSR data.
  /// Destructive: clears the triplet list to free memory after building.
  /// Returns the CSR data directly for potential GPU upload.
  struct CsrData {
    std::vector<int> row_ptr;   // size = n+1
    std::vector<int> col_ind;   // size = nnz
    std::vector<double> values; // size = nnz
    int n;
    int nnz;
  };

  [[nodiscard]] CsrData build_csr();

  /// The raw triplet list (useful for testing / inspection)
  [[nodiscard]] const std::vector<Triplet> &triplets() const noexcept {
    return triplets_;
  }

private:
  int size_;
  std::vector<Triplet> triplets_;
};

} // namespace vibestran
