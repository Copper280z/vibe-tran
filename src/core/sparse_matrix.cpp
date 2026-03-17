// src/core/sparse_matrix.cpp
#include "core/sparse_matrix.hpp"
#include "core/dof_map.hpp"
#include <algorithm>
#include <execution>
#include <numeric>
#include <stdexcept>

namespace nastran {

// cppcheck-suppress unusedFunction -- called from linear_static.cpp
void SparseMatrixBuilder::add_element_stiffness(
    std::span<const int32_t> global_dofs, std::span<const double> ke) {
  const int n = static_cast<int>(global_dofs.size());
  if (static_cast<int>(ke.size()) != n * n)
    throw std::invalid_argument("ke size mismatch");

  for (int i = 0; i < n; ++i) {
    if (global_dofs[i] == CONSTRAINED_DOF)
      continue;
    for (int j = 0; j < n; ++j) {
      if (global_dofs[j] == CONSTRAINED_DOF)
        continue;
      double v = ke[static_cast<size_t>(i * n + j)];
      if (v != 0.0)
        triplets_.push_back({global_dofs[i], global_dofs[j], v});
    }
  }
}

// cppcheck-suppress unusedFunction -- called from linear_static.cpp
void SparseMatrixBuilder::add_element_force(
    std::span<const int32_t> global_dofs, std::span<const double> fe,
    std::vector<double> &F) {
  const int n = static_cast<int>(global_dofs.size());
  for (int i = 0; i < n; ++i) {
    if (global_dofs[i] == CONSTRAINED_DOF)
      continue;
    if (global_dofs[i] >= static_cast<int>(F.size()))
      throw std::out_of_range("Force vector index out of range");
    F[static_cast<size_t>(global_dofs[i])] += fe[static_cast<size_t>(i)];
  }
}

SparseMatrixBuilder::CsrData SparseMatrixBuilder::build_csr() {
  const auto n_trips = static_cast<int>(triplets_.size());

  // Pass 1: histogram of row indices → row-bucket boundaries.
  // Two O(n) sequential passes over compact data instead of one O(n log n)
  // global sort over 9+ GB — the global sort is memory-bandwidth-bound and
  // does not scale with core count.
  std::vector<int> bucket_ptr(static_cast<size_t>(size_ + 1), 0);
  for (const auto &t : triplets_)
    ++bucket_ptr[static_cast<size_t>(t.row + 1)];
  std::partial_sum(bucket_ptr.begin(), bucket_ptr.end(), bucket_ptr.begin());

  // Pass 2: scatter triplets into row-ordered staging buffer using cursor
  // positions derived from bucket_ptr.
  std::vector<int> cursor(bucket_ptr.begin(), bucket_ptr.begin() + size_);
  std::vector<Triplet> staged(static_cast<size_t>(n_trips));
  for (const auto &t : triplets_)
    staged[static_cast<size_t>(cursor[static_cast<size_t>(t.row)]++)] = t;

  // Free source memory now that scatter is complete.
  triplets_.clear();
  triplets_.shrink_to_fit();

  // Pass 3: sort within each row by col.
  // Each row has O(elements_per_node²) entries (~85 for CHEXA8) — tiny arrays
  // that fit in L1 cache. Parallel dispatch across rows via TBB.
  std::vector<int> row_indices(static_cast<size_t>(size_));
  std::iota(row_indices.begin(), row_indices.end(), 0);
  std::for_each(std::execution::par_unseq,
                row_indices.begin(), row_indices.end(),
                [&](int row) {
                  auto beg = staged.begin() + bucket_ptr[static_cast<size_t>(row)];
                  auto end = staged.begin() + bucket_ptr[static_cast<size_t>(row + 1)];
                  std::sort(beg, end, [](const Triplet &a, const Triplet &b) {
                    return a.col < b.col;
                  });
                });

  // Pass 4: dedup within each row (staged is now sorted by col within rows),
  // and write to final CSR arrays.
  CsrData csr;
  csr.n = size_;
  csr.row_ptr.resize(static_cast<size_t>(size_ + 1));
  csr.col_ind.resize(static_cast<size_t>(n_trips)); // upper bound; shrunk below
  csr.values.resize(static_cast<size_t>(n_trips));
  csr.row_ptr[0] = 0;

  int out = 0;
  for (int row = 0; row < size_; ++row) {
    const int beg = bucket_ptr[static_cast<size_t>(row)];
    const int end = bucket_ptr[static_cast<size_t>(row + 1)];
    for (int i = beg; i < end; ++i) {
      if (i > beg &&
          staged[static_cast<size_t>(i)].col ==
              staged[static_cast<size_t>(i - 1)].col) {
        csr.values[static_cast<size_t>(out - 1)] +=
            staged[static_cast<size_t>(i)].value;
      } else {
        csr.col_ind[static_cast<size_t>(out)] =
            staged[static_cast<size_t>(i)].col;
        csr.values[static_cast<size_t>(out)] =
            staged[static_cast<size_t>(i)].value;
        ++out;
      }
    }
    csr.row_ptr[static_cast<size_t>(row + 1)] = out;
  }

  csr.nnz = out;
  csr.col_ind.resize(static_cast<size_t>(out));
  csr.values.resize(static_cast<size_t>(out));

  return csr;
}

} // namespace nastran
