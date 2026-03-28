#pragma once

#include "core/mpc_handler.hpp"
#include "core/model.hpp"
#include "core/sparse_matrix.hpp"
#include "elements/element_base.hpp"
#include "elements/element_factory.hpp"
#include <cstddef>
#include <utility>
#include <vector>

#ifdef HAVE_TBB
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#endif

namespace vibestran::detail {

inline int element_num_dofs(ElementType type) {
  switch (type) {
  case ElementType::CQUAD4:
    return 24;
  case ElementType::CTRIA3:
    return 18;
  case ElementType::CHEXA8:
    return 24;
  case ElementType::CHEXA20:
    return 60;
  case ElementType::CTETRA4:
    return 12;
  case ElementType::CTETRA10:
    return 30;
  case ElementType::CPENTA6:
    return 18;
  case ElementType::CBAR:
  case ElementType::CBEAM:
  case ElementType::CBUSH:
    return 12;
  case ElementType::CELAS1:
  case ElementType::CELAS2:
  case ElementType::CMASS1:
  case ElementType::CMASS2:
    return 2;
  }
  return 0;
}

inline size_t estimate_triplet_capacity(const Model& model) {
  size_t total = 0;
  for (const auto& elem : model.elements) {
    const size_t ndof = static_cast<size_t>(element_num_dofs(elem.type));
    total += ndof * ndof;
  }
  return total;
}

inline std::vector<double> to_row_major(const LocalKe& matrix) {
  const int rows = static_cast<int>(matrix.rows());
  const int cols = static_cast<int>(matrix.cols());
  std::vector<double> values(static_cast<size_t>(rows * cols));
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      values[static_cast<size_t>(r * cols + c)] = matrix(r, c);
  return values;
}

template <typename MatrixFn>
inline void assemble_element_matrix(const Model& model,
                                    const MpcHandler& mpc_handler,
                                    SparseMatrixBuilder& builder,
                                    MatrixFn&& matrix_fn) {
  const DofMap& dof_map = mpc_handler.full_dof_map();

#ifdef HAVE_TBB
  constexpr size_t thread_local_triplet_reserve = 4096;

  tbb::enumerable_thread_specific<SparseMatrixBuilder> partial_builders(
      [&]() {
        return SparseMatrixBuilder(builder.size(),
                                   thread_local_triplet_reserve);
      });

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, model.elements.size()),
      [&](const tbb::blocked_range<size_t>& range) {
        auto& partial_builder = partial_builders.local();
        for (size_t i = range.begin(); i != range.end(); ++i) {
          const auto& elem_data = model.elements[i];
          auto elem = make_element(elem_data, model);
          auto gdofs = elem->global_dof_indices(dof_map);
          LocalKe matrix = matrix_fn(*elem);
          auto matrix_row_major = to_row_major(matrix);
          mpc_handler.apply_to_stiffness(gdofs, matrix_row_major,
                                         partial_builder);
        }
      });

  for (auto& partial_builder : partial_builders)
    builder.merge_from(std::move(partial_builder));
#else
  for (const auto& elem_data : model.elements) {
    auto elem = make_element(elem_data, model);
    auto gdofs = elem->global_dof_indices(dof_map);
    LocalKe matrix = matrix_fn(*elem);
    auto matrix_row_major = to_row_major(matrix);
    mpc_handler.apply_to_stiffness(gdofs, matrix_row_major, builder);
  }
#endif
}

} // namespace vibestran::detail
