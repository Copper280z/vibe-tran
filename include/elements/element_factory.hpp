#pragma once
// include/elements/element_factory.hpp
// Creates concrete element objects from raw ElementData + Model.

#include "elements/element_base.hpp"
#include "core/model.hpp"
#include <memory>

namespace vibestran {

/// Create the appropriate element object for the given ElementData.
/// Throws SolverError if the element type is unsupported.
std::unique_ptr<ElementBase> make_element(const ElementData& data,
                                           const Model& model);

} // namespace vibestran
