#pragma once

#include "core/dof_map.hpp"
#include "core/mpc_handler.hpp"
#include "core/model.hpp"

namespace vibestran {

[[nodiscard]] DofMap build_analysis_dof_map(const Model &model,
                                            const SubCase &sc);

void build_analysis_mpc_system(const Model &model, const SubCase &sc,
                               DofMap &dof_map, MpcHandler &mpc_handler);

} // namespace vibestran
