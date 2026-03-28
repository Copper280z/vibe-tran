// src/io/csv_writer.cpp
// CSV output for nodal displacements and element stresses.
//
// Output files:
//   <stem>.node.csv  – one row per (subcase, node)
//     Header: # node_id, subcase_id, T1, T2, T3, R1, R2, R3
//
//   <stem>.elem.csv  – one row per (subcase, element)
//     Header: # elem_id, subcase_id, sx, sy, sxy, sz, syz, szx,
//               mx, my, mxy, von_mises
//     Fields not applicable to an element type are written as 0.0.

#include "io/results.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <format>

namespace vibestran {

void CsvWriter::write(const SolverResults& results, const Model& model,
                      const std::filesystem::path& stem) {
    // Collect output flags per subcase
    auto get_flags = [&](int sc_id, bool& do_disp, bool& do_stress) {
        do_disp = do_stress = false;
        const auto it = std::find_if(
            model.analysis.subcases.begin(), model.analysis.subcases.end(),
            [&](const SubCase& sc) { return sc.id == sc_id; });
        if (it == model.analysis.subcases.end()) return;
        do_disp = it->disp_print;
        do_stress = it->stress_print;
    };

    // ── Nodal CSV ─────────────────────────────────────────────────────────────
    {
        auto node_path = std::filesystem::path(stem).replace_extension("").string()
                         + ".node.csv";
        std::ofstream nf(node_path);
        if (!nf)
            throw SolverError(std::format("Cannot write CSV: {}", node_path));

        nf << "# node_id, subcase_id, T1, T2, T3, R1, R2, R3\n";
        nf << std::scientific << std::setprecision(8);

        for (const auto& sc : results.subcases) {
            bool do_disp, do_stress;
            get_flags(sc.id, do_disp, do_stress);
            if (!do_disp) continue;

            for (const auto& nd : sc.displacements) {
                nf << nd.node.value << ", " << sc.id;
                for (int i = 0; i < 6; ++i)
                    nf << ", " << nd.d[i];
                nf << "\n";
            }
        }
    }

    // ── Elemental CSV ─────────────────────────────────────────────────────────
    {
        auto elem_path = std::filesystem::path(stem).replace_extension("").string()
                         + ".elem.csv";
        std::ofstream ef(elem_path);
        if (!ef)
            throw SolverError(std::format("Cannot write CSV: {}", elem_path));

        ef << "# elem_id, subcase_id, sx, sy, sxy, sz, syz, szx,"
              " mx, my, mxy, von_mises\n";
        ef << std::scientific << std::setprecision(8);

        for (const auto& sc : results.subcases) {
            bool do_disp, do_stress;
            get_flags(sc.id, do_disp, do_stress);
            if (!do_stress) continue;

            for (const auto& ps : sc.plate_stresses) {
                ef << ps.eid.value << ", " << sc.id << ", "
                   << ps.sx  << ", " << ps.sy  << ", " << ps.sxy << ", "
                   << 0.0    << ", " << 0.0    << ", " << 0.0    << ", "
                   << ps.mx  << ", " << ps.my  << ", " << ps.mxy << ", "
                   << ps.von_mises << "\n";
            }

            for (const auto& ss : sc.solid_stresses) {
                ef << ss.eid.value << ", " << sc.id << ", "
                   << ss.sx  << ", " << ss.sy  << ", " << ss.sxy << ", "
                   << ss.sz  << ", " << ss.syz << ", " << ss.szx << ", "
                   << 0.0    << ", " << 0.0    << ", " << 0.0    << ", "
                   << ss.von_mises << "\n";
            }
        }
    }
}

} // namespace vibestran
