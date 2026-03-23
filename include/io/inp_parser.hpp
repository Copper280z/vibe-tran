#pragma once
// include/io/inp_parser.hpp
// CalculiX/Abaqus .inp file parser.
//
// Populates the same Model struct used by BdfParser so the rest of the solver
// pipeline (assembly, solve, output) is unchanged.
//
// ── Supported keywords ──────────────────────────────────────────────────────
//   *NODE, *ELEMENT, *NSET, *ELSET, *MATERIAL, *ELASTIC, *DENSITY,
//   *EXPANSION, *SOLID SECTION, *SHELL SECTION, *BOUNDARY, *CLOAD,
//   *TEMPERATURE, *INITIAL CONDITIONS, *STEP, *STATIC, *NODE FILE,
//   *NODE PRINT, *EL FILE, *EL PRINT, *END STEP
//
// Unrecognized keywords are silently skipped (consistent with BDF parser).
//
// ── Element type mapping ────────────────────────────────────────────────────
//   C3D8  / C3D8R  → CHEXA8      S4  / S4R  → CQUAD4
//   C3D4           → CTETRA4     S3  / S3R  → CTRIA3
//   C3D10          → CTETRA10
//   C3D6           → CPENTA6
//
//   Reduced-integration variants (R suffix) are accepted and mapped to the
//   same internal element type as the full-integration version; the solver
//   uses its own integration scheme regardless.
//
//   CTETRA10 node ordering is identical between Abaqus and Nastran (midside
//   nodes on edges 0-1, 1-2, 2-0, 0-3, 1-3, 2-3), so no reorder is needed.
//
// ── Intentionally unsupported ───────────────────────────────────────────────
//   C3D20 / C3D20R — Midside node ordering differs substantially between
//     Abaqus and Nastran for 20-node hexahedra. Parsing these throws a
//     ParseError with an explanatory message.
//
//   *DLOAD, *DFLUX, *FILM, *RADIATE — Distributed loads and thermal BCs are
//     not implemented. Only concentrated loads (*CLOAD) and nodal temps are
//     supported.
//
//   *EQUATION, *TIE, *CONTACT — Constraint equations and contact are not
//     implemented. Use *BOUNDARY for SPCs.
//
//   *AMPLITUDE, *FREQUENCY, *MODAL DYNAMIC, *BUCKLE — Only *STATIC (SOL 101
//     equivalent) is supported. Non-linear and dynamic analysis keywords are
//     ignored or will produce an incomplete subcase.
//
//   Coordinate systems — All nodes are assumed to be in global Cartesian.
//     CalculiX *TRANSFORM is not parsed; GridPoint.cp and .cd are always 0.
//
//   MOMENT loads — *CLOAD only supports DOFs 1-3 (translational forces).
//     Rotational DOFs (4-6) are not mapped since CalculiX does not have a
//     separate *CMOMENT keyword; adding moment support would require
//     interpreting DOF > 3 as MomentLoad instead of ForceLoad.
//
// ── Material / section / property synthesis ─────────────────────────────────
//   CalculiX links materials to elements via:
//     *MATERIAL (name) → *SECTION (ELSET + MATERIAL) → elements in ELSET
//
//   The parser synthesizes Nastran-style PropertyId/MaterialId:
//     - *MATERIAL begins accumulating a Mat1 under a string name.
//     - *ELASTIC, *DENSITY, *EXPANSION fill fields on that Mat1.
//     - *SOLID SECTION or *SHELL SECTION finalizes the material (assigns a
//       MaterialId), creates a PSolid/PShell (assigns a PropertyId), and sets
//       element.pid for every element in the referenced ELSET.
//     - If multiple sections reference the same material name, they share one
//       MaterialId but get distinct PropertyIds.
//     - G (shear modulus) is derived as E / (2*(1+nu)) when not explicitly set.
//
// ── Set management ──────────────────────────────────────────────────────────
//   - Named sets use case-insensitive lookup (stored as uppercase keys).
//   - Multiple *NSET/*ELSET blocks with the same name append to the set.
//   - *ELEMENT with ELSET= auto-registers parsed elements into the named set.
//   - *NODE with NSET= auto-registers parsed nodes into the named set.
//   - *BOUNDARY and *CLOAD accept either a numeric node ID or a set name as
//     the first field; set names are resolved via the nset table.
//   - GENERATE is supported for both *NSET and *ELSET.
//
// ── Step → SubCase mapping ──────────────────────────────────────────────────
//   - Each *STEP creates a new SubCase with id = step number (1-based).
//   - *BOUNDARY/*CLOAD inside a step use SpcSetId/LoadSetId = step number.
//   - *BOUNDARY/*CLOAD outside any step (model-level) use set ID 1 and are
//     inherited by steps that don't define their own BCs/loads.
//   - If no *STEP is present, a default subcase {id=1, load_set=1, spc_set=1}
//     is created with all output flags enabled.
//   - *NODE FILE → disp_plot, *NODE PRINT → disp_print,
//     *EL FILE → stress_plot, *EL PRINT → stress_print.
//   - *INITIAL CONDITIONS, TYPE=TEMPERATURE → model.tempd[1] (reference temp).

#include "core/model.hpp"
#include <filesystem>
#include <istream>
#include <string>

namespace vibestran {

class InpParser {
public:
    /// Parse an .inp file and return the populated model.
    [[nodiscard]] static Model parse_file(const std::filesystem::path& path);

    /// Parse .inp content from a string (useful in tests)
    [[nodiscard]] static Model parse_string(const std::string& content);

    /// Parse from an input stream
    [[nodiscard]] static Model parse_stream(std::istream& in);

    // Forward declaration — defined in the .cpp file.
    // Public so implementation helpers in the .cpp can use it.
    struct ParseContext;
};

} // namespace vibestran
