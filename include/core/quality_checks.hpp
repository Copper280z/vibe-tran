#pragma once
// include/core/quality_checks.hpp
// Input deck quality checker: element quality, topology, and physical sanity.
//
// Two modes are supported, controlled by PARAM,CHECKMODE in the BDF:
//
//   STRICT (default, Nastran-like)
//     Any quality violation throws SolverError and halts analysis.
//     PARAM cards control thresholds; see QualityThresholds below.
//
//   LENIENT (PARAM,CHECKMODE,LENIENT — ANSYS-like)
//     Violations that do not prevent a meaningful solve are logged as warnings
//     and analysis proceeds. A small set of conditions remain fatal regardless
//     of mode: E ≤ 0, nu ≥ 0.5, and SOL 101 with no constraints.
//
// Usage (from main.cpp, after BdfParser::parse_file() but before solve()):
//   QualityThresholds t = build_thresholds(model);
//   run_quality_checks(model, t);   // may throw or mutate model

#include "core/model.hpp"
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace vibestran {

// ── Check mode ────────────────────────────────────────────────────────────────

enum class CheckMode { Strict, Lenient };

// ── Thresholds ────────────────────────────────────────────────────────────────

/// Populated from PARAM cards in the model; defaults mirror MSC Nastran.
struct QualityThresholds {
    CheckMode mode            {CheckMode::Strict};

    // Element quality
    double max_aspect_ratio   {20.0};   // PARAM,ASPECT     longest/shortest edge
    double min_jacobian_ratio {0.0};    // PARAM,JACMIN     (0 = check disabled)
    double max_warp_angle     {30.0};   // PARAM,WARP       degrees, shells only
    double max_taper_ratio    {0.5};    // PARAM,TAPER      area imbalance ratio
    double min_interior_angle {10.0};   // PARAM,MINANGLE   degrees
    double max_interior_angle {170.0};  // PARAM,MAXANGLE   degrees

    // Topology
    double dup_node_tol       {1e-8};   // PARAM,DUPNODTOL  coordinate tolerance
    bool   auto_merge_nodes   {true};   // PARAM,AUTOMERGE  0 → warn but don't merge

    // Solver-stage (read and stored here; consumed by linear_static / modal):
    double maxratio           {1e7};    // PARAM,MAXRATIO   stiffness diagonal ratio
    int    bailout            {0};      // PARAM,BAILOUT    0=halt, -1=continue
};

// ── Per-element result ────────────────────────────────────────────────────────

struct ElementQualityResult {
    ElementId   id{0};
    ElementType type{ElementType::CQUAD4};
    // Metrics that are not applicable to an element type are set to NaN.
    // Metrics that are applicable but not yet computed are set to -1.0.
    double aspect_ratio       {-1.0};
    // NaN = not applicable (e.g. shells don't have a Jacobian ratio).
    // -1.0 = inverted element (negative Jacobian). 0..1 = valid.
    double min_jacobian_ratio {std::numeric_limits<double>::quiet_NaN()};
    double min_interior_angle {-1.0};  // degrees (shells only)
    double max_interior_angle {-1.0};  // degrees (shells only)
    double warp_angle         {-1.0};  // degrees, CQUAD4 only
    double taper_ratio        {-1.0};  // CQUAD4 only
};

// ── Topology summary ──────────────────────────────────────────────────────────

struct TopologyResult {
    std::vector<ElementId>                      free_edge_elements;
    std::vector<NodeId>                         orphaned_nodes;
    std::vector<PropertyId>                     orphaned_properties;
    std::vector<MaterialId>                     orphaned_materials;
    std::vector<std::pair<NodeId,NodeId>>       duplicate_node_pairs;
    std::vector<std::pair<ElementId,ElementId>> duplicate_element_pairs;
};

// ── Physical sanity summary ───────────────────────────────────────────────────

struct PhysicalResult {
    std::vector<MaterialId> bad_E;          // E ≤ 0
    std::vector<MaterialId> bad_nu;         // nu ≥ 0.5
    std::vector<MaterialId> bad_rho;        // rho < 0
    std::vector<PropertyId> bad_thickness;  // PShell t ≤ 0
    std::vector<int>        subcases_no_load;       // load_set references no loads
    std::vector<int>        subcases_no_constraint; // no SPCs and no RBEs
};

// ── Entry points ──────────────────────────────────────────────────────────────

/// Build thresholds by reading PARAM cards from model.params.
QualityThresholds build_thresholds(const Model& model);

/// Run all quality checks.
/// In Strict mode, throws SolverError on first violation.
/// In Lenient mode, logs warnings and returns; may still throw for conditions
/// that prevent a valid solve (E ≤ 0, nu ≥ 0.5, SOL 101 with no constraints).
/// In Lenient mode with AUTOMERGE=1, coincident node pairs (excluding zero-length
/// spring/bush elements and RBE pairs) are merged in-place and model is mutated.
void run_quality_checks(Model& model, const QualityThresholds& thresholds);

/// Compute element quality metrics from node geometry.
/// Returns nullopt for non-geometric elements (CBAR, CBEAM, CBUSH, CELAS*, CMASS*).
/// All node positions must already be in basic Cartesian (post resolve_coordinates).
std::optional<ElementQualityResult> compute_element_quality(
    const ElementData& elem, const Model& model);

/// Compute topology diagnostics for the entire model.
TopologyResult check_topology(const Model& model, double dup_node_tol);

/// Check physical validity of materials and properties.
PhysicalResult check_physical(const Model& model);

} // namespace vibestran
