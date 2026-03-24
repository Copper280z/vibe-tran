#pragma once
// include/core/model.hpp
// The FE model: nodes, elements, materials, properties, loads, SPCs.
// This is a pure data structure layer — no computation here.

#include "core/types.hpp"
#include "core/coord_sys.hpp"
#include <memory>
#include <limits>
#include <optional>
#include <unordered_map>
#include <variant>

namespace vibestran {

// ── Grid point ───────────────────────────────────────────────────────────────

struct GridPoint {
  NodeId id{0};
  CoordId cp{0}; // input coordinate system
  Vec3 position{0, 0, 0};
  CoordId cd{0}; // displacement coordinate system
};

// ── Materials ────────────────────────────────────────────────────────────────

struct Mat1 {
  MaterialId id{0};
  double E{0};        // Young's modulus
  double G{0};        // Shear modulus (0 = derived from E,nu)
  double nu{0};       // Poisson's ratio
  double rho{0};      // density
  double A{0};        // thermal expansion coefficient
  double ref_temp{0}; // reference temperature
};

// ── Properties ───────────────────────────────────────────────────────────────

enum class SolidFormulation { SRI, EAS };
enum class ShellFormulation { MINDLIN, MITC4 };

struct PShell {
  PropertyId pid{0};
  MaterialId mid1{0};     // membrane material
  double t{0};            // thickness
  MaterialId mid2{0};     // bending material (0 = same as mid1)
  double twelveI_t3{1.0}; // 12I/t^3 bending stiffness factor
  MaterialId mid3{0};     // transverse shear material
  double tst{0.833333};   // transverse shear thickness ratio
  double nsm{0.0};        // non-structural mass / unit area
  double z1{std::numeric_limits<double>::quiet_NaN()}; // stress recovery fiber distance 1
  double z2{std::numeric_limits<double>::quiet_NaN()}; // stress recovery fiber distance 2
  MaterialId mid4{0};     // membrane-bending coupling
  ShellFormulation shell_form{ShellFormulation::MITC4};
};

struct PSolid {
  PropertyId pid{0};
  MaterialId mid{0};
  int cordm{0}; // material coordinate system
  SolidFormulation isop{SolidFormulation::EAS};
};

using Property = std::variant<PShell, PSolid>;

// ── Elements ─────────────────────────────────────────────────────────────────

enum class ElementType {
  CQUAD4,
  CTRIA3,
  CHEXA8,
  CHEXA20,
  CTETRA4,
  CTETRA10,
  CPENTA6,
};

struct ElementData {
  ElementId id{0};
  ElementType type{0};
  PropertyId pid{0};
  std::vector<NodeId> nodes;
  std::optional<double> theta{std::nullopt};
  std::optional<CoordId> mcid{std::nullopt};
  double zoffs{0.0};
};

// ── Loads
// ─────────────────────────────────────────────────────────────────────

struct ForceLoad {
  LoadSetId sid{0};
  NodeId node{0};
  CoordId cid{0};
  double scale{1.0};
  Vec3 direction;
  // Effective force = scale * direction (direction includes magnitude in
  // Nastran)
};

struct MomentLoad {
  LoadSetId sid{0};
  NodeId node{0};
  CoordId cid{0};
  double scale{1.0};
  Vec3 direction;
};

/// Nodal temperature (TEMP card)
struct TempLoad {
  LoadSetId sid{0};
  NodeId node{0};
  double temperature{0};
};

/// Temperature field reference for elements (not needed at nodes — handled via
/// TEMP)
struct TempRefLoad {
  LoadSetId sid{0};
  double t_ref{0}; // from TEMPD or analysis case
};

using Load = std::variant<ForceLoad, MomentLoad, TempLoad>;

// ── Single point constraints
// ──────────────────────────────────────────────────

struct Spc {
  SpcSetId sid{0};
  NodeId node{0};
  DofSet dofs;
  double value{0}; // enforced displacement (0 = fixed)
};

// ── Multi-point constraints
// ─────────────────────────────────────────────────

/// One term in an MPC equation: coeff * u[node, dof]
struct MpcTerm {
  NodeId node{0};
  int dof{0};    // 1-based DOF (1-6)
  double coeff{0.0};
};

/// One MPC equation: sum_i coeff_i * u[node_i, dof_i] = 0
/// The term with the largest |coeff| is chosen as the dependent DOF
/// during elimination.
struct Mpc {
  MpcSetId sid{0};
  std::vector<MpcTerm> terms;
};

// ── Rigid elements
// ──────────────────────────────────────────────────────────

/// RBE2 — rigid body element with independent node GN and dependent nodes GM
struct Rbe2 {
  ElementId eid{0};
  NodeId gn{0};       // independent node
  DofSet cm;          // constrained DOFs (on dependent nodes)
  std::vector<NodeId> gm; // dependent nodes
};

/// One weight group in an RBE3
struct Rbe3WeightGroup {
  double weight{1.0};
  DofSet component;         // active DOF components for this group
  std::vector<NodeId> nodes;
};

/// RBE3 — interpolation constraint element
struct Rbe3 {
  ElementId eid{0};
  NodeId ref_node{0};   // reference (dependent) node
  DofSet refc;          // constrained DOFs on reference node
  std::vector<Rbe3WeightGroup> weight_groups;
};

// ── EIGRL card (modal analysis) ──────────────────────────────────────────────

struct EigRL {
  int sid{0};
  double v1{0.0};   // lower frequency bound (Hz); 0 = unconstrained
  double v2{1e30};  // upper frequency bound (Hz)
  int nd{10};       // number of desired eigenvalues
  enum class Norm { Mass, Max } norm{Norm::Mass};
};

// ── Analysis case
// ─────────────────────────────────────────────────────────────

struct SubCase {
  int id{1};
  std::string label;
  LoadSetId load_set{0};
  SpcSetId spc_set{0};
  MpcSetId mpc_set{0}; // 0 = no MPCs
  int temp_load_set{0}; // TEMPERATURE(LOAD) / TEMP(LOAD) set ID
  double t_ref{0}; // reference temperature for thermal load

  // Output selection (case control deck).
  // PRINT → text output (F06, CSV); PLOT → binary output (OP2).
  // Both default to false; set by DISPLACEMENT/STRESS case control entries.
  // No output card → no output.  NONE → clears both for that result type.
  bool disp_print{false};   // DISPLACEMENT(PRINT)=ALL  or  DISPLACEMENT=ALL
  bool disp_plot{false};    // DISPLACEMENT(PLOT)=ALL
  bool stress_print{false}; // STRESS(PRINT)=ALL  or  STRESS=ALL
  bool stress_plot{false};  // STRESS(PLOT)=ALL

  // Modal analysis (SOL 103) output selection
  int eigrl_id{0};        // METHOD = <sid>  references an EIGRL card
  bool eigvec_print{false}; // EIGENVECTOR(PRINT)=ALL
  bool eigvec_plot{false};  // EIGENVECTOR(PLOT)=ALL
};

struct AnalysisCase {
  SolutionType sol{SolutionType::LinearStatic};
  std::vector<SubCase> subcases;
};

// ── The Model
// ─────────────────────────────────────────────────────────────────

class Model {
public:
  // Nodes
  std::unordered_map<NodeId, GridPoint> nodes;

  // Elements
  std::vector<ElementData> elements;

  // Materials
  std::unordered_map<MaterialId, Mat1> materials;

  // Properties
  std::unordered_map<PropertyId, Property> properties;

  // Loads (keyed by set id for quick lookup)
  std::vector<Load> loads;

  // SPCs
  std::vector<Spc> spcs;

  // MPCs
  std::vector<Mpc> mpcs;

  // Rigid elements
  std::vector<Rbe2> rbe2s;
  std::vector<Rbe3> rbe3s;

  // Coordinate systems (CoordId{0} = basic, never stored here)
  std::unordered_map<CoordId, CoordSys> coord_systems;

  // Default temperatures per set (TEMPD cards): set_id → temperature
  std::unordered_map<int, double> tempd;

  // Analysis
  AnalysisCase analysis;

  // PARAM entries (name → value as string)
  std::unordered_map<std::string, std::string> params;

  // EIGRL cards (keyed by SID)
  std::unordered_map<int, EigRL> eigrls;

  // ── Accessors ────────────────────────────────────────────────────────────

  const GridPoint &node(NodeId id) const {
    auto it = nodes.find(id);
    if (it == nodes.end())
      throw SolverError(std::format("Node {} not found in model", id.value));
    return it->second;
  }

  const Mat1 &material(MaterialId id) const {
    auto it = materials.find(id);
    if (it == materials.end())
      throw SolverError(std::format("Material {} not found", id.value));
    return it->second;
  }

  const Property &property(PropertyId id) const {
    auto it = properties.find(id);
    if (it == properties.end())
      throw SolverError(std::format("Property {} not found", id.value));
    return it->second;
  }

  /// Retrieve all loads for a given set
  std::vector<const Load *> loads_for_set(LoadSetId sid) const {
    std::vector<const Load *> result;
    for (const auto &load : loads) {
      auto has_sid = [&](const auto &l) { return l.sid == sid; };
      if (std::visit(has_sid, load))
        result.push_back(&load);
    }
    return result;
  }

  /// Retrieve all SPCs for a given set
  std::vector<const Spc *> spcs_for_set(SpcSetId sid) const {
    std::vector<const Spc *> result;
    for (const auto &spc : spcs) {
      if (spc.sid == sid)
        result.push_back(&spc);
    }
    return result;
  }

  /// Retrieve all MPCs for a given set
  std::vector<const Mpc *> mpcs_for_set(MpcSetId sid) const {
    std::vector<const Mpc *> result;
    for (const auto &mpc : mpcs) {
      if (mpc.sid == sid)
        result.push_back(&mpc);
    }
    return result;
  }

  /// After parsing, transform all GridPoint positions from their CP frame to
  /// basic Cartesian.  Also updates GridPoint.cp to CoordId::basic() after
  /// transformation so position is always in basic once called.
  /// Must be called after all GRID and CORDxx cards have been parsed.
  void resolve_coordinates();

  /// Validate consistency of the model (throws on error)
  void validate() const;
};

} // namespace vibestran
