#pragma once
// include/core/model.hpp
// The FE model: nodes, elements, materials, properties, loads, SPCs.
// This is a pure data structure layer — no computation here.

#include "core/types.hpp"
#include <memory>
#include <optional>
#include <unordered_map>
#include <variant>

namespace nastran {

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
  MaterialId mid4{0};     // membrane-bending coupling
  ShellFormulation shell_form{ShellFormulation::MITC4};
};

struct PSolid {
  PropertyId pid{0};
  MaterialId mid{0};
  int cordm{0}; // material coordinate system
  SolidFormulation isop{SolidFormulation::SRI};
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
};

struct ElementData {
  ElementId id{0};
  ElementType type{0};
  PropertyId pid{0};
  std::vector<NodeId> nodes;
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

// ── Analysis case
// ─────────────────────────────────────────────────────────────

struct SubCase {
  int id{1};
  std::string label;
  LoadSetId load_set{0};
  SpcSetId spc_set{0};
  double t_ref{0}; // reference temperature for thermal load
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

  // Analysis
  AnalysisCase analysis;

  // PARAM entries (name → value as string)
  std::unordered_map<std::string, std::string> params;

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

  /// Validate consistency of the model (throws on error)
  void validate() const;
};

} // namespace nastran
