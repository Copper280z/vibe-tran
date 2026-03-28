// src/core/model.cpp
#include "core/model.hpp"
#include "core/coord_sys.hpp"
#include <format>
#include <algorithm>
#include <type_traits>
#include <unordered_set>

namespace vibestran {

namespace {

bool element_requires_property(const ElementType type) {
    switch (type) {
    case ElementType::CQUAD4:
    case ElementType::CTRIA3:
    case ElementType::CHEXA8:
    case ElementType::CHEXA20:
    case ElementType::CTETRA4:
    case ElementType::CTETRA10:
    case ElementType::CPENTA6:
    case ElementType::CBAR:
    case ElementType::CBEAM:
    case ElementType::CBUSH:
    case ElementType::CELAS1:
    case ElementType::CMASS1:
        return true;
    case ElementType::CELAS2:
    case ElementType::CMASS2:
        return false;
    }
    return false;
}

bool component_is_valid(const int component) {
    return component >= 1 && component <= 6;
}

} // namespace

void Model::validate() const {
    std::unordered_set<ElementId> element_ids;
    element_ids.reserve(elements.size());
    for (const auto& elem : elements)
        element_ids.insert(elem.id);

    // Check all element nodes exist
    for (const auto& elem : elements) {
        auto missing_node = std::find_if(elem.nodes.begin(), elem.nodes.end(),
            [&](NodeId nid) { return !nodes.count(nid); });
        if (missing_node != elem.nodes.end())
            throw SolverError(std::format(
                "Element {} references undefined node {}", elem.id.value, missing_node->value));
        if (element_requires_property(elem.type) && !properties.count(elem.pid))
            throw SolverError(std::format(
                "Element {} references undefined property {}", elem.id.value, elem.pid.value));

        if ((elem.type == ElementType::CBAR || elem.type == ElementType::CBEAM ||
             elem.type == ElementType::CBUSH) && elem.nodes.size() != 2) {
            throw SolverError(std::format(
                "Element {} expects exactly 2 grid points", elem.id.value));
        }
        if ((elem.type == ElementType::CELAS1 || elem.type == ElementType::CELAS2 ||
             elem.type == ElementType::CMASS1 || elem.type == ElementType::CMASS2) &&
            (elem.nodes.empty() || elem.nodes.size() > 2)) {
            throw SolverError(std::format(
                "Scalar element {} must reference one or two grid points", elem.id.value));
        }
        if (elem.type == ElementType::CELAS1 || elem.type == ElementType::CELAS2 ||
            elem.type == ElementType::CMASS1 || elem.type == ElementType::CMASS2) {
            if (!component_is_valid(elem.components[0])) {
                throw SolverError(std::format(
                    "Element {} has invalid first component {}", elem.id.value,
                    elem.components[0]));
            }
            if (elem.nodes.size() > 1 && !component_is_valid(elem.components[1])) {
                throw SolverError(std::format(
                    "Element {} has invalid second component {}", elem.id.value,
                    elem.components[1]));
            }
        }
    }

    // Check property-material references
    for (const auto& [pid, prop] : properties) {
        std::visit([&](const auto& p) {
            using T = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<T, PShell>) {
                if (!materials.count(p.mid1))
                    throw SolverError(std::format(
                        "PSHELL {} references undefined material {}", pid.value, p.mid1.value));
                if (p.mid2.value != 0 && !materials.count(p.mid2))
                    throw SolverError(std::format(
                        "PSHELL {} references undefined bending material {}", pid.value, p.mid2.value));
                if (p.mid3.value != 0 && !materials.count(p.mid3))
                    throw SolverError(std::format(
                        "PSHELL {} references undefined shear material {}", pid.value, p.mid3.value));
                if (p.mid4.value != 0 && !materials.count(p.mid4))
                    throw SolverError(std::format(
                        "PSHELL {} references undefined coupling material {}", pid.value, p.mid4.value));
            } else if constexpr (std::is_same_v<T, PSolid>) {
                if (!materials.count(p.mid))
                    throw SolverError(std::format(
                        "PSOLID {} references undefined material {}", pid.value, p.mid.value));
            } else if constexpr (std::is_same_v<T, PBar> ||
                                 std::is_same_v<T, PBarL> ||
                                 std::is_same_v<T, PBeam>) {
                if (!materials.count(p.mid))
                    throw SolverError(std::format(
                        "Property {} references undefined material {}", pid.value, p.mid.value));
            }
        }, prop);
    }

    // Check SPCs reference existing nodes
    auto missing_spc = std::find_if(spcs.begin(), spcs.end(),
        [&](const Spc& spc) { return !nodes.count(spc.node); });
    if (missing_spc != spcs.end())
        throw SolverError(std::format(
            "SPC references undefined node {}", missing_spc->node.value));

    // Check load nodes exist
    for (const auto& load : loads) {
        std::visit([&](const auto& l) {
            using T = std::decay_t<decltype(l)>;
            if constexpr (std::is_same_v<T, ForceLoad> ||
                          std::is_same_v<T, MomentLoad> ||
                          std::is_same_v<T, TempLoad>) {
                if (!nodes.count(l.node))
                    throw SolverError(std::format(
                        "Load references undefined node {}", l.node.value));
            } else if constexpr (std::is_same_v<T, PloadLoad>) {
                auto missing_node = std::find_if(l.nodes.begin(), l.nodes.end(),
                    [&](NodeId nid) { return !nodes.count(nid); });
                if (missing_node != l.nodes.end())
                    throw SolverError(std::format(
                        "PLOAD references undefined node {}", missing_node->value));
            } else if constexpr (std::is_same_v<T, Pload1Load> ||
                                 std::is_same_v<T, Pload2Load> ||
                                 std::is_same_v<T, Pload4Load>) {
                if (!element_ids.count(l.element))
                    throw SolverError(std::format(
                        "Load references undefined element {}", l.element.value));
                if constexpr (std::is_same_v<T, Pload4Load>) {
                    if (l.face_node1 && !nodes.count(*l.face_node1))
                        throw SolverError(std::format(
                            "PLOAD4 references undefined face node {}", l.face_node1->value));
                    if (l.face_node34 && !nodes.count(*l.face_node34))
                        throw SolverError(std::format(
                            "PLOAD4 references undefined face node {}", l.face_node34->value));
                }
            } else if constexpr (std::is_same_v<T, Accel1Load>) {
                auto missing_node = std::find_if(l.nodes.begin(), l.nodes.end(),
                    [&](NodeId nid) { return !nodes.count(nid); });
                if (missing_node != l.nodes.end())
                    throw SolverError(std::format(
                        "ACCEL1 references undefined node {}", missing_node->value));
            }
        }, load);
    }

    // Check analysis case references
    for (const auto& sc : analysis.subcases) {
        if (sc.load_set.value != 0) {
            bool found = std::any_of(loads.begin(), loads.end(), [&](const Load& l){
                return std::visit([&](const auto& ll){ return ll.sid == sc.load_set; }, l);
            });
            if (!found && sc.load_set.value != 0) {
                // Not necessarily an error — thermal only case may have no FORCE loads
            }
        }
    }
}

void Model::resolve_coordinates() {
    // Step 1: Resolve any CORD1x systems (defining points are node IDs in basic).
    // CORD1x nodes are always in basic, so just look them up.
    for (auto& [cid, cs] : coord_systems) {
        if (!cs.is_cord1)
            continue;
        auto lookup_node = [&](int nid, const char* role) -> Vec3 {
            auto it = nodes.find(NodeId{nid});
            if (it == nodes.end())
                throw ParseError(std::format(
                    "CORD1x {}: {} node {} not found", cid.value, role, nid));
            return it->second.position; // positions not yet resolved → must be basic
        };
        Vec3 a = lookup_node(cs.def_node_a, "A");
        Vec3 b = lookup_node(cs.def_node_b, "B");
        Vec3 c = lookup_node(cs.def_node_c, "C");
        build_axes(cs, a, b, c);
    }

    // Step 2: Resolve CORD2x systems with RID≠0 in topological order.
    // Build dependency graph: cs → its RID (parent).
    // Iteratively resolve in topological order (basic = CoordId{0} is the root).
    std::unordered_set<CoordId> resolved;
    // CoordId{0} (basic) is always resolved
    resolved.insert(CoordId{0});
    // Mark CORD1x and systems with rid=0 as resolved after building their axes
    for (auto& [cid, cs] : coord_systems) {
        if (cs.rid == CoordId{0} && !cs.is_cord1) {
            // Defining points already in basic → build axes directly
            build_axes(cs, cs.pt_a, cs.pt_b, cs.pt_c);
            resolved.insert(cid);
        } else if (cs.is_cord1) {
            resolved.insert(cid);
        }
    }

    // Iteratively resolve remaining
    bool progress = true;
    while (progress) {
        progress = false;
        for (auto& [cid, cs] : coord_systems) {
            if (resolved.count(cid))
                continue;
            if (!resolved.count(cs.rid))
                continue; // parent not yet resolved
            // Parent is resolved: transform defining points from RID to basic
            const CoordSys& parent = coord_systems.at(cs.rid);
            Vec3 a_basic = to_basic(parent, cs.pt_a);
            Vec3 b_basic = to_basic(parent, cs.pt_b);
            Vec3 c_basic = to_basic(parent, cs.pt_c);
            build_axes(cs, a_basic, b_basic, c_basic);
            resolved.insert(cid);
            progress = true;
        }
    }

    // Detect unresolved (circular references)
    for (const auto& [cid, cs] : coord_systems) {
        if (!resolved.count(cid))
            throw ParseError(std::format(
                "Coordinate system {}: circular or unresolvable reference chain "
                "(RID={} is part of a cycle)", cid.value, cs.rid.value));
    }

    // Step 3: Transform all GridPoint positions from CP to basic Cartesian.
    for (auto& [nid, gp] : nodes) {
        if (gp.cp == CoordId{0})
            continue; // already in basic
        auto it = coord_systems.find(gp.cp);
        if (it == coord_systems.end())
            throw ParseError(std::format(
                "Node {}: CP={} references undefined coordinate system",
                nid.value, gp.cp.value));
        gp.position = to_basic(it->second, gp.position);
        gp.cp = CoordId{0}; // mark as resolved
    }
}

} // namespace vibestran
