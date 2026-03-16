// src/elements/element_factory.cpp
#include "elements/element_factory.hpp"
#include "elements/cquad4.hpp"
#include "elements/ctria3.hpp"
#include "elements/solid_elements.hpp"
#include <format>

namespace nastran {

std::unique_ptr<ElementBase> make_element(const ElementData& data, const Model& model) {
    switch (data.type) {
        case ElementType::CQUAD4: {
            if (data.nodes.size() != 4)
                throw SolverError(std::format("CQUAD4 {} needs 4 nodes, got {}", data.id.value, data.nodes.size()));
            std::array<NodeId,4> nids{data.nodes[0],data.nodes[1],data.nodes[2],data.nodes[3]};
            // Dispatch based on shell formulation
            const auto& prop = model.property(data.pid);
            if (std::holds_alternative<PShell>(prop)) {
                if (std::get<PShell>(prop).shell_form == ShellFormulation::MITC4)
                    return std::make_unique<CQuad4Mitc4>(data.id, data.pid, nids, model);
            }
            return std::make_unique<CQuad4>(data.id, data.pid, nids, model);
        }
        case ElementType::CTRIA3: {
            if (data.nodes.size() != 3)
                throw SolverError(std::format("CTRIA3 {} needs 3 nodes, got {}", data.id.value, data.nodes.size()));
            std::array<NodeId,3> nids{data.nodes[0],data.nodes[1],data.nodes[2]};
            return std::make_unique<CTria3>(data.id, data.pid, nids, model);
        }
        case ElementType::CHEXA8: {
            if (data.nodes.size() != 8)
                throw SolverError(std::format("CHEXA8 {} needs 8 nodes, got {}", data.id.value, data.nodes.size()));
            std::array<NodeId,8> nids{data.nodes[0],data.nodes[1],data.nodes[2],data.nodes[3],
                                       data.nodes[4],data.nodes[5],data.nodes[6],data.nodes[7]};
            // Dispatch based on solid formulation
            const auto& prop = model.property(data.pid);
            if (std::holds_alternative<PSolid>(prop)) {
                if (std::get<PSolid>(prop).isop == SolidFormulation::EAS)
                    return std::make_unique<CHexa8Eas>(data.id, data.pid, nids, model);
            }
            return std::make_unique<CHexa8>(data.id, data.pid, nids, model);
        }
        case ElementType::CTETRA4: {
            if (data.nodes.size() != 4)
                throw SolverError(std::format("CTETRA4 {} needs 4 nodes, got {}", data.id.value, data.nodes.size()));
            std::array<NodeId,4> nids{data.nodes[0],data.nodes[1],data.nodes[2],data.nodes[3]};
            return std::make_unique<CTetra4>(data.id, data.pid, nids, model);
        }
        case ElementType::CTETRA10: {
            if (data.nodes.size() != 10)
                throw SolverError(std::format("CTETRA10 {} needs 10 nodes, got {}", data.id.value, data.nodes.size()));
            std::array<NodeId,10> nids{data.nodes[0],data.nodes[1],data.nodes[2],data.nodes[3],
                                        data.nodes[4],data.nodes[5],data.nodes[6],data.nodes[7],
                                        data.nodes[8],data.nodes[9]};
            return std::make_unique<CTetra10>(data.id, data.pid, nids, model);
        }
        default:
            throw SolverError(std::format("Unsupported element type for element {}", data.id.value));
    }
}

} // namespace nastran
