#pragma once
// include/elements/ctria3.hpp
// CTRIA3: 3-node constant-strain triangular shell element.
//
// Formulation:
//   - Membrane: CST (Constant Strain Triangle), plane-stress
//   - Bending:  DKT (Discrete Kirchhoff Triangle)
//   - 6 DOF per node: [u, v, w, θx, θy, θz]
//   - Closed-form integration (1-point centroid for membrane CST)
//
// Reference:
//   Batoz, "An explicit formulation for an efficient triangular plate-bending
//   element", IJNME, 1982.

#include "elements/element_base.hpp"
#include <array>

namespace vibestran {

class CTria3 : public ElementBase {
public:
    static constexpr int NUM_NODES = 3;
    static constexpr int DOF_PER_NODE = 6;
    static constexpr int NUM_DOFS = NUM_NODES * DOF_PER_NODE; // 18

    CTria3(ElementId eid,
           PropertyId pid,
           std::array<NodeId, NUM_NODES> node_ids,
           const Model& model);

    [[nodiscard]] ElementType type()     const noexcept override { return ElementType::CTRIA3; }
    [[nodiscard]] ElementId   id()       const noexcept override { return eid_; }
    [[nodiscard]] int         num_dofs() const noexcept override { return NUM_DOFS; }

    [[nodiscard]] LocalKe stiffness_matrix() const override;
    [[nodiscard]] LocalKe mass_matrix() const override;
    [[nodiscard]] LocalFe thermal_load(std::span<const double> temperatures,
                                        double t_ref) const override;
    [[nodiscard]] std::vector<EqIndex> global_dof_indices(const DofMap&) const override;
    [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
        return std::span<const NodeId>{nodes_.data(), NUM_NODES};
    }

private:
    ElementId   eid_;
    PropertyId  pid_;
    std::array<NodeId, NUM_NODES>  nodes_;
    const Model& model_;

    std::array<Vec3, 3> node_coords() const;
    Eigen::Matrix3d plane_stress_D() const;
    double thickness() const;
    const Mat1& material() const;
    const PShell& pshell() const;

    /// Compute membrane stiffness (CST) from local in-plane coordinates xl[3], yl[3]
    Eigen::MatrixXd membrane_stiffness(const std::array<double,3>& xl,
                                        const std::array<double,3>& yl) const;

    /// Compute bending stiffness (DKT) from local in-plane coordinates xl[3], yl[3]
    Eigen::MatrixXd bending_stiffness(const std::array<double,3>& xl,
                                       const std::array<double,3>& yl) const;
};

} // namespace vibestran
