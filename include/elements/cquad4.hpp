#pragma once
// include/elements/cquad4.hpp
// CQUAD4: 4-node isoparametric quadrilateral shell element.
//
// Formulation:
//   - Membrane: bilinear isoparametric (Q4), plane-stress
//   - Bending:  Mindlin-Reissner (first-order shear deformation)
//   - 6 DOF per node: [u, v, w, θx, θy, θz]
//   - 2x2 Gauss quadrature for membrane+bending, 1-point for shear (reduced)
//
// Reference:
//   Cook et al., "Concepts and Applications of FEA", 4th ed., ch. 7

#include "elements/element_base.hpp"
#include <array>

namespace nastran {

class CQuad4 : public ElementBase {
public:
    static constexpr int NUM_NODES = 4;
    static constexpr int DOF_PER_NODE = 6;
    static constexpr int NUM_DOFS = NUM_NODES * DOF_PER_NODE; // 24

    CQuad4(ElementId eid,
           PropertyId pid,
           std::array<NodeId, NUM_NODES> node_ids,
           const Model& model);

    [[nodiscard]] ElementType type()     const noexcept override { return ElementType::CQUAD4; }
    [[nodiscard]] ElementId   id()       const noexcept override { return eid_; }
    [[nodiscard]] int         num_dofs() const noexcept override { return NUM_DOFS; }

    [[nodiscard]] LocalKe stiffness_matrix() const override;
    [[nodiscard]] LocalFe thermal_load(std::span<const double> temperatures,
                                        double t_ref) const override;
    [[nodiscard]] std::vector<EqIndex> global_dof_indices(const DofMap&) const override;
    [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
        return std::span<const NodeId>{nodes_.data(), NUM_NODES};
    }

    // ── Public helpers (also used by stress recovery) ────────────────────────

    /// Shape function values and derivatives at (xi, eta) — exposed for recovery
    struct ShapeData {
        std::array<double, 4> N;   // shape functions
        std::array<double, 4> dNdxi;
        std::array<double, 4> dNdeta;
    };
    static ShapeData shape_functions(double xi, double eta) noexcept;

private:
    ElementId   eid_;
    PropertyId  pid_;
    std::array<NodeId, NUM_NODES>  nodes_;
    const Model& model_;

    /// Node coordinates in local element frame
    std::array<Vec3, 4> node_coords() const;

    /// Build the plane-stress constitutive matrix D (3×3) for membrane
    Eigen::Matrix3d membrane_D() const;

    /// Build the bending constitutive matrix Db (3×3)
    Eigen::Matrix3d bending_D() const;

    /// Compute element thickness and material
    double thickness() const;
    const Mat1& material() const;
    const PShell& pshell() const;
};

// ── CQuad4Mitc4 ──────────────────────────────────────────────────────────────
// CQUAD4 with MITC4 (Assumed Natural Strain) transverse shear.
// Membrane and bending formulations identical to CQuad4.
// Transverse shear uses 4 tying points at edge midpoints (Bathe & Dvorkin 1985),
// eliminating shear locking in thin-plate applications.

class CQuad4Mitc4 : public ElementBase {
public:
    static constexpr int NUM_NODES = 4;
    static constexpr int DOF_PER_NODE = 6;
    static constexpr int NUM_DOFS = NUM_NODES * DOF_PER_NODE; // 24

    CQuad4Mitc4(ElementId eid,
                PropertyId pid,
                std::array<NodeId, NUM_NODES> node_ids,
                const Model& model);

    [[nodiscard]] ElementType type()     const noexcept override { return ElementType::CQUAD4; }
    [[nodiscard]] ElementId   id()       const noexcept override { return eid_; }
    [[nodiscard]] int         num_dofs() const noexcept override { return NUM_DOFS; }

    [[nodiscard]] LocalKe stiffness_matrix() const override;
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

    std::array<Vec3, 4> node_coords() const;
    Eigen::Matrix3d membrane_D() const;
    Eigen::Matrix3d bending_D() const;
    double thickness() const;
    const Mat1& material() const;
    const PShell& pshell() const;
};

} // namespace nastran
