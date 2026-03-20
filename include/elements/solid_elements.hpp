#pragma once
// include/elements/solid_elements.hpp
// CHEXA8: 8-node isoparametric hexahedral solid element
// CTETRA4: 4-node linear tetrahedral element
//
// Both use 3 DOF per node (T1, T2, T3 translations only).
//
// CHEXA8 formulation:
//   - Trilinear isoparametric, 2×2×2 Gauss quadrature
//   - Selectively reduced integration for volumetric locking prevention
//
// CTETRA4 formulation:
//   - Constant strain tetrahedron, closed-form integration

#include "elements/element_base.hpp"
#include <array>

namespace nastran {

// ── CHEXA8 ───────────────────────────────────────────────────────────────────

class CHexa8 : public ElementBase {
public:
    static constexpr int NUM_NODES    = 8;
    static constexpr int DOF_PER_NODE = 3;
    static constexpr int NUM_DOFS     = NUM_NODES * DOF_PER_NODE; // 24

    CHexa8(ElementId eid,
           PropertyId pid,
           std::array<NodeId, NUM_NODES> node_ids,
           const Model& model);

    [[nodiscard]] ElementType type()     const noexcept override { return ElementType::CHEXA8; }
    [[nodiscard]] ElementId   id()       const noexcept override { return eid_; }
    [[nodiscard]] int         num_dofs() const noexcept override { return NUM_DOFS; }

    [[nodiscard]] LocalKe stiffness_matrix() const override;
    [[nodiscard]] LocalFe thermal_load(std::span<const double> temperatures,
                                        double t_ref) const override;
    [[nodiscard]] std::vector<EqIndex> global_dof_indices(const DofMap&) const override;
    [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
        return std::span<const NodeId>{nodes_.data(), NUM_NODES};
    }

    // Exposed for stress recovery in linear_static.cpp
    struct ShapeData {
        std::array<double, 8> N;
        std::array<double, 8> dNdxi;
        std::array<double, 8> dNdeta;
        std::array<double, 8> dNdzeta;
    };
    static ShapeData shape_functions(double xi, double eta, double zeta) noexcept;

private:
    ElementId   eid_;
    PropertyId  pid_;
    std::array<NodeId, NUM_NODES> nodes_;
    const Model& model_;

    std::array<Vec3, 8> node_coords() const;
    Eigen::Matrix<double,6,6> constitutive_D() const;
    const Mat1& material() const;
    const PSolid& psolid() const;
};

// ── CTETRA4 ──────────────────────────────────────────────────────────────────

class CTetra4 : public ElementBase {
public:
    static constexpr int NUM_NODES    = 4;
    static constexpr int DOF_PER_NODE = 3;
    static constexpr int NUM_DOFS     = NUM_NODES * DOF_PER_NODE; // 12

    CTetra4(ElementId eid,
            PropertyId pid,
            std::array<NodeId, NUM_NODES> node_ids,
            const Model& model);

    [[nodiscard]] ElementType type()     const noexcept override { return ElementType::CTETRA4; }
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
    std::array<NodeId, NUM_NODES> nodes_;
    const Model& model_;

    std::array<Vec3, 4> node_coords() const;
    Eigen::Matrix<double,6,6> constitutive_D() const;
    const Mat1& material() const;
    const PSolid& psolid() const;
};

// ── CPENTA6 ──────────────────────────────────────────────────────────────────
// 6-node linear pentahedral (wedge) element.
// Uses selectively reduced integration (SRI):
//   - Deviatoric: 6-point Gauss quadrature (3 triangle × 2 axial)
//   - Volumetric: 1-point centroidal integration

class CPenta6 : public ElementBase {
public:
    static constexpr int NUM_NODES    = 6;
    static constexpr int DOF_PER_NODE = 3;
    static constexpr int NUM_DOFS     = NUM_NODES * DOF_PER_NODE; // 18

    CPenta6(ElementId eid,
            PropertyId pid,
            std::array<NodeId, NUM_NODES> node_ids,
            const Model& model);

    [[nodiscard]] ElementType type()     const noexcept override { return ElementType::CPENTA6; }
    [[nodiscard]] ElementId   id()       const noexcept override { return eid_; }
    [[nodiscard]] int         num_dofs() const noexcept override { return NUM_DOFS; }

    [[nodiscard]] LocalKe stiffness_matrix() const override;
    [[nodiscard]] LocalFe thermal_load(std::span<const double> temperatures,
                                        double t_ref) const override;
    [[nodiscard]] std::vector<EqIndex> global_dof_indices(const DofMap&) const override;
    [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
        return std::span<const NodeId>{nodes_.data(), NUM_NODES};
    }

    // Exposed for stress recovery in linear_static.cpp
    struct ShapeData6 {
        std::array<double, 6> N;
        std::array<double, 6> dNdL1;
        std::array<double, 6> dNdL2;
        std::array<double, 6> dNdzeta;
    };
    static ShapeData6 shape_functions(double L1, double L2, double zeta) noexcept;

private:
    ElementId   eid_;
    PropertyId  pid_;
    std::array<NodeId, NUM_NODES> nodes_;
    const Model& model_;

    std::array<Vec3, 6> node_coords() const;
    Eigen::Matrix<double,6,6> constitutive_D() const;
    const Mat1& material() const;
    const PSolid& psolid() const;
};

// ── CPENTA6EAS ────────────────────────────────────────────────────────────────
// 6-node wedge with Enhanced Assumed Strain (9-mode incompatible modes).
// Identical interface to CPenta6; uses full 6-point Gauss + static
// condensation to eliminate both volumetric and bending locking.

class CPenta6Eas : public ElementBase {
public:
    static constexpr int NUM_NODES    = 6;
    static constexpr int DOF_PER_NODE = 3;
    static constexpr int NUM_DOFS     = NUM_NODES * DOF_PER_NODE; // 18

    CPenta6Eas(ElementId eid,
               PropertyId pid,
               std::array<NodeId, NUM_NODES> node_ids,
               const Model& model);

    [[nodiscard]] ElementType type()     const noexcept override { return ElementType::CPENTA6; }
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
    std::array<NodeId, NUM_NODES> nodes_;
    const Model& model_;

    std::array<Vec3, 6> node_coords() const;
    Eigen::Matrix<double,6,6> constitutive_D() const;
    const Mat1& material() const;
    const PSolid& psolid() const;
};

// ── CTETRA10 ─────────────────────────────────────────────────────────────────
// 10-node quadratic tetrahedron: 4 corners + 6 midside nodes.
// Uses 4-point Gauss quadrature (exact for degree-2 integrands).

class CTetra10 : public ElementBase {
public:
    static constexpr int NUM_NODES    = 10;
    static constexpr int DOF_PER_NODE = 3;
    static constexpr int NUM_DOFS     = NUM_NODES * DOF_PER_NODE; // 30

    CTetra10(ElementId eid,
             PropertyId pid,
             std::array<NodeId, NUM_NODES> node_ids,
             const Model& model);

    [[nodiscard]] ElementType type()     const noexcept override { return ElementType::CTETRA10; }
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
    std::array<NodeId, NUM_NODES> nodes_;
    const Model& model_;

    std::array<Vec3, NUM_NODES> node_coords() const;
    Eigen::Matrix<double,6,6> constitutive_D() const;
    const Mat1& material() const;
    const PSolid& psolid() const;

    // Shape functions in barycentric coords (L1,L2,L3; L4 = 1-L1-L2-L3)
    struct ShapeData10 {
        std::array<double, 10> N;
        std::array<double, 10> dNdL1;
        std::array<double, 10> dNdL2;
        std::array<double, 10> dNdL3;
    };
    static ShapeData10 shape_functions(double L1, double L2, double L3) noexcept;
};

// ── CHEXA8EAS ────────────────────────────────────────────────────────────────
// 8-node hex with Enhanced Assumed Strain (9-mode Wilson-Taylor incompatible
// modes). Identical interface to CHexa8; uses full 2x2x2 Gauss + static
// condensation to eliminate both volumetric and bending locking.

class CHexa8Eas : public ElementBase {
public:
    static constexpr int NUM_NODES    = 8;
    static constexpr int DOF_PER_NODE = 3;
    static constexpr int NUM_DOFS     = NUM_NODES * DOF_PER_NODE; // 24

    CHexa8Eas(ElementId eid,
              PropertyId pid,
              std::array<NodeId, NUM_NODES> node_ids,
              const Model& model);

    [[nodiscard]] ElementType type()     const noexcept override { return ElementType::CHEXA8; }
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
    std::array<NodeId, NUM_NODES> nodes_;
    const Model& model_;

    std::array<Vec3, 8> node_coords() const;
    Eigen::Matrix<double,6,6> constitutive_D() const;
    const Mat1& material() const;
    const PSolid& psolid() const;
};

} // namespace nastran
