#pragma once
// include/elements/line_elements.hpp
// 1-D structural and scalar element families:
//   - CBAR / CBEAM
//   - CBUSH
//   - CELAS1 / CELAS2
//   - CMASS1 / CMASS2

#include "elements/element_base.hpp"
#include <array>

namespace vibestran {

class CBarBeamElement : public ElementBase {
public:
  static constexpr int NUM_NODES = 2;
  static constexpr int DOF_PER_NODE = 6;
  static constexpr int NUM_DOFS = NUM_NODES * DOF_PER_NODE;

  CBarBeamElement(ElementType type, ElementId eid, PropertyId pid,
                  std::array<NodeId, NUM_NODES> node_ids, const Model &model,
                  std::optional<Vec3> orientation,
                  std::optional<NodeId> g0);

  [[nodiscard]] ElementType type() const noexcept override { return type_; }
  [[nodiscard]] ElementId id() const noexcept override { return eid_; }
  [[nodiscard]] int num_dofs() const noexcept override { return NUM_DOFS; }

  [[nodiscard]] LocalKe stiffness_matrix() const override;
  [[nodiscard]] LocalKe mass_matrix() const override;
  [[nodiscard]] LocalFe thermal_load(std::span<const double> temperatures,
                                     double t_ref) const override;
  [[nodiscard]] std::vector<EqIndex>
  global_dof_indices(const DofMap &dof_map) const override;
  [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
    return std::span<const NodeId>{nodes_.data(), NUM_NODES};
  }

private:
  ElementType type_;
  ElementId eid_;
  PropertyId pid_;
  std::array<NodeId, NUM_NODES> nodes_;
  const Model &model_;
  std::optional<Vec3> orientation_;
  std::optional<NodeId> g0_;
};

class CBushElement : public ElementBase {
public:
  static constexpr int NUM_NODES = 2;
  static constexpr int DOF_PER_NODE = 6;
  static constexpr int NUM_DOFS = NUM_NODES * DOF_PER_NODE;

  CBushElement(ElementId eid, PropertyId pid,
               std::array<NodeId, NUM_NODES> node_ids, const Model &model,
               std::optional<Vec3> orientation,
               std::optional<NodeId> g0,
               std::optional<CoordId> cid);

  [[nodiscard]] ElementType type() const noexcept override {
    return ElementType::CBUSH;
  }
  [[nodiscard]] ElementId id() const noexcept override { return eid_; }
  [[nodiscard]] int num_dofs() const noexcept override { return NUM_DOFS; }

  [[nodiscard]] LocalKe stiffness_matrix() const override;
  [[nodiscard]] LocalKe mass_matrix() const override;
  [[nodiscard]] LocalFe thermal_load(std::span<const double> temperatures,
                                     double t_ref) const override;
  [[nodiscard]] std::vector<EqIndex>
  global_dof_indices(const DofMap &dof_map) const override;
  [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
    return std::span<const NodeId>{nodes_.data(), NUM_NODES};
  }

private:
  ElementId eid_;
  PropertyId pid_;
  std::array<NodeId, NUM_NODES> nodes_;
  const Model &model_;
  std::optional<Vec3> orientation_;
  std::optional<NodeId> g0_;
  std::optional<CoordId> cid_;
};

class ScalarSpringElement : public ElementBase {
public:
  ScalarSpringElement(ElementType type, ElementId eid, PropertyId pid,
                      std::vector<NodeId> nodes, std::array<int, 2> components,
                      double value, const Model &model);

  [[nodiscard]] ElementType type() const noexcept override { return type_; }
  [[nodiscard]] ElementId id() const noexcept override { return eid_; }
  [[nodiscard]] int num_dofs() const noexcept override {
    return static_cast<int>(nodes_.size());
  }

  [[nodiscard]] LocalKe stiffness_matrix() const override;
  [[nodiscard]] LocalKe mass_matrix() const override;
  [[nodiscard]] LocalFe thermal_load(std::span<const double> temperatures,
                                     double t_ref) const override;
  [[nodiscard]] std::vector<EqIndex>
  global_dof_indices(const DofMap &dof_map) const override;
  [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
    return nodes_;
  }

private:
  [[nodiscard]] double stiffness_value() const;

  ElementType type_;
  ElementId eid_;
  PropertyId pid_;
  std::vector<NodeId> nodes_;
  std::array<int, 2> components_;
  double value_;
  const Model &model_;
};

class ScalarMassElement : public ElementBase {
public:
  ScalarMassElement(ElementType type, ElementId eid, PropertyId pid,
                    std::vector<NodeId> nodes, std::array<int, 2> components,
                    double value, const Model &model);

  [[nodiscard]] ElementType type() const noexcept override { return type_; }
  [[nodiscard]] ElementId id() const noexcept override { return eid_; }
  [[nodiscard]] int num_dofs() const noexcept override {
    return static_cast<int>(nodes_.size());
  }

  [[nodiscard]] LocalKe stiffness_matrix() const override;
  [[nodiscard]] LocalKe mass_matrix() const override;
  [[nodiscard]] LocalFe thermal_load(std::span<const double> temperatures,
                                     double t_ref) const override;
  [[nodiscard]] std::vector<EqIndex>
  global_dof_indices(const DofMap &dof_map) const override;
  [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
    return nodes_;
  }

private:
  [[nodiscard]] double mass_value() const;

  ElementType type_;
  ElementId eid_;
  PropertyId pid_;
  std::vector<NodeId> nodes_;
  std::array<int, 2> components_;
  double value_;
  const Model &model_;
};

[[nodiscard]] LocalFe compute_pload1_equivalent_load(const ElementData &elem,
                                                     const Model &model,
                                                     const Pload1Load &load);

} // namespace vibestran
