#include "solver/linear_static.hpp"

#include "core/coord_sys.hpp"
#include "elements/element_factory.hpp"
#include "elements/line_elements.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <numeric>
#include <numbers>
#include <optional>
#include <unordered_map>

namespace vibestran {

namespace {

constexpr double GP2 = 1.0 / std::numbers::sqrt3_v<double>;
constexpr std::array<double, 2> QUAD_GAUSS{-GP2, GP2};
constexpr std::array<double, 2> QUAD_WEIGHTS{1.0, 1.0};

struct DirectionSpec {
  CoordId cid{0};
  Vec3 direction{0.0, 0.0, 0.0};
};

struct QuadShapeData {
  std::array<double, 4> N{};
  std::array<double, 4> dN_dxi{};
  std::array<double, 4> dN_deta{};
};

struct Tri6ShapeData {
  std::array<double, 6> N{};
  std::array<double, 6> dN_dxi{};
  std::array<double, 6> dN_deta{};
};

bool is_shell_element(ElementType type) noexcept {
  return type == ElementType::CQUAD4 || type == ElementType::CTRIA3;
}

bool is_supported_solid_pressure_element(ElementType type) noexcept {
  return type == ElementType::CHEXA8 || type == ElementType::CPENTA6 ||
         type == ElementType::CTETRA4 || type == ElementType::CTETRA10;
}

int dofs_per_node(const ElementType type) {
  return is_shell_element(type) ? 6 : 3;
}

int num_dofs_for_element(const ElementData &elem) {
  return dofs_per_node(elem.type) * static_cast<int>(elem.nodes.size());
}

Vec3 scaled(const Vec3 &v, const double s) noexcept { return v * s; }

Vec3 normalize_or_throw(const Vec3 &v, const std::string &context) {
  const double norm = v.norm();
  if (norm < 1e-14) {
    throw SolverError(std::format("{}: direction vector has near-zero norm",
                                  context));
  }
  return v * (1.0 / norm);
}

Vec3 direction_in_basic(const Model &model, const DirectionSpec &spec,
                        const Vec3 &position, const std::string &context) {
  Vec3 direction = spec.direction;
  if (spec.cid.value != 0) {
    auto cs_it = model.coord_systems.find(spec.cid);
    if (cs_it == model.coord_systems.end()) {
      throw SolverError(std::format(
          "{}: coordinate system {} not found for pressure direction", context,
          spec.cid.value));
    }
    const Mat3 T3 = rotation_matrix(cs_it->second, position);
    direction = apply_rotation(T3, direction);
  }
  return normalize_or_throw(direction, context);
}

QuadShapeData quad_shape_functions(const double xi, const double eta) noexcept {
  QuadShapeData s;
  s.N[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
  s.N[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
  s.N[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
  s.N[3] = 0.25 * (1.0 - xi) * (1.0 + eta);

  s.dN_dxi[0] = -0.25 * (1.0 - eta);
  s.dN_dxi[1] = 0.25 * (1.0 - eta);
  s.dN_dxi[2] = 0.25 * (1.0 + eta);
  s.dN_dxi[3] = -0.25 * (1.0 + eta);

  s.dN_deta[0] = -0.25 * (1.0 - xi);
  s.dN_deta[1] = -0.25 * (1.0 + xi);
  s.dN_deta[2] = 0.25 * (1.0 + xi);
  s.dN_deta[3] = 0.25 * (1.0 - xi);
  return s;
}

Tri6ShapeData tri6_shape_functions(const double xi, const double eta) noexcept {
  const double a = 1.0 - xi - eta;
  const double b = xi;
  const double c = eta;

  Tri6ShapeData s;
  s.N[0] = a * (2.0 * a - 1.0);
  s.N[1] = b * (2.0 * b - 1.0);
  s.N[2] = c * (2.0 * c - 1.0);
  s.N[3] = 4.0 * a * b;
  s.N[4] = 4.0 * b * c;
  s.N[5] = 4.0 * c * a;

  s.dN_dxi[0] = 1.0 - 4.0 * a;
  s.dN_dxi[1] = 4.0 * b - 1.0;
  s.dN_dxi[2] = 0.0;
  s.dN_dxi[3] = 4.0 * (a - b);
  s.dN_dxi[4] = 4.0 * c;
  s.dN_dxi[5] = -4.0 * c;

  s.dN_deta[0] = 1.0 - 4.0 * a;
  s.dN_deta[1] = 0.0;
  s.dN_deta[2] = 4.0 * c - 1.0;
  s.dN_deta[3] = -4.0 * b;
  s.dN_deta[4] = 4.0 * b;
  s.dN_deta[5] = 4.0 * (a - c);
  return s;
}

Vec3 quad_position(const std::array<Vec3, 4> &coords,
                   const std::array<double, 4> &shape) noexcept {
  Vec3 x;
  for (int i = 0; i < 4; ++i)
    x = x + coords[i] * shape[i];
  return x;
}

Vec3 tri_position(const std::array<Vec3, 3> &coords, const double n1,
                  const double n2, const double n3) noexcept {
  return coords[0] * n1 + coords[1] * n2 + coords[2] * n3;
}

Vec3 tri6_position(const std::array<Vec3, 6> &coords,
                   const std::array<double, 6> &shape) noexcept {
  Vec3 x;
  for (int i = 0; i < 6; ++i)
    x = x + coords[i] * shape[i];
  return x;
}

std::array<Vec3, 4> integrate_quad_surface_load(
    const std::array<Vec3, 4> &coords, const std::array<double, 4> &pressures,
    const Model &model, const std::optional<DirectionSpec> &direction,
    const double normal_sign, const std::string &context) {
  std::array<Vec3, 4> nodal_forces{};

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      const double xi = QUAD_GAUSS[i];
      const double eta = QUAD_GAUSS[j];
      const QuadShapeData shape = quad_shape_functions(xi, eta);

      Vec3 dx_dxi;
      Vec3 dx_deta;
      double pressure = 0.0;
      for (int a = 0; a < 4; ++a) {
        dx_dxi = dx_dxi + coords[a] * shape.dN_dxi[a];
        dx_deta = dx_deta + coords[a] * shape.dN_deta[a];
        pressure += shape.N[a] * pressures[a];
      }

      const Vec3 area_vector = dx_dxi.cross(dx_deta);
      const Vec3 position = quad_position(coords, shape.N);

      Vec3 traction_measure;
      if (direction) {
        const Vec3 dir_basic =
            direction_in_basic(model, *direction, position, context);
        traction_measure = scaled(dir_basic, pressure * area_vector.norm());
      } else {
        traction_measure = scaled(area_vector, normal_sign * pressure);
      }

      const double weight = QUAD_WEIGHTS[i] * QUAD_WEIGHTS[j];
      for (int a = 0; a < 4; ++a) {
        nodal_forces[a] =
            nodal_forces[a] + traction_measure * (shape.N[a] * weight);
      }
    }
  }

  return nodal_forces;
}

std::array<Vec3, 3> integrate_tri3_surface_load(
    const std::array<Vec3, 3> &coords, const std::array<double, 3> &pressures,
    const Model &model, const std::optional<DirectionSpec> &direction,
    const double normal_sign, const std::string &context) {
  static constexpr std::array<std::array<double, 2>, 3> tri_gauss{{
      {1.0 / 6.0, 1.0 / 6.0},
      {2.0 / 3.0, 1.0 / 6.0},
      {1.0 / 6.0, 2.0 / 3.0},
  }};
  static constexpr double tri_weight = 1.0 / 6.0;

  std::array<Vec3, 3> nodal_forces{};
  const Vec3 dx_dxi = coords[1] - coords[0];
  const Vec3 dx_deta = coords[2] - coords[0];
  const Vec3 area_vector = dx_dxi.cross(dx_deta);
  const double area_scale = area_vector.norm();

  for (const auto &gp : tri_gauss) {
    const double xi = gp[0];
    const double eta = gp[1];
    const double n1 = 1.0 - xi - eta;
    const double n2 = xi;
    const double n3 = eta;
    const std::array<double, 3> shape{n1, n2, n3};
    const double pressure =
        n1 * pressures[0] + n2 * pressures[1] + n3 * pressures[2];

    Vec3 traction_measure;
    if (direction) {
      const Vec3 position = tri_position(coords, n1, n2, n3);
      const Vec3 dir_basic =
          direction_in_basic(model, *direction, position, context);
      traction_measure = scaled(dir_basic, pressure * area_scale);
    } else {
      traction_measure = scaled(area_vector, normal_sign * pressure);
    }

    for (int a = 0; a < 3; ++a) {
      nodal_forces[a] =
          nodal_forces[a] + traction_measure * (shape[a] * tri_weight);
    }
  }

  return nodal_forces;
}

std::array<Vec3, 6> integrate_tri6_surface_load(
    const std::array<Vec3, 6> &coords, const std::array<double, 3> &pressures,
    const Model &model, const std::optional<DirectionSpec> &direction,
    const double normal_sign, const std::string &context) {
  // Dunavant degree-4 rule. The tabulated weights sum to 1 on the reference
  // triangle, so multiply by 0.5 for the standard area-1/2 simplex.
  static constexpr std::array<std::array<double, 3>, 6> tri_gauss{{
      {0.816847572980459, 0.091576213509771, 0.109951743655322 * 0.5},
      {0.091576213509771, 0.816847572980459, 0.109951743655322 * 0.5},
      {0.091576213509771, 0.091576213509771, 0.109951743655322 * 0.5},
      {0.108103018168070, 0.445948490915965, 0.223381589678011 * 0.5},
      {0.445948490915965, 0.108103018168070, 0.223381589678011 * 0.5},
      {0.445948490915965, 0.445948490915965, 0.223381589678011 * 0.5},
  }};

  std::array<Vec3, 6> nodal_forces{};

  for (const auto &gp : tri_gauss) {
    const double xi = gp[0];
    const double eta = gp[1];
    const double weight = gp[2];
    const double n1 = 1.0 - xi - eta;
    const double n2 = xi;
    const double n3 = eta;

    const Tri6ShapeData shape = tri6_shape_functions(xi, eta);
    Vec3 dx_dxi;
    Vec3 dx_deta;
    for (int a = 0; a < 6; ++a) {
      dx_dxi = dx_dxi + coords[a] * shape.dN_dxi[a];
      dx_deta = dx_deta + coords[a] * shape.dN_deta[a];
    }

    const Vec3 area_vector = dx_dxi.cross(dx_deta);
    const double pressure =
        n1 * pressures[0] + n2 * pressures[1] + n3 * pressures[2];

    Vec3 traction_measure;
    if (direction) {
      const Vec3 position = tri6_position(coords, shape.N);
      const Vec3 dir_basic =
          direction_in_basic(model, *direction, position, context);
      traction_measure =
          scaled(dir_basic, pressure * area_vector.norm());
    } else {
      traction_measure = scaled(area_vector, normal_sign * pressure);
    }

    for (int a = 0; a < 6; ++a) {
      nodal_forces[a] =
          nodal_forces[a] + traction_measure * (shape.N[a] * weight);
    }
  }

  return nodal_forces;
}

std::vector<int> reverse_face_order(const std::vector<int> &face) {
  switch (face.size()) {
  case 3:
    return {face[0], face[2], face[1]};
  case 4:
    return {face[0], face[3], face[2], face[1]};
  case 6:
    return {face[0], face[2], face[1], face[5], face[4], face[3]};
  default:
    throw SolverError(std::format("Unsupported face order size {}", face.size()));
  }
}

std::vector<int> rotate_face_to_first_corner(const std::vector<int> &face,
                                             const int local_corner) {
  if (face.empty())
    return face;

  const size_t corner_count = (face.size() == 6) ? 3 : face.size();
  size_t first_pos = corner_count;
  for (size_t i = 0; i < corner_count; ++i) {
    if (face[i] == local_corner) {
      first_pos = i;
      break;
    }
  }

  if (first_pos >= corner_count) {
    throw SolverError(std::format(
        "Requested face corner {} is not part of the selected face",
        local_corner));
  }

  if (first_pos == 0)
    return face;

  if (face.size() == 3) {
    if (first_pos == 1)
      return {face[1], face[2], face[0]};
    return {face[2], face[0], face[1]};
  }

  if (face.size() == 4) {
    return {face[first_pos % 4], face[(first_pos + 1) % 4],
            face[(first_pos + 2) % 4], face[(first_pos + 3) % 4]};
  }

  if (face.size() == 6) {
    if (first_pos == 1)
      return {face[1], face[2], face[0], face[4], face[5], face[3]};
    return {face[2], face[0], face[1], face[5], face[3], face[4]};
  }

  throw SolverError(std::format("Unsupported face order size {}", face.size()));
}

std::vector<std::vector<int>> candidate_faces(const ElementData &elem) {
  switch (elem.type) {
  case ElementType::CHEXA8:
    return {
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {0, 1, 5, 4},
        {1, 2, 6, 5},
        {2, 3, 7, 6},
        {3, 0, 4, 7},
    };
  case ElementType::CPENTA6:
    return {
        {0, 1, 2},
        {3, 4, 5},
        {0, 1, 4, 3},
        {1, 2, 5, 4},
        {2, 0, 3, 5},
    };
  case ElementType::CTETRA4:
    return {
        {0, 1, 2},
        {0, 3, 1},
        {1, 3, 2},
        {0, 2, 3},
    };
  case ElementType::CTETRA10:
    return {
        {0, 1, 2, 4, 5, 6},
        {0, 3, 1, 7, 8, 4},
        {1, 3, 2, 8, 9, 5},
        {0, 2, 3, 6, 9, 7},
    };
  default:
    return {};
  }
}

Vec3 element_centroid(const ElementData &elem, const Model &model) {
  const Vec3 centroid = std::accumulate(elem.nodes.begin(), elem.nodes.end(), Vec3{},
                                        [&](const Vec3 &sum, NodeId nid) {
                                          return sum + model.node(nid).position;
                                        });
  return centroid * (1.0 / static_cast<double>(elem.nodes.size()));
}

Vec3 face_centroid(const ElementData &elem, const Model &model,
                   const std::vector<int> &face) {
  const size_t corner_count = (face.size() == 6) ? 3 : face.size();
  Vec3 centroid;
  for (size_t i = 0; i < corner_count; ++i)
    centroid = centroid + model.node(elem.nodes[face[i]]).position;
  return centroid * (1.0 / static_cast<double>(corner_count));
}

Vec3 face_normal_from_corners(const ElementData &elem, const Model &model,
                              const std::vector<int> &face) {
  const Vec3 &x1 = model.node(elem.nodes[face[0]]).position;
  const Vec3 &x2 = model.node(elem.nodes[face[1]]).position;
  const Vec3 &x3 = model.node(elem.nodes[face[2]]).position;
  return (x2 - x1).cross(x3 - x1);
}

std::vector<int> orient_face_outward(const ElementData &elem, const Model &model,
                                     const std::vector<int> &face) {
  std::vector<int> oriented = face;
  const Vec3 normal = face_normal_from_corners(elem, model, oriented);
  const Vec3 inward_hint = element_centroid(elem, model) - face_centroid(elem, model, oriented);
  if (normal.dot(inward_hint) > 0.0)
    oriented = reverse_face_order(oriented);
  return oriented;
}

std::vector<int> select_solid_face(const ElementData &elem, const Model &model,
                                   const Pload4Load &load) {
  const auto faces = candidate_faces(elem);
  if (faces.empty()) {
    throw SolverError(std::format(
        "PLOAD4 element {} uses unsupported solid type", elem.id.value));
  }

  const auto first_face =
      orient_face_outward(elem, model, faces.front());

  if (!load.face_node1 && !load.face_node34)
    return first_face;

  if (elem.type == ElementType::CTETRA4 || elem.type == ElementType::CTETRA10) {
    const NodeId g1 = *load.face_node1;
    const NodeId g4 = *load.face_node34;
    std::optional<std::vector<int>> match;

    for (const auto &face : faces) {
      const size_t corner_count = (face.size() == 6) ? 3 : face.size();
      bool has_g1 = false;
      bool has_g4 = false;
      for (size_t i = 0; i < corner_count; ++i) {
        if (elem.nodes[face[i]] == g1)
          has_g1 = true;
        if (elem.nodes[face[i]] == g4)
          has_g4 = true;
      }
      if (has_g1 && !has_g4) {
        match = orient_face_outward(elem, model, face);
        break;
      }
    }

    if (!match) {
      throw SolverError(std::format(
          "PLOAD4 on element {} could not resolve face from G1={} and G4={}",
          elem.id.value, g1.value, g4.value));
    }

    const int local_g1 = [&]() {
      for (int i = 0; i < static_cast<int>(elem.nodes.size()); ++i) {
        if (elem.nodes[static_cast<size_t>(i)] == g1)
          return i;
      }
      return -1;
    }();
    return rotate_face_to_first_corner(*match, local_g1);
  }

  const NodeId g1 = *load.face_node1;
  const NodeId g34 = *load.face_node34;

  for (const auto &face : faces) {
    const size_t corner_count = (face.size() == 6) ? 3 : face.size();
    if (corner_count != 4)
      continue;

    bool has_g1 = false;
    bool has_g34 = false;
    for (size_t i = 0; i < corner_count; ++i) {
      has_g1 = has_g1 || elem.nodes[face[i]] == g1;
      has_g34 = has_g34 || elem.nodes[face[i]] == g34;
    }
    if (!has_g1 || !has_g34)
      continue;

    std::vector<int> oriented = orient_face_outward(elem, model, face);
    int local_g1 = -1;
    for (int i = 0; i < static_cast<int>(elem.nodes.size()); ++i) {
      if (elem.nodes[static_cast<size_t>(i)] == g1) {
        local_g1 = i;
        break;
      }
    }
    if (local_g1 < 0)
      break;

    oriented = rotate_face_to_first_corner(oriented, local_g1);
    if (elem.nodes[oriented[2]] == g34)
      return oriented;
  }

  throw SolverError(std::format(
      "PLOAD4 on element {} could not resolve face from G1={} and G34={}",
      elem.id.value, g1.value, g34.value));
}

std::array<Vec3, 4> quad_face_coords(const ElementData &elem, const Model &model,
                                     const std::vector<int> &face) {
  std::array<Vec3, 4> coords{};
  for (int i = 0; i < 4; ++i)
    coords[i] = model.node(elem.nodes[face[static_cast<size_t>(i)]]).position;
  return coords;
}

std::array<Vec3, 3> tri3_face_coords(const ElementData &elem, const Model &model,
                                     const std::vector<int> &face) {
  std::array<Vec3, 3> coords{};
  for (int i = 0; i < 3; ++i)
    coords[i] = model.node(elem.nodes[face[static_cast<size_t>(i)]]).position;
  return coords;
}

std::array<Vec3, 6> tri6_face_coords(const ElementData &elem, const Model &model,
                                     const std::vector<int> &face) {
  std::array<Vec3, 6> coords{};
  for (int i = 0; i < 6; ++i)
    coords[i] = model.node(elem.nodes[face[static_cast<size_t>(i)]]).position;
  return coords;
}

void add_force_to_node(const DofMap &dof_map, const MpcHandler &mpc_handler,
                       const NodeId node, const Vec3 &force,
                       std::vector<double> &F) {
  std::array<EqIndex, 6> eqs{};
  dof_map.global_indices(node, eqs);
  std::vector<EqIndex> gdofs(eqs.begin(), eqs.end());
  std::array<double, 6> fe{force.x, force.y, force.z, 0.0, 0.0, 0.0};
  mpc_handler.apply_to_force(gdofs, fe, F);
}

std::vector<double> shell_force_vector(const ElementData &elem,
                                       const std::vector<Vec3> &forces) {
  const int ndof = num_dofs_for_element(elem);
  std::vector<double> fe(static_cast<size_t>(ndof), 0.0);
  for (size_t i = 0; i < forces.size(); ++i) {
    fe[6 * i + 0] = forces[i].x;
    fe[6 * i + 1] = forces[i].y;
    fe[6 * i + 2] = forces[i].z;
  }
  return fe;
}

std::vector<double> solid_face_force_vector(const ElementData &elem,
                                            const std::vector<int> &face,
                                            const std::vector<Vec3> &forces) {
  const int ndof = num_dofs_for_element(elem);
  std::vector<double> fe(static_cast<size_t>(ndof), 0.0);
  for (size_t i = 0; i < forces.size(); ++i) {
    const int local = face[i];
    fe[3 * local + 0] += forces[i].x;
    fe[3 * local + 1] += forces[i].y;
    fe[3 * local + 2] += forces[i].z;
  }
  return fe;
}

void apply_element_force_vector(const ElementData &elem_data, const Model &model,
                                const DofMap &dof_map,
                                const MpcHandler &mpc_handler,
                                const std::vector<double> &fe,
                                std::vector<double> &F) {
  auto elem = make_element(elem_data, model);
  auto gdofs = elem->global_dof_indices(dof_map);
  mpc_handler.apply_to_force(gdofs, fe, F);
}

Vec3 pload_triangle_force(const std::array<Vec3, 3> &coords,
                          const double pressure) {
  return scaled((coords[1] - coords[0]).cross(coords[2] - coords[0]),
                0.5 * pressure);
}

Vec3 pload_quad_force(const std::array<Vec3, 4> &coords, const double pressure) {
  const Vec3 tri1 =
      pload_triangle_force({coords[0], coords[1], coords[2]}, pressure);
  const Vec3 tri2 =
      pload_triangle_force({coords[0], coords[2], coords[3]}, pressure);
  return tri1 + tri2;
}

} // namespace

// cppcheck-suppress unusedFunction -- called from linear_static.cpp
void LinearStaticSolver::apply_pressure_loads(const Model &model,
                                              const SubCase &sc,
                                              const MpcHandler &mpc_handler,
                                              std::vector<double> &F) {
  const DofMap &dof_map = mpc_handler.full_dof_map();

  std::unordered_map<ElementId, const ElementData *> elements_by_id;
  elements_by_id.reserve(model.elements.size());
  for (const auto &elem : model.elements)
    elements_by_id.emplace(elem.id, &elem);

  for (const Load *lp : model.loads_for_set(sc.load_set)) {
    std::visit(
        [&](const auto &load) {
          using T = std::decay_t<decltype(load)>;

          if constexpr (std::is_same_v<T, PloadLoad>) {
            if (load.nodes.size() == 3) {
              const std::array<Vec3, 3> coords{
                  model.node(load.nodes[0]).position,
                  model.node(load.nodes[1]).position,
                  model.node(load.nodes[2]).position,
              };
              const Vec3 total_force = pload_triangle_force(coords, load.pressure);
              for (NodeId nid : load.nodes)
                add_force_to_node(dof_map, mpc_handler, nid,
                                  scaled(total_force, 1.0 / 3.0), F);
            } else {
              const std::array<Vec3, 4> coords{
                  model.node(load.nodes[0]).position,
                  model.node(load.nodes[1]).position,
                  model.node(load.nodes[2]).position,
                  model.node(load.nodes[3]).position,
              };
              const Vec3 total_force = pload_quad_force(coords, load.pressure);
              for (NodeId nid : load.nodes)
                add_force_to_node(dof_map, mpc_handler, nid,
                                  scaled(total_force, 0.25), F);
            }

          } else if constexpr (std::is_same_v<T, Pload1Load>) {
            const auto it = elements_by_id.find(load.element);
            if (it == elements_by_id.end()) {
              throw SolverError(std::format(
                  "PLOAD1 references undefined element {}",
                  load.element.value));
            }
            const ElementData &elem = *it->second;
            const LocalFe fe = compute_pload1_equivalent_load(elem, model, load);
            const std::vector<double> fe_vec(fe.data(), fe.data() + fe.size());
            apply_element_force_vector(elem, model, dof_map, mpc_handler, fe_vec,
                                       F);

          } else if constexpr (std::is_same_v<T, Pload2Load>) {
            const auto it = elements_by_id.find(load.element);
            if (it == elements_by_id.end()) {
              throw SolverError(std::format(
                  "PLOAD2 references undefined element {}", load.element.value));
            }
            const ElementData &elem = *it->second;

            if (elem.type == ElementType::CQUAD4) {
              const std::array<Vec3, 4> coords{
                  model.node(elem.nodes[0]).position,
                  model.node(elem.nodes[1]).position,
                  model.node(elem.nodes[2]).position,
                  model.node(elem.nodes[3]).position,
              };
              const std::array<double, 4> pressures{
                  load.pressure, load.pressure, load.pressure, load.pressure};
              const auto nodal_forces = integrate_quad_surface_load(
                  coords, pressures, model, std::nullopt, 1.0,
                  std::format("PLOAD2 element {}", elem.id.value));
              const std::vector<Vec3> forces(nodal_forces.begin(),
                                             nodal_forces.end());
              apply_element_force_vector(elem, model, dof_map, mpc_handler,
                                         shell_force_vector(elem, forces), F);
            } else if (elem.type == ElementType::CTRIA3) {
              const std::array<Vec3, 3> coords{
                  model.node(elem.nodes[0]).position,
                  model.node(elem.nodes[1]).position,
                  model.node(elem.nodes[2]).position,
              };
              const std::array<double, 3> pressures{
                  load.pressure, load.pressure, load.pressure};
              const auto nodal_forces = integrate_tri3_surface_load(
                  coords, pressures, model, std::nullopt, 1.0,
                  std::format("PLOAD2 element {}", elem.id.value));
              const std::vector<Vec3> forces(nodal_forces.begin(),
                                             nodal_forces.end());
              apply_element_force_vector(elem, model, dof_map, mpc_handler,
                                         shell_force_vector(elem, forces), F);
            } else {
              throw SolverError(std::format(
                  "PLOAD2 on element {} is only supported for CQUAD4 and CTRIA3",
                  elem.id.value));
            }

          } else if constexpr (std::is_same_v<T, Pload4Load>) {
            const auto it = elements_by_id.find(load.element);
            if (it == elements_by_id.end()) {
              throw SolverError(std::format(
                  "PLOAD4 references undefined element {}", load.element.value));
            }
            const ElementData &elem = *it->second;
            const std::optional<DirectionSpec> direction =
                load.use_vector ? std::optional<DirectionSpec>{
                                      DirectionSpec{load.cid, load.direction}}
                                : std::nullopt;
            const std::string context =
                std::format("PLOAD4 element {}", elem.id.value);

            if (elem.type == ElementType::CQUAD4) {
              const std::array<Vec3, 4> coords{
                  model.node(elem.nodes[0]).position,
                  model.node(elem.nodes[1]).position,
                  model.node(elem.nodes[2]).position,
                  model.node(elem.nodes[3]).position,
              };
              const auto nodal_forces = integrate_quad_surface_load(
                  coords, load.pressures, model, direction, 1.0, context);
              const std::vector<Vec3> forces(nodal_forces.begin(),
                                             nodal_forces.end());
              apply_element_force_vector(elem, model, dof_map, mpc_handler,
                                         shell_force_vector(elem, forces), F);
            } else if (elem.type == ElementType::CTRIA3) {
              const std::array<Vec3, 3> coords{
                  model.node(elem.nodes[0]).position,
                  model.node(elem.nodes[1]).position,
                  model.node(elem.nodes[2]).position,
              };
              const std::array<double, 3> tri_pressures{
                  load.pressures[0], load.pressures[1], load.pressures[2]};
              const auto nodal_forces = integrate_tri3_surface_load(
                  coords, tri_pressures, model, direction, 1.0, context);
              const std::vector<Vec3> forces(nodal_forces.begin(),
                                             nodal_forces.end());
              apply_element_force_vector(elem, model, dof_map, mpc_handler,
                                         shell_force_vector(elem, forces), F);
            } else if (is_supported_solid_pressure_element(elem.type)) {
              const std::vector<int> face = select_solid_face(elem, model, load);

              if (face.size() == 4) {
                const auto coords = quad_face_coords(elem, model, face);
                const auto nodal_forces = integrate_quad_surface_load(
                    coords, load.pressures, model, direction, -1.0, context);
                const std::vector<Vec3> forces(nodal_forces.begin(),
                                               nodal_forces.end());
                apply_element_force_vector(
                    elem, model, dof_map, mpc_handler,
                    solid_face_force_vector(elem, face, forces), F);
              } else if (face.size() == 3) {
                const auto coords = tri3_face_coords(elem, model, face);
                const std::array<double, 3> tri_pressures{
                    load.pressures[0], load.pressures[1], load.pressures[2]};
                const auto nodal_forces = integrate_tri3_surface_load(
                    coords, tri_pressures, model, direction, -1.0, context);
                const std::vector<Vec3> forces(nodal_forces.begin(),
                                               nodal_forces.end());
                apply_element_force_vector(
                    elem, model, dof_map, mpc_handler,
                    solid_face_force_vector(elem, face, forces), F);
              } else if (face.size() == 6) {
                const auto coords = tri6_face_coords(elem, model, face);
                const std::array<double, 3> tri_pressures{
                    load.pressures[0], load.pressures[1], load.pressures[2]};
                const auto nodal_forces = integrate_tri6_surface_load(
                    coords, tri_pressures, model, direction, -1.0, context);
                const std::vector<Vec3> forces(nodal_forces.begin(),
                                               nodal_forces.end());
                apply_element_force_vector(
                    elem, model, dof_map, mpc_handler,
                    solid_face_force_vector(elem, face, forces), F);
              } else {
                throw SolverError(std::format(
                    "PLOAD4 element {} resolved an unsupported face size {}",
                    elem.id.value, face.size()));
              }
            } else {
              throw SolverError(std::format(
                  "PLOAD4 on element {} uses unsupported element type",
                  elem.id.value));
            }
          }
        },
        *lp);
  }
}

} // namespace vibestran
