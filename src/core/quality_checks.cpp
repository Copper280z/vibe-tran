// src/core/quality_checks.cpp
// Input deck quality checker — element quality, topology, and physical sanity.

#include "core/quality_checks.hpp"
#include "core/logger.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <map>
#include <numbers>
#include <numeric>
#include <spdlog/spdlog.h>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#ifdef HAVE_TBB
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#endif

namespace vibestran {

namespace {

// ── Geometry helpers ──────────────────────────────────────────────────────────

double edge_length(const Vec3& a, const Vec3& b) {
    return (b - a).norm();
}

double triangle_area(const Vec3& a, const Vec3& b, const Vec3& c) {
    return 0.5 * (b - a).cross(c - a).norm();
}

// Angle in degrees between two vectors (from a shared vertex)
double angle_deg(const Vec3& v1, const Vec3& v2) {
    double n1 = v1.norm();
    double n2 = v2.norm();
    if (n1 < 1e-15 || n2 < 1e-15)
        return 0.0;
    double cos_a = std::clamp(v1.dot(v2) / (n1 * n2), -1.0, 1.0);
    return std::acos(cos_a) * (180.0 / std::numbers::pi);
}

// Canonical sorted edge key from two node IDs
std::pair<int,int> edge_key(NodeId a, NodeId b) {
    return {std::min(a.value, b.value), std::max(a.value, b.value)};
}

// 3×3 determinant (row-major storage)
double det3(const double J[3][3]) {
    return J[0][0] * (J[1][1]*J[2][2] - J[1][2]*J[2][1])
         - J[0][1] * (J[1][0]*J[2][2] - J[1][2]*J[2][0])
         + J[0][2] * (J[1][0]*J[2][1] - J[1][1]*J[2][0]);
}

// ── Element type name helper ──────────────────────────────────────────────────

static const char* element_type_name(ElementType t) {
    switch (t) {
    case ElementType::CQUAD4:  return "CQUAD4";
    case ElementType::CTRIA3:  return "CTRIA3";
    case ElementType::CHEXA8:  return "CHEXA8";
    case ElementType::CHEXA20: return "CHEXA20";
    case ElementType::CTETRA4: return "CTETRA4";
    case ElementType::CTETRA10:return "CTETRA10";
    case ElementType::CPENTA6: return "CPENTA6";
    case ElementType::CBAR:    return "CBAR";
    case ElementType::CBEAM:   return "CBEAM";
    case ElementType::CBUSH:   return "CBUSH";
    case ElementType::CELAS1:  return "CELAS1";
    case ElementType::CELAS2:  return "CELAS2";
    case ElementType::CMASS1:  return "CMASS1";
    case ElementType::CMASS2:  return "CMASS2";
    }
    return "UNKNOWN";
}

// ── CHEXA8 Jacobian ───────────────────────────────────────────────────────────
// Reference element corners: node i has (r_i, s_i, t_i) per MSC/Nastran ordering.
// N1:(-1,-1,-1) N2:(+1,-1,-1) N3:(+1,+1,-1) N4:(-1,+1,-1)
// N5:(-1,-1,+1) N6:(+1,-1,+1) N7:(+1,+1,+1) N8:(-1,+1,+1)

static constexpr double kHexR[8] = {-1,+1,+1,-1,-1,+1,+1,-1};
static constexpr double kHexS[8] = {-1,-1,+1,+1,-1,-1,+1,+1};
static constexpr double kHexT[8] = {-1,-1,-1,-1,+1,+1,+1,+1};

double chexa8_detJ(double r, double s, double t, const Vec3 pts[8]) {
    double dNr[8], dNs[8], dNt[8];
    for (int i = 0; i < 8; ++i) {
        dNr[i] = kHexR[i] * (1.0 + s*kHexS[i]) * (1.0 + t*kHexT[i]) / 8.0;
        dNs[i] = kHexS[i] * (1.0 + r*kHexR[i]) * (1.0 + t*kHexT[i]) / 8.0;
        dNt[i] = kHexT[i] * (1.0 + r*kHexR[i]) * (1.0 + s*kHexS[i]) / 8.0;
    }
    double J[3][3] = {};
    for (int i = 0; i < 8; ++i) {
        J[0][0] += dNr[i]*pts[i].x; J[0][1] += dNr[i]*pts[i].y; J[0][2] += dNr[i]*pts[i].z;
        J[1][0] += dNs[i]*pts[i].x; J[1][1] += dNs[i]*pts[i].y; J[1][2] += dNs[i]*pts[i].z;
        J[2][0] += dNt[i]*pts[i].x; J[2][1] += dNt[i]*pts[i].y; J[2][2] += dNt[i]*pts[i].z;
    }
    return det3(J);
}

// ── Element types that legitimately allow coincident (zero-length) nodes ──────
// To add a new type, simply append it to this array.
static constexpr std::array kCoincidentNodeAllowed = {
    ElementType::CBUSH,
    ElementType::CELAS1,
    ElementType::CELAS2,
};

// Build the set of (low,high) node pairs connected by zero-length-allowed elements,
// including RBE2 and RBE3 (co-located independent/dependent pairs are valid).
std::unordered_set<long long> build_intentional_coincident_pairs(const Model& model) {
    auto pack = [](NodeId a, NodeId b) -> long long {
        int lo = std::min(a.value, b.value);
        int hi = std::max(a.value, b.value);
        return (static_cast<long long>(lo) << 32) | static_cast<unsigned>(hi);
    };

    std::unordered_set<long long> result;

    for (const auto& elem : model.elements) {
        bool allowed = std::any_of(kCoincidentNodeAllowed.begin(),
                                   kCoincidentNodeAllowed.end(),
                                   [&](ElementType t){ return t == elem.type; });
        if (allowed && elem.nodes.size() >= 2)
            result.insert(pack(elem.nodes[0], elem.nodes[1]));
    }

    // RBE2: independent node GN and each dependent GM are a valid coincident pair
    for (const auto& rbe2 : model.rbe2s)
        for (const auto& gm : rbe2.gm)
            result.insert(pack(rbe2.gn, gm));

    // RBE3: reference node and each weighted node
    for (const auto& rbe3 : model.rbe3s)
        for (const auto& wg : rbe3.weight_groups)
            for (const auto& n : wg.nodes)
                result.insert(pack(rbe3.ref_node, n));

    return result;
}

// ── CQUAD4 quality ────────────────────────────────────────────────────────────

ElementQualityResult cquad4_quality(const ElementData& elem, const Model& model) {
    ElementQualityResult r;
    r.id   = elem.id;
    r.type = elem.type;

    if (elem.nodes.size() < 4) return r;

    const Vec3& n0 = model.node(elem.nodes[0]).position;
    const Vec3& n1 = model.node(elem.nodes[1]).position;
    const Vec3& n2 = model.node(elem.nodes[2]).position;
    const Vec3& n3 = model.node(elem.nodes[3]).position;

    // Aspect ratio: max edge / min edge
    double e01 = edge_length(n0, n1);
    double e12 = edge_length(n1, n2);
    double e23 = edge_length(n2, n3);
    double e30 = edge_length(n3, n0);
    double emax = std::max({e01, e12, e23, e30});
    double emin = std::min({e01, e12, e23, e30});
    r.aspect_ratio = (emin > 1e-15) ? emax / emin : std::numeric_limits<double>::infinity();

    // Interior angles at each corner (degrees)
    auto corner_angle = [](const Vec3& prev, const Vec3& curr, const Vec3& next) {
        return angle_deg(prev - curr, next - curr);
    };
    double a0 = corner_angle(n3, n0, n1);
    double a1 = corner_angle(n0, n1, n2);
    double a2 = corner_angle(n1, n2, n3);
    double a3 = corner_angle(n2, n3, n0);
    r.min_interior_angle = std::min({a0, a1, a2, a3});
    r.max_interior_angle = std::max({a0, a1, a2, a3});

    // Warp angle: angle between normals of the two triangles formed by diagonal N0-N2
    Vec3 normal1 = (n1 - n0).cross(n2 - n0);
    Vec3 normal2 = (n2 - n0).cross(n3 - n0);
    r.warp_angle = angle_deg(normal1, normal2);

    // Taper ratio: area imbalance of the two diagonal triangulations
    double a_012 = triangle_area(n0, n1, n2);
    double a_023 = triangle_area(n0, n2, n3);
    double total = a_012 + a_023;
    r.taper_ratio = (total > 1e-15) ? std::abs(a_012 - a_023) / total : 0.0;

    return r;
}

// ── CTRIA3 quality ────────────────────────────────────────────────────────────

ElementQualityResult ctria3_quality(const ElementData& elem, const Model& model) {
    ElementQualityResult r;
    r.id   = elem.id;
    r.type = elem.type;

    if (elem.nodes.size() < 3) return r;

    const Vec3& n0 = model.node(elem.nodes[0]).position;
    const Vec3& n1 = model.node(elem.nodes[1]).position;
    const Vec3& n2 = model.node(elem.nodes[2]).position;

    double e01 = edge_length(n0, n1);
    double e12 = edge_length(n1, n2);
    double e20 = edge_length(n2, n0);
    double emax = std::max({e01, e12, e20});
    double emin = std::min({e01, e12, e20});
    r.aspect_ratio = (emin > 1e-15) ? emax / emin : std::numeric_limits<double>::infinity();

    double a0 = angle_deg(n2 - n0, n1 - n0);
    double a1 = angle_deg(n0 - n1, n2 - n1);
    double a2 = angle_deg(n1 - n2, n0 - n2);
    r.min_interior_angle = std::min({a0, a1, a2});
    r.max_interior_angle = std::max({a0, a1, a2});

    // Triangles are always planar
    r.warp_angle = 0.0;
    r.taper_ratio = 0.0;

    return r;
}

// ── CHEXA8 quality ────────────────────────────────────────────────────────────

ElementQualityResult chexa8_quality(const ElementData& elem, const Model& model) {
    ElementQualityResult r;
    r.id   = elem.id;
    r.type = elem.type;

    if (elem.nodes.size() < 8) return r;

    // 12 edges of a hex: pairs of node indices
    static constexpr int kEdges[12][2] = {
        {0,1},{1,2},{2,3},{3,0}, // bottom face
        {4,5},{5,6},{6,7},{7,4}, // top face
        {0,4},{1,5},{2,6},{3,7}  // lateral
    };

    double emax = 0.0, emin = std::numeric_limits<double>::infinity();
    for (const auto& e : kEdges) {
        double len = edge_length(model.node(elem.nodes[e[0]]).position,
                                 model.node(elem.nodes[e[1]]).position);
        emax = std::max(emax, len);
        emin = std::min(emin, len);
    }
    r.aspect_ratio = (emin > 1e-15) ? emax / emin : std::numeric_limits<double>::infinity();

    // Jacobian at 8 Gauss points (2×2×2 at ±1/√3)
    Vec3 pts[8];
    for (int i = 0; i < 8; ++i)
        pts[i] = model.node(elem.nodes[i]).position;

    static const double gp = 1.0 / std::sqrt(3.0);
    static const double gp_coords[8][3] = {
        {-gp,-gp,-gp},{+gp,-gp,-gp},{+gp,+gp,-gp},{-gp,+gp,-gp},
        {-gp,-gp,+gp},{+gp,-gp,+gp},{+gp,+gp,+gp},{-gp,+gp,+gp}
    };

    double det_min = std::numeric_limits<double>::infinity();
    double det_max = -std::numeric_limits<double>::infinity();
    for (const auto& gpc : gp_coords) {
        double d = chexa8_detJ(gpc[0], gpc[1], gpc[2], pts);
        det_min = std::min(det_min, d);
        det_max = std::max(det_max, d);
    }
    r.min_jacobian_ratio = (std::abs(det_max) > 1e-15) ? det_min / det_max : det_min;

    return r;
}

// ── CTETRA4 quality ───────────────────────────────────────────────────────────

ElementQualityResult ctetra4_quality(const ElementData& elem, const Model& model) {
    ElementQualityResult r;
    r.id   = elem.id;
    r.type = elem.type;

    if (elem.nodes.size() < 4) return r;

    static constexpr int kEdges[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    double emax = 0.0, emin = std::numeric_limits<double>::infinity();
    for (const auto& e : kEdges) {
        double len = edge_length(model.node(elem.nodes[e[0]]).position,
                                 model.node(elem.nodes[e[1]]).position);
        emax = std::max(emax, len);
        emin = std::min(emin, len);
    }
    r.aspect_ratio = (emin > 1e-15) ? emax / emin : std::numeric_limits<double>::infinity();

    // Jacobian at centroid: J has columns (N2-N1), (N3-N1), (N4-N1)
    const Vec3& p0 = model.node(elem.nodes[0]).position;
    const Vec3& p1 = model.node(elem.nodes[1]).position;
    const Vec3& p2 = model.node(elem.nodes[2]).position;
    const Vec3& p3 = model.node(elem.nodes[3]).position;
    Vec3 c1 = p1 - p0, c2 = p2 - p0, c3 = p3 - p0;
    double J[3][3] = {
        {c1.x, c2.x, c3.x},
        {c1.y, c2.y, c3.y},
        {c1.z, c2.z, c3.z}
    };
    double d = det3(J);
    // For a linear tet there is only one Jacobian value; ratio is det/|det| = sign
    r.min_jacobian_ratio = (std::abs(d) > 1e-15) ? 1.0 : 0.0;
    // A negative det means the element is inverted; encode as -1.0
    if (d < 0.0) r.min_jacobian_ratio = -1.0;

    return r;
}

// ── CTETRA10 quality ──────────────────────────────────────────────────────────

ElementQualityResult ctetra10_quality(const ElementData& elem, const Model& model) {
    ElementQualityResult r;
    r.id   = elem.id;
    r.type = elem.type;

    if (elem.nodes.size() < 4)
        return r;

    // Use the 6 corner edges for aspect ratio (linear approximation is sufficient for quality)
    static constexpr int kEdges[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    double emax = 0.0, emin = std::numeric_limits<double>::infinity();
    for (const auto& e : kEdges) {
        double len = edge_length(model.node(elem.nodes[e[0]]).position,
                                 model.node(elem.nodes[e[1]]).position);
        emax = std::max(emax, len);
        emin = std::min(emin, len);
    }
    r.aspect_ratio = (emin > 1e-15) ? emax / emin : std::numeric_limits<double>::infinity();

    // Jacobian: use corner nodes only (linear approximation)
    const Vec3& p0 = model.node(elem.nodes[0]).position;
    const Vec3& p1 = model.node(elem.nodes[1]).position;
    const Vec3& p2 = model.node(elem.nodes[2]).position;
    const Vec3& p3 = model.node(elem.nodes[3]).position;
    Vec3 c1 = p1 - p0, c2 = p2 - p0, c3 = p3 - p0;
    double J[3][3] = {
        {c1.x, c2.x, c3.x},
        {c1.y, c2.y, c3.y},
        {c1.z, c2.z, c3.z}
    };
    double d = det3(J);
    r.min_jacobian_ratio = (d < 0.0) ? -1.0 : ((std::abs(d) > 1e-15) ? 1.0 : 0.0);

    return r;
}

// ── CPENTA6 quality ───────────────────────────────────────────────────────────

ElementQualityResult cpenta6_quality(const ElementData& elem, const Model& model) {
    ElementQualityResult r;
    r.id   = elem.id;
    r.type = elem.type;

    if (elem.nodes.size() < 6) return r;

    // 9 edges: 3 bottom, 3 top, 3 lateral
    static constexpr int kEdges[9][2] = {
        {0,1},{1,2},{2,0}, // bottom triangle
        {3,4},{4,5},{5,3}, // top triangle
        {0,3},{1,4},{2,5}  // lateral
    };
    double emax = 0.0, emin = std::numeric_limits<double>::infinity();
    for (const auto& e : kEdges) {
        double len = edge_length(model.node(elem.nodes[e[0]]).position,
                                 model.node(elem.nodes[e[1]]).position);
        emax = std::max(emax, len);
        emin = std::min(emin, len);
    }
    r.aspect_ratio = (emin > 1e-15) ? emax / emin : std::numeric_limits<double>::infinity();

    // Jacobian at centroid of the wedge (r=1/3, s=1/3, t=0 in triangular/linear coords)
    // Use a simple scalar Jacobian approximation: volume = (base_area * height)
    // A negative det indicates inversion.
    const Vec3& b0 = model.node(elem.nodes[0]).position;
    const Vec3& b1 = model.node(elem.nodes[1]).position;
    const Vec3& b2 = model.node(elem.nodes[2]).position;
    const Vec3& t0 = model.node(elem.nodes[3]).position;
    const Vec3& t1 = model.node(elem.nodes[4]).position;
    const Vec3& t2 = model.node(elem.nodes[5]).position;
    // The signed volume of the wedge: bottom normal dotted with height direction
    Vec3 base_normal = (b1 - b0).cross(b2 - b0);
    Vec3 top_centroid  = Vec3{(t0.x + t1.x + t2.x) / 3.0,
                               (t0.y + t1.y + t2.y) / 3.0,
                               (t0.z + t1.z + t2.z) / 3.0};
    Vec3 base_centroid = Vec3{(b0.x + b1.x + b2.x) / 3.0,
                               (b0.y + b1.y + b2.y) / 3.0,
                               (b0.z + b1.z + b2.z) / 3.0};
    Vec3 height = top_centroid - base_centroid;
    double signed_vol = base_normal.dot(height);
    r.min_jacobian_ratio = (signed_vol < 0.0) ? -1.0 : ((std::abs(signed_vol) > 1e-15) ? 1.0 : 0.0);

    return r;
}

// ── Lenient auto-merge ────────────────────────────────────────────────────────

/// Replace all references to `old_id` with `keep_id` throughout the model.
void remap_node(Model& model, NodeId old_id, NodeId keep_id) {
    auto remap = [&](NodeId& n) { if (n == old_id) n = keep_id; };
    auto remap_vec = [&](std::vector<NodeId>& v) {
        for (auto& n : v) remap(n);
    };

    for (auto& elem : model.elements)
        remap_vec(elem.nodes);
    for (auto& spc : model.spcs)
        remap(spc.node);
    for (auto& mpc : model.mpcs)
        for (auto& term : mpc.terms)
            remap(term.node);
    for (auto& load : model.loads) {
        std::visit([&](auto& l) {
            using T = std::decay_t<decltype(l)>;
            if constexpr (std::is_same_v<T, ForceLoad>  ||
                          std::is_same_v<T, MomentLoad> ||
                          std::is_same_v<T, TempLoad>)
                remap(l.node);
            else if constexpr (std::is_same_v<T, PloadLoad>)
                remap_vec(l.nodes);
            else if constexpr (std::is_same_v<T, Accel1Load>)
                remap_vec(l.nodes);
        }, load);
    }
    for (auto& rbe2 : model.rbe2s) {
        remap(rbe2.gn);
        remap_vec(rbe2.gm);
    }
    for (auto& rbe3 : model.rbe3s) {
        remap(rbe3.ref_node);
        for (auto& wg : rbe3.weight_groups)
            remap_vec(wg.nodes);
    }
}

} // anonymous namespace

// ── Public API ────────────────────────────────────────────────────────────────

QualityThresholds build_thresholds(const Model& model) {
    QualityThresholds t;

    auto get = [&](const std::string& key) -> std::string {
        auto it = model.params.find(key);
        return (it != model.params.end()) ? it->second : std::string{};
    };
    auto get_double = [&](const std::string& key, double def) -> double {
        auto s = get(key);
        if (s.empty()) return def;
        try { return std::stod(s); } catch (...) { return def; }
    };
    auto get_int = [&](const std::string& key, int def) -> int {
        auto s = get(key);
        if (s.empty()) return def;
        try { return std::stoi(s); } catch (...) { return def; }
    };

    if (get("CHECKMODE") == "LENIENT")
        t.mode = CheckMode::Lenient;

    t.max_aspect_ratio   = get_double("ASPECT",    t.max_aspect_ratio);
    t.min_jacobian_ratio = get_double("JACMIN",    t.min_jacobian_ratio);
    t.max_warp_angle     = get_double("WARP",      t.max_warp_angle);
    t.max_taper_ratio    = get_double("TAPER",     t.max_taper_ratio);
    t.min_interior_angle = get_double("MINANGLE",  t.min_interior_angle);
    t.max_interior_angle = get_double("MAXANGLE",  t.max_interior_angle);
    t.dup_node_tol       = get_double("DUPNODTOL", t.dup_node_tol);
    t.auto_merge_nodes   = (get_int("AUTOMERGE", 1) != 0);
    t.maxratio           = get_double("MAXRATIO",  t.maxratio);
    t.bailout            = get_int("BAILOUT",       t.bailout);

    // In Lenient mode, BAILOUT=-1 (continue past singularities) unless explicitly set
    if (t.mode == CheckMode::Lenient && get("BAILOUT").empty())
        t.bailout = -1;

    return t;
}

std::optional<ElementQualityResult> compute_element_quality(
    const ElementData& elem, const Model& model)
{
    switch (elem.type) {
    case ElementType::CQUAD4:  return cquad4_quality(elem, model);
    case ElementType::CTRIA3:  return ctria3_quality(elem, model);
    case ElementType::CHEXA8:  return chexa8_quality(elem, model);
    case ElementType::CTETRA4: return ctetra4_quality(elem, model);
    case ElementType::CTETRA10:return ctetra10_quality(elem, model);
    case ElementType::CPENTA6: return cpenta6_quality(elem, model);
    default: return std::nullopt;
    }
}

PhysicalResult check_physical(const Model& model) {
    PhysicalResult res;

    for (const auto& [mid, mat] : model.materials) {
        if (mat.E <= 0.0) res.bad_E.push_back(mid);
        if (mat.nu >= 0.5) res.bad_nu.push_back(mid);
        if (mat.rho < 0.0) res.bad_rho.push_back(mid);
    }

    for (const auto& [pid, prop] : model.properties) {
        if (const auto* ps = std::get_if<PShell>(&prop))
            if (ps->t <= 0.0) res.bad_thickness.push_back(pid);
    }

    for (const auto& sc : model.analysis.subcases) {
        if (sc.load_set.value != 0 && model.loads_for_set(sc.load_set).empty())
            res.subcases_no_load.push_back(sc.id);

        bool has_spc = !model.spcs_for_set(sc.spc_set).empty();
        bool has_rbe = !model.rbe2s.empty() || !model.rbe3s.empty();
        if (!has_spc && !has_rbe)
            res.subcases_no_constraint.push_back(sc.id);
    }

    return res;
}

TopologyResult check_topology(const Model& model, double dup_node_tol) {
    TopologyResult res;

    // ── Orphaned nodes ────────────────────────────────────────────────────────
    // Collect all used node IDs (from elements, SPCs, MPCs, RBEs)

#ifdef HAVE_TBB
    tbb::enumerable_thread_specific<std::unordered_set<NodeId>> local_used;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, model.elements.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            auto& lu = local_used.local();
            for (size_t i = range.begin(); i != range.end(); ++i)
                for (NodeId n : model.elements[i].nodes)
                    lu.insert(n);
        });
    std::unordered_set<NodeId> used_nodes;
    for (auto& lu : local_used)
        used_nodes.insert(lu.begin(), lu.end());
#else
    std::unordered_set<NodeId> used_nodes;
    for (const auto& elem : model.elements)
        for (NodeId n : elem.nodes)
            used_nodes.insert(n);
#endif

    for (const auto& spc : model.spcs)
        used_nodes.insert(spc.node);
    for (const auto& mpc : model.mpcs)
        for (const auto& term : mpc.terms)
            used_nodes.insert(term.node);
    for (const auto& rbe2 : model.rbe2s) {
        used_nodes.insert(rbe2.gn);
        for (NodeId n : rbe2.gm) used_nodes.insert(n);
    }
    for (const auto& rbe3 : model.rbe3s) {
        used_nodes.insert(rbe3.ref_node);
        for (const auto& wg : rbe3.weight_groups)
            for (NodeId n : wg.nodes) used_nodes.insert(n);
    }
    for (const auto& [nid, _] : model.nodes)
        if (!used_nodes.count(nid))
            res.orphaned_nodes.push_back(nid);
    std::sort(res.orphaned_nodes.begin(), res.orphaned_nodes.end(),
              [](NodeId a, NodeId b){ return a.value < b.value; });

    // ── Orphaned properties ───────────────────────────────────────────────────
    std::unordered_set<PropertyId> used_props;
    for (const auto& elem : model.elements)
        if (elem.pid.value != 0) used_props.insert(elem.pid);
    for (const auto& [pid, _] : model.properties)
        if (!used_props.count(pid))
            res.orphaned_properties.push_back(pid);
    std::sort(res.orphaned_properties.begin(), res.orphaned_properties.end(),
              [](PropertyId a, PropertyId b){ return a.value < b.value; });

    // ── Orphaned materials ────────────────────────────────────────────────────
    std::unordered_set<MaterialId> used_mats;
    for (const auto& [pid, prop] : model.properties) {
        std::visit([&](const auto& p) {
            using T = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<T, PShell>) {
                used_mats.insert(p.mid1);
                if (p.mid2.value != 0) used_mats.insert(p.mid2);
                if (p.mid3.value != 0) used_mats.insert(p.mid3);
                if (p.mid4.value != 0) used_mats.insert(p.mid4);
            } else if constexpr (std::is_same_v<T, PSolid>) {
                used_mats.insert(p.mid);
            } else if constexpr (std::is_same_v<T, PBar>  ||
                                 std::is_same_v<T, PBarL> ||
                                 std::is_same_v<T, PBeam>) {
                used_mats.insert(p.mid);
            }
        }, prop);
    }
    for (const auto& [mid, _] : model.materials)
        if (!used_mats.count(mid))
            res.orphaned_materials.push_back(mid);
    std::sort(res.orphaned_materials.begin(), res.orphaned_materials.end(),
              [](MaterialId a, MaterialId b){ return a.value < b.value; });

    // ── Free edges (shell elements only) ─────────────────────────────────────
    // Edge defined by (min_node_id, max_node_id) → count, owning element id

#ifdef HAVE_TBB
    tbb::enumerable_thread_specific<std::map<std::pair<int,int>, int>> local_edge_count;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, model.elements.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            auto& ec = local_edge_count.local();
            for (size_t i = range.begin(); i != range.end(); ++i) {
                const auto& elem = model.elements[i];
                if (elem.type != ElementType::CQUAD4 && elem.type != ElementType::CTRIA3)
                    continue;
                int nn = static_cast<int>(elem.nodes.size());
                for (int j = 0; j < nn; ++j)
                    ec[edge_key(elem.nodes[j], elem.nodes[(j+1) % nn])]++;
            }
        });
    std::map<std::pair<int,int>, int> edge_count;
    for (auto& ec : local_edge_count)
        for (const auto& [k, v] : ec)
            edge_count[k] += v;
#else
    std::map<std::pair<int,int>, int> edge_count;
    for (const auto& elem : model.elements) {
        if (elem.type != ElementType::CQUAD4 && elem.type != ElementType::CTRIA3)
            continue;
        int nn = static_cast<int>(elem.nodes.size());
        for (int j = 0; j < nn; ++j)
            edge_count[edge_key(elem.nodes[j], elem.nodes[(j+1) % nn])]++;
    }
#endif

    // Collect elements that own at least one free edge
    std::unordered_set<ElementId> free_edge_set;
    for (const auto& elem : model.elements) {
        if (elem.type != ElementType::CQUAD4 && elem.type != ElementType::CTRIA3)
            continue;
        int nn = static_cast<int>(elem.nodes.size());
        for (int j = 0; j < nn; ++j) {
            auto key = edge_key(elem.nodes[j], elem.nodes[(j+1) % nn]);
            if (edge_count[key] == 1) {
                free_edge_set.insert(elem.id);
                break;
            }
        }
    }
    res.free_edge_elements.assign(free_edge_set.begin(), free_edge_set.end());
    std::sort(res.free_edge_elements.begin(), res.free_edge_elements.end(),
              [](ElementId a, ElementId b){ return a.value < b.value; });

    // ── Duplicate nodes ───────────────────────────────────────────────────────
    // Build sorted list by x, then sweep for close pairs
    auto intentional = build_intentional_coincident_pairs(model);
    auto is_intentional = [&](NodeId a, NodeId b) -> bool {
        int lo = std::min(a.value, b.value), hi = std::max(a.value, b.value);
        long long key = (static_cast<long long>(lo) << 32) | static_cast<unsigned>(hi);
        return intentional.count(key) > 0;
    };

    std::vector<std::pair<Vec3, NodeId>> sorted_nodes;
    sorted_nodes.reserve(model.nodes.size());
    for (const auto& [nid, gp] : model.nodes)
        sorted_nodes.push_back({gp.position, nid});
    std::sort(sorted_nodes.begin(), sorted_nodes.end(),
              [](const auto& a, const auto& b){ return a.first.x < b.first.x; });

    for (size_t i = 0; i < sorted_nodes.size(); ++i) {
        for (size_t j = i + 1; j < sorted_nodes.size(); ++j) {
            if (sorted_nodes[j].first.x - sorted_nodes[i].first.x > dup_node_tol)
                break;
            Vec3 diff = sorted_nodes[i].first - sorted_nodes[j].first;
            if (diff.norm() <= dup_node_tol) {
                NodeId a = sorted_nodes[i].second, b = sorted_nodes[j].second;
                if (!is_intentional(a, b))
                    res.duplicate_node_pairs.push_back({a, b});
            }
        }
    }

    // ── Duplicate elements ────────────────────────────────────────────────────
    // Key: element type + sorted node IDs encoded as int64 hash
    std::unordered_map<std::string, ElementId> elem_seen;
    for (const auto& elem : model.elements) {
        std::vector<int> node_vals;
        node_vals.reserve(elem.nodes.size());
        for (NodeId n : elem.nodes) node_vals.push_back(n.value);
        std::sort(node_vals.begin(), node_vals.end());
        // Build a string key: type:n1,n2,...
        std::string key = std::to_string(static_cast<int>(elem.type)) + ":";
        for (int v : node_vals) key += std::to_string(v) + ",";
        auto [it, inserted] = elem_seen.emplace(key, elem.id);
        if (!inserted)
            res.duplicate_element_pairs.push_back({it->second, elem.id});
    }

    return res;
}

void run_quality_checks(Model& model, const QualityThresholds& t) {
    const bool strict = (t.mode == CheckMode::Strict);

    auto issue = [&](const std::string& msg, bool fatal) {
        if (fatal)
            throw SolverError(msg);
        else
            log_warn("[quality] " + msg);
    };

    // ── Physical sanity ───────────────────────────────────────────────────────
    PhysicalResult phys = check_physical(model);

    // E and nu violations are always fatal (nonphysical material)
    for (MaterialId mid : phys.bad_E)
        issue(std::format("MAT1 {}: E ≤ 0 (E must be > 0)", mid.value), true);
    for (MaterialId mid : phys.bad_nu)
        issue(std::format("MAT1 {}: nu ≥ 0.5 (must be < 0.5 for isotropic material)",
                          mid.value), true);
    for (MaterialId mid : phys.bad_rho)
        issue(std::format("MAT1 {}: density < 0", mid.value), strict);
    for (PropertyId pid : phys.bad_thickness)
        issue(std::format("PSHELL {}: thickness ≤ 0", pid.value), strict);
    for (int sc_id : phys.subcases_no_load)
        log_warn(std::format("[quality] Subcase {}: load set references no loads", sc_id));
    // No-constraint is fatal for SOL 101 regardless of mode; warn for SOL 103 in lenient
    for (int sc_id : phys.subcases_no_constraint) {
        bool fatal_constraint = strict ||
            (model.analysis.sol == SolutionType::LinearStatic);
        issue(std::format("Subcase {}: no constraints (SPC/RBE) — model may be singular",
                          sc_id), fatal_constraint);
    }

    // ── Topology ──────────────────────────────────────────────────────────────
    TopologyResult topo = check_topology(model, t.dup_node_tol);

    for (ElementId eid : topo.free_edge_elements)
        issue(std::format("Element {}: has free edge(s) — possible mesh gap", eid.value), strict);
    for (NodeId nid : topo.orphaned_nodes)
        issue(std::format("Node {}: not referenced by any element or constraint", nid.value), strict);
    for (PropertyId pid : topo.orphaned_properties)
        issue(std::format("Property {}: defined but not used by any element", pid.value), strict);
    for (MaterialId mid : topo.orphaned_materials)
        issue(std::format("Material {}: defined but not referenced by any property", mid.value), strict);
    for (const auto& [eid1, eid2] : topo.duplicate_element_pairs)
        issue(std::format("Elements {} and {} have identical connectivity", eid1.value, eid2.value), strict);

    // Duplicate nodes: auto-merge in lenient mode (unless disabled)
    if (!topo.duplicate_node_pairs.empty()) {
        for (const auto& [a, b] : topo.duplicate_node_pairs) {
            // Merge higher-ID node into lower-ID node
            NodeId keep = (a.value < b.value) ? a : b;
            NodeId drop = (a.value < b.value) ? b : a;
            if (strict) {
                issue(std::format("Nodes {} and {} are coincident (distance ≤ {:.2e})",
                                  drop.value, keep.value, t.dup_node_tol), true);
            } else {
                if (t.auto_merge_nodes) {
                    remap_node(model, drop, keep);
                    model.nodes.erase(drop);
                    log_warn(std::format("[quality] Merged node {} into {} (coincident nodes)",
                                         drop.value, keep.value));
                } else {
                    log_warn(std::format("[quality] Nodes {} and {} are coincident "
                                         "(AUTOMERGE=0, not merged)", drop.value, keep.value));
                }
            }
        }
    }

    // ── Element quality metrics ───────────────────────────────────────────────
    std::vector<ElementQualityResult> elem_results;

#ifdef HAVE_TBB
    tbb::enumerable_thread_specific<std::vector<ElementQualityResult>> local_results;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, model.elements.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            auto& lr = local_results.local();
            for (size_t i = range.begin(); i != range.end(); ++i) {
                if (auto q = compute_element_quality(model.elements[i], model))
                    lr.push_back(*q);
            }
        });
    for (auto& lr : local_results)
        std::move(lr.begin(), lr.end(), std::back_inserter(elem_results));
#else
    for (const auto& elem : model.elements)
        if (auto q = compute_element_quality(elem, model))
            elem_results.push_back(*q);
#endif

    for (const auto& q : elem_results) {
        const char* tname = element_type_name(q.type);

        if (q.aspect_ratio > t.max_aspect_ratio)
            issue(std::format("Element {} ({}): aspect ratio {:.2f} > threshold {:.2f}",
                              q.id.value, tname, q.aspect_ratio, t.max_aspect_ratio), strict);

        if (t.min_jacobian_ratio > 0.0 && q.min_jacobian_ratio >= 0.0 &&
            q.min_jacobian_ratio < t.min_jacobian_ratio)
            issue(std::format("Element {} ({}): Jacobian ratio {:.4f} < threshold {:.4f}",
                              q.id.value, tname, q.min_jacobian_ratio, t.min_jacobian_ratio), strict);

        if (!std::isnan(q.min_jacobian_ratio) && q.min_jacobian_ratio < 0.0)
            issue(std::format("Element {} ({}): inverted element (negative Jacobian)",
                              q.id.value, tname), true); // always fatal regardless of mode

        if (q.min_interior_angle >= 0.0 && q.min_interior_angle < t.min_interior_angle)
            issue(std::format("Element {} ({}): min interior angle {:.1f} deg < threshold {:.1f} deg",
                              q.id.value, tname, q.min_interior_angle, t.min_interior_angle), strict);

        if (q.max_interior_angle >= 0.0 && q.max_interior_angle > t.max_interior_angle)
            issue(std::format("Element {} ({}): max interior angle {:.1f} deg > threshold {:.1f} deg",
                              q.id.value, tname, q.max_interior_angle, t.max_interior_angle), strict);

        if (q.warp_angle >= 0.0 && q.warp_angle > t.max_warp_angle)
            issue(std::format("Element {} ({}): warp angle {:.1f} deg > threshold {:.1f} deg",
                              q.id.value, tname, q.warp_angle, t.max_warp_angle), strict);

        if (q.taper_ratio >= 0.0 && q.taper_ratio > t.max_taper_ratio)
            issue(std::format("Element {} ({}): taper ratio {:.3f} > threshold {:.3f}",
                              q.id.value, tname, q.taper_ratio, t.max_taper_ratio), strict);
    }
}

} // namespace vibestran
