// src/elements/rbe_constraints.cpp
// Expand RBE2 / RBE3 elements into MPC equations.

#include "elements/rbe_constraints.hpp"
#include <cmath>
#include <format>

namespace vibestran {

// ── RBE2 ──────────────────────────────────────────────────────────────────────
//
// Rigid body constraint: gm moves with gn as a rigid body.
// Let r = pos(gm) - pos(gn) in basic Cartesian.
// For translations (T1,T2,T3) and rotations (R1,R2,R3) in cm:
//
// u_gm[T1] - u_gn[T1] - r.z*θ_gn[R2] + r.y*θ_gn[R3] = 0
// u_gm[T2] - u_gn[T2] + r.z*θ_gn[R1] - r.x*θ_gn[R3] = 0
// u_gm[T3] - u_gn[T3] - r.y*θ_gn[R1] + r.x*θ_gn[R2] = 0
// u_gm[R1] - u_gn[R1] = 0
// u_gm[R2] - u_gn[R2] = 0
// u_gm[R3] - u_gn[R3] = 0

void expand_rbe2(const Rbe2& rbe, const Model& model, std::vector<Mpc>& out) {
    const Vec3 pos_n = model.node(rbe.gn).position;

    for (NodeId gm : rbe.gm) {
        const Vec3 pos_m = model.node(gm).position;
        Vec3 r{pos_m.x - pos_n.x, pos_m.y - pos_n.y, pos_m.z - pos_n.z};

        // Use a synthetic MpcSetId of 0 (the solver wires these in directly)
        MpcSetId sid{0};

        auto make_mpc = [&](std::initializer_list<MpcTerm> terms) {
            Mpc mpc;
            mpc.sid = sid;
            mpc.terms = std::vector<MpcTerm>(terms);
            out.push_back(std::move(mpc));
        };

        if (rbe.cm.has(1)) { // T1
            make_mpc({
                {gm, 1, +1.0},
                {rbe.gn, 1, -1.0},
                {rbe.gn, 5, -r.z}, // -r.z * θ_R2
                {rbe.gn, 6, +r.y}, // +r.y * θ_R3
            });
        }
        if (rbe.cm.has(2)) { // T2
            make_mpc({
                {gm, 2, +1.0},
                {rbe.gn, 2, -1.0},
                {rbe.gn, 4, +r.z}, // +r.z * θ_R1
                {rbe.gn, 6, -r.x}, // -r.x * θ_R3
            });
        }
        if (rbe.cm.has(3)) { // T3
            make_mpc({
                {gm, 3, +1.0},
                {rbe.gn, 3, -1.0},
                {rbe.gn, 4, -r.y}, // -r.y * θ_R1
                {rbe.gn, 5, +r.x}, // +r.x * θ_R2
            });
        }
        if (rbe.cm.has(4)) { // R1
            make_mpc({
                {gm, 4, +1.0},
                {rbe.gn, 4, -1.0},
            });
        }
        if (rbe.cm.has(5)) { // R2
            make_mpc({
                {gm, 5, +1.0},
                {rbe.gn, 5, -1.0},
            });
        }
        if (rbe.cm.has(6)) { // R3
            make_mpc({
                {gm, 6, +1.0},
                {rbe.gn, 6, -1.0},
            });
        }
    }
}

// ── RBE3 ──────────────────────────────────────────────────────────────────────
//
// Interpolation constraint: the reference node motion equals a weighted average
// of independent node motions (including rigid-body lever-arm correction).
//
// For translational DOF Ti of reference node (ref):
//   u_ref[Ti] = (1/W) * sum_j wj * (u_j[Ti] + (omega_j × r_j)[i])
// where r_j = pos(j) - pos(ref) and omega_j are the rotational DOFs of j
// (only included if j's weight group component includes rotation DOFs).
//
// Rearranging (multiply through by W, move ref to left):
//   W * u_ref[Ti] - sum_j wj * u_j[Ti] - sum_j wj * (omega_j × r_j)[i] = 0

void expand_rbe3(const Rbe3& rbe, const Model& model, std::vector<Mpc>& out) {
    const Vec3 pos_ref = model.node(rbe.ref_node).position;
    MpcSetId sid{0};

    // Compute total weights (sum over all independent nodes, all weight groups)
    double W_trans = 0.0; // for translational DOFs
    double W_rot   = 0.0; // for rotational DOFs (only groups with rotation components)
    for (const auto& wg : rbe.weight_groups) {
        double wt = wg.weight * static_cast<double>(wg.nodes.size());
        // For translational averaging, all groups contribute
        W_trans += wt;
        // For rotational averaging, only groups that include rotation components
        bool has_rot = wg.component.has(4) || wg.component.has(5) || wg.component.has(6);
        if (has_rot) W_rot += wt;
    }
    if (W_trans < 1e-30) return; // degenerate

    // Generate MPCs for each constrained DOF in refc
    for (int d = 1; d <= 6; ++d) {
        if (!rbe.refc.has(d))
            continue;

        Mpc mpc;
        mpc.sid = sid;

        if (d <= 3) {
            // Translational DOF d: W * u_ref[d] = sum_j wj * u_j[d] + lever terms
            // → W * u_ref[d] - sum_j wj * u_j[d] - sum_j wj * (ω_j × r_j)[d-1] = 0
            mpc.terms.push_back({rbe.ref_node, d, W_trans});

            for (const auto& wg : rbe.weight_groups) {
                for (NodeId nid : wg.nodes) {
                    const Vec3 pos_j = model.node(nid).position;
                    Vec3 r_j{pos_j.x - pos_ref.x,
                             pos_j.y - pos_ref.y,
                             pos_j.z - pos_ref.z};

                    // Translational term
                    mpc.terms.push_back({nid, d, -wg.weight});

                    // Lever-arm rotation correction: omega_j × r_j
                    // (omega_j × r_j)[0] = omega_y * r_z - omega_z * r_y  (for T1)
                    // (omega_j × r_j)[1] = omega_z * r_x - omega_x * r_z  (for T2)
                    // (omega_j × r_j)[2] = omega_x * r_y - omega_y * r_x  (for T3)
                    bool has_rot = wg.component.has(4) || wg.component.has(5) || wg.component.has(6);
                    if (!has_rot) continue;

                    if (d == 1) {
                        // (ω × r)[x] = ω_y * r_z - ω_z * r_y
                        if (wg.component.has(5)) // R2 = ω_y
                            mpc.terms.push_back({nid, 5, -wg.weight * r_j.z});
                        if (wg.component.has(6)) // R3 = ω_z
                            mpc.terms.push_back({nid, 6, +wg.weight * r_j.y});
                    } else if (d == 2) {
                        // (ω × r)[y] = ω_z * r_x - ω_x * r_z
                        if (wg.component.has(6)) // R3 = ω_z
                            mpc.terms.push_back({nid, 6, -wg.weight * r_j.x});
                        if (wg.component.has(4)) // R1 = ω_x
                            mpc.terms.push_back({nid, 4, +wg.weight * r_j.z});
                    } else { // d == 3
                        // (ω × r)[z] = ω_x * r_y - ω_y * r_x
                        if (wg.component.has(4)) // R1 = ω_x
                            mpc.terms.push_back({nid, 4, -wg.weight * r_j.y});
                        if (wg.component.has(5)) // R2 = ω_y
                            mpc.terms.push_back({nid, 5, +wg.weight * r_j.x});
                    }
                }
            }
        } else {
            // Rotational DOF d: W_rot * θ_ref[d] = sum_j wj * θ_j[d]
            if (W_rot < 1e-30) continue;
            mpc.terms.push_back({rbe.ref_node, d, W_rot});
            for (const auto& wg : rbe.weight_groups) {
                bool has_rot = wg.component.has(d);
                if (!has_rot) continue;
                for (NodeId nid : wg.nodes)
                    mpc.terms.push_back({nid, d, -wg.weight});
            }
        }

        if (!mpc.terms.empty())
            out.push_back(std::move(mpc));
    }
}

} // namespace vibestran
