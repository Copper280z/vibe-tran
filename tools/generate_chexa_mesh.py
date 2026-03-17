#!/usr/bin/env python3
"""Generate a CHEXA8 stress-test BDF file.

Geometry
--------
Rectangular prismatic bar aligned along the X-axis with fixed physical dimensions:
  - Length  L = 1.0 m  (X extent)
  - Width   W = 0.2 m  (Y extent)
  - Height  H = 0.2 m  (Z extent)

The cross-section mesh (NY x NZ elements) scales with the target element count
so that elements remain approximately cubic:
  - Since W = H, NY = NZ = max(2, round((n_target / 5)^(1/3)))
  - Element side length d = W / NY
  - Axial count nx = round(L / d)  (≈ 5 * NY for L/W = 5)
  - Actual element count = nx * NY * NZ

The minimum cross-section mesh is 2x2, which is required for solid elements
to represent bending correctly (avoids single-element-wide locking artifacts).

Material (steel defaults)
-------------------------
  E     = 2.0e11 Pa
  nu    = 0.3
  alpha = 1.2e-5 /°C

Load cases  (select with --load-case)
--------------------------------------
thermal
  Uniaxially constrained bar, uniform temperature rise dT=100 °C.
  The x=0 face is fully clamped (T1,T2,T3=0).  The x=L face is constrained
  in T1 only (no axial expansion).  All other nodes are free to move in T2
  and T3 (Poisson contraction is unrestrained), so the only non-zero stress
  component is the axial one.

  Analytical (uniaxial constraint, free lateral surfaces):
    sigma_xx = -E * alpha * dT
    sigma_yy = sigma_zz = 0

axial
  Free end (x=L) loaded with total force F=1000 N in the +X direction,
  distributed uniformly over the (NY+1)*(NZ+1) free-end nodes.
  Fixed end (x=0) fully clamped (T1,T2,T3=0 on all face nodes).

  Analytical (uniform axial stress):
    sigma_xx = F / (W * H)
    delta_x  = F * L / (E * W * H)

bending
  Cantilever: fixed end at x=0 (fully clamped), transverse load F=1000 N
  in the +Z direction at free end x=L, distributed uniformly over the
  (NY+1)*(NZ+1) free-end nodes.

  Analytical (Euler-Bernoulli, valid for L/H >> 1):
    I             = W * H^3 / 12           (second moment about bending axis)
    delta_z_tip   = F * L^3 / (3 * E * I)
    sigma_xx_max  = F * L * (H/2) / I      (at fixed-end top/bottom surface)

  Note: agreement improves as L/H increases.

Usage
-----
  python3 generate_chexa_mesh.py N [--load-case {thermal,axial,bending}] [-o FILE]

Examples
--------
  python3 generate_chexa_mesh.py 100 --load-case axial -o bar_100_axial.bdf
  python3 generate_chexa_mesh.py 10000 --load-case thermal -o bar_10k_thermal.bdf
  python3 generate_chexa_mesh.py 500 --load-case bending   # writes to stdout
"""

import argparse
import math
import sys

# ── Material defaults ────────────────────────────────────────────────────────
E_DEFAULT     = 2.0e11   # Pa  Young's modulus (steel)
NU_DEFAULT    = 0.3      # Poisson's ratio
ALPHA_DEFAULT = 1.2e-5   # /°C  coefficient of thermal expansion

# ── Fixed bar geometry ───────────────────────────────────────────────────────
L_DEFAULT = 1.0   # m  bar length   (X extent)
W_DEFAULT = 0.2   # m  cross-section width  (Y extent)
H_DEFAULT = 0.2   # m  cross-section height (Z extent)

# ── Load magnitudes ──────────────────────────────────────────────────────────
F_TOTAL = 1000.0  # N    total applied force (axial and bending cases)
DT      = 100.0   # °C   temperature rise    (thermal case)


# ── Mesh helpers ─────────────────────────────────────────────────────────────

def node_id(i: int, j: int, k: int, ny: int, nz: int) -> int:
    """Return 1-based node ID for mesh indices (i along X, j along Y, k along Z)."""
    return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k + 1


def element_nodes(ie: int, je: int, ke: int, ny: int, nz: int) -> list[int]:
    """
    Return the 8 node IDs for the CHEXA8 element at grid position (ie, je, ke).

    NASTRAN CHEXA8 node ordering:
      Nodes 1-4: bottom face (z_min), CCW when viewed from -Z
      Nodes 5-8: top face   (z_max), CCW when viewed from +Z
      Lateral edges connect node i (bottom) to node i+4 (top).

    Natural-coordinate mapping (as expected by the solver):
      n1 -> (-1,-1,-1)   n5 -> (-1,-1,+1)
      n2 -> (+1,-1,-1)   n6 -> (+1,-1,+1)
      n3 -> (+1,+1,-1)   n7 -> (+1,+1,+1)
      n4 -> (-1,+1,-1)   n8 -> (-1,+1,+1)
    """
    def n(di: int, dj: int, dk: int) -> int:
        return node_id(ie + di, je + dj, ke + dk, ny, nz)

    return [
        n(0, 0, 0),  # 1: x_min, y_min, z_min
        n(1, 0, 0),  # 2: x_max, y_min, z_min
        n(1, 1, 0),  # 3: x_max, y_max, z_min
        n(0, 1, 0),  # 4: x_min, y_max, z_min
        n(0, 0, 1),  # 5: x_min, y_min, z_max
        n(1, 0, 1),  # 6: x_max, y_min, z_max
        n(1, 1, 1),  # 7: x_max, y_max, z_max
        n(0, 1, 1),  # 8: x_min, y_max, z_max
    ]


# ── BDF card writers ─────────────────────────────────────────────────────────

def write_chexa(out, eid: int, pid: int, nodes: list[int]) -> None:
    """
    Write a CHEXA card in free-field format.

    CHEXA has 10 data fields (EID, PID, G1..G8) which exceeds the 8 data
    fields per line in fixed-field format, so G7 and G8 go on a continuation
    card.  We use a bare '+' as the continuation marker (no label) because
    the parser accumulates any immediately following line that starts with '+'
    into the current card's fields — a matching label is not required.
    """
    n = nodes
    out.write(f"CHEXA,{eid},{pid},{n[0]},{n[1]},{n[2]},{n[3]},{n[4]},{n[5]}\n")
    out.write(f"+,{n[6]},{n[7]}\n")


def fmt_float(v: float) -> str:
    """Format a float for BDF output (scientific notation, no extra zeros)."""
    return f"{v:.6E}"


# ── Main generator ────────────────────────────────────────────────────────────

def generate(n_target: int, load_case: str, out_path: str) -> None:
    # ── Fixed bar geometry ────────────────────────────────────────────────────
    L = L_DEFAULT
    W = W_DEFAULT
    H = H_DEFAULT

    # ── Mesh dimensions ───────────────────────────────────────────────────────
    # Keep elements approximately cubic: d = W/NY = H/NZ = L/nx.
    # Since W = H, NY = NZ.  With L/W = 5, nx ≈ 5*NY, so n ≈ 5*NY^3.
    # Solve for NY, enforce minimum of 2 (needed to avoid locking artifacts).
    NY = max(2, round((n_target / (L / W)) ** (1.0 / 3.0)))
    NZ = NY  # W = H so same count in both cross-section directions
    d  = W / NY
    nx = max(1, round(L / d))

    n_nodes    = (nx + 1) * (NY + 1) * (NZ + 1)
    n_elements = nx * NY * NZ

    # ── Material ─────────────────────────────────────────────────────────────
    E     = E_DEFAULT
    nu    = NU_DEFAULT
    alpha = ALPHA_DEFAULT

    # ── Derived analytical quantities ────────────────────────────────────────
    A   = W * H               # cross-sectional area
    I_z = W * H**3 / 12.0    # second moment of area (bending in X-Z plane, load in Z)

    if load_case == "thermal":
        sigma_xx_thermal = -E * alpha * DT  # uniaxial, free lateral surfaces
    elif load_case == "axial":
        sigma_xx_axial = F_TOTAL / A
        delta_x_axial  = F_TOTAL * L / (E * A)
    elif load_case == "bending":
        delta_z_tip    = F_TOTAL * L**3 / (3.0 * E * I_z)
        sigma_xx_surf  = F_TOTAL * L * (H / 2.0) / I_z   # fixed-end top/bottom
        slenderness    = L / H

    # ── Open output ──────────────────────────────────────────────────────────
    if out_path == "-":
        out = sys.stdout
    else:
        out = open(out_path, "w")

    def wr(s: str = "") -> None:
        out.write(s + "\n")

    # ── Case control ─────────────────────────────────────────────────────────
    wr("$ CHEXA8 stress-test mesh — generated by generate_chexa_mesh.py")
    wr(f"$ Target elements : {n_target}")
    wr(f"$ Actual elements : {n_elements}  ({nx} x {NY} x {NZ})")
    wr(f"$ Total nodes     : {n_nodes}")
    wr(f"$ Geometry        : L = {L:.4g} m,  W = {W:.4g} m,  H = {H:.4g} m  (fixed)")
    wr(f"$ Element size    : d = {d:.4g} m  (cubic CHEXA8, d = W/NY = H/NZ ≈ L/nx)")
    wr(f"$ Material        : E = {E:.4g} Pa,  nu = {nu},  alpha = {alpha:.4g} /deg")
    wr("$")
    wr("$ ── Analytical reference values ─────────────────────────────────────────")
    if load_case == "thermal":
        wr(f"$   Load case : THERMAL  (dT = {DT} °C, uniaxially constrained bar)")
        wr(f"$   BCs       : x=0 face fully clamped; x=L face locked in T1 only")
        wr(f"$   Formula   : sigma_xx = -E * alpha * dT  (lateral surfaces free)")
        wr(f"$   Result    : sigma_xx = {sigma_xx_thermal:.6G} Pa,  sigma_yy = sigma_zz = 0")
    elif load_case == "axial":
        wr(f"$   Load case : AXIAL  (F_total = {F_TOTAL} N in +X, free end x = {L:.4g} m)")
        wr(f"$   A         = W * H = {A:.4g} m^2")
        wr(f"$   sigma_xx  = F / A = {sigma_xx_axial:.6G} Pa  (uniform through cross-section)")
        wr(f"$   delta_x   = F * L / (E * A) = {delta_x_axial:.6G} m  (free-end displacement)")
    elif load_case == "bending":
        wr(f"$   Load case    : BENDING  (F_total = {F_TOTAL} N in +Z, cantilever at x = 0)")
        wr(f"$   L/H ratio    : {slenderness:.2f}  (Euler-Bernoulli valid for L/H >> 1)")
        wr(f"$   I = W*H^3/12 : {I_z:.6G} m^4")
        wr(f"$   delta_z_tip  = F*L^3/(3*E*I) = {delta_z_tip:.6G} m")
        wr(f"$   sigma_xx_max = F*L*(H/2)/I   = {sigma_xx_surf:.6G} Pa  (fixed-end surface)")
        wr("$   sigma_xx_max occurs at z = 0 (bottom, tensile) and z = H (top, compressive).")
    wr("$")
    wr("SOL 101")
    wr("CEND")
    wr("SUBCASE 1")
    labels = {
        "thermal": "CHEXA THERMAL STRESS UNIAXIALLY CONSTRAINED BAR",
        "axial":   "CHEXA AXIAL TENSION BAR",
        "bending": "CHEXA CANTILEVER BENDING",
    }
    wr(f"  LABEL = {labels[load_case]}")
    wr("  DISPLACEMENT(PRINT) = ALL")
    wr("  STRESS(PRINT) = ALL")
    wr("  LOAD  = 1")
    wr("  SPC   = 1")
    wr("BEGIN BULK")

    # ── Material ─────────────────────────────────────────────────────────────
    wr("$")
    wr("$ ── Material ───────────────────────────────────────────────────────────")
    wr(f"MAT1,1,{fmt_float(E)},,{fmt_float(nu)},{fmt_float(0.0)},{fmt_float(alpha)}")

    # ── Property ─────────────────────────────────────────────────────────────
    wr("$")
    wr("$ ── Property ───────────────────────────────────────────────────────────")
    wr("PSOLID,1,1,,2")  # IN=2: 2x2x2 full Gauss integration for CHEXA8

    # ── Nodes ────────────────────────────────────────────────────────────────
    wr("$")
    wr("$ ── Nodes ──────────────────────────────────────────────────────────────")
    dy = W / NY
    dz = H / NZ
    dx = L / nx
    for i in range(nx + 1):
        x = i * dx
        for j in range(NY + 1):
            y = j * dy
            for k in range(NZ + 1):
                z = k * dz
                nid = node_id(i, j, k, NY, NZ)
                wr(f"GRID,{nid},,{fmt_float(x)},{fmt_float(y)},{fmt_float(z)}")

    # ── Elements ─────────────────────────────────────────────────────────────
    wr("$")
    wr("$ ── Elements ───────────────────────────────────────────────────────────")
    eid = 1
    for ie in range(nx):
        for je in range(NY):
            for ke in range(NZ):
                write_chexa(out, eid, 1, element_nodes(ie, je, ke, NY, NZ))
                eid += 1

    # ── Boundary conditions ───────────────────────────────────────────────────
    wr("$")
    wr("$ ── Boundary Conditions ────────────────────────────────────────────────")

    # Nodes at x=0 face (i=0): node IDs 1 .. (NY+1)*(NZ+1), which are contiguous.
    fixed_face_first = node_id(0, 0, 0, NY, NZ)
    fixed_face_last  = node_id(0, NY, NZ, NY, NZ)

    if load_case == "thermal":
        # Uniaxial constraint: lock T1 at BOTH faces so the bar cannot expand
        # axially, but leave ALL lateral DOFs (T2, T3) free everywhere.
        # This gives sigma_xx = -E*alpha*dT with sigma_yy = sigma_zz = 0.
        #
        # Rigid-body motion is suppressed at the single center node of the x=0
        # face (T2=0, T3=0) rather than clamping the entire face, so that the
        # Poisson contraction is unrestrained at x=0 just as it is everywhere
        # else along the bar.
        free_face_first  = node_id(nx, 0,    0,      NY, NZ)
        free_face_last   = node_id(nx, NY,   NZ,     NY, NZ)
        center_node_x0   = node_id(0,  NY//2, NZ//2, NY, NZ)  # centre of x=0 face
        wr(f"$ x=0 face: lock T1 on all face nodes (no axial motion at this end).")
        wr(f"SPC1,1,1,{fixed_face_first},THRU,{fixed_face_last}")
        wr(f"$ x=L face: lock T1 on all face nodes (no axial expansion allowed).")
        wr(f"SPC1,1,1,{free_face_first},THRU,{free_face_last}")
        wr(f"$ Rigid-body T2/T3 at centre of x=0 face (node {center_node_x0}),")
        wr(f"$ leaving all lateral DOFs elsewhere free for Poisson contraction.")
        wr(f"SPC1,1,23,{center_node_x0}")

    else:  # axial or bending
        # Fully clamp the x=0 face.
        wr("$ Fixed end (x=0): all T1,T2,T3 constrained.")
        wr(f"SPC1,1,123,{fixed_face_first},THRU,{fixed_face_last}")

    # ── Loads ─────────────────────────────────────────────────────────────────
    wr("$")
    wr("$ ── Loads ──────────────────────────────────────────────────────────────")

    if load_case == "thermal":
        # TEMPD declares the reference (stress-free) temperature.
        wr("$ Reference temperature (stress-free state): 0 degrees.")
        wr("TEMPD,1,0.0")
        wr(f"$ Nodal temperatures: all nodes at dT = {DT} degrees.")
        # TEMP card holds up to 3 (node, temperature) pairs.
        all_nids = list(range(1, n_nodes + 1))
        for chunk_start in range(0, len(all_nids), 3):
            chunk = all_nids[chunk_start:chunk_start + 3]
            pairs = ",".join(f"{nid},{DT:.1f}" for nid in chunk)
            wr(f"TEMP,1,{pairs}")

    else:
        # Free end nodes (i=nx): also contiguous — IDs from nx*(NY+1)*(NZ+1)+1 onward.
        free_face_first = node_id(nx, 0,  0,  NY, NZ)
        free_face_last  = node_id(nx, NY, NZ, NY, NZ)
        n_free_nodes    = (NY + 1) * (NZ + 1)
        f_per_node      = F_TOTAL / n_free_nodes

        if load_case == "axial":
            wr(f"$ Axial force: {F_TOTAL} N in +X, split equally over "
               f"{n_free_nodes} free-end nodes ({f_per_node:.6G} N each).")
            direction = "1.0,0.0,0.0"
        else:  # bending
            wr(f"$ Transverse force: {F_TOTAL} N in +Z, split equally over "
               f"{n_free_nodes} free-end nodes ({f_per_node:.6G} N each).")
            direction = "0.0,0.0,1.0"

        for nid in range(free_face_first, free_face_last + 1):
            wr(f"FORCE,1,{nid},0,{fmt_float(f_per_node)},{direction}")

    wr("ENDDATA")

    # ── Summary to stderr / stdout ───────────────────────────────────────────
    if out_path != "-":
        out.close()

    summary = [
        f"Written: {out_path if out_path != '-' else '(stdout)'}",
        f"  Mesh       : {nx} x {NY} x {NZ} = {n_elements} CHEXA8 elements, {n_nodes} nodes",
        f"  Bar        : L = {L:.4g} m,  W x H = {W:.4g} m x {H:.4g} m  (d = {d:.4g} m cubic)",
    ]
    if load_case == "thermal":
        summary.append(
            f"  Expected   : sigma_xx = {sigma_xx_thermal:.6G} Pa,  sigma_yy = sigma_zz = 0"
        )
    elif load_case == "axial":
        summary += [
            f"  Expected   : sigma_xx = {sigma_xx_axial:.6G} Pa",
            f"             : delta_x  = {delta_x_axial:.6G} m  (free-end T1 displacement)",
        ]
    elif load_case == "bending":
        summary += [
            f"  L/H ratio  : {slenderness:.2f}",
            f"  Expected   : delta_z_tip   = {delta_z_tip:.6G} m",
            f"             : sigma_xx_max  = {sigma_xx_surf:.6G} Pa  (fixed-end surface)",
        ]

    print("\n".join(summary), file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a CHEXA8 mesh BDF file for solver stress testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "n_elements",
        type=int,
        help="Target number of CHEXA8 elements. NY=NZ is chosen so elements are "
             "approximately cubic; nx is set to keep L=1m, W=H=0.2m fixed.",
    )
    parser.add_argument(
        "--load-case",
        choices=["thermal", "axial", "bending"],
        default="axial",
        help="Load case to generate (default: axial).",
    )
    parser.add_argument(
        "-o", "--output",
        default="-",
        help="Output BDF file path.  Use '-' for stdout (default).",
    )
    args = parser.parse_args()

    if args.n_elements < 1:
        parser.error("n_elements must be at least 1")

    generate(args.n_elements, args.load_case, args.output)


if __name__ == "__main__":
    main()
