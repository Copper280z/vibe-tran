#!/usr/bin/env python3
"""
e2e_test_runner.py — End-to-end test runner for the Nastran FEA solver.

Runs the solver binary against every *.bdf file in a given directory, places
output files in a results directory, then validates results against companion
*.expected.json files (one per BDF) using pyNastran to read the OP2 binary.

Usage
-----
    python tools/e2e_test_runner.py \\
        --solver  build/nastran_solver \\
        --bdf-dir tests/e2e/cases \\
        --results-dir /tmp/e2e_results \\
        [--solver-args "--backend=cpu-pcg"]

Expected JSON format
--------------------
Each <stem>.expected.json lives alongside <stem>.bdf and looks like:

    {
      "description": "one-line human description",
      "checks": [
        {
          "type": "node_displacement",
          "subcase": 1,
          "node_id": 2,
          "dof": "T1",          // T1 T2 T3 R1 R2 R3
          "expected": 0.01,
          "rel_tol": 1e-6       // |actual-expected| <= rel_tol * |expected|
        },
        {
          "type": "element_stress",
          "subcase": 1,
          "elem_id": 1,
          "component": "sz",    // sx sy sxy sz syz szx von_mises
          "expected": 0.0,
          "abs_tol": 1.0        // |actual-expected| <= abs_tol
        }
      ]
    }

If both rel_tol and abs_tol are given the check passes when *either* condition
holds.  If neither is given abs_tol=1e-6 is used as a default.

BDF files must include DISPLACEMENT=ALL and STRESS=ALL in the case control so
the OP2 output contains all result data.

OP2 result layout (via pyNastran)
----------------------------------
Displacements:
    op2.displacements[isubcase].node_gridtype[:, 0]  → node IDs
    op2.displacements[isubcase].data[0, :, :]         → shape (nnodes, 6): T1-R3

Plate stresses (CQUAD4 / CTRIA3):
    op2.cquad4_stress[isubcase]  /  op2.ctria3_stress[isubcase]
    .element_node  → shape (nrows, 2): [elem_id, node_id] (0 = centroid)
    .data[0, :, :] → shape (nrows, 8): fiber_dist, sx, sy, sxy, angle, omax, omin, vm

Solid stresses (CHEXA8 / CTETRA4 / CTETRA10):
    op2.chexa_stress[isubcase]  /  op2.ctetra_stress[isubcase]
    .element_node  → shape (nrows, 2): [elem_id, node_id] (0 = centroid)
    .data[0, :, :] → shape (nrows, 7): sx, sy, sz, sxy, syz, szx, vm
"""

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── OP2 parser (pyNastran) ────────────────────────────────────────────────────

_NODE_DOFS = ["T1", "T2", "T3", "R1", "R2", "R3"]

# Plate stress column indices in pyNastran op2 .data array
# Headers: ['fiber_distance', 'oxx', 'oyy', 'txy', 'angle', 'omax', 'omin', 'max_shear']
# Note: 'max_shear' column stores the von Mises stress as written by this solver (s_code=0).
_PLATE_COLS = {
    "sx": 1, "sy": 2, "sxy": 3,
    "angle": 4, "major": 5, "minor": 6, "von_mises": 7,
}

# Solid stress column indices in pyNastran op2 .data array
# Headers: ['oxx', 'oyy', 'ozz', 'txy', 'tyz', 'txz', 'omax', 'omid', 'omin', 'max_shear']
# Note: 'max_shear' column stores the von Mises stress as written by this solver.
_SOLID_COLS = {
    "sx": 0, "sy": 1, "sz": 2,
    "sxy": 3, "syz": 4, "szx": 5, "von_mises": 9,
}


def _load_op2(op2_path: Path) -> tuple[dict, dict]:
    """
    Load an OP2 file with pyNastran and return (node_data, elem_data).

    node_data: {(node_id, subcase_id): {"T1": float, ...}}
    elem_data: {(elem_id, subcase_id): {"sx": float, ..., "von_mises": float}}
    """
    try:
        from pyNastran.op2.op2 import OP2
    except ImportError as exc:
        raise RuntimeError(
            "pyNastran is required to read OP2 results.  "
            "Install it with: pip install pyNastran"
        ) from exc

    op2 = OP2(debug=False)
    op2.read_op2(str(op2_path))

    node_data: dict = {}
    elem_data: dict = {}

    # ── Displacements ──────────────────────────────────────────────────────
    for isubcase, table in op2.displacements.items():
        node_ids = table.node_gridtype[:, 0]   # (nnodes,)
        values = table.data[0]                  # (nnodes, 6)
        for i, nid in enumerate(node_ids):
            key = (int(nid), int(isubcase))
            node_data[key] = {dof: float(values[i, j]) for j, dof in enumerate(_NODE_DOFS)}

    stress = op2.op2_results.stress

    # ── Plate stresses (CQUAD4) ────────────────────────────────────────────
    for isubcase, table in stress.cquad4_stress.items():
        _extract_plate_stress(table, int(isubcase), elem_data)

    # ── Plate stresses (CTRIA3) ────────────────────────────────────────────
    for isubcase, table in stress.ctria3_stress.items():
        _extract_plate_stress(table, int(isubcase), elem_data)

    # ── Solid stresses (CHEXA8) ────────────────────────────────────────────
    for isubcase, table in stress.chexa_stress.items():
        _extract_solid_stress(table, int(isubcase), elem_data)

    # ── Solid stresses (CTETRA4 / CTETRA10) ───────────────────────────────
    for isubcase, table in stress.ctetra_stress.items():
        _extract_solid_stress(table, int(isubcase), elem_data)

    return node_data, elem_data


def _extract_plate_stress(table, isubcase: int, elem_data: dict) -> None:
    """
    Extract centroidal plate stress from a pyNastran plate stress table.

    For CQUAD4/CTRIA3 the OP2 stores one row per (element, layer) pair at
    both the bottom and top fiber.  We average them to get a single centroidal
    stress for each element.
    """
    elem_node = table.element_node   # (nrows, 2): [elem_id, node_id/fiber_flag]
    data = table.data[0]             # (nrows, 8)

    # Accumulate sum and count per element so we can average layers
    accum: dict[int, list] = {}
    counts: dict[int, int] = {}
    for row_idx in range(elem_node.shape[0]):
        eid = int(elem_node[row_idx, 0])
        row = data[row_idx]
        if eid not in accum:
            accum[eid] = [0.0] * len(_PLATE_COLS)
            counts[eid] = 0
        for col_name, col_idx in _PLATE_COLS.items():
            accum[eid][list(_PLATE_COLS.keys()).index(col_name)] += float(row[col_idx])
        counts[eid] += 1

    for eid, vals in accum.items():
        n = counts[eid]
        key = (eid, isubcase)
        elem_data[key] = {
            col_name: vals[i] / n
            for i, col_name in enumerate(_PLATE_COLS.keys())
        }


def _extract_solid_stress(table, isubcase: int, elem_data: dict) -> None:
    """
    Extract centroidal solid stress from a pyNastran solid stress table.

    The OP2 stores rows for each corner node plus a CEN entry.  We look for
    rows where element_node[:, 1] == 0 (centroid), falling back to averaging
    all corner rows if no centroid row exists.
    """
    elem_node = table.element_node   # (nrows, 2)
    data = table.data[0]             # (nrows, 7)

    # Try centroid rows first (node_id == 0)
    centroid_found: set[int] = set()
    for row_idx in range(elem_node.shape[0]):
        eid = int(elem_node[row_idx, 0])
        nid = int(elem_node[row_idx, 1])
        if nid == 0:
            row = data[row_idx]
            key = (eid, isubcase)
            elem_data[key] = {
                col_name: float(row[col_idx])
                for col_name, col_idx in _SOLID_COLS.items()
            }
            centroid_found.add(eid)

    # Fall back: average all rows for elements that had no centroid entry
    accum: dict[int, list] = {}
    counts: dict[int, int] = {}
    for row_idx in range(elem_node.shape[0]):
        eid = int(elem_node[row_idx, 0])
        if eid in centroid_found:
            continue
        row = data[row_idx]
        if eid not in accum:
            accum[eid] = [0.0] * len(_SOLID_COLS)
            counts[eid] = 0
        for col_name, col_idx in _SOLID_COLS.items():
            accum[eid][list(_SOLID_COLS.keys()).index(col_name)] += float(row[col_idx])
        counts[eid] += 1

    for eid, vals in accum.items():
        n = counts[eid]
        key = (eid, isubcase)
        elem_data[key] = {
            col_name: vals[i] / n
            for i, col_name in enumerate(_SOLID_COLS.keys())
        }


# ── Check evaluation ──────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    passed: bool
    description: str
    message: str


def _evaluate_check(check: dict, node_data: dict, elem_data: dict) -> CheckResult:
    subcase = check.get("subcase", 1)
    expected = check["expected"]
    rel_tol: Optional[float] = check.get("rel_tol")
    abs_tol: Optional[float] = check.get("abs_tol")

    if rel_tol is None and abs_tol is None:
        abs_tol = 1e-6

    ctype = check["type"]

    if ctype == "node_displacement":
        node_id = check["node_id"]
        dof = check["dof"]
        key = (node_id, subcase)
        if key not in node_data:
            return CheckResult(
                False,
                f"node {node_id} dof {dof}",
                f"node {node_id} not found in subcase {subcase} — "
                "check BDF has DISPLACEMENT=ALL",
            )
        actual = node_data[key].get(dof)
        if actual is None:
            return CheckResult(False, f"node {node_id} dof {dof}", f"unknown DOF '{dof}'")
        description = f"node_disp  node={node_id:>4}  dof={dof}"

    elif ctype == "element_stress":
        elem_id = check["elem_id"]
        component = check["component"]
        key = (elem_id, subcase)
        if key not in elem_data:
            return CheckResult(
                False,
                f"elem {elem_id} {component}",
                f"element {elem_id} not found in subcase {subcase} — "
                "check BDF has STRESS=ALL",
            )
        actual = elem_data[key].get(component)
        if actual is None:
            return CheckResult(
                False,
                f"elem {elem_id} {component}",
                f"component '{component}' not available for this element type "
                "(plates: sx sy sxy von_mises; solids: sx sy sz sxy syz szx von_mises)",
            )
        description = f"elem_stress elem={elem_id:>4}  comp={component}"

    else:
        return CheckResult(False, f"? {ctype}", f"unknown check type '{ctype}'")

    diff = abs(actual - expected)
    tol_parts = []
    passes_any = False

    if abs_tol is not None:
        if diff <= abs_tol:
            passes_any = True
        tol_parts.append(f"abs_tol={abs_tol:.2e}")

    if rel_tol is not None:
        ref = abs(expected) if abs(expected) > 1e-300 else 1.0
        if diff <= rel_tol * ref:
            passes_any = True
        tol_parts.append(f"rel_tol={rel_tol:.2e}")

    tol_str = ", ".join(tol_parts)
    if passes_any:
        return CheckResult(
            True, description,
            f"actual={actual:+.6e}  expected={expected:+.6e}  [{tol_str}]",
        )
    else:
        return CheckResult(
            False, description,
            f"actual={actual:+.6e}  expected={expected:+.6e}  diff={diff:.2e}  [{tol_str}]",
        )


# ── Test execution ────────────────────────────────────────────────────────────

def run_one_test(
    solver: str,
    solver_extra_args: list[str],
    bdf_path: Path,
    results_dir: Path,
) -> tuple[bool, list[str], Path]:
    """
    Invoke the solver on bdf_path, writing output to results_dir.

    Returns (success, list_of_message_lines, op2_path).
    """
    stem = bdf_path.stem
    f06_out = results_dir / f"{stem}.f06"
    op2_path = results_dir / f"{stem}.op2"

    cmd = [solver] + solver_extra_args + [str(bdf_path), str(f06_out)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        return False, [f"  ERROR: solver binary not found: {solver}"], op2_path
    except subprocess.TimeoutExpired:
        return False, ["  ERROR: solver timed out after 120 s"], op2_path

    if proc.returncode != 0:
        lines = [f"  ERROR: solver exited with code {proc.returncode}"]
        if proc.stdout:
            lines += ["  --- stdout ---"] + proc.stdout.splitlines()[-20:]
        if proc.stderr:
            lines += ["  --- stderr ---"] + proc.stderr.splitlines()[-20:]
        return False, lines, op2_path

    return True, [], op2_path


def check_results(
    op2_path: Path,
    expected_path: Path,
) -> tuple[bool, list[str]]:
    """Load op2_path via pyNastran and validate against expected_path."""

    if not op2_path.exists():
        return False, [f"  ERROR: OP2 not found: {op2_path}"]

    try:
        node_data, elem_data = _load_op2(op2_path)
    except Exception as exc:
        return False, [f"  ERROR loading OP2: {exc}"]

    with open(expected_path, encoding="utf-8") as fh:
        expected = json.load(fh)

    messages: list[str] = []
    all_passed = True

    for check in expected.get("checks", []):
        if "type" not in check:
            continue  # comment/note-only entries have no type
        result = _evaluate_check(check, node_data, elem_data)
        status = "PASS" if result.passed else "FAIL"
        messages.append(f"  {status}  {result.description:<40}  {result.message}")
        if not result.passed:
            all_passed = False

    return all_passed, messages


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end test runner for nastran_solver (reads OP2 output via pyNastran)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--solver", required=True,
        help="Path to the solver binary (e.g. build/nastran_solver)",
    )
    parser.add_argument(
        "--solver-args", default="",
        metavar="'ARGS'",
        help="Extra arguments forwarded to the solver as a quoted string "
             "(e.g. '--backend=cpu-pcg').",
    )
    parser.add_argument(
        "--bdf-dir", required=True, type=Path,
        help="Directory containing *.bdf and *.expected.json files",
    )
    parser.add_argument(
        "--results-dir", required=True, type=Path,
        help="Directory where solver output (f06/op2) is written",
    )
    parser.add_argument(
        "--pattern", default="*.bdf",
        help="Glob pattern to select BDF files (default: *.bdf)",
    )
    args = parser.parse_args()

    solver_extra = shlex.split(args.solver_args) if args.solver_args.strip() else []
    args.results_dir.mkdir(parents=True, exist_ok=True)

    bdf_files = sorted(args.bdf_dir.glob(args.pattern))
    if not bdf_files:
        print(f"No files matching '{args.pattern}' found in {args.bdf_dir}", file=sys.stderr)
        sys.exit(1)

    passed_count = 0
    failed_count = 0
    skipped_count = 0
    sep = "─" * 70

    for bdf_path in bdf_files:
        expected_path = bdf_path.with_suffix(".expected.json")
        has_expected = expected_path.exists()

        print(f"\n{sep}")
        print(f"BDF : {bdf_path.name}")
        if has_expected:
            with open(expected_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            print(f"DESC: {meta.get('description', '(no description)')}")

        ok, run_msgs, op2_path = run_one_test(
            args.solver, solver_extra, bdf_path, args.results_dir
        )
        for m in run_msgs:
            print(m)

        if not ok:
            failed_count += 1
            print("→ FAILED (solver error)")
            continue

        if not has_expected:
            skipped_count += 1
            print("→ SKIP (no .expected.json — solver ran successfully)")
            continue

        check_ok, check_msgs = check_results(op2_path, expected_path)
        for m in check_msgs:
            print(m)

        if check_ok:
            passed_count += 1
            print("→ PASSED")
        else:
            failed_count += 1
            print("→ FAILED")

    total = passed_count + failed_count + skipped_count
    print(f"\n{'═' * 70}")
    print(f"Results: {passed_count} passed / {failed_count} failed / {skipped_count} skipped "
          f"({total} total)")

    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
