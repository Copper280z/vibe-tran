#!/usr/bin/env python3
"""
e2e_test_runner.py — End-to-end test runner for the Nastran FEA solver.

Runs the solver binary against every *.bdf file in a given directory, places
output files in a results directory, then validates results against companion
*.expected.json files (one per BDF).

Usage
-----
    python tools/e2e_test_runner.py \\
        --solver  build/nastran_solver \\
        --bdf-dir tests/e2e/cases \\
        --results-dir /tmp/e2e_results \\
        [--solver-args "--backend=cpu"]

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
          "component": "sx",    // sx sy sxy sz syz szx mx my mxy von_mises
          "expected": 0.0,
          "abs_tol": 1.0        // |actual-expected| <= abs_tol
        }
      ]
    }

If both rel_tol and abs_tol are given the check passes when *either* condition
holds.  If neither is given, abs_tol=1e-6 is used as a default.

BDF files must include DISPLACEMENT=ALL and STRESS=ALL in the case control so
the CSV output contains all result data.  The runner always appends --csv to
the solver invocation to force CSV output.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── CSV column indices (after stripping the '#' comment header) ────────────────

_NODE_DOF_NAMES = ["T1", "T2", "T3", "R1", "R2", "R3"]
_ELEM_COMP_NAMES = ["sx", "sy", "sxy", "sz", "syz", "szx", "mx", "my", "mxy", "von_mises"]


def _parse_csv(path: Path, *, is_node: bool) -> dict:
    """
    Parse a .node.csv or .elem.csv file produced by the solver.

    Returns a dict keyed by (id, subcase_id) mapping to a {field: float} dict.

    Node CSV columns:  node_id, subcase_id, T1, T2, T3, R1, R2, R3
    Elem CSV columns:  elem_id, subcase_id, elem_type, sx, sy, sxy, sz, syz, szx,
                       mx, my, mxy, von_mises
    """
    data: dict = {}
    names = _NODE_DOF_NAMES if is_node else _ELEM_COMP_NAMES

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            try:
                id_ = int(parts[0])
                subcase = int(parts[1])
                # elem CSV has elem_type string at parts[2]; skip it
                value_start = 2 if is_node else 3
                values = [float(p) for p in parts[value_start : value_start + len(names)]]
            except (ValueError, IndexError) as exc:
                raise RuntimeError(f"Malformed CSV line in {path}: {raw!r}") from exc

            data[(id_, subcase)] = dict(zip(names, values))

    return data


@dataclass
class CheckResult:
    passed: bool
    description: str
    message: str


def _evaluate_check(check: dict, node_data: dict, elem_data: dict) -> CheckResult:
    """Run a single check from the expected JSON against the parsed CSV data."""

    subcase = check.get("subcase", 1)
    expected = check["expected"]
    rel_tol: Optional[float] = check.get("rel_tol")
    abs_tol: Optional[float] = check.get("abs_tol")

    if rel_tol is None and abs_tol is None:
        abs_tol = 1e-6

    ctype = check["type"]

    # ── Fetch actual value ────────────────────────────────────────────────────
    if ctype == "node_displacement":
        node_id = check["node_id"]
        dof = check["dof"]
        key = (node_id, subcase)
        if key not in node_data:
            return CheckResult(
                False,
                f"node {node_id} dof {dof}",
                f"node {node_id} not found in subcase {subcase} — "
                "check that the BDF has DISPLACEMENT=ALL",
            )
        actual = node_data[key].get(dof)
        if actual is None:
            return CheckResult(False, f"node {node_id} dof {dof}", f"unknown DOF name '{dof}'")
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
                "check that the BDF has STRESS=ALL",
            )
        actual = elem_data[key].get(component)
        if actual is None:
            return CheckResult(False, f"elem {elem_id} {component}", f"unknown component '{component}'")
        description = f"elem_stress elem={elem_id:>4}  comp={component}"

    else:
        return CheckResult(False, f"? {ctype}", f"unknown check type '{ctype}'")

    # ── Evaluate tolerance ────────────────────────────────────────────────────
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
            True,
            description,
            f"actual={actual:+.6e}  expected={expected:+.6e}  [{tol_str}]",
        )
    else:
        return CheckResult(
            False,
            description,
            f"actual={actual:+.6e}  expected={expected:+.6e}  diff={diff:.2e}  [{tol_str}]",
        )


def run_one_test(
    solver: str,
    solver_extra_args: list[str],
    bdf_path: Path,
    results_dir: Path,
) -> tuple[bool, list[str]]:
    """
    Run the solver on bdf_path, placing output in results_dir.

    Returns (success, list_of_message_lines).
    """
    stem = bdf_path.stem
    f06_out = results_dir / f"{stem}.f06"

    # Always request CSV output so we have machine-readable results.
    cmd = [solver] + solver_extra_args + ["--csv", str(bdf_path), str(f06_out)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        return False, [f"  ERROR: solver binary not found: {solver}"]
    except subprocess.TimeoutExpired:
        return False, ["  ERROR: solver timed out after 120 s"]

    if proc.returncode != 0:
        lines = [f"  ERROR: solver exited with code {proc.returncode}"]
        if proc.stdout:
            lines += ["  --- stdout ---"] + proc.stdout.splitlines()[-20:]
        if proc.stderr:
            lines += ["  --- stderr ---"] + proc.stderr.splitlines()[-20:]
        return False, lines

    return True, []


def check_results(
    bdf_path: Path,
    results_dir: Path,
    expected_path: Path,
) -> tuple[bool, list[str]]:
    """Load CSV outputs and validate against expected JSON. Returns (passed, messages)."""

    stem = bdf_path.stem

    node_csv = results_dir / f"{stem}.node.csv"
    elem_csv = results_dir / f"{stem}.elem.csv"

    node_data: dict = {}
    elem_data: dict = {}

    if node_csv.exists():
        try:
            node_data = _parse_csv(node_csv, is_node=True)
        except RuntimeError as exc:
            return False, [f"  ERROR parsing {node_csv.name}: {exc}"]
    else:
        # CSV not written likely means no DISPLACEMENT=ALL in BDF
        pass

    if elem_csv.exists():
        try:
            elem_data = _parse_csv(elem_csv, is_node=False)
        except RuntimeError as exc:
            return False, [f"  ERROR parsing {elem_csv.name}: {exc}"]

    with open(expected_path, encoding="utf-8") as fh:
        expected = json.load(fh)

    checks = expected.get("checks", [])
    messages: list[str] = []
    all_passed = True

    for check in checks:
        result = _evaluate_check(check, node_data, elem_data)
        status = "PASS" if result.passed else "FAIL"
        messages.append(f"  {status}  {result.description:<40}  {result.message}")
        if not result.passed:
            all_passed = False

    return all_passed, messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end test runner for nastran_solver",
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
        help="Extra arguments to forward to the solver, as a quoted string "
             "(e.g. '--backend=cpu-pcg'). --csv is always added automatically.",
    )
    parser.add_argument(
        "--bdf-dir", required=True, type=Path,
        help="Directory containing *.bdf and *.expected.json files",
    )
    parser.add_argument(
        "--results-dir", required=True, type=Path,
        help="Directory where solver output (f06/op2/csv) is written",
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

        # ── Run solver ────────────────────────────────────────────────────────
        ok, run_msgs = run_one_test(args.solver, solver_extra, bdf_path, args.results_dir)
        for m in run_msgs:
            print(m)

        if not ok:
            failed_count += 1
            print("→ FAILED (solver error)")
            continue

        # ── Validate checks ───────────────────────────────────────────────────
        if not has_expected:
            skipped_count += 1
            print("→ SKIP (no .expected.json — solver ran successfully)")
            continue

        check_ok, check_msgs = check_results(bdf_path, args.results_dir, expected_path)
        for m in check_msgs:
            print(m)

        if check_ok:
            passed_count += 1
            print("→ PASSED")
        else:
            failed_count += 1
            print("→ FAILED")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = passed_count + failed_count + skipped_count
    print(f"\n{'═' * 70}")
    print(f"Results: {passed_count} passed / {failed_count} failed / {skipped_count} skipped "
          f"({total} total)")

    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
