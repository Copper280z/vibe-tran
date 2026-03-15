#!/usr/bin/env python3
"""Validate BDF files and inline BDF strings against pyNastran.

Checks two sources of BDF content:
  1. All *.bdf files under example_cases/
  2. Raw-string BDF literals embedded in tests/unit/test_bdf_parser.cpp and
     tests/integration/test_integration.cpp

Each BDF is parsed with pyNastran's BDF reader (xref=False so cross-reference
errors don't mask parse errors).  Results are reported per file/test, and a
summary exit code of 1 is returned if any failures are found.

Usage:
    python3 tools/validate_bdf_compliance.py [--verbose]
"""

import argparse
import glob
import io
import os
import re
import sys
import tempfile
import traceback

try:
    from pyNastran.bdf.bdf import BDF
except ImportError:
    sys.exit("pyNastran is not installed.  Run:  pip install pyNastran")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Cards our solver supports that pyNastran might warn about ─────────────────
# pyNastran is used as a compliance oracle, not a strict superset checker.
# We suppress warnings for cards that are intentionally unsupported by pyNastran
# (none expected) but we DO treat any parse error as a failure.

# ── BDF parsing wrapper ───────────────────────────────────────────────────────

def _has_executive_control(bdf_text: str) -> bool:
    """Return True if the BDF contains a SOL line before BEGIN BULK."""
    in_bulk = False
    for line in bdf_text.splitlines():
        upper = line.upper().strip()
        if upper.startswith("BEGIN BULK") or upper.startswith("BEGIN  BULK"):
            in_bulk = True
        if not in_bulk and upper.startswith("SOL "):
            return True
    return False


def parse_with_pynastran(bdf_text: str, label: str, verbose: bool) -> list[str]:
    """
    Attempt to parse bdf_text with pyNastran.

    BDFs that contain only bulk data (no SOL / Executive Control Deck) are
    validated with punch=True so that pyNastran does not require a Case
    Control Deck.  BDFs that do contain a SOL line are expected to be fully
    compliant (Executive + Case Control + Bulk Data) and are parsed normally.

    Returns a list of error strings (empty = success).
    """
    import contextlib

    errors = []
    punch = not _has_executive_control(bdf_text)

    # In punch mode pyNastran reads raw bulk data with no deck markers.
    # Strip BEGIN BULK and ENDDATA so they don't confuse the parser.
    if punch:
        filtered = []
        for line in bdf_text.splitlines():
            upper = line.upper().strip()
            if upper.startswith("BEGIN BULK") or upper.startswith("BEGIN  BULK"):
                continue
            if upper.startswith("ENDDATA"):
                continue
            filtered.append(line)
        bdf_text = "\n".join(filtered)

    # pyNastran reads from file; write to a temp file.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".bdf", delete=False, prefix="bdf_validate_"
    ) as fh:
        tmp_path = fh.name
        fh.write(bdf_text)

    try:
        model = BDF(debug=False)
        # Capture stdout/stderr from pyNastran (it can be chatty)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            model.read_bdf(tmp_path, xref=False, punch=punch)
        output = buf.getvalue()
        if verbose and output.strip():
            mode_tag = " [punch]" if punch else ""
            print(f"  [pyNastran output{mode_tag}]\n{output.rstrip()}")
    except Exception as exc:  # pylint: disable=broad-except
        errors.append(f"Parse error: {exc}")
        if verbose:
            traceback.print_exc()
    finally:
        os.unlink(tmp_path)

    return errors


# ── Inline BDF extractor ──────────────────────────────────────────────────────

def extract_inline_bdfs(cpp_path: str) -> list[tuple[str, str]]:
    """
    Extract raw-string BDF literals from a C++ source file.

    Looks for patterns like:
        const std::string bdf = R"(   -or-   const std::string bdf_X = R"(
    followed by BDF content ending with   )";

    Returns a list of (label, bdf_text) tuples.
    """
    with open(cpp_path) as fh:
        src = fh.read()

    results = []
    # Find every raw string literal that starts with a BDF-like pattern.
    # The delimiter is R"( ... )" — we match greedily up to the first )".
    raw_re = re.compile(
        r'(?:const\s+std::string\s+\w+\s*=\s*R"\(|R"\()(.*?)\)"',
        re.DOTALL,
    )

    test_name_re = re.compile(r'TEST\s*\(\s*\w+\s*,\s*(\w+)\s*\)', re.DOTALL)

    # Build a list of (position, test_name) for attribution
    test_positions = [(m.start(), m.group(1)) for m in test_name_re.finditer(src)]

    for m in raw_re.finditer(src):
        text = m.group(1)
        # Only treat it as a BDF if it contains SOL or BEGIN BULK or a card keyword
        if not re.search(r'(?:SOL|BEGIN BULK|GRID|CHEXA|CQUAD4|CTETRA|MAT1)', text):
            continue

        # Find the nearest preceding TEST(...) for labelling
        pos = m.start()
        label = "unknown"
        for tp, tn in reversed(test_positions):
            if tp <= pos:
                label = tn
                break

        results.append((label, text))

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    sources: list[tuple[str, str]] = []  # (label, bdf_text)

    # 1. Example BDF files
    bdf_files = sorted(glob.glob(os.path.join(REPO_ROOT, "example_cases", "*.bdf")))
    for path in bdf_files:
        with open(path) as fh:
            text = fh.read()
        label = os.path.relpath(path, REPO_ROOT)
        sources.append((label, text))

    # 2. Inline BDFs from test source files
    cpp_files = [
        os.path.join(REPO_ROOT, "tests", "unit", "test_bdf_parser.cpp"),
        os.path.join(REPO_ROOT, "tests", "integration", "test_integration.cpp"),
    ]
    for cpp_path in cpp_files:
        if not os.path.exists(cpp_path):
            print(f"[SKIP] {cpp_path} not found")
            continue
        inline = extract_inline_bdfs(cpp_path)
        rel = os.path.relpath(cpp_path, REPO_ROOT)
        for test_name, text in inline:
            sources.append((f"{rel}::{test_name}", text))

    if not sources:
        print("No BDF sources found.")
        return 1

    # ── Run validation ────────────────────────────────────────────────────────
    failures: list[tuple[str, list[str]]] = []
    passes: list[str] = []

    col_w = max(len(s[0]) for s in sources) + 2

    for label, text in sources:
        errors = parse_with_pynastran(text, label, args.verbose)
        if errors:
            failures.append((label, errors))
            status = "FAIL"
        else:
            passes.append(label)
            status = "PASS"
        print(f"  [{status}]  {label}")
        if errors:
            for e in errors:
                print(f"         {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(sources)
    print()
    print(f"Results: {len(passes)}/{total} passed, {len(failures)}/{total} failed")

    if failures:
        print("\nFailed cases:")
        for label, errs in failures:
            print(f"  {label}")
            for e in errs:
                print(f"    {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
