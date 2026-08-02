"""vtk2d3plot command-line interface.

Usage::

    python -m vtk2d3plot.cli <vtk-dir-or-files> -o out.d3plot
    python -m vtk2d3plot.cli <dir> -o out.d3plot --f64 --title "My model"
    python -m vtk2d3plot.cli <dir> -o out.d3plot -q
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="vtk2d3plot",
        description="Convert legacy ASCII VTK time series to LS-DYNA d3plot.",
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        help="VTK directory, glob stem, or explicit list of .vtk files.",
    )
    ap.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output d3plot path (family: <name>.d3plot + <name>.d3plot01).",
    )
    ap.add_argument(
        "--f64",
        action="store_true",
        default=False,
        help="Write 64-bit output (default: 32-bit).",
    )
    ap.add_argument(
        "--title",
        default="",
        help="Title string embedded in the d3plot header.",
    )
    ap.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress informational output (only errors).",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``vtk2d3plot`` console script."""
    ap = _build_parser()
    args = ap.parse_args(argv)

    from .api import convert

    # Resolve inputs: single directory or list of files
    inputs: str | Path | list[Path]
    if len(args.inputs) == 1:
        p = Path(args.inputs[0])
        inputs = p  # directory or single file — let load_series figure it out
    else:
        inputs = [Path(f) for f in args.inputs]

    try:
        rep = convert(inputs, args.output, title=args.title, f64=args.f64)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if not args.quiet:
        print(f"Wrote {rep.output} and {rep.output}01")
        print(
            f"  {rep.n_states} states, {rep.n_nodes} nodes, "
            f"{rep.n_solids} solids, {rep.n_shells} shells, "
            f"{rep.n_beams} beams, {rep.n_parts} parts"
        )
        print(f"  dialect: {rep.dialect}")
        if rep.warnings:
            for w in rep.warnings:
                print(f"  WARNING: {w}")


if __name__ == "__main__":
    main()
