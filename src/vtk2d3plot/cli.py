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
        "--part-names",
        default=None,
        help="File mapping part IDs to names (format: 'id<TAB>name' per line, # comments).",
    )
    ap.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress informational output (only errors).",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output machine-readable JSON report.",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``vtk2d3plot`` console script."""
    ap = _build_parser()
    args = ap.parse_args(argv)

    import glob as _glob

    # Windows glob expansion (PowerShell doesn't expand wildcards)
    expanded_inputs: list[str] = []
    for inp in args.inputs:
        if "*" in inp or "?" in inp:
            expanded_inputs.extend(_glob.glob(inp))
        else:
            expanded_inputs.append(inp)
    if not expanded_inputs:
        print("ERROR: no input files matched", file=sys.stderr)
        raise SystemExit(1)
    args.inputs = expanded_inputs

    from .api import convert

    # Resolve inputs: single directory or list of files
    inputs: str | Path | list[Path]
    if len(args.inputs) == 1:
        p = Path(args.inputs[0])
        inputs = p  # directory or single file — let load_series figure it out
    else:
        inputs = [Path(f) for f in args.inputs]

    part_names = None
    if args.part_names:
        part_names = {}
        pn_path = Path(args.part_names)
        if not pn_path.exists():
            print(f"ERROR: part-names file not found: {pn_path}", file=sys.stderr)
            raise SystemExit(1)
        try:
            text = pn_path.read_text(encoding="utf-8-sig")  # handles BOM
        except UnicodeDecodeError:
            try:
                text = pn_path.read_text(encoding="cp1252")
                if not args.quiet:
                    print("WARNING: part-names file is cp1252-encoded, not UTF-8", file=sys.stderr)
            except UnicodeDecodeError:
                print(
                    f"ERROR: cannot decode part-names file {pn_path} (tried UTF-8, cp1252)",
                    file=sys.stderr,
                )
                raise SystemExit(1)
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                parts = line.split(None, 1)  # fallback: any whitespace
            if len(parts) == 2:
                try:
                    part_names[int(parts[0])] = parts[1].strip()
                except ValueError:
                    pass

    # Default title from input directory name
    if not args.title and len(args.inputs) == 1:
        p = Path(args.inputs[0])
        if p.is_dir():
            dir_name = p.name
            # If directory name is 'vtk', use parent directory name
            if dir_name.lower() == "vtk" and p.parent.name:
                dir_name = p.parent.name
            args.title = dir_name

    # Per-state progress callback for non-quiet mode
    progress_fn = None
    if not args.quiet and not args.json:

        def progress_fn(state_index: int, n_states: int) -> None:
            print(f"  Processing state {state_index + 1}/{n_states}...")

    try:
        rep = convert(
            inputs,
            args.output,
            title=args.title,
            f64=args.f64,
            part_names=part_names,
            progress=progress_fn,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.json:
        print(rep.to_json())
    elif not args.quiet:
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
