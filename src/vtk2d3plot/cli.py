"""vtk2d3plot command-line interface.

Usage::

    python -m vtk2d3plot.cli <vtk-dir-or-files> -o out.d3plot
    python -m vtk2d3plot.cli <dir> -o out.d3plot --f64 --title "My model"
    python -m vtk2d3plot.cli <dir> -o out.d3plot -q
"""

from __future__ import annotations

import argparse
import os
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
        help=(
            "Output d3plot path. Writes a family: the geometry file <name>.d3plot "
            "plus one state file per state <name>.d3plot01, <name>.d3plot02, ... "
            "Keep all files together."
        ),
    )
    ap.add_argument(
        "--f64",
        action="store_true",
        default=True,
        help="Write 64-bit output (default: enabled).",
    )
    ap.add_argument(
        "--f32",
        action="store_true",
        default=False,
        help="Write 32-bit output (overrides --f64).",
    )

    ap.add_argument(
        "--rotate-shell-stress",
        action="store_true",
        default=False,
        help="Rotate shell stress from element-local to global coordinate frame.",
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


def _default_title(inputs: list[str]) -> str:
    """Default d3plot title: the name of the folder that holds the inputs.

    A directory input is that folder; any other input (a .vtk/.vtu file, a
    file matched by a glob pattern, a .pvd manifest, a "<folder>/<stem>" stem)
    is held by its parent folder. When all inputs share one folder its name is
    the title, and a folder called "vtk" gives its parent's name. Inputs from
    several folders give no title. Paths are made absolute (lexically
    normalised) first, so relative inputs such as "vtk", "." or ".." get the
    same title as the same folder given absolute.
    """
    folders = set()
    for inp in inputs:
        p = Path(os.path.abspath(inp))
        folders.add(p if p.is_dir() else p.parent)
    if len(folders) != 1:
        return ""
    folder = folders.pop()
    if folder.name.lower() == "vtk" and folder.parent.name:
        return folder.parent.name
    return folder.name


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``vtk2d3plot`` console script."""
    ap = _build_parser()
    args = ap.parse_args(argv)

    import glob as _glob

    from .series import _sort_and_filter_files

    # Windows glob expansion (PowerShell doesn't expand wildcards). The matches
    # of each pattern come in file-system order; sort them with the same
    # numeric state ordering a directory input gets. Patterns and explicit
    # files keep the order the user gave.
    expanded_inputs: list[str] = []
    for inp in args.inputs:
        if "*" in inp or "?" in inp:
            matches = [Path(m) for m in _glob.glob(inp)]
            if not matches:
                # Bug report 2026-10-11, suspicion S26: a pattern that matches
                # nothing is an error even when another input matches.
                print(f"ERROR: pattern '{inp}' matched no files", file=sys.stderr)
                raise SystemExit(1)
            try:
                ordered = _sort_and_filter_files(matches)
            except ValueError as exc:
                print(f"ERROR: pattern {inp!r}: {exc}", file=sys.stderr)
                raise SystemExit(1) from exc
            dropped = sorted(set(map(str, matches)) - set(map(str, ordered)))
            if dropped and not args.quiet:
                print(
                    f"WARNING: pattern {inp!r}: ignored {len(dropped)} file(s) without "
                    f"the <stem>A<number>.vtk/.vtu state name: {', '.join(dropped)}",
                    file=sys.stderr,
                )
            expanded_inputs.extend(str(m) for m in ordered)
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
            raw = pn_path.read_bytes()
            if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
                text = raw.decode("utf-16")
            elif raw.startswith(b"\xef\xbb\xbf"):
                text = raw.decode("utf-8-sig")
            else:
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    text = raw.decode("cp1252")
                    if not args.quiet:
                        print(
                            "WARNING: part-names file is cp1252-encoded, not UTF-8", file=sys.stderr
                        )
        except UnicodeDecodeError:
            print(
                f"ERROR: cannot decode part-names file {pn_path}",
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

    if not args.title:
        args.title = _default_title(args.inputs)

    # Per-state progress callback for non-quiet mode
    progress_fn = None
    if not args.quiet and not args.json:

        def progress_fn(state_index: int, n_states: int) -> None:
            print(f"  Processing state {state_index + 1}/{n_states}...")

    # --f32 overrides --f64
    use_f64 = args.f64 and not args.f32

    try:
        rep = convert(
            inputs,
            args.output,
            title=args.title,
            f64=use_f64,
            part_names=part_names,
            progress=progress_fn,
            rotate_shell_stress=args.rotate_shell_stress,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.json:
        print(rep.to_json())
    elif not args.quiet:
        state_files = rep.output_files[1:]
        if len(state_files) == 1:
            print(f"Wrote {rep.output_files[0]} + 1 state file {state_files[0]}")
        elif state_files:
            print(
                f"Wrote {rep.output_files[0]} + {len(state_files)} state files "
                f"{state_files[0]}..{state_files[-1]}"
            )
        else:
            print(f"Wrote {rep.output}")
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
