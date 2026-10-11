"""Compare a d3plot with the VTK series it was made from, joined by node ID.

Usage:
    .venv\\Scripts\\python.exe tools\\reference\\verify_byid.py <d3plot> <vtk_dir>

For every state it prints max|dxyz| between the d3plot node_displacement
(absolute coordinates) and the VTK POINTS, joined by node ID, never by index.

Pairing rules (same as the converter, so a state is compared with the file it
was converted from):
- VTK files are ordered by the numeric ``A(\\d+)`` suffix
  (``vtk2d3plot.series._sort_and_filter_files``), not by string order.
- Zero NODE_IDs are renumbered max+1, max+2, ... exactly as the converter does
  (``vtk2d3plot.mapping.map_topology``).
- The number of VTK files must equal the number of d3plot states, and each
  state time must match the VTK time (relative to the largest time, 1e-5,
  because anim_to_vtk prints TIME with 6 significant digits). Otherwise the
  script prints ERROR and exits with code 2. The same holds for .vtk files of
  more than one series (stem) in the folder; files without an ``A<digits>``
  state number (e.g. notes.vtk) are ignored, as by the converter. The VTK
  time is the FIELD TIME value (official dialect) or, without it,
  ``t=<value>`` in the title line (port dialect). States without any time are
  not time-checked; the script then prints a WARNING with their count.

Exit codes: 0 = compared, 2 = pairing error (count, time, stem or node-ID
mismatch).
"""

import re
import sys
from pathlib import Path

import numpy as np
from lasso.dyna.d3plot import D3plot

TIME_RTOL = 1.0e-5
# Port dialect: the time is only in the title line, "pyradioss state t=%.9E".
# Same rule as the converter (bug report 2026-10-11 #29): "t=" must start a word
# and be followed by a well-formed number, so "Result=extracted" or "count=5"
# give no time instead of a crash or a wrong time.
TITLE_TIME_RE = re.compile(r"\bt=([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(?!\w|\.\w)")


def _read_tokens(lines, start, count):
    """Collect ``count`` whitespace tokens from ``lines[start:]``; return (tokens, next_line)."""
    vals = []
    j = start
    while len(vals) < count:
        vals.extend(lines[j].split())
        j += 1
    return vals, j


def parse_vtk(path):
    """Independent minimal legacy-VTK parser: (n_pts, n_cells, points, raw NODE_ID, time).

    time = FIELD TIME if present, else ``t=<value>`` from the title line, else None.
    """
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()
    n_pts = n_cells = pts = node_ids = tv = None
    title_m = TITLE_TIME_RE.search(lines[1]) if len(lines) > 1 else None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("TIME"):
            tv = float(lines[i + 1].split()[0])
        elif line.startswith("POINTS"):
            n_pts = int(line.split()[1])
            vals, i = _read_tokens(lines, i + 1, n_pts * 3)
            pts = np.array(vals, dtype=float).reshape(n_pts, 3)
            continue
        elif line.startswith("CELLS"):
            n_cells = int(line.split()[1])
        elif line.startswith("SCALARS NODE_ID"):
            vals, i = _read_tokens(lines, i + 2, n_pts)  # skip LOOKUP_TABLE line
            node_ids = np.array(vals, dtype=np.int64)
            continue
        i += 1
    if tv is None and title_m is not None:
        try:
            tv = float(title_m.group(1))
        except ValueError:
            tv = None
    return n_pts, n_cells, pts, node_ids, tv


def warn_untimed(n_untimed: int) -> None:
    """Say loudly that some states were paired without a time check."""
    if n_untimed:
        print(
            f"WARNING: {n_untimed} state(s) have no TIME in the VTK file - "
            "pairing not time-checked (file order only)"
        )


def ordered_vtk_files(vtk_dir) -> list[Path]:
    """VTK files in the converter's order (numeric A(\\d+) suffix)."""
    from vtk2d3plot.series import _sort_and_filter_files

    files = [f for f in Path(vtk_dir).iterdir() if f.is_file() and f.suffix.lower() == ".vtk"]
    return _sort_and_filter_files(files)


def converter_node_ids(vtk_path) -> np.ndarray:
    """Node IDs in VTK point order, zero IDs renumbered exactly as the converter."""
    from vtk2d3plot.mapping import map_topology
    from vtk2d3plot.vtk_reader import read_vtk

    return np.asarray(map_topology(read_vtk(vtk_path)).node_ids, dtype=np.int64)


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    d3, vtk_dir = argv[0], argv[1]
    p = D3plot(d3)
    d_ids = np.asarray(p.arrays["node_ids"]).astype(np.int64)
    nd = p.arrays["node_displacement"]
    t = np.asarray(p.arrays["timesteps"], dtype=np.float64)

    try:
        vtks = ordered_vtk_files(vtk_dir)
    except ValueError as exc:  # e.g. .vtk files of two series in one folder
        print(f"ERROR: {exc}")
        return 2
    if len(vtks) != len(t):
        print(f"ERROR: {len(vtks)} VTK files but {len(t)} d3plot states - cannot pair them")
        return 2
    ids = converter_node_ids(vtks[0])
    raw0 = parse_vtk(vtks[0])[3]
    print("d3plot node_ids[:8] :", d_ids[:8].tolist(), " sorted:", bool(np.all(np.diff(d_ids) > 0)))
    print("vtk    NODE_ID[:8] :", ids[:8].tolist(), " sorted:", bool(np.all(np.diff(ids) > 0)))
    if raw0 is not None:
        print("zero NODE_IDs renumbered:", int((raw0 == 0).sum()))
    print("same id set        :", bool(np.array_equal(np.sort(ids), np.sort(d_ids))))
    print("same index order   :", bool(np.array_equal(ids, d_ids)))
    lut = {int(v): k for k, v in enumerate(d_ids)}
    missing = [int(v) for v in ids if int(v) not in lut]
    if missing:
        print(f"ERROR: {len(missing)} VTK node IDs not in the d3plot, e.g. {missing[:5]}")
        return 2
    perm = np.array([lut[int(v)] for v in ids])

    t_scale = max(float(np.abs(t).max()), np.finfo(float).tiny)
    errors = 0
    n_untimed = 0
    for idx, f in enumerate(vtks):
        n_pts, n_cells, pts, vids, tv = parse_vtk(f)
        if raw0 is not None and (vids is None or not np.array_equal(vids, raw0)):
            print(f"ERROR: NODE_ID array of {f.name} differs from {vtks[0].name}")
            errors += 1
            continue
        diff = np.abs(nd[idx][perm] - pts)
        t_d3 = float(t[idx])
        flag = ""
        if tv is None:
            n_untimed += 1
        elif abs(tv - t_d3) > TIME_RTOL * t_scale:
            flag = "  ERROR: time mismatch, wrong VTK/state pairing"
            errors += 1
        print(
            f"state {idx:2d} {f.name} t_vtk={tv!s:<12} t_d3={t_d3:<20} nodes={n_pts} cells={n_cells}"
            f"  max|dxyz|={diff.max():.4e}  mean={diff.mean():.3e}{flag}"
        )
    warn_untimed(n_untimed)
    if errors:
        print(f"ERROR: {errors} state(s) could not be paired with their VTK file")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
