"""Zero-NODE_ID diagnostic: compare a d3plot with its VTK series on non-zero IDs only.

Usage:
    .venv\\Scripts\\python.exe tools\\reference\\verify_zug.py <d3plot> <vtk_dir>

VTK files are paired with d3plot states in the converter's order (numeric
``A(\\d+)`` suffix, see ``verify_byid.ordered_vtk_files``); the file count and
every state time are checked (FIELD TIME, else ``t=`` in the title), and a
pairing mismatch prints ERROR and exits with code 2. The NODE_ID array must
exist and be the same in every state (else ERROR, exit 2), and every non-zero
VTK node ID must exist in the d3plot (else ERROR, exit 2); states without a
time are reported with a WARNING.
"""

import sys
from pathlib import Path

import numpy as np
from lasso.dyna.d3plot import D3plot

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_byid import TIME_RTOL, ordered_vtk_files, parse_vtk, warn_untimed


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    d3, vtk_dir = argv[0], argv[1]
    p = D3plot(d3)
    d_ids = np.asarray(p.arrays["node_ids"]).astype(np.int64)
    nd = p.arrays["node_displacement"]
    t = np.asarray(p.arrays["timesteps"], dtype=np.float64)

    vtks = ordered_vtk_files(vtk_dir)
    if len(vtks) != len(t):
        print(f"ERROR: {len(vtks)} VTK files but {len(t)} d3plot states - cannot pair them")
        return 2
    _, _, _, vids0, _ = parse_vtk(vtks[0])
    if vids0 is None:
        print(f"ERROR: {vtks[0].name} has no NODE_ID array - use verify_byid.py instead")
        return 2
    nzero = int((vids0 == 0).sum())
    print("VTK nodes with NODE_ID == 0 :", nzero, "of", vids0.size)
    print("d3plot n_nodes              :", d_ids.size)
    print("d3plot max node id          :", int(d_ids.max()))
    print("VTK    max node id          :", int(vids0.max()))
    nz = vids0 != 0
    print("non-zero VTK ids all present in d3plot ids:", bool(np.isin(vids0[nz], d_ids).all()))
    lut = {int(v): k for k, v in enumerate(d_ids)}
    missing = [int(v) for v in vids0[nz] if int(v) not in lut]
    if missing:
        print(f"ERROR: {len(missing)} non-zero VTK node IDs not in the d3plot, e.g. {missing[:5]}")
        return 2
    sel = np.flatnonzero(nz)
    perm = np.array([lut[int(v)] for v in vids0[nz]])
    t_scale = max(float(np.abs(t).max()), np.finfo(float).tiny)
    errors = 0
    n_untimed = 0
    for idx, f in enumerate(vtks):
        _, ncell, pts, vids, tv = parse_vtk(f)
        if vids is None or not np.array_equal(vids, vids0):
            print(f"ERROR: NODE_ID array of {f.name} differs from {vtks[0].name}")
            errors += 1
            continue
        diff = np.abs(nd[idx][perm] - pts[sel])
        t_d3 = float(t[idx])
        flag = ""
        if tv is None:
            n_untimed += 1
        elif abs(tv - t_d3) > TIME_RTOL * t_scale:
            flag = "  ERROR: time mismatch, wrong VTK/state pairing"
            errors += 1
        print(
            f"state {idx:2d} {f.name} t_vtk={tv!s:<12} t_d3={t_d3:<20} matched={sel.size} "
            f"cells={ncell}  max|dxyz|={diff.max():.4e}  mean={diff.mean():.3e}{flag}"
        )
    warn_untimed(n_untimed)
    if errors:
        print(f"ERROR: {errors} state(s) could not be paired with their VTK file")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
