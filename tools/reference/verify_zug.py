import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from lasso.dyna.d3plot import D3plot


def parse_vtk(path):
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()
    n_pts = None; pts = None; node_ids = None; tv = None; n_cells = None
    i = 0
    while i < len(lines):
        L = lines[i]
        if L.startswith("TIME"):
            tv = float(lines[i + 1].split()[0])
        elif L.startswith("POINTS"):
            n_pts = int(L.split()[1]); vals = []; j = i + 1
            while len(vals) < n_pts * 3:
                vals.extend(lines[j].split()); j += 1
            pts = np.array(vals, dtype=float).reshape(n_pts, 3); i = j; continue
        elif L.startswith("CELLS"):
            n_cells = int(L.split()[1])
        elif L.startswith("SCALARS NODE_ID"):
            vals = []; j = i + 2
            while len(vals) < n_pts:
                vals.extend(lines[j].split()); j += 1
            node_ids = np.array(vals, dtype=np.int64); i = j; continue
        i += 1
    return n_pts, n_cells, pts, node_ids, tv


d3, vtk_dir = sys.argv[1], sys.argv[2]
p = D3plot(d3)
d_ids = np.asarray(p.arrays["node_ids"]).astype(np.int64)
nd = p.arrays["node_displacement"]
t = np.asarray(p.arrays["timesteps"])

vtks = sorted(f for f in os.listdir(vtk_dir) if f.endswith(".vtk"))
_, _, _, vids0, _ = parse_vtk(os.path.join(vtk_dir, vtks[0]))
nzero = int((vids0 == 0).sum())
print("VTK nodes with NODE_ID == 0 :", nzero, "of", vids0.size)
print("d3plot n_nodes              :", d_ids.size)
print("d3plot max node id          :", int(d_ids.max()))
print("VTK    max node id          :", int(vids0.max()))
nz = vids0 != 0
print("non-zero VTK ids all present in d3plot ids:",
      bool(np.isin(vids0[nz], d_ids).all()))
lut = {int(v): k for k, v in enumerate(d_ids)}
sel = np.flatnonzero(nz)
perm = np.array([lut[int(v)] for v in vids0[nz]])
for idx, f in enumerate(vtks):
    _, ncell, pts, vids, tv = parse_vtk(os.path.join(vtk_dir, f))
    diff = np.abs(nd[idx][perm] - pts[sel])
    print("state %2d t_vtk=%-12s t_d3=%-20s matched=%d cells=%d  max|dxyz|=%.4e  mean=%.3e"
          % (idx, tv, float(t[idx]), sel.size, ncell, diff.max(), diff.mean()))
