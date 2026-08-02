import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from lasso.dyna.d3plot import D3plot

d3, vtk_dir = sys.argv[1], sys.argv[2]
p = D3plot(d3)
d_ids = np.asarray(p.arrays["node_ids"]).astype(np.int64)
nd = p.arrays["node_displacement"]
t = np.asarray(p.arrays["timesteps"])


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
            vals = []; j = i + 2   # skip LOOKUP_TABLE line
            while len(vals) < n_pts:
                vals.extend(lines[j].split()); j += 1
            node_ids = np.array(vals, dtype=np.int64); i = j; continue
        i += 1
    return n_pts, n_cells, pts, node_ids, tv


vtks = sorted(f for f in os.listdir(vtk_dir) if f.endswith(".vtk"))
print("d3plot node_ids[:8] :", d_ids[:8].tolist(), " sorted:", bool(np.all(np.diff(d_ids) > 0)))
for idx, f in enumerate(vtks):
    n_pts, n_cells, pts, vids, tv = parse_vtk(os.path.join(vtk_dir, f))
    if idx == 0:
        print("vtk    NODE_ID[:8] :", vids[:8].tolist(), " sorted:", bool(np.all(np.diff(vids) > 0)))
        print("same id set        :", bool(np.array_equal(np.sort(vids), np.sort(d_ids))))
        print("same index order   :", bool(np.array_equal(vids, d_ids)))
    order = np.argsort(d_ids)
    lut = {int(v): k for k, v in enumerate(d_ids)}
    perm = np.array([lut[int(v)] for v in vids])
    diff = np.abs(nd[idx][perm] - pts)
    print("state %2d t_vtk=%-12s t_d3=%-20s nodes=%d cells=%d  max|dxyz|=%.4e  mean=%.3e"
          % (idx, tv, float(t[idx]), n_pts, n_cells, diff.max(), diff.mean()))
