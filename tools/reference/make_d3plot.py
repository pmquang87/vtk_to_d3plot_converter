"""Build a d3plot from scratch with lasso-python, then read it back and verify.

Usage: make_d3plot.py [output_folder]   (default: a fresh temporary folder)
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import tempfile

import numpy as np
from lasso.dyna import ArrayType, D3plot

OUT = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="make_d3plot_")
os.makedirs(OUT, exist_ok=True)
PATH = os.path.join(OUT, "cube.d3plot")

# ---------------------------------------------------------------- geometry
# unit cube, LS-DYNA hex node ordering: bottom face 1-4 ccw, then top face 5-8
coords = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
], dtype=np.float32)

n_nodes = coords.shape[0]
n_states = 3

# ---------------------------------------------------------------- states
timesteps = np.array([0.0, 0.5, 1.0], dtype=np.float32)

# node_displacement in lasso == deformed (absolute) nodal coordinates per state
disp = np.zeros((n_states, n_nodes, 3), dtype=np.float32)
for i, t in enumerate(timesteps):
    d = coords.copy()
    d[4:, 2] += 0.5 * t          # top face rises
    d[:, 0] += 0.1 * t           # whole cube translates in x
    disp[i] = d

# element results: 1 solid, 1 integration layer
eps = np.array([[[0.0]], [[0.01]], [[0.025]]], dtype=np.float32)      # (3,1,1)
stress = np.zeros((n_states, 1, 1, 6), dtype=np.float32)              # xx yy zz xy yz xz
stress[:, 0, 0, 0] = [0.0, 100.0, 200.0]
stress[:, 0, 0, 1] = [0.0,  20.0,  40.0]
stress[:, 0, 0, 2] = [0.0, -10.0, -20.0]

# ---------------------------------------------------------------- assemble
d3 = D3plot()
a = d3.arrays
a[ArrayType.node_coordinates] = coords
a[ArrayType.node_ids] = np.arange(1, n_nodes + 1, dtype=np.int32)

a[ArrayType.element_solid_node_indexes] = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
a[ArrayType.element_solid_part_indexes] = np.array([0], dtype=np.int32)
a[ArrayType.element_solid_ids] = np.array([1], dtype=np.int32)

a[ArrayType.part_ids] = np.array([77], dtype=np.int32)
a[ArrayType.part_titles_ids] = np.array([77], dtype=np.int32)
a[ArrayType.part_titles] = np.array([b"CUBE PART"])

a[ArrayType.global_timesteps] = timesteps
a[ArrayType.node_displacement] = disp
a[ArrayType.element_solid_effective_plastic_strain] = eps
a[ArrayType.element_solid_stress] = stress

print("writing ->", PATH)
d3.write_d3plot(PATH)
print("write OK")

for f in sorted(os.listdir(OUT)):
    print("   file:", f, os.path.getsize(os.path.join(OUT, f)), "bytes")

# ---------------------------------------------------------------- read back
print("\n--- reading back ---")
r = D3plot(PATH)
ra = r.arrays
print("array keys:", sorted(ra.keys()))
print("n_timesteps (header):", r.header.n_timesteps if hasattr(r.header, "n_timesteps") else "n/a")
print("timesteps:", ra[ArrayType.global_timesteps])
print("node_coordinates:\n", ra[ArrayType.node_coordinates])
print("node_ids:", ra.get(ArrayType.node_ids))
print("element_solid_node_indexes:", ra[ArrayType.element_solid_node_indexes])
print("element_solid_part_indexes:", ra[ArrayType.element_solid_part_indexes])
print("element_solid_ids:", ra.get(ArrayType.element_solid_ids))
print("part_ids:", ra.get(ArrayType.part_ids))
print("part_titles:", ra.get(ArrayType.part_titles))
print("node_displacement shape:", ra[ArrayType.node_displacement].shape)
print("node_displacement last state:\n", ra[ArrayType.node_displacement][-1])
print("eps shape:", ra[ArrayType.element_solid_effective_plastic_strain].shape)
print("eps:", ra[ArrayType.element_solid_effective_plastic_strain].ravel())
print("stress shape:", ra[ArrayType.element_solid_stress].shape)
print("stress:", ra[ArrayType.element_solid_stress].reshape(n_states, 6))

# ---------------------------------------------------------------- assertions
ok = True
def chk(name, cond):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name)
    ok = ok and bool(cond)

chk("n_states == 3", ra[ArrayType.node_displacement].shape[0] == 3)
chk("timesteps match", np.allclose(ra[ArrayType.global_timesteps], timesteps))
chk("coordinates match", np.allclose(ra[ArrayType.node_coordinates], coords))
chk("displacements match", np.allclose(ra[ArrayType.node_displacement], disp))
chk("connectivity matches", np.array_equal(
    ra[ArrayType.element_solid_node_indexes], np.array([[0, 1, 2, 3, 4, 5, 6, 7]])))
chk("part index matches", np.array_equal(ra[ArrayType.element_solid_part_indexes], np.array([0])))
chk("eps matches", np.allclose(
    ra[ArrayType.element_solid_effective_plastic_strain].reshape(eps.shape), eps))
chk("stress matches", np.allclose(
    ra[ArrayType.element_solid_stress].reshape(stress.shape), stress))
chk("part id 77 present", 77 in np.asarray(ra.get(ArrayType.part_ids, [])).tolist()
    or 77 in np.asarray(ra.get(ArrayType.part_titles_ids, [])).tolist())

print("\nOVERALL:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
