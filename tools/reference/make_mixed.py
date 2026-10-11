"""Harder case: solids + shells, 2 parts, non-contiguous node/element/part ids.

Usage: make_mixed.py [output_folder]   (default: a fresh temporary folder)
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os
import tempfile

import numpy as np
from lasso.dyna import ArrayType, D3plot

OUT = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="make_mixed_")
os.makedirs(OUT, exist_ok=True)
PATH = os.path.join(OUT, "mixed.d3plot")

# 8 hex nodes + 4 extra nodes for a shell above the cube
coords = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    [0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2],
], dtype=np.float32)
n_nodes = 12
n_states = 4
timesteps = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

disp = np.zeros((n_states, n_nodes, 3), dtype=np.float32)
for i, t in enumerate(timesteps):
    d = coords.copy()
    d[:, 2] += 0.1 * t * d[:, 2]
    disp[i] = d

d3 = D3plot()
a = d3.arrays
a[ArrayType.node_coordinates] = coords
a[ArrayType.node_ids] = np.array([101, 102, 103, 104, 105, 106, 107, 108,
                                  201, 202, 203, 204], dtype=np.int32)

a[ArrayType.element_solid_node_indexes] = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
a[ArrayType.element_solid_part_indexes] = np.array([0], dtype=np.int32)
a[ArrayType.element_solid_ids] = np.array([5001], dtype=np.int32)

a[ArrayType.element_shell_node_indexes] = np.array([[8, 9, 10, 11]], dtype=np.int32)
a[ArrayType.element_shell_part_indexes] = np.array([1], dtype=np.int32)
a[ArrayType.element_shell_ids] = np.array([9001], dtype=np.int32)

a[ArrayType.part_ids] = np.array([77, 88], dtype=np.int32)
a[ArrayType.part_titles_ids] = np.array([77, 88], dtype=np.int32)
a[ArrayType.part_titles] = np.array([b"HEX PART", b"SHELL PART"])

a[ArrayType.global_timesteps] = timesteps
a[ArrayType.node_displacement] = disp
a[ArrayType.node_velocity] = np.zeros((n_states, n_nodes, 3), dtype=np.float32)

# solid results (1 layer)
a[ArrayType.element_solid_stress] = np.zeros((n_states, 1, 1, 6), dtype=np.float32)
a[ArrayType.element_solid_stress][:, 0, 0, 0] = [0, 100, 200, 300]
a[ArrayType.element_solid_effective_plastic_strain] = np.array(
    [[[0.0]], [[0.01]], [[0.02]], [[0.03]]], dtype=np.float32)

# shell results: 3 integration layers
a[ArrayType.element_shell_stress] = np.zeros((n_states, 1, 3, 6), dtype=np.float32)
a[ArrayType.element_shell_stress][:, 0, :, 0] = np.array([[0, 0, 0], [50, 50, 50],
                                                          [80, 80, 80], [110, 110, 110]])
a[ArrayType.element_shell_effective_plastic_strain] = np.zeros((n_states, 1, 3), dtype=np.float32)

print("writing ->", PATH)
d3.write_d3plot(PATH)
print("write OK")
for f in sorted(os.listdir(OUT)):
    print("   file:", f, os.path.getsize(os.path.join(OUT, f)), "bytes")

print("\n--- read back ---")
r = D3plot(PATH)
ra = r.arrays
print("timesteps:", ra[ArrayType.global_timesteps])
print("node_ids:", ra[ArrayType.node_ids])
print("part_ids:", ra[ArrayType.part_ids])
print("part_titles:", [t.strip() for t in ra[ArrayType.part_titles]])
print("solid ids:", ra[ArrayType.element_solid_ids],
      " shell ids:", ra[ArrayType.element_shell_ids])
print("solid part idx:", ra[ArrayType.element_solid_part_indexes],
      " shell part idx:", ra[ArrayType.element_shell_part_indexes])
print("disp shape:", ra[ArrayType.node_displacement].shape)
print("disp state4 z:", ra[ArrayType.node_displacement][-1][:, 2])
print("solid stress xx:", ra[ArrayType.element_solid_stress][:, 0, 0, 0])
print("shell stress shape:", ra[ArrayType.element_shell_stress].shape)
print("shell stress xx:", ra[ArrayType.element_shell_stress][:, 0, :, 0])

ok = (ra[ArrayType.node_displacement].shape == (4, 12, 3)
      and np.allclose(ra[ArrayType.node_displacement], disp)
      and np.array_equal(ra[ArrayType.node_ids], a[ArrayType.node_ids])
      and np.array_equal(ra[ArrayType.part_ids], np.array([77, 88]))
      and np.array_equal(ra[ArrayType.element_solid_ids], np.array([5001]))
      and np.array_equal(ra[ArrayType.element_shell_ids], np.array([9001]))
      and np.allclose(ra[ArrayType.element_shell_stress], a[ArrayType.element_shell_stress]))
print("\nOVERALL:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
