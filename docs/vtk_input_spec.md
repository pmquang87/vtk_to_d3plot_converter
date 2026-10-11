# VTK input specification — the two dialects

Source-verified 2026-08-02 against `pyradioss/output/anim_vtk.py` (local repo
`C:\Users\pmqua\PycharmProjects\OpenRadioss_Python`, the "port dialect") and
`OpenRadioss/Tools` `output_converters/anim_to_vtk/src/anim_to_vtk.cpp`
(the "official dialect"). Fixture files for both live in `tests/data/`.

## Port dialect (pyradioss, `anim_vtk.py`, 144 lines)

- Legacy ASCII, `DATASET UNSTRUCTURED_GRID`, **no blank lines between blocks**.
- File naming: `{run}A{nnn:03d}.vtk`, counter from 000 (`engine.py:686,769`).
- **Time is ONLY in the title (line 2)**: `pyradioss state t=%.9E`
  (`anim_vtk.py:103`). No FIELD data, no cycle number. Parse with
  `r"t=([0-9.eE+\-]+)"`.
- State cadence quirk: state 000 is NOT at t=0 (output check runs after
  `state.t += dt`), and a final state is always written at t_end. Times are
  strictly increasing but unevenly spaced at both ends.
- `POINTS n double`, `%.9E` — **current deformed** coordinates (`model.x`).
  Point row i ↔ `model.node_ids[i]`.
- Cell blocks in `model.element_groups()` order: bricks(12), tetras(10),
  shells/shells_qbat/shells_qeph(9), sh3n(5), trusses/springs/beams(3).
  Only non-empty groups appear; the three shell flavours are
  indistinguishable in the file. Beams write only their 2 geometric nodes.
  **No degenerate-cell collapse** (a collapsed hex stays type 12).
  Topology is constant across states (deleted elements stay in the mesh).
- `POINT_DATA`: `VECTORS DISPLACEMENT double` (= x − x0), `VECTORS VELOCITY
  double`. Always both present in practice.
- `CELL_DATA`: `SCALARS VONM double 1`, `SCALARS EPSP double 1`,
  `SCALARS OFF double 1` (+ `LOOKUP_TABLE default` each).
  `OFF`: 1.0 alive, 0.0 deleted — unambiguous.
  `VONM` is a reduced scalar (shells: max over layers; solids: from Voigt
  tensor; beams |N|/A; springs 0). `EPSP` likewise max-over-layers.
- **Not present** (gaps vs d3plot): node/element/part IDs, part titles, stress
  tensors, per-layer values, shell thickness, accelerations, global energies,
  material info.

### Where the missing data lives (for later fidelity upgrades)

- `{run}_{nnnn}.rst` — pickled Model: `node_ids`, per-group element `ids`,
  `state['part_ids']`, `parts_list` (id, prop_id, mat_id, title), properties
  (`thick`, `nip`), full state arrays (`sig`, `epsp`, `off`, `eint`, ...).
  Version-fragile pickle — requires matching pyradioss on sys.path.
- `{run}T01.csv` — global TIME, IE, KE, HE, CE, EN, DE, EW, ERR%, MASS,
  MOM[XYZ]; sampled at /TFILE dt (NOT /ANIM dt) → needs resampling.
- Shell stress detail: `sig` is (n, nip, 3) in-plane LOCAL-frame components;
  transverse shear separate in `qshear` (n, 2); szz = 0 (plane stress). A
  full-tensor export needs the local→global rotation (recomputed from
  geometry — this is exactly what Vortex-Radioss v1.02 had to add for ANIM).
- Local frame used by `--rotate-shell-stress`: OpenRadioss CLSKEW3 (quads) /
  c3coor3 (triangles), i.e. the QEPH/QBAT element frame. Shells from a BT4-type
  corotational formulation (pyfemsolver `shell_bt4.py`: e3 = unit(v13 x v24),
  e1 = projected N1→N2) use a different frame and would be rotated wrongly; VTK
  carries no formulation info. anim_to_vtk-collapsed triangles (ascending node
  order) lose the original N1→N2 edge, so their rotated components may be
  wrong. Shells with zero-area geometry cannot be rotated; the builder counts
  them in a warning (bug report 2026-10-11 #1).

### Recommended upstream patch (~15 lines, closes the 4 biggest gaps)

In `anim_vtk.py`: emit `POINT_DATA SCALARS NODE_ID int`, `CELL_DATA SCALARS
ELEMENT_ID int` + `SCALARS PART_ID int` (from `group.state['part_ids']`), and
a `FIELD FieldData 1` block with `TIME 1 1 double`. The converter accepts
these automatically (they make a port file look like an official file).

## Official dialect (`anim_to_vtk.cpp`, 66.7 KB)

- Legacy ASCII, blank line after each block. Output to **stdout** — redirect
  with `cmd /c "anim_to_vtk_win64.exe runA001 > runA001.vtk"` (PowerShell `>`
  adds a UTF-8 BOM; tolerate BOM in the reader anyway).
- `FIELD FieldData 2`: `TIME 1 1 double` (6 sig digits, from float) and
  `CYCLE 1 1 int` (hardcoded 0).
- `POINTS n float`, deformed coordinates.
- Cell order 1D → 2D → 3D → SPH(VTK_VERTEX). **Degenerate collapse**: 4-node
  facet with 3 unique nodes → TRIANGLE(5); 8-node solid with 4 unique →
  TETRA(10) — with connectivity emitted in ASCENDING node order (original
  orientation lost!).
- IDs: `POINT_DATA SCALARS NODE_ID int 1`, `CELL_DATA SCALARS ELEMENT_ID int
  1`, `SCALARS PART_ID int 1`.
- `SCALARS EROSION_STATUS int 1` — **semantics RESOLVED**: 1=alive, 0=eroded/deleted.
  Same polarity as pyradioss `OFF`. Written for ALL element families (solids,
  shells, beams, trusses, springs). Elements in rigid/unintegrated parts may
  emit 0 from state 0 (handled by ever_alive logic in builder.py; the warning
  names the part IDs written to the d3plot). An element is dead only after it
  has been observed alive in a state that holds real EROSION_STATUS data:
  leading zeros of an element activated after state 0 are kept and do not
  latch, and one-filled (missing) states do not count as observations; a 0→1
  return after death is latched to 0 and warned once per element at its first
  revival. Known edge: a missing state before activation is one-filled and
  shows as alive (e.g. [0, missing, 0, 1] → [0, 1, 0, 1]); the one-fill warning
  reports it (bug report 2026-10-11 #6).

- Result arrays family-prefixed, names from the ANIM text fields (do not
  hardcode; underscores replace blanks): `1DELEM_*`, `2DELEM_{Plastic_Strain,
  Specific_Energy, Von_Mises, ...}`, `3DELEM_*`, plus
  `TENSORS 2DELEM_Stress_(membrane|upper|lower) float` (3×3, zero out-of-plane
  row/col) and `TENSORS 3DELEM_Stress float` (symmetric 3×3 from Voigt), same
  for `Strain` when STRFLG is on. Zero-filled for non-applicable families.
- Nodal arrays seen in real files: `Velocity`, `Displacement`, `Acceleration`,
  `Contact_Forces`, `Tied_Contact_Forces`, `Time_Step`, `Mass_Change`,
  `Contact_Pressure_/_Normal`, `Contact_Pressure_/_Tangent`.
- Zero NODE_IDs occur (Radioss-internal nodes): `zug_test3_RS` has one.
  Policy (match Vortex): renumber to max_id + 1.
- Legacy FIELD blocks: a dataset-level FIELD (before POINT_DATA / CELL_DATA)
  is used only for TIME; FIELD arrays inside POINT_DATA / CELL_DATA (meshio
  vtk42 writes all data this way) are point / cell arrays and are read like
  SCALARS, a 9-component one as an (n, 3, 3) tensor. Accepted legacy type
  names (case-insensitive): bit, char, signed_char, unsigned_char, short,
  unsigned_short, int (int32); unsigned_int, long, unsigned_long, vtktypeint64,
  vtktypeuint64, vtkIdType (int64); float, double (float64); any other name
  raises. Title time is only a fallback: `\bt=<float>` (word boundary,
  well-formed float, no unit glued on: "time t=0.5s" gives no time); FIELD
  TIME takes precedence. An empty title line is allowed; a file that ends
  after the title raises "Truncated VTK file" (bug report 2026-10-11 #17, #18,
  #29, #30). VTK 8+ METADATA / INFORMATION blocks are not supported (open).
- Large real-world fixture for reader stress tests (not in repo):
  `E:\openradioss_run\getriebekette\badamid_konditioniert_MAT024_no-fail_STRFLG=11\vtk\`
  (66 MB, 46686 nodes, 205488 cells, types {3,5,9,10}).

## Initial geometry recovery (both dialects)

`x0 = POINTS(state0) − DISPLACEMENT(state0)` — verified exact to print
precision on both fixtures (e.g. box_beam node 1: (0,0,0.4971) − (0,0,−0.0029)
= (0,0,0.5) in state 0 AND state 20).
