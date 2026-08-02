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
- `SCALARS EROSION_STATUS int 1` — **semantics unverified**: a no-fail run
  shows 3154 zeros at t=0 (all in the 2D block), so 0 ≠ deleted. Do not map
  until resolved against a with-failure run.
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
- Large real-world fixture for reader stress tests (not in repo):
  `E:\openradioss_run\getriebekette\badamid_konditioniert_MAT024_no-fail_STRFLG=11\vtk\`
  (66 MB, 46686 nodes, 205488 cells, types {3,5,9,10}).

## Initial geometry recovery (both dialects)

`x0 = POINTS(state0) − DISPLACEMENT(state0)` — verified exact to print
precision on both fixtures (e.g. box_beam node 1: (0,0,0.4971) − (0,0,−0.0029)
= (0,0,0.5) in state 0 AND state 20).
