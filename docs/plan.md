# vtk2d3plot Implementation Plan

> **For agentic workers (Antigravity):** Work **one milestone per conversation**.
> Read `AGENTS.md` first. For each milestone: produce an Implementation Plan
> artifact, wait for review, implement test-first, run the full test suite,
> tick the milestone checkbox here, and stop. Never start milestone N+1 in the
> same conversation.

[x] **Milestone 6: The Review Fixes (Post-M1..5)**  
Fix C1/C2/M2–M6 from the 2026-08-02 code review (M1/M7 deferred to M7 round).

[x] **Milestone 7: Fix Round M7 (Post-verification)**  
Fix N1–N4 new defects, M1 minimum remedy, M7 phantom masking, M8 perf fix.
N4 partial (3 of 6 allocation sites), CLI --title and N1-eps tests missing — completed in M8.

[x] **Milestone 8: Fix Round M8 (Post-M7-verification)**  
Fix P1 (remaining N4 crash sites), P1b (single-layer-only stress/eps),
P2 (non-ASCII title), P3 (missing tests, truthful records).

[x] **Milestone 9: Fix Round M9 + Roadmap Tier-1**  
Fix R1 (generic eps discarded when Upper exists), R2 (upper-only data slots
into wrong layer), P2 (transliteration-based title). Roadmap Tier-1: EROSION_STATUS
→ is_alive, unmapped-array warnings, part titles from CLI/API, SPH robustness
(VTK_VERTEX skip), atomic writes + stale-family cleanup, CLI batch polish
(glob expansion, default title, --part-names).

[x] **Milestone 10: Fix Round M10 (Post-M9-verification)**  
Fix C1 (erosion latch — never-alive rigid elements forced alive), T5 (atomic
write: regex stale cleanup, failure-safe, glob-escape), T2 (unmapped absmax
max-accumulates across states), title budget f64 76→72, real surrogate/bytes
title tests, stress/eps lower-only value tests, de-vacuous M8 P2 test,
cp1252/BOM part-names encoding tolerance, OFF+EROSION coexistence warning,
CLI title degeneration ('vtk'→parent dir).

**Goal:** A Python package `vtk2d3plot` that converts a time series of legacy
ASCII VTK unstructured-grid files (one file per output state, as written by
pyradioss or by OpenRadioss `anim_to_vtk`) into an LS-DYNA d3plot family that
LS-PrePost opens correctly.

**Architecture:** Adapter pattern proven by Vortex-Radioss: parse VTK states
into NumPy arrays, fill `lasso.dyna.D3plot.arrays` with `ArrayType` keys, let
lasso serialize the binary. We write **no binary code ourselves**. The VTK
reader is hand-rolled (~200 lines, numpy) because it must handle two dialects
that differ from strict VTK (see `docs/vtk_input_spec.md`).

**Tech stack:** Python 3.14 (`.venv`), numpy, lasso-python 2.0.4 (writer),
meshio (test cross-check only), pytest. LS-PrePost 4.13 for ground-truth
verification via `tools/lspp_check.py`.

**Verified foundation (do not re-litigate):** lasso-python 2.0.4 writes
from-scratch d3plots on this machine that LS-PrePost 4.13 loads and fringes
correctly — proven by `tools/reference/make_d3plot.py` and `make_mixed.py`
(run them if in doubt). The full local verification evidence is in
`docs/research/local-verification-report.md`.

## Global constraints

- Interpreter: ALWAYS `.venv\Scripts\python.exe`. Never bare `python`/`pip`.
- Dependencies are preinstalled. No `pip install` without explicit user approval.
- `lasso-python==2.0.4` is the writer. Pin it. Its API quirks are load-bearing:
  - `ArrayType.node_displacement` = **absolute deformed coordinates** per state, NOT deltas.
  - `part_titles` must be a numpy array of **bytes** (`np.array([b"NAME"])`); str raises AttributeError. `part_titles` and `part_titles_ids` must both be present or both absent.
  - Solids: exactly **1 or 8** integration layers. We always use 1.
  - Default `write_d3plot(path)` writes TWO files: `<path>` (geometry) + `<path>01` (states). Both are the deliverable; copy/ship them together.
  - `element_solid_effective_plastic_strain` real shape is `(n_states, n_solids, n_layers)` — the docstring in `array_type.py` is wrong (says layers before solids). Trust `d3plot.py:3431`, not the docstring.
  - 64-bit output needs `header.itype=np.int64`, `header.ftype=np.float64` AND `header.wordsize=8` — all three, or the file is silently corrupt. Default output is 32-bit; cast all float arrays to `np.float32`, ids to `np.int32`.
- Element blocks in d3plot are segregated by type (solids, then beams, then shells). VTK interleaves them. All splitting/mapping lives in `mapping.py` only.
- Golden comparisons join **by node/element ID, never by array index** (Vortex re-sorts nodes; index-wise diffs give garbage — measured max|dxyz|=2.5e2 vs 5e-4 by-ID on `zug_test3_RS`).
- Tests: `.venv\Scripts\python.exe -m pytest -q` must be green at every milestone end. Markers: `milestone1..milestone5`, `lspp` (auto-skip when LS-PrePost missing), `pyradioss` (auto-skip when pyradioss missing), `slow`.
- Commit after every green milestone (`git add -A; git commit`).
- Windows/PowerShell: one flat command per call, no `&&`, no pipes, backslash paths. LS-PrePost exe path contains literal `[4.13]` — always quote it.

## Input dialects (summary — full spec in `docs/vtk_input_spec.md`)

| Property | port dialect (pyradioss) | official dialect (anim_to_vtk) |
|---|---|---|
| Fixture | `tests/data/port_dialect/box_beam_impact/` (21 states) | `tests/data/golden/*/vtk/` |
| Time | title line 2: `pyradioss state t=%.9E` | `FIELD FieldData` → `TIME` |
| Points | `POINTS n double`, deformed coords | `POINTS n float`, deformed coords |
| Node IDs | none → sequential 1..N | `POINT_DATA SCALARS NODE_ID int` |
| Elem/Part IDs | none → sequential / single part 1 | `CELL_DATA SCALARS ELEMENT_ID, PART_ID int` |
| Point vectors | `DISPLACEMENT`, `VELOCITY` | `Displacement`, `Velocity`, `Acceleration`, contact arrays |
| Cell scalars | `VONM`, `EPSP`, `OFF` (1.0 alive) | `1D/2D/3DELEM_*` per family, `EROSION_STATUS` (semantics unverified — do NOT map it in v1) |
| Tensors | none | `TENSORS 2DELEM_Stress_(membrane/upper/lower)`, `3DELEM_Stress`, strains |
| Blank lines between blocks | no | yes |
| Cell order | 3D → 2D → 1D | 1D → 2D → 3D |

Initial geometry recovery (both dialects): `x0 = POINTS(state0) − DISPLACEMENT(state0)`
(verified exact to print precision on both fixtures). If no displacement array
exists, use `POINTS(state0)` and warn.

## File structure

```
src/vtk2d3plot/
  __init__.py       __version__, re-export convert()
  vtk_reader.py     read_vtk(path) -> VtkState (one file)
  series.py         load_series(paths|stem) -> VtkSeries (sorted states, topology check, times)
  mapping.py        split_cells(), connectivity tables, ID policies
  builder.py        build_d3plot(series, options) -> lasso D3plot (arrays filled, not yet written)
  api.py            convert(inputs, output, **options) -> ConversionReport
  cli.py            vtk2d3plot console entry point
tests/
  conftest.py       fixture paths, availability markers
  test_env.py       environment sanity (already present, keep green)
  test_m1_reader.py ... test_m5_end2end.py
tools/
  lspp_check.py     headless LS-PrePost load/fringe checker (already present, working)
  reference/        verified working scripts — READ THESE FIRST (make_d3plot.py, make_mixed.py, verify_byid.py)
```

## Core interfaces (all milestones must conform)

```python
# vtk_reader.py
@dataclass
class VtkState:
    path: str
    title: str                      # raw line 2
    time: float | None              # from FIELD TIME or title regex r"t=([0-9.eE+\-]+)"
    points: np.ndarray              # (n_points, 3) float64
    cell_types: np.ndarray          # (n_cells,) int32, VTK type ids
    cells: list[np.ndarray]         # per cell, 0-based point indices (ragged)
    point_data: dict[str, np.ndarray]   # name -> (n_points,) or (n_points, 3)
    cell_data: dict[str, np.ndarray]    # name -> (n_cells,) or (n_cells, 3, 3) for TENSORS

def read_vtk(path: str | Path) -> VtkState

# series.py
@dataclass
class VtkSeries:
    states: list[VtkState]          # sorted by state number then time
    dialect: str                    # "port" | "official" | "unknown"
    times: np.ndarray               # (n_states,) float64, strictly increasing

def load_series(inputs: str | Path | Sequence[str | Path]) -> VtkSeries
# inputs: a directory, a glob-able stem, or an explicit list of .vtk files.
# Raises TopologyError if CELLS/CELL_TYPES differ between states.

# mapping.py
@dataclass
class MappedModel:
    node_ids: np.ndarray            # (n_nodes,) int32; zero IDs renumbered max+1
    solid_conn: np.ndarray          # (n_solids, 8) int32, 0-based node indexes
    solid_ids: np.ndarray; solid_part_idx: np.ndarray
    shell_conn: np.ndarray          # (n_shells, 4)
    shell_ids: np.ndarray; shell_part_idx: np.ndarray
    beam_conn: np.ndarray           # (n_beams, 5); orientation cols zero
    beam_ids: np.ndarray; beam_part_idx: np.ndarray
    part_ids: np.ndarray            # (n_parts,) int32
    solid_src: np.ndarray           # (n_solids,) int32 — VTK cell index of each solid, for result gather
    shell_src: np.ndarray; beam_src: np.ndarray

def map_topology(state0: VtkState) -> MappedModel

# builder.py
def build_d3plot(series: VtkSeries, model: MappedModel, *,
                 title: str = "", f64: bool = False) -> "lasso.dyna.D3plot"

# api.py
@dataclass
class ConversionReport:
    output: str; n_states: int; n_nodes: int
    n_solids: int; n_shells: int; n_beams: int; n_parts: int
    dialect: str; warnings: list[str]

def convert(inputs, output: str | Path, *, title="", f64=False) -> ConversionReport
```

## VTK → LS-DYNA connectivity mapping (implement exactly; verify visually in M2)

d3plot identifies shape by repeated-node patterns in fixed-width connectivity.

| VTK type | Target | Mapping (VTK local indices → 8 or 4 or 5 slots) | Status |
|---|---|---|---|
| 12 HEXAHEDRON | solid | `[0,1,2,3,4,5,6,7]` | verified (golden W3 solids) |
| 10 TETRA | solid | `[0,1,2,3,3,3,3,3]` | standard LS-DYNA degenerate tetra |
| 13 WEDGE | solid | `[0,1,4,3,2,2,5,5]` | verified (M2 LS-PrePost render, 2026-08-02: clean prism, no twist) |
| 14 PYRAMID | solid | `[0,1,2,3,4,4,4,4]` | standard |
| 9 QUAD | shell | `[0,1,2,3]` | verified |
| 5 TRIANGLE | shell | `[0,1,2,2]` | standard degenerate |
| 3 LINE | beam | `[0,1,0,0,0]` + zeros for orientation | geometry only |

Stress tensor Voigt order for d3plot: `[xx, yy, zz, xy, xz, yz]` (verified M4, 2026-08-02).
From a VTK `TENSORS` 3×3: `[t[0,0], t[1,1], t[2,2], t[0,1], t[0,2], t[1,2]]`.
Note: lasso docstring says `[σxy, σyz, σxz]` but binary layout is `[σxy, σxz, σyz]` — slots 4/5 swapped.
Shell stress: `anim_to_vtk` writes in element-local coords (z=0), Vortex golden uses global coords.
Solid stress matches golden directly (rel. err < 1e-6); shell stress matches via von Mises invariant (tolerance 1e-4; von Mises is rotation-invariant so local-frame stress gives exact VM values; raw σ components differ ~50% due to element-local vs global frame).

## Result mapping

Official dialect:
- `3DELEM_Stress` → `element_solid_stress (n_states, n_solids, 1, 6)`
- `2DELEM_Stress_(lower)` → shell layer 0, `2DELEM_Stress_(upper)` → layer 1 →
  `element_shell_stress (n_states, n_shells, 2, 6)` (Vortex writes exactly 2
  layers; the golden d3plot has shape `(11, 400, 2, 6)` — match it; confirm
  lower/upper order against the golden file in M4, swap if needed and record).
- `2DELEM_Plastic_Strain` → `element_shell_effective_plastic_strain (n_states, n_shells, 2)`
  (replicate the scalar into both layers unless `_Upper/_Lower` variants exist),
  `3DELEM_Plastic_Strain` → `element_solid_effective_plastic_strain (n_states, n_solids, 1)`
- `EROSION_STATUS`: DO NOT map in v1 (semantics unverified — a no-fail run shows
  3154 zeros at t=0; see local-verification-report). All elements alive.

Port dialect:
- `VONM` scalar → synthetic uniaxial tensor `[vonm, 0, 0, 0, 0, 0]` per element.
  Von Mises of uniaxial stress equals the scalar, so LS-PrePost's derived
  von-Mises fringe reproduces the port value exactly. Document this clearly in
  the report warnings (`"stress tensor is synthetic-uniaxial from VONM"`).
- `EPSP` → effective plastic strain, 1 layer (shells `(n_states, n_shells, 1)`).
- `OFF` (1.0 alive / 0.0 deleted) → `element_shell_is_alive` /
  `element_solid_is_alive` / `element_beam_is_alive` `(n_states, n_elems)`.
- `DISPLACEMENT` → geometry recovery (above); per-state absolute coords → `node_displacement`.
- `VELOCITY` → `node_velocity`. No acceleration exists → do not set the array.

Kinematics (both dialects): `timestamps = series.times`;
`node_displacement[i] = states[i].points` (absolute, see lasso quirk);
`node_coordinates = x0`.

---

### Milestone 1 — package skeleton + VTK reader for both dialects ✅ criteria: `pytest -q -m milestone1` green

- [x] **M1.1 Write failing tests** in `tests/test_m1_reader.py` (the red test
  `test_m1_contract` in the file already imports `vtk2d3plot.api` — extend it):

```python
import numpy as np, pytest
from conftest import PORT_DIR, GOLDEN_W3_VTK

pytestmark = pytest.mark.milestone1

def test_read_port_state0():
    from vtk2d3plot.vtk_reader import read_vtk
    s = read_vtk(PORT_DIR / "BOXIMPA000.vtk")
    assert s.points.shape == (220, 3)
    assert s.time == pytest.approx(2.902412789e-04)
    assert len(s.cells) == 200 and set(s.cell_types) == {9}
    assert s.point_data["DISPLACEMENT"].shape == (220, 3)
    assert s.cell_data["OFF"].shape == (200,)

def test_read_official_state():
    from vtk2d3plot.vtk_reader import read_vtk
    s = read_vtk(sorted(GOLDEN_W3_VTK.glob("*.vtk"))[0])
    assert s.time is not None                       # from FIELD TIME
    assert "NODE_ID" in s.point_data
    assert "PART_ID" in s.cell_data and "ELEMENT_ID" in s.cell_data
    assert any(k.startswith("3DELEM_Stress") for k in s.cell_data)
    t = next(v for k, v in s.cell_data.items() if k.startswith("3DELEM_Stress"))
    assert t.ndim == 3 and t.shape[1:] == (3, 3)    # TENSORS as (n,3,3)

def test_series_port():
    from vtk2d3plot.series import load_series
    ser = load_series(PORT_DIR)
    assert ser.dialect == "port" and len(ser.states) == 21
    assert np.all(np.diff(ser.times) > 0)

def test_reader_matches_meshio():
    meshio = pytest.importorskip("meshio")
    from vtk2d3plot.vtk_reader import read_vtk
    p = PORT_DIR / "BOXIMPA000.vtk"
    ours, ref = read_vtk(p), meshio.read(p)
    assert np.allclose(ours.points, ref.points)
```

- [x] **M1.2 Run to verify failure** (`ModuleNotFoundError: vtk2d3plot`).
- [x] **M1.3 Implement** `src/vtk2d3plot/{__init__,vtk_reader,series}.py` +
  minimal `api.py`/`cli.py` stubs raising `NotImplementedError`, and register
  the package: it is already installed editable via `pyproject.toml` at repo
  root (`pip install -e .` was run during setup — do NOT rerun pip; if imports
  fail, check `src/` layout matches pyproject `[tool.setuptools]`).
  Reader notes: parse line-oriented; tolerate a UTF-8 BOM; tolerate missing
  blank lines; `SCALARS <name> <type> [ncomp]` then `LOOKUP_TABLE` line;
  `VECTORS`/`TENSORS` have no lookup table; `FIELD FieldData n` blocks carry
  `TIME 1 1 double`. Store TENSORS as `(n, 3, 3)`.
- [x] **M1.4 Run until green**, including `test_env.py`.
- [x] **M1.5 Commit** `feat(m1): VTK reader + series loader for port and official dialects`.

### Milestone 2 — topology mapping + geometry-only d3plot ✅ criteria: `pytest -q -m milestone2` green incl. one `lspp` test

- [x] **M2.1 Failing tests** `tests/test_m2_geometry.py`:

```python
import numpy as np, pytest
from conftest import PORT_DIR, GOLDEN_W3_VTK, lspp_available

pytestmark = pytest.mark.milestone2

def test_map_port_topology():
    from vtk2d3plot.vtk_reader import read_vtk
    from vtk2d3plot.mapping import map_topology
    m = map_topology(read_vtk(PORT_DIR / "BOXIMPA000.vtk"))
    assert m.shell_conn.shape == (200, 4)
    assert m.node_ids[0] == 1 and m.part_ids.tolist() == [1]

def test_map_official_topology_w3():
    from vtk2d3plot.vtk_reader import read_vtk
    from vtk2d3plot.mapping import map_topology
    m = map_topology(read_vtk(sorted(GOLDEN_W3_VTK.glob("*.vtk"))[0]))
    assert m.solid_conn.shape[0] == 875 and m.shell_conn.shape[0] == 400
    assert m.part_ids.shape[0] == 2
    assert 0 not in m.node_ids                      # zero-ID renumbering

def test_geometry_roundtrip(tmp_path):
    from vtk2d3plot.api import convert
    from lasso.dyna import D3plot, ArrayType
    rep = convert(PORT_DIR, tmp_path / "box.d3plot")
    d = D3plot(str(tmp_path / "box.d3plot"))
    assert d.arrays[ArrayType.node_coordinates].shape == (220, 3)
    x0 = d.arrays[ArrayType.node_coordinates]
    assert np.allclose(x0[0], [0.0, 0.0, 0.5], atol=1e-6)   # recovered undeformed z=0.5

@pytest.mark.lspp
def test_geometry_loads_in_lsprepost(tmp_path):
    from vtk2d3plot.api import convert
    from tools.lspp_check import check_d3plot
    convert(PORT_DIR, tmp_path / "box.d3plot")
    r = check_d3plot(tmp_path / "box.d3plot")
    assert r["loaded"] and r["n_states"] == 21
```

- [x] **M2.2 Verify failure**, **M2.3 implement** `mapping.py` (+ builder/api
  geometry path; states carry only timestamps + node_displacement for now).
  Use `tools/reference/make_d3plot.py` as the exact template for array
  population. **M2.4 green**, and additionally: build one synthetic VTK with a
  tetra + wedge + pyramid, convert, and render in LS-PrePost
  (`tools/lspp_check.py --png out.png`), view the PNG, and confirm clean
  (non-twisted) solids; record the verified wedge mapping in the table above.
- [x] **M2.5 Commit** `feat(m2): topology mapping and geometry d3plot`.

### Milestone 3 — kinematic states ✅ criteria: `pytest -q -m milestone3` green

- [x] **M3.1 Failing tests** `tests/test_m3_states.py`:

```python
import numpy as np, pytest
from conftest import PORT_DIR, GOLDEN_W3_VTK, GOLDEN_W3_D3, GOLDEN_ZUG_VTK, GOLDEN_ZUG_D3

pytestmark = pytest.mark.milestone3

def test_port_states_roundtrip(tmp_path):
    from vtk2d3plot.api import convert
    from lasso.dyna import D3plot, ArrayType
    convert(PORT_DIR, tmp_path / "box.d3plot")
    d = D3plot(str(tmp_path / "box.d3plot"))
    assert len(d.arrays[ArrayType.global_timesteps]) == 21
    assert d.arrays[ArrayType.node_displacement].shape == (21, 220, 3)
    assert d.arrays[ArrayType.node_velocity].shape == (21, 220, 3)

def _byid_max_coord_diff(mine_path, golden_path):
    # join on node ID, compare per-state absolute coordinates
    from lasso.dyna import D3plot, ArrayType
    a, b = D3plot(str(mine_path)), D3plot(str(golden_path))
    ia = np.argsort(a.arrays[ArrayType.node_ids]); ib = np.argsort(b.arrays[ArrayType.node_ids])
    assert np.array_equal(a.arrays[ArrayType.node_ids][ia], b.arrays[ArrayType.node_ids][ib])
    da = a.arrays[ArrayType.node_displacement][:, ia, :]
    db = b.arrays[ArrayType.node_displacement][:, ib, :]
    assert da.shape == db.shape
    return float(np.abs(da - db).max())

def test_w3_matches_vortex_golden(tmp_path):
    from vtk2d3plot.api import convert
    convert(GOLDEN_W3_VTK, tmp_path / "w3.d3plot")
    assert _byid_max_coord_diff(tmp_path / "w3.d3plot",
        GOLDEN_W3_D3 / "W3_SETUP_Ball_impact__ton-mm-s.d3plot") < 1.0e-2   # ASCII precision bound

def test_zug_matches_vortex_golden(tmp_path):
    # the node-renumbering trap: passes ONLY with by-ID join and zero-ID policy max+1
    from vtk2d3plot.api import convert
    convert(GOLDEN_ZUG_VTK, tmp_path / "zug.d3plot")
    assert _byid_max_coord_diff(tmp_path / "zug.d3plot",
        GOLDEN_ZUG_D3 / "zug_test3_RS.d3plot") < 1.0e-2
```

- [x] **M3.2 red → M3.3 implement → M3.4 green → M3.5 commit**
  `feat(m3): multi-state kinematics verified against Vortex golden data`.

### Milestone 4 — element results ✅ criteria: `pytest -q -m milestone4` green incl. `lspp` fringe check

- [x] **M4.1 Failing tests** `tests/test_m4_results.py`:

```python
import numpy as np, pytest
from conftest import PORT_DIR, GOLDEN_W3_VTK, GOLDEN_W3_D3

pytestmark = pytest.mark.milestone4

def test_w3_shell_stress_matches_golden(tmp_path):
    from vtk2d3plot.api import convert
    from lasso.dyna import D3plot, ArrayType
    convert(GOLDEN_W3_VTK, tmp_path / "w3.d3plot")
    mine, gold = D3plot(str(tmp_path / "w3.d3plot")), \
                 D3plot(str(GOLDEN_W3_D3 / "W3_SETUP_Ball_impact__ton-mm-s.d3plot"))
    im = np.argsort(mine.arrays[ArrayType.element_shell_ids])
    ig = np.argsort(gold.arrays[ArrayType.element_shell_ids])
    sm = mine.arrays[ArrayType.element_shell_stress][:, im]
    sg = gold.arrays[ArrayType.element_shell_stress][:, ig]
    assert sm.shape == sg.shape == (11, 400, 2, 6)
    scale = max(1.0, float(np.abs(sg).max()))
    assert np.abs(sm - sg).max() / scale < 1.0e-3

def test_w3_solid_stress_and_eps(tmp_path):
    from vtk2d3plot.api import convert
    from lasso.dyna import D3plot, ArrayType
    convert(GOLDEN_W3_VTK, tmp_path / "w3.d3plot")
    d = D3plot(str(tmp_path / "w3.d3plot"))
    assert d.arrays[ArrayType.element_solid_stress].shape == (11, 875, 1, 6)
    assert d.arrays[ArrayType.element_solid_effective_plastic_strain].shape == (11, 875, 1)

def test_port_vonm_synthetic_tensor(tmp_path):
    from vtk2d3plot.api import convert
    from vtk2d3plot.vtk_reader import read_vtk
    from lasso.dyna import D3plot, ArrayType
    rep = convert(PORT_DIR, tmp_path / "box.d3plot")
    d = D3plot(str(tmp_path / "box.d3plot"))
    s = d.arrays[ArrayType.element_shell_stress]        # (21, 200, 1, 6)
    vtk9 = read_vtk(PORT_DIR / "BOXIMPA009.vtk")
    assert np.allclose(s[9, :, 0, 0], vtk9.cell_data["VONM"], rtol=1e-5)
    assert np.allclose(s[9, :, 0, 1:], 0.0)
    assert any("synthetic" in w for w in rep.warnings)

def test_port_off_deletion(tmp_path):
    from vtk2d3plot.api import convert
    from lasso.dyna import D3plot, ArrayType
    convert(PORT_DIR, tmp_path / "box.d3plot")
    d = D3plot(str(tmp_path / "box.d3plot"))
    alive = d.arrays[ArrayType.element_shell_is_alive]
    assert alive.shape[:2] == (21, 200)

@pytest.mark.lspp
def test_w3_fringe_in_lsprepost(tmp_path):
    from vtk2d3plot.api import convert
    from tools.lspp_check import check_d3plot
    convert(GOLDEN_W3_VTK, tmp_path / "w3.d3plot")
    r = check_d3plot(tmp_path / "w3.d3plot", fringe="vonmises", png=tmp_path / "w3.png")
    assert r["loaded"] and r["n_states"] == 11 and (tmp_path / "w3.png").exists()
```

- [x] **M4.2 red → M4.3 implement result mapping (see Result mapping section;
  check lower/upper layer order against the golden and record the outcome
  here) → M4.4 green → M4.5 commit** `feat(m4): element results with golden verification`.
  Outcome: lower=layer0, upper=layer1 is correct. Shell stress golden comparison
  uses von Mises (coordinate-invariant) because anim_to_vtk outputs local-frame
  stress while Vortex golden uses global-frame. Solid stress matches raw Voigt < 1e-3.

### Milestone 5 — CLI + pyradioss end-to-end + robustness ✅ criteria: full suite green, `pytest -q -m milestone5` incl. `pyradioss` marker

- [x] **M5.1 Failing tests** `tests/test_m5_end2end.py`:

```python
import subprocess, sys, shutil, pytest
from pathlib import Path
from conftest import PORT_DIR, TENSILE_DIR, REPO, VENV_PY

pytestmark = pytest.mark.milestone5

def test_cli_smoke(tmp_path):
    r = subprocess.run([str(VENV_PY), "-m", "vtk2d3plot.cli",
                        str(PORT_DIR), "-o", str(tmp_path / "box.d3plot")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "box.d3plot").exists() and (tmp_path / "box.d3plot01").exists()

def test_tensile_vtk_only_converts(tmp_path):
    # the run vortex-radioss CRASHES on — from VTK we must succeed
    from vtk2d3plot.api import convert
    rep = convert(TENSILE_DIR, tmp_path / "tensile.d3plot")
    assert rep.n_states == 20

@pytest.mark.pyradioss
def test_pyradioss_run_and_convert(tmp_path):
    # full loop: run the Python solver on a bundled example deck, convert its VTK output
    pytest.importorskip("pyradioss")
    for f in (REPO / "tests" / "data" / "port_dialect" / "box_beam_impact").glob("BOXIMP_000*.rad"):
        shutil.copy(f, tmp_path)
    r1 = subprocess.run([str(VENV_PY), "-m", "pyradioss.starter", "-i", "BOXIMP_0000.rad"],
                        cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert r1.returncode == 0, r1.stderr[-2000:]
    r2 = subprocess.run([str(VENV_PY), "-m", "pyradioss.engine", "-i", "BOXIMP_0001.rad"],
                        cwd=tmp_path, capture_output=True, text=True, timeout=1800)
    assert r2.returncode == 0, r2.stderr[-2000:]
    vtks = sorted(tmp_path.glob("BOXIMPA*.vtk"))
    assert vtks, "solver produced no VTK states"
    from vtk2d3plot.api import convert
    rep = convert(tmp_path, tmp_path / "box.d3plot")
    assert rep.n_states == len(vtks)
```

  (If `python -m pyradioss.starter` is not runnable as a module, use the
  console scripts `.venv\Scripts\pyradioss-starter.exe` / `-engine.exe`
  instead — both are installed; prefer whichever works, record which.)
- [x] **M5.2 red → M5.3 implement `cli.py` (argparse: inputs, `-o/--output`,
  `--f64`, `--title`, `-q`) and robustness fixes the tests demand → M5.4 full
  suite green → M5.5** update `README.md` usage section →
  pyradioss invocation: `python -m pyradioss.starter` / `.engine` works as modules.
- [x] **M5.6 Commit** `feat(m5): CLI, pyradioss end-to-end loop, robustness`.

## Post-v1 backlog (do NOT implement now)

- Global variables from `{run}T01.csv` (resample /TFILE dt → state times).
- Full shell stress tensors from pyradioss via upstream `anim_vtk.py` patch
  (port emits only VONM today; the patch sketch lives in `docs/vtk_input_spec.md`).
- `EROSION_STATUS` semantics resolution (needs a with-failure official run).
- SPH (lasso writer bug #101), thick shells, 64-bit default, `.pvd`/XML `.vtu` input.

## Self-review notes (already applied)

- Every fixture path used by tests exists in the repo (copied and size-checked).
- All lasso array names verified against the locally installed 2.0.4 package,
  not docs. Shapes cross-checked against the golden d3plot read-back.
- The wedge mapping is explicitly marked UNVERIFIED with a decision procedure.
- pyradioss module-vs-console invocation ambiguity is called out in M5.

## Design decisions (Q&A, 2026-08-02)

Settled before M1 implementation. Milestones 2–5 inherit these without re-asking.

1. **FIELD FieldData**: parse N entries dynamically (not hard-coded 2).
2. **Missing arrays across states**: reader reports faithfully (no zero-fill).
   Builder (M2/M3) zero-fills states missing an array + one report warning.
3. **Geometry recovery**: scan states in order; `x0 = POINTS(first_state_with_DISPLACEMENT) −
   DISPLACEMENT(that_state)`. Warn only if no state in the series has DISPLACEMENT (fall back
   to POINTS(state0)). This is a builder-stage rule, not in the reader.
4. **SCALARS ncomp**: reader supports ncomp > 1 `(n, ncomp)`; mapper raises if a
   known name has unexpected ncomp. Lenient reader, strict mapper.
5. **Line endings/BOM**: `open(path, "r", encoding="utf-8-sig", errors="replace")` —
   universal newlines, BOM tolerance, no-crash on stray bytes.
6. **Blank lines**: skip globally. VTK is count-driven; blank lines never semantic.
7. **Dialect detection**: title regex `pyradioss` → "port"; FIELD TIME → "official";
   else "unknown". Dialect is provenance metadata only — every mapping decision keys
   on array presence, never the dialect string. Patched port files stay "port".
8. **Topology check**: hard error on POINTS count, CELLS, CELL_TYPES, or identity
   arrays (NODE_ID/ELEMENT_ID/PART_ID) differing. Data-array availability differences
   → warning (not error).
9. **File sorting**: glob `*.vtk`, group by stem, error on >1 stem, sort by numeric
   suffix `A(\d+)`, verify monotonic time. Caller order preserved for explicit lists.
10. **TENSORS**: read 9 tokens per cell (flexible line breaks), store `(n, 3, 3)` row-major.
    Symmetric-tensor sanity check belongs in the mapper (M4), not the reader.
11. **Stubs (M1)**: full signatures + ConversionReport dataclass, bodies raise
    NotImplementedError naming the milestone. No `mapping.py`/`builder.py` in M1.
12. **Test imports**: `from conftest import ...` pattern. No `__init__.py` in `tests/`.
