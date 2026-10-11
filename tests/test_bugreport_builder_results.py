"""Bug report 2026-10-11, cluster B1: builder result gathers, alive latch, fill warnings.

Covers report items #5 (c12, c13), #6 (c14) and #34 (c27).
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from conftest import GOLDEN_W3_VTK, PORT_DIR

pytestmark = pytest.mark.bugreport


# ── helpers ──────────────────────────────────────────────────────────────


def _read(path):
    from lasso.dyna import D3plot

    return D3plot(str(path))


def _assert_all_arrays_equal(path_a, path_b):
    """Every lasso array of two d3plot families must be identical."""
    a, b = _read(path_a), _read(path_b)
    keys_a, keys_b = set(a.arrays), set(b.arrays)
    assert keys_a == keys_b, (sorted(keys_a - keys_b), sorted(keys_b - keys_a))
    for k in sorted(keys_a):
        va, vb = np.asarray(a.arrays[k]), np.asarray(b.arrays[k])
        assert va.shape == vb.shape, k
        assert np.array_equal(va, vb), k
    return a


def _copy_port(
    dst, rename: dict[str, str], off_zero_from_state: int | None = None, off_zero_cells=range(0)
):
    """Copy the port fixture (incl. T01 csv), renaming SCALARS and zeroing OFF cells."""
    dst.mkdir()
    shutil.copy(PORT_DIR / "BOXIMPT01.csv", dst / "BOXIMPT01.csv")
    for src in sorted(PORT_DIR.glob("BOXIMPA*.vtk")):
        state_no = int(src.stem[-3:])
        lines = src.read_text(encoding="utf-8").splitlines()
        out = []
        i = 0
        while i < len(lines):
            line = lines[i]
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "SCALARS":
                name = parts[1]
                out.append(" ".join(["SCALARS", rename.get(name, name), *parts[2:]]))
                out.append(lines[i + 1])  # LOOKUP_TABLE
                vals = lines[i + 2 : i + 2 + 200]
                if (
                    name == "OFF"
                    and off_zero_from_state is not None
                    and state_no >= off_zero_from_state
                ):
                    vals = ["0.0" if j in off_zero_cells else v for j, v in enumerate(vals)]
                out.extend(vals)
                i += 2 + 200
                continue
            out.append(line)
            i += 1
        (dst / src.name).write_text("\n".join(out) + "\n", encoding="utf-8")


def _write_official(
    path,
    *,
    time_val,
    points,
    cells,
    cell_scalars=None,
    cell_tensors=None,
    part_key="PART_ID",
    part_values=None,
):
    """Write a small official-dialect legacy VTK file.

    cells: list of (vtk_type, [0-based point indices]).
    Node IDs 1..n, element IDs 1..m, all cells in part 1 (or part_values under
    the name part_key).
    """
    cell_scalars = cell_scalars or {}
    cell_tensors = cell_tensors or {}
    n_pts, n_cells = len(points), len(cells)
    lines = [
        "# vtk DataFile Version 3.0",
        "synthetic official dialect",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        "FIELD FieldData 1",
        "TIME 1 1 double",
        f"{time_val}",
        f"POINTS {n_pts} float",
    ]
    lines += [f"{p[0]} {p[1]} {p[2]}" for p in points]
    size = sum(len(c) + 1 for _, c in cells)
    lines.append(f"CELLS {n_cells} {size}")
    lines += [" ".join(str(x) for x in [len(c), *c]) for _, c in cells]
    lines.append(f"CELL_TYPES {n_cells}")
    lines += [str(t) for t, _ in cells]
    lines += [f"POINT_DATA {n_pts}", "SCALARS NODE_ID int 1", "LOOKUP_TABLE default"]
    lines += [str(i + 1) for i in range(n_pts)]
    lines.append("VECTORS Displacement float")
    lines += ["0 0 0"] * n_pts
    lines += [f"CELL_DATA {n_cells}", "SCALARS ELEMENT_ID int 1", "LOOKUP_TABLE default"]
    lines += [str(i + 1) for i in range(n_cells)]
    lines += [f"SCALARS {part_key} int 1", "LOOKUP_TABLE default"]
    lines += [str(p) for p in (part_values or [1] * n_cells)]
    for name, vals in cell_scalars.items():
        dtype = "int" if name.upper() == "EROSION_STATUS" else "float"
        lines += [f"SCALARS {name} {dtype} 1", "LOOKUP_TABLE default"]
        lines += [str(v) for v in vals]
    for name, ts in cell_tensors.items():
        lines.append(f"TENSORS {name} float")
        for t in ts:
            lines += [f"{t[r][0]} {t[r][1]} {t[r][2]}" for r in range(3)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_QUAD_PTS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]


def _quads(n):
    pts, cells = [], []
    for e in range(n):
        base = len(pts)
        pts += [(x + 2.0 * e, y, z) for x, y, z in _QUAD_PTS]
        cells.append((9, [base, base + 1, base + 2, base + 3]))
    return pts, cells


def _hex_and_quad():
    hex_pts = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ]  # fmt: skip
    quad_pts = [(3, 0, 0), (4, 0, 0), (4, 1, 0), (3, 1, 0)]
    cells = [(12, list(range(8))), (9, [8, 9, 10, 11])]
    return hex_pts + quad_pts, cells


# ── #5 (c12, c13): case-insensitive presence, exact-case gather ──────────


def test_c13_port_lowercase_results_identical(tmp_path):
    """Bug report 2026-10-11 #5 (c12, c13): vonm/epsp/Off must map like VONM/EPSP/OFF."""
    from lasso.dyna import ArrayType

    from vtk2d3plot.api import convert

    dead = range(5, 10)
    up, low = tmp_path / "up", tmp_path / "low"
    _copy_port(up, {}, off_zero_from_state=10, off_zero_cells=dead)
    _copy_port(
        low,
        {"VONM": "vonm", "EPSP": "epsp", "OFF": "Off"},
        off_zero_from_state=10,
        off_zero_cells=dead,
    )
    rep_up = convert(up, tmp_path / "up.d3plot")
    rep_low = convert(low, tmp_path / "low.d3plot")

    d = _assert_all_arrays_equal(tmp_path / "up.d3plot", tmp_path / "low.d3plot")

    # The real values arrived (not zero-filled): pinned against the source file.
    from vtk2d3plot.vtk_reader import read_vtk

    vtk9 = read_vtk(PORT_DIR / "BOXIMPA009.vtk")
    ids = d.arrays[ArrayType.element_shell_ids]
    order = np.argsort(ids)
    assert np.array_equal(ids[order], np.arange(1, 201))
    s = d.arrays[ArrayType.element_shell_stress][9][order, 0, 0]
    assert np.allclose(s, vtk9.cell_data["VONM"], rtol=1e-6)
    eps = d.arrays[ArrayType.element_shell_effective_plastic_strain][9][order, 0]
    assert np.allclose(eps, vtk9.cell_data["EPSP"], rtol=1e-6)
    assert float(np.abs(eps).max()) > 0.0

    # OFF zeros reach is_alive: cells 5..9 (element IDs 6..10) dead from state 10.
    alive = d.arrays[ArrayType.element_shell_is_alive][:, order]
    assert np.all(alive[:10] == 1.0)
    assert np.all(alive[10:, 5:10] == 0.0)
    assert np.all(alive[:, :5] == 1.0) and np.all(alive[:, 10:] == 1.0)

    # No zero-fill / missing-source warnings for the lower-case run.
    assert not [w for w in rep_low.warnings if "zero-filled" in w or "one-filled" in w]
    non_unmapped_up = [w for w in rep_up.warnings if not w.startswith("unmapped arrays")]
    non_unmapped_low = [w for w in rep_low.warnings if not w.startswith("unmapped arrays")]
    assert non_unmapped_low == non_unmapped_up


_W3_RENAME = {
    "EROSION_STATUS": "erosion_status",
    "2DELEM_Plastic_Strain": "2delem_plastic_strain",
    "2DELEM_Specific_Energy": "2delem_specific_energy",
    "2DELEM_Plastic_Strain_Upper": "2delem_plastic_strain_upper",
    "2DELEM_Plastic_Strain_Lower": "2delem_plastic_strain_lower",
    "2DELEM_Stress_(upper)": "2delem_stress_(UPPER)",
    "2DELEM_Stress_(lower)": "2delem_stress_(lower)",
    "3DELEM_Plastic_Strain": "3delem_plastic_strain",
    "3DELEM_Specific_Energy": "3delem_specific_energy",
    "3DELEM_Stress": "3delem_stress",
}


def _copy_w3_renamed(dst):
    dst.mkdir()
    for src in sorted(GOLDEN_W3_VTK.glob("*.vtk")):
        out = []
        for line in src.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] in ("SCALARS", "TENSORS") and parts[1] in _W3_RENAME:
                line = " ".join([parts[0], _W3_RENAME[parts[1]], *parts[2:]])
            out.append(line)
        (dst / src.name).write_text("\n".join(out) + "\n", encoding="utf-8")


def test_c13_official_lowercase_results_identical(tmp_path):
    """Bug report 2026-10-11 #5 (c13): lower-cased official arrays must map like the originals."""
    from lasso.dyna import ArrayType

    from vtk2d3plot.api import convert

    low = tmp_path / "low"
    _copy_w3_renamed(low)
    rep_up = convert(GOLDEN_W3_VTK, tmp_path / "up.d3plot")
    rep_low = convert(low, tmp_path / "low.d3plot")

    d = _assert_all_arrays_equal(tmp_path / "up.d3plot", tmp_path / "low.d3plot")

    # Real data arrived: solid + shell stress are not all zero.
    assert float(np.abs(d.arrays[ArrayType.element_solid_stress]).max()) > 1.0
    assert float(np.abs(d.arrays[ArrayType.element_shell_stress][:, :, 1]).max()) > 1.0
    assert not [w for w in rep_low.warnings if "zero-filled" in w or "one-filled" in w]
    non_unmapped_up = [w for w in rep_up.warnings if not w.startswith("unmapped arrays")]
    non_unmapped_low = [w for w in rep_low.warnings if not w.startswith("unmapped arrays")]
    assert non_unmapped_low == non_unmapped_up


@pytest.mark.xfail(
    strict=True,
    reason="needs the cell-array check of the 'unmapped arrays ignored' block in "
    "build_d3plot (owned by another cluster) to ignore case; see cross-cluster request",
)
def test_c13_unmapped_warning_omits_mapped_lowercase(tmp_path):
    """Bug report 2026-10-11 #5 (c13): mapped lower-case arrays are not 'unmapped arrays ignored'."""
    from vtk2d3plot.api import convert

    low = tmp_path / "low"
    _copy_port(low, {"VONM": "vonm", "EPSP": "epsp", "OFF": "Off"})
    rep = convert(low, tmp_path / "low.d3plot")
    unmapped = [w for w in rep.warnings if w.startswith("unmapped arrays")]
    assert unmapped == []


def test_c13_case_collision_exact_wins(tmp_path):
    """Bug report 2026-10-11 #5 (c13): 'VONM' and 'vonm' in one state: exact case wins, warned."""
    from lasso.dyna import ArrayType

    from vtk2d3plot.api import convert

    pts, cells = _quads(2)
    for k in range(2):
        _write_official(
            tmp_path / f"runA{k:03d}.vtk",
            time_val=0.1 * (k + 1),
            points=pts,
            cells=cells,
            cell_scalars={"VONM": [5.0 + k, 7.0 + k], "vonm": [900.0, 901.0]},
        )
    rep = convert(sorted(tmp_path.glob("runA*.vtk")), tmp_path / "out.d3plot")
    d = _read(tmp_path / "out.d3plot")
    s = d.arrays[ArrayType.element_shell_stress]
    order = np.argsort(d.arrays[ArrayType.element_shell_ids])
    assert np.array_equal(s[:, order, 0, 0], [[5.0, 7.0], [6.0, 8.0]])
    hits = [w for w in rep.warnings if "differ only in case" in w]
    assert hits == [
        (
            "cell arrays 'VONM', 'vonm' differ only in case at states [0, 1]; "
            "using 'VONM' (exact name), the others are ignored"
        )
    ]


def test_c13_case_collision_exact_wins_not_sorted_first(tmp_path):
    """Bug report 2026-10-11 #5 (c13): exact name wins even when it is not first in sort order.

    '3DELEM_STRESS' sorts before '3DELEM_Stress' ('T' < 't'), so this test tells
    "exact case wins" apart from "first in sorted order".
    """
    from lasso.dyna import ArrayType

    from vtk2d3plot.api import convert

    hex_pts = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ]  # fmt: skip
    exact = [[1.0, 4.0, 5.0], [4.0, 2.0, 6.0], [5.0, 6.0, 3.0]]
    other = [[-100.0, -400.0, -500.0], [-400.0, -200.0, -600.0], [-500.0, -600.0, -300.0]]
    for k in range(2):
        _write_official(
            tmp_path / f"runA{k:03d}.vtk",
            time_val=0.1 * (k + 1),
            points=hex_pts,
            cells=[(12, list(range(8)))],
            cell_tensors={"3DELEM_STRESS": [other], "3DELEM_Stress": [exact]},
        )
    rep = convert(sorted(tmp_path.glob("runA*.vtk")), tmp_path / "out.d3plot")
    d = _read(tmp_path / "out.d3plot")
    s = d.arrays[ArrayType.element_solid_stress]
    # Voigt [xx, yy, zz, xy, xz, yz] of the exact-name tensor, both states.
    assert np.array_equal(s[:, 0, 0, :], [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 2)
    hits = [w for w in rep.warnings if "differ only in case" in w]
    assert hits == [
        (
            "cell arrays '3DELEM_STRESS', '3DELEM_Stress' differ only in case at states "
            "[0, 1]; using '3DELEM_Stress' (exact name), the others are ignored"
        )
    ]


# ── #6 (c14): EROSION_STATUS latch kills late-activated elements ─────────


def test_c14_erosion_latch_late_activation(tmp_path):
    """Bug report 2026-10-11 #6 (c14): latch 1->0 only after the element was seen alive."""
    from lasso.dyna import ArrayType

    from vtk2d3plot.api import convert

    # per element over 3 states:
    #   e1 [0,1,1] late activation -> stays [0,1,1]
    #   e2 [1,1,0] erodes          -> [1,1,0]
    #   e3 [1,0,1] resurrection    -> latched [1,0,0]
    #   e4 [0,0,0] never alive     -> forced alive [1,1,1] (rigid-part rule)
    #   e5 [0,1,0] activated, then eroded -> [0,1,0]
    es = [[0, 1, 1, 0, 0], [1, 1, 0, 0, 1], [1, 0, 1, 0, 0]]
    pts, cells = _quads(5)
    for k in range(3):
        _write_official(
            tmp_path / f"runA{k:03d}.vtk",
            time_val=0.1 * (k + 1),
            points=pts,
            cells=cells,
            cell_scalars={"EROSION_STATUS": es[k]},
        )
    rep = convert(sorted(tmp_path.glob("runA*.vtk")), tmp_path / "out.d3plot")
    d = _read(tmp_path / "out.d3plot")
    order = np.argsort(d.arrays[ArrayType.element_shell_ids])
    alive = d.arrays[ArrayType.element_shell_is_alive][:, order]
    expected = np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [1, 0, 0, 1, 0],
        ],
        dtype=float,
    )
    assert np.array_equal(alive, expected)
    assert any("1 shell elements never alive" in w for w in rep.warnings)
    assert (
        "EROSION_STATUS: 1 shell elements return to 1 after being eroded "
        "(states [2]); kept eroded (no resurrection)"
    ) in rep.warnings


def test_c14_erosion_latch_ignores_filled_states(tmp_path):
    """Bug report 2026-10-11 #6 (c14): a one-filled state is not an observation of 'alive'.

    State 0 has no EROSION_STATUS (one-filled, warned). Element 1 reads 0 then 1:
    it was never seen alive before state 1, so state 2 must stay 1 (late activation),
    not be latched to 0 because of the filled 1.0 at state 0.
    """
    from lasso.dyna import ArrayType

    from vtk2d3plot.api import convert

    es = [None, [0, 1], [1, 1]]
    pts, cells = _quads(2)
    for k in range(3):
        _write_official(
            tmp_path / f"runA{k:03d}.vtk",
            time_val=0.1 * (k + 1),
            points=pts,
            cells=cells,
            cell_scalars={} if es[k] is None else {"EROSION_STATUS": es[k]},
        )
    rep = convert(sorted(tmp_path.glob("runA*.vtk")), tmp_path / "out.d3plot")
    d = _read(tmp_path / "out.d3plot")
    order = np.argsort(d.arrays[ArrayType.element_shell_ids])
    alive = d.arrays[ArrayType.element_shell_is_alive][:, order]
    assert np.array_equal(alive, [[1.0, 1.0], [0.0, 1.0], [1.0, 1.0]])
    assert (
        "element_shell_is_alive: one-filled (all-alive) at states [0] "
        "(source EROSION_STATUS array missing in those states)"
    ) in rep.warnings
    assert not [w for w in rep.warnings if "after being eroded" in w]


@pytest.mark.parametrize("part_key", ["PART_ID", "part_id"])
def test_c14_never_alive_warning_names_written_parts(tmp_path, part_key):
    """Bug report 2026-10-11 #6 (c14): the never-alive warning names the parts in the d3plot.

    The part IDs in the warning must be the ones the model wrote, whatever
    rule the topology mapping uses for the part-ID array name.
    """
    from lasso.dyna import ArrayType

    from vtk2d3plot.api import convert

    pts, cells = _quads(2)
    for k in range(2):
        _write_official(
            tmp_path / f"runA{k:03d}.vtk",
            time_val=0.1 * (k + 1),
            points=pts,
            cells=cells,
            cell_scalars={"EROSION_STATUS": [0, 1]},
            part_key=part_key,
            part_values=[7, 8],
        )
    rep = convert(sorted(tmp_path.glob("runA*.vtk")), tmp_path / "out.d3plot")
    d = _read(tmp_path / "out.d3plot")
    ids = d.arrays[ArrayType.element_shell_ids]
    pidx = d.arrays[ArrayType.element_shell_part_indexes]
    part_ids = d.arrays[ArrayType.part_titles_ids]
    written = int(part_ids[pidx[int(np.flatnonzero(ids == 1)[0])]])
    hits = [w for w in rep.warnings if "never alive" in w]
    assert hits == [
        (
            f"EROSION_STATUS: 1 shell elements never alive in any state (parts [{written}]); "
            "forced alive (rigid/non-integrated)"
        )
    ]
    if part_key == "PART_ID":
        assert written == 7


# ── #34 (c27): internal-energy fill warnings dropped ─────────────────────


def test_c27_internal_energy_fill_warnings(tmp_path):
    """Bug report 2026-10-11 #34 (c27): 2D/3DELEM_Specific_Energy missing in state 1 is warned."""
    from lasso.dyna import ArrayType

    from vtk2d3plot.api import convert

    pts, cells = _hex_and_quad()
    for k in range(2):
        # 3DELEM_Plastic_Strain in every state keeps this test clear of report #8
        # (history variables shift by one word without a solid eps array).
        scal = {"3DELEM_Plastic_Strain": [0.5 + k, 0.0]}
        if k == 0:
            scal["3DELEM_Specific_Energy"] = [42.0, 0.0]
            scal["2DELEM_Specific_Energy"] = [0.0, 17.0]
        _write_official(
            tmp_path / f"runA{k:03d}.vtk",
            time_val=0.1 * (k + 1),
            points=pts,
            cells=cells,
            cell_scalars=scal,
        )
    rep = convert(sorted(tmp_path.glob("runA*.vtk")), tmp_path / "out.d3plot")
    d = _read(tmp_path / "out.d3plot")
    assert np.array_equal(d.arrays[ArrayType.element_shell_internal_energy][:, 0], [17.0, 0.0])
    assert np.array_equal(
        d.arrays[ArrayType.element_solid_history_variables][:, 0, 0, 0], [42.0, 0.0]
    )
    assert (
        "element_solid_history_variables: zero-filled at states [1] "
        "(source array missing in those states)"
    ) in rep.warnings
    assert (
        "element_shell_internal_energy: zero-filled at states [1] "
        "(source array missing in those states)"
    ) in rep.warnings
