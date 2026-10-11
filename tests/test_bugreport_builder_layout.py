"""Bug report 2026-10-11, cluster B2: builder layout flags, globals, part titles.

Covers report numbers #3 (c26), #7 (c15), #8 (c23), #9 (c01), #26 (c09),
#27 (c02+c25, builder guard) and #35 (c28).  Every test goes through the
public interface (``vtk2d3plot.convert`` or the CLI) and pins read-back
VALUES; where a lasso read-back could hide a word shift, the raw state words
are checked as well.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import PORT_DIR

pytestmark = pytest.mark.bugreport


# ── synthetic official-dialect VTK writer ─────────────────────────────

# hex (0-7), quad (8-11), triangle (12-14), line (15-16)
_POINTS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
        [2, 0, 0],
        [3, 0, 0],
        [3, 1, 0],
        [2, 1, 0],
        [4, 0, 0],
        [5, 0, 0],
        [4, 1, 0],
        [6, 0, 0],
        [7, 0, 0],
    ],
    dtype=float,
)
_HEX = (12, [0, 1, 2, 3, 4, 5, 6, 7])
_QUAD = (9, [8, 9, 10, 11])
_TRI = (5, [12, 13, 14])
_LINE = (3, [15, 16])


def _write_vtk(
    path: Path,
    *,
    time: float,
    cells: list[tuple[int, list[int]]],
    part_ids: list[int],
    cell_scalars: dict[str, list[float]] | None = None,
    cell_tensors: dict[str, np.ndarray] | None = None,
    point_vectors: dict[str, np.ndarray] | None = None,
) -> None:
    """Write one official-dialect legacy VTK state (FIELD TIME, NODE_ID, ...)."""
    used = sorted({n for _, conn in cells for n in conn})
    remap = {old: new for new, old in enumerate(used)}
    pts = _POINTS[used]
    n_pts, n_cells = len(pts), len(cells)
    lines = [
        "# vtk DataFile Version 3.0",
        "bug report B2 synthetic state",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        "FIELD FieldData 1",
        "TIME 1 1 double",
        f"{time!r}",
        f"POINTS {n_pts} float",
    ]
    lines += [" ".join(repr(float(v)) for v in row) for row in pts]
    size = sum(len(c) + 1 for _, c in cells)
    lines.append(f"CELLS {n_cells} {size}")
    lines += [" ".join(str(v) for v in [len(c)] + [remap[n] for n in c]) for _, c in cells]
    lines.append(f"CELL_TYPES {n_cells}")
    lines += [str(t) for t, _ in cells]

    lines.append(f"POINT_DATA {n_pts}")
    lines += ["SCALARS NODE_ID int 1", "LOOKUP_TABLE default"]
    lines += [str(i + 1) for i in range(n_pts)]
    vectors = {"Displacement": np.zeros((n_pts, 3))}
    vectors.update(point_vectors or {})
    for name, arr in vectors.items():
        lines.append(f"VECTORS {name} float")
        lines += [" ".join(repr(float(v)) for v in row) for row in np.asarray(arr, dtype=float)]

    lines.append(f"CELL_DATA {n_cells}")
    lines += ["SCALARS ELEMENT_ID int 1", "LOOKUP_TABLE default"]
    lines += [str(100 + i) for i in range(n_cells)]
    lines += ["SCALARS PART_ID int 1", "LOOKUP_TABLE default"]
    lines += [str(p) for p in part_ids]
    for name, vals in (cell_scalars or {}).items():
        lines += [f"SCALARS {name} float 1", "LOOKUP_TABLE default"]
        lines += [repr(float(v)) for v in vals]
    for name, arr in (cell_tensors or {}).items():
        lines.append(f"TENSORS {name} float")
        for t in np.asarray(arr, dtype=float):
            lines += [" ".join(repr(float(v)) for v in row) for row in t]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sym(xx, yy, zz, xy, yz, zx) -> np.ndarray:
    """3x3 tensor whose builder Voigt extraction gives d3plot [xx yy zz xy yz zx].

    The builder reads slot 4 from t[0,2] and slot 5 from t[1,2]
    (anim_to_vtk convention, see builder._tensor_to_voigt).
    """
    return np.array([[xx, xy, yz], [xy, yy, zx], [yz, zx, zz]], dtype=float)


def _family_words(out: Path, ftype=np.float64) -> np.ndarray:
    """All words of the written d3plot family (geometry + state files)."""
    files = sorted(p for p in out.parent.iterdir() if p.name.startswith(out.name))
    assert len(files) >= 2, files
    return np.concatenate([np.frombuffer(p.read_bytes(), dtype=ftype) for p in files])


def _count_subsequence(words: np.ndarray, seq) -> int:
    seq = np.asarray(seq, dtype=words.dtype)
    win = np.lib.stride_tricks.sliding_window_view(words, len(seq))
    return int(np.all(np.isclose(win, seq, rtol=1e-12, atol=0.0), axis=1).sum())


def _read(out: Path):
    from lasso.dyna import D3plot

    return D3plot(str(out))


# ── #3 (c26): total energy ───────────────────────────────────────────


def _t01_columns(path: Path) -> dict[str, np.ndarray]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[1]]
    data = np.array([[float(v) for v in r] for r in rows[2:] if r], dtype=float)
    return {h: data[:, i] for i, h in enumerate(header)}


def test_c26_total_energy_is_energy_sum_not_en(tmp_path):
    """Bug report 2026-10-11 #3 (c26): global total energy must be the energy sum.

    LS-DYNA convention TE = KE + IE + HE + sliding (CE) + damping (DE);
    the pyradioss EN column is numerical dissipation and must not be used.
    """
    from lasso.dyna import ArrayType

    from vtk2d3plot import convert

    out = tmp_path / "box.d3plot"
    rep = convert(PORT_DIR, out)
    d = _read(out)
    times = d.arrays[ArrayType.global_timesteps]
    te = d.arrays[ArrayType.global_total_energy]
    cols = _t01_columns(PORT_DIR / "BOXIMPT01.csv")
    expected = np.interp(
        times,
        cols["TIME"],
        cols["IE"] + cols["KE"] + cols["HE"] + cols["CE"] + cols["DE"],
    )
    assert te.shape == (21,)
    # at the first state the box only carries its kinetic energy 0.156
    assert te[0] == pytest.approx(0.156, rel=1e-9)
    assert te[0] == pytest.approx(cols["KE"][0] + cols["IE"][0], rel=1e-9)
    assert np.all(te > 0.15)
    np.testing.assert_allclose(te, expected, rtol=1e-9, atol=1e-15)
    # KE / IE are still the plain columns
    np.testing.assert_allclose(
        d.arrays[ArrayType.global_kinetic_energy],
        np.interp(times, cols["TIME"], cols["KE"]),
        rtol=1e-9,
    )
    assert any("global_total_energy" in w and "EN" in w for w in rep.warnings), rep.warnings


# ── #27 (c02+c25): header-only T01 csv ────────────────────────────────


def test_c02_header_only_t01_converts_without_globals(tmp_path):
    """Bug report 2026-10-11 #27 (c02, c25): a header-only T01 CSV must not abort."""
    from lasso.dyna import ArrayType

    from vtk2d3plot import convert

    src = tmp_path / "box"
    shutil.copytree(PORT_DIR, src)
    t01 = src / "BOXIMPT01.csv"
    head = t01.read_text(encoding="utf-8").splitlines()[:2]
    t01.write_text("\n".join(head) + "\n", encoding="utf-8")

    out = tmp_path / "out" / "box.d3plot"
    rep = convert(src, out)
    assert out.exists() and Path(str(out) + "01").exists()
    assert rep.n_states == 21
    d = _read(out)
    assert d.arrays[ArrayType.node_displacement].shape == (21, 220, 3)
    # no T01 globals were written: the read-back energies are the default zeros
    for at in (
        ArrayType.global_internal_energy,
        ArrayType.global_kinetic_energy,
        ArrayType.global_total_energy,
    ):
        if at in d.arrays:
            assert np.all(d.arrays[at] == 0.0), at
    assert any("T01" in w and "no data rows" in w for w in rep.warnings), rep.warnings


# ── #7 (c15): solid internal energy with shell-only strain ───────────


def _c15_series(tmp_path: Path) -> Path:
    src = tmp_path / "c15"
    src.mkdir()
    cells = [_HEX, _QUAD]
    for i in range(3):
        lo = _sym(0.1 + i, 0.2, 0.0, 0.3, 0.0, 0.0)
        hi = _sym(0.4 + i, 0.5, 0.0, 0.6, 0.0, 0.0)
        zero = np.zeros((3, 3))
        _write_vtk(
            src / f"runA{i:03d}.vtk",
            time=0.1 * i,
            cells=cells,
            part_ids=[1, 2],
            cell_scalars={"3DELEM_Specific_Energy": [42.0, 0.0]},
            cell_tensors={
                "2DELEM_Strain_(lower)": np.array([zero, lo]),
                "2DELEM_Strain_(upper)": np.array([zero, hi]),
            },
        )
    return src


def test_c15_solid_internal_energy_kept_with_shell_strain(tmp_path):
    """Bug report 2026-10-11 #7 (c15): 3DELEM_Specific_Energy survives shell-only strain."""
    from lasso.dyna import ArrayType

    from vtk2d3plot import convert

    out = tmp_path / "c15.d3plot"
    rep = convert(_c15_series(tmp_path), out)
    d = _read(out)
    hv = d.arrays[ArrayType.element_solid_history_variables]
    assert hv.shape == (3, 1, 1, 1)
    np.testing.assert_array_equal(hv[:, 0, 0, 0], [42.0, 42.0, 42.0])
    np.testing.assert_array_equal(d.arrays[ArrayType.element_solid_strain], 0.0)
    ss = d.arrays[ArrayType.element_shell_strain]
    for i in range(3):
        np.testing.assert_allclose(ss[i, 0, 0], [0.1 + i, 0.2, 0, 0.3, 0, 0], rtol=1e-12)
        np.testing.assert_allclose(ss[i, 0, 1], [0.4 + i, 0.5, 0, 0.6, 0, 0], rtol=1e-12)
    # raw words: each state's solid record is [stress 6, eps, IE, strain 6]
    words = _family_words(out)
    assert _count_subsequence(words, [0.0] * 7 + [42.0] + [0.0] * 6) == 3
    assert any(
        w.startswith("element_solid_strain: zero-filled at all states") for w in rep.warnings
    ), rep.warnings
    assert any(
        w.startswith("element_solid_effective_plastic_strain: zero-filled at all states")
        for w in rep.warnings
    ), rep.warnings


@pytest.mark.lspp
def test_c15_lsprepost_loads(tmp_path):
    """Bug report 2026-10-11 #7 (c15): LS-PrePost opens the repaired layout."""
    from tools.lspp_check import check_d3plot
    from vtk2d3plot import convert

    out = tmp_path / "c15.d3plot"
    convert(_c15_series(tmp_path), out)
    r = check_d3plot(out)
    assert r["loaded"] and r["n_states"] == 3, r


# ── #8 (c23): solid history var / strain shift without plastic strain ─

_C23_STRESS = [100.0, 200.0, 300.0, 10.0, 20.0, 30.0]
_C23_STRAIN = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]


def _c23_series(tmp_path: Path) -> Path:
    src = tmp_path / "c23"
    src.mkdir()
    for i, ie in enumerate([7.0, 8.0, 9.0]):
        _write_vtk(
            src / f"runA{i:03d}.vtk",
            time=0.1 * i,
            cells=[_HEX],
            part_ids=[1],
            cell_scalars={"3DELEM_Specific_Energy": [ie]},
            cell_tensors={
                "3DELEM_Stress": np.array([_sym(*_C23_STRESS)]),
                "3DELEM_Strain": np.array([_sym(*_C23_STRAIN)]),
            },
        )
    return src


@pytest.mark.parametrize("f64", [True, False])
def test_c23_solid_histvar_and_strain_not_shifted(tmp_path, f64):
    """Bug report 2026-10-11 #8 (c23): no word shift without 3DELEM_Plastic_Strain."""
    from lasso.dyna import ArrayType

    from vtk2d3plot import convert

    out = tmp_path / "c23.d3plot"
    rep = convert(_c23_series(tmp_path), out, f64=f64)
    d = _read(out)
    assert d.header.raw_header["ioshl2"] in (999, 1000)
    hv = d.arrays[ArrayType.element_solid_history_variables]
    np.testing.assert_allclose(hv[:, 0, 0, 0], [7.0, 8.0, 9.0], rtol=1e-6)
    strain = d.arrays[ArrayType.element_solid_strain]
    for i in range(3):
        np.testing.assert_allclose(strain[i, 0, 0], _C23_STRAIN, rtol=1e-6)
        np.testing.assert_allclose(
            d.arrays[ArrayType.element_solid_stress][i, 0, 0], _C23_STRESS, rtol=1e-6
        )
    np.testing.assert_array_equal(d.arrays[ArrayType.element_solid_effective_plastic_strain], 0.0)
    # raw words: [stress 6, eps 0, IE, strain 6] per state, IE = 7, 8, 9
    ftype = np.float64 if f64 else np.float32
    words = _family_words(out, ftype)
    for ie in (7.0, 8.0, 9.0):
        seq = np.array(_C23_STRESS + [0.0, ie] + _C23_STRAIN, dtype=ftype)
        assert _count_subsequence(words, seq) == 1, ie
    assert any(
        w.startswith("element_solid_effective_plastic_strain: zero-filled at all states")
        for w in rep.warnings
    ), rep.warnings


def _c23_energy_only_series(tmp_path: Path) -> Path:
    src = tmp_path / "c23e"
    src.mkdir()
    for i, ie in enumerate([7.0, 8.0, 9.0]):
        _write_vtk(
            src / f"runA{i:03d}.vtk",
            time=0.1 * i,
            cells=[_HEX],
            part_ids=[1],
            cell_scalars={"3DELEM_Specific_Energy": [ie]},
        )
    return src


@pytest.mark.parametrize("f64", [True, False])
def test_c23_solid_internal_energy_only(tmp_path, f64):
    """Bug report 2026-10-11 #8 (c23): solids with internal energy only (no stress,
    no strain, no plastic strain) -> the history variable reads back exactly."""
    from lasso.dyna import ArrayType

    from vtk2d3plot import convert

    out = tmp_path / "c23e.d3plot"
    rep = convert(_c23_energy_only_series(tmp_path), out, f64=f64)
    d = _read(out)
    hv = d.arrays[ArrayType.element_solid_history_variables]
    assert hv.shape == (3, 1, 1, 1)
    np.testing.assert_array_equal(hv[:, 0, 0, 0], [7.0, 8.0, 9.0])
    np.testing.assert_array_equal(d.arrays[ArrayType.element_solid_stress], 0.0)
    np.testing.assert_array_equal(d.arrays[ArrayType.element_solid_effective_plastic_strain], 0.0)
    assert ArrayType.element_solid_strain not in d.arrays
    # raw words: each state's solid record is [stress 6 = 0, eps = 0, IE]
    ftype = np.float64 if f64 else np.float32
    words = _family_words(out, ftype)
    for ie in (7.0, 8.0, 9.0):
        assert _count_subsequence(words, [0.0] * 7 + [ie]) == 1, ie
    assert any(
        w.startswith("element_solid_effective_plastic_strain: zero-filled at all states")
        for w in rep.warnings
    ), rep.warnings


_C23_SHELL_LO = [11.0, 12.0, 0.0, 13.0, 0.0, 0.0]
_C23_SHELL_HI = [21.0, 22.0, 0.0, 23.0, 0.0, 0.0]


def _c23_shell_stress_series(tmp_path: Path) -> Path:
    src = tmp_path / "c23s"
    src.mkdir()
    zero = np.zeros((3, 3))
    for i, ie in enumerate([7.0, 8.0, 9.0]):
        k = i + 1.0
        solid = _sym(*[v * k for v in _C23_STRESS])
        lo = _sym(*[v * k for v in _C23_SHELL_LO])
        hi = _sym(*[v * k for v in _C23_SHELL_HI])
        _write_vtk(
            src / f"runA{i:03d}.vtk",
            time=0.1 * i,
            cells=[_QUAD, _HEX],
            part_ids=[2, 1],
            cell_scalars={"3DELEM_Specific_Energy": [0.0, ie]},
            cell_tensors={
                "3DELEM_Stress": np.array([zero, solid]),
                "2DELEM_Stress_(lower)": np.array([lo, zero]),
                "2DELEM_Stress_(upper)": np.array([hi, zero]),
            },
        )
    return src


@pytest.mark.parametrize("f64", [True, False])
def test_c23_shell_stress_without_plastic_strain(tmp_path, f64):
    """Bug report 2026-10-11 #8 (c23): shells with stress but no plastic strain anywhere,
    plus solids with stress and internal energy -> solid history variable, solid
    stress and shell stress (both layers) read back exactly."""
    from lasso.dyna import ArrayType

    from vtk2d3plot import convert

    out = tmp_path / "c23s.d3plot"
    convert(_c23_shell_stress_series(tmp_path), out, f64=f64)
    d = _read(out)
    hv = d.arrays[ArrayType.element_solid_history_variables]
    assert hv.shape == (3, 1, 1, 1)
    np.testing.assert_array_equal(hv[:, 0, 0, 0], [7.0, 8.0, 9.0])
    sst = d.arrays[ArrayType.element_shell_stress]
    assert sst.shape == (3, 1, 2, 6)
    for i in range(3):
        k = i + 1.0
        np.testing.assert_array_equal(
            d.arrays[ArrayType.element_solid_stress][i, 0, 0], [v * k for v in _C23_STRESS]
        )
        np.testing.assert_array_equal(sst[i, 0, 0], [v * k for v in _C23_SHELL_LO])
        np.testing.assert_array_equal(sst[i, 0, 1], [v * k for v in _C23_SHELL_HI])
    np.testing.assert_array_equal(d.arrays[ArrayType.element_solid_effective_plastic_strain], 0.0)
    if ArrayType.element_shell_effective_plastic_strain in d.arrays:
        np.testing.assert_array_equal(
            d.arrays[ArrayType.element_shell_effective_plastic_strain], 0.0
        )
    # raw words: the solid record [stress 6, eps 0, IE] of every state
    ftype = np.float64 if f64 else np.float32
    words = _family_words(out, ftype)
    for i, ie in enumerate((7.0, 8.0, 9.0)):
        seq = [v * (i + 1.0) for v in _C23_STRESS] + [0.0, ie]
        assert _count_subsequence(words, np.array(seq, dtype=ftype)) == 1, ie


@pytest.mark.lspp
def test_c23_lsprepost_loads(tmp_path):
    """Bug report 2026-10-11 #8 (c23): LS-PrePost opens the repaired layout."""
    from tools.lspp_check import check_d3plot
    from vtk2d3plot import convert

    out = tmp_path / "c23.d3plot"
    convert(_c23_series(tmp_path), out)
    r = check_d3plot(out)
    assert r["loaded"] and r["n_states"] == 3, r


# ── #26 (c09): solid strain + shells without results ─────────────────


def _c09_series(tmp_path: Path) -> Path:
    src = tmp_path / "c09"
    src.mkdir()
    zero = np.zeros((3, 3))
    for i in range(2):
        stress = _sym(*[v * (i + 1) for v in _C23_STRESS])
        strain = _sym(*[v * (i + 1) for v in _C23_STRAIN])
        _write_vtk(
            src / f"runA{i:03d}.vtk",
            time=0.5 * i,
            cells=[_LINE, _QUAD, _TRI, _HEX],
            part_ids=[4, 2, 3, 1],
            cell_tensors={
                "3DELEM_Stress": np.array([zero, zero, zero, stress]),
                "3DELEM_Strain": np.array([zero, zero, zero, strain]),
            },
        )
    return src


def test_c09_solid_strain_with_resultless_shells_writes(tmp_path):
    """Bug report 2026-10-11 #26 (c09): solid strain + result-less shells must write."""
    from lasso.dyna import ArrayType

    from vtk2d3plot import convert

    out = tmp_path / "c09.d3plot"
    rep = convert(_c09_series(tmp_path), out)
    assert out.exists() and Path(str(out) + "01").exists()
    d = _read(out)
    strain = d.arrays[ArrayType.element_solid_strain]
    stress = d.arrays[ArrayType.element_solid_stress]
    for i in range(2):
        np.testing.assert_allclose(strain[i, 0, 0], [v * (i + 1) for v in _C23_STRAIN], rtol=1e-12)
        np.testing.assert_allclose(stress[i, 0, 0], [v * (i + 1) for v in _C23_STRESS], rtol=1e-12)
    shell_strain = d.arrays[ArrayType.element_shell_strain]
    assert shell_strain.shape == (2, 2, 2, 6)
    np.testing.assert_array_equal(shell_strain, 0.0)
    assert any(
        w.startswith("element_shell_strain: zero-filled at all states") for w in rep.warnings
    ), rep.warnings


@pytest.mark.lspp
def test_c09_lsprepost_loads(tmp_path):
    """Bug report 2026-10-11 #26 (c09): LS-PrePost opens the written family."""
    from tools.lspp_check import check_d3plot
    from vtk2d3plot import convert

    out = tmp_path / "c09.d3plot"
    convert(_c09_series(tmp_path), out)
    r = check_d3plot(out)
    assert r["loaded"] and r["n_states"] == 2, r


# ── #9 (c01): non-ASCII part names ───────────────────────────────────


def _two_part_series(tmp_path: Path) -> Path:
    src = tmp_path / "parts"
    src.mkdir()
    for i in range(2):
        _write_vtk(
            src / f"runA{i:03d}.vtk",
            time=0.1 * i,
            cells=[_HEX, _QUAD],
            part_ids=[1, 2],
        )
    return src


def _check_titles(out: Path, first: bytes, second: bytes) -> None:
    from lasso.dyna import ArrayType

    d = _read(out)
    np.testing.assert_array_equal(d.arrays[ArrayType.part_titles_ids], [1, 2])
    titles = d.arrays[ArrayType.part_titles]
    assert titles[0].rstrip() == first, titles
    assert titles[1].startswith(b"Alu") and titles[1].rstrip() == second, titles


@pytest.mark.parametrize("f64", [True, False])
def test_c01_non_ascii_part_names_api(tmp_path, f64):
    """Bug report 2026-10-11 #9 (c01): non-ASCII part names keep the part table intact."""
    from vtk2d3plot import convert

    out = tmp_path / "p.d3plot"
    rep = convert(_two_part_series(tmp_path), out, f64=f64, part_names={1: "Größe", 2: "Alu"})
    _check_titles(out, b"Groesse", b"Alu")
    assert any(w.startswith("part 1 title: transliterated") for w in rep.warnings), rep.warnings
    assert any(w.code.value == "TITLE_SANITIZED" for w in rep.structured_warnings)


def test_c01_non_ascii_part_names_cjk(tmp_path):
    """Bug report 2026-10-11 #9 (c01): a CJK name becomes '?' and is reported."""
    from vtk2d3plot import convert

    out = tmp_path / "p.d3plot"
    rep = convert(_two_part_series(tmp_path), out, part_names={1: "钢板", 2: "Alu"})
    _check_titles(out, b"??", b"Alu")
    assert any(w.startswith("part 1 title: replaced") for w in rep.warnings), rep.warnings


def test_c01_ascii_part_names_control(tmp_path):
    """Bug report 2026-10-11 #9 (c01): ASCII control, unchanged and no sanitising warning."""
    from vtk2d3plot import convert

    out = tmp_path / "p.d3plot"
    rep = convert(_two_part_series(tmp_path), out, part_names={1: "Steel", 2: "Alu"})
    _check_titles(out, b"Steel", b"Alu")
    assert not any(w.startswith("part ") for w in rep.warnings), rep.warnings


@pytest.mark.parametrize("extra", [[], ["--f32"]])
def test_c01_non_ascii_part_names_cli(tmp_path, extra):
    """Bug report 2026-10-11 #9 (c01): CLI --part-names with a UTF-8 file."""
    names = tmp_path / "names.txt"
    names.write_text("1\tGröße\n2\tAlu\n", encoding="utf-8")
    out = tmp_path / "cli.d3plot"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "vtk2d3plot.cli",
            str(_two_part_series(tmp_path)),
            "-o",
            str(out),
            "--part-names",
            str(names),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    _check_titles(out, b"Groesse", b"Alu")


# ── #35 (c28): lower-case mapped point arrays ────────────────────────


def test_c28_lowercase_velocity_not_reported_unmapped(tmp_path):
    """Bug report 2026-10-11 #35 (c28): 'velocity' is mapped, so not 'unmapped'."""
    from lasso.dyna import ArrayType

    from vtk2d3plot import convert

    src = tmp_path / "c28"
    src.mkdir()
    for i in range(2):
        vel = np.zeros((4, 3))
        vel[:, 0] = 1.0 + i
        _write_vtk(
            src / f"runA{i:03d}.vtk",
            time=0.1 * i,
            cells=[_QUAD],
            part_ids=[1],
            point_vectors={"velocity": vel, "my_extra": np.full((4, 3), 3.0)},
        )
    out = tmp_path / "c28.d3plot"
    rep = convert(src, out)
    d = _read(out)
    v = d.arrays[ArrayType.node_velocity]
    np.testing.assert_array_equal(v[:, :, 0], [[1.0] * 4, [2.0] * 4])
    unmapped = [w for w in rep.warnings if w.startswith("unmapped arrays ignored")]
    # the genuinely unmapped array is still reported, the mapped one is not
    assert unmapped == ["unmapped arrays ignored: my_extra (absmax=3)"], rep.warnings
