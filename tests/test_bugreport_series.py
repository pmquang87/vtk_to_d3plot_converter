"""Bug report 2026-10-11, cluster E: series loader (src/vtk2d3plot/series.py).

Covers report numbers #12 (c36), #15 (c17), #19 (c24), #27 (c02+c25, parser side),
#31 (c33) and #32 (c35). All inputs are written by hand into tmp_path.
"""

import pathlib
import shutil

import numpy as np
import pytest
from conftest import PORT_DIR

from vtk2d3plot.series import (
    TopologyError,
    load_series,
    load_series_lazy,
    parse_t01_csv,
)

pytestmark = pytest.mark.bugreport


# ── helpers ─────────────────────────────────────────────────────────────────


def _write_vtk(path, time, node_ids=None, element_ids=None, title="hand-written state"):
    """Legacy ASCII VTK: 4 nodes, 1 quad, optional FIELD TIME and ID arrays."""
    lines = ["# vtk DataFile Version 3.0", title, "ASCII", "DATASET UNSTRUCTURED_GRID"]
    if time is not None:
        lines += ["FIELD FieldData 1", "TIME 1 1 double", repr(float(time))]
    lines += [
        "POINTS 4 float",
        "0 0 0",
        "1 0 0",
        "1 1 0",
        "0 1 0",
        "CELLS 1 5",
        "4 0 1 2 3",
        "CELL_TYPES 1",
        "9",
    ]
    if node_ids is not None:
        lines += [
            "POINT_DATA 4",
            "SCALARS NODE_ID int 1",
            "LOOKUP_TABLE default",
            " ".join(str(int(n)) for n in node_ids),
        ]
    if element_ids is not None:
        lines += [
            "CELL_DATA 1",
            "SCALARS ELEMENT_ID int 1",
            "LOOKUP_TABLE default",
            " ".join(str(int(n)) for n in element_ids),
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_vtu(path, z=0.0, time_point_array=None):
    """XML VTU: 4 nodes, 1 quad. No time anywhere unless time_point_array is given."""
    time_block = ""
    if time_point_array is not None:
        time_block = (
            '        <DataArray type="Float64" Name="TIME" format="ascii">\n'
            f"          {time_point_array!r} {time_point_array!r} "
            f"{time_point_array!r} {time_point_array!r}\n"
            "        </DataArray>\n"
        )
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n'
        "  <UnstructuredGrid>\n"
        '    <Piece NumberOfPoints="4" NumberOfCells="1">\n'
        "      <Points>\n"
        '        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n'
        f"          0 0 {z} 1 0 {z} 1 1 {z} 0 1 {z}\n"
        "        </DataArray>\n"
        "      </Points>\n"
        "      <Cells>\n"
        '        <DataArray type="Int32" Name="connectivity" format="ascii">0 1 2 3</DataArray>\n'
        '        <DataArray type="Int32" Name="offsets" format="ascii">4</DataArray>\n'
        '        <DataArray type="UInt8" Name="types" format="ascii">9</DataArray>\n'
        "      </Cells>\n"
        "      <PointData>\n"
        f"{time_block}"
        '        <DataArray type="Float64" Name="Displacement" NumberOfComponents="3" format="ascii">\n'
        f"          0 0 {z} 0 0 {z} 0 0 {z} 0 0 {z}\n"
        "        </DataArray>\n"
        "      </PointData>\n"
        "    </Piece>\n"
        "  </UnstructuredGrid>\n"
        "</VTKFile>\n",
        encoding="utf-8",
    )
    return path


def _write_pvd(path, entries):
    rows = "".join(
        f'    <DataSet timestep="{t}" group="" part="0" file="{f}"/>\n' for t, f in entries
    )
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="0.1">\n'
        "  <Collection>\n"
        f"{rows}"
        "  </Collection>\n"
        "</VTKFile>\n",
        encoding="utf-8",
    )
    return path


# ── #12 (c36): mistyped relative stem folder ────────────────────────────────


def test_c36_stem_missing_dir_raises(tmp_path, monkeypatch):
    """Bug report 2026-10-11 #12 (c36): 'no_such_dir/run' must not load ./runA*.vtk."""
    _write_vtk(tmp_path / "runA000.vtk", 0.0)
    _write_vtk(tmp_path / "runA001.vtk", 1.0)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="no_such_dir"):
        load_series("no_such_dir/run")
    with pytest.raises(FileNotFoundError, match="no_such_dir"):
        load_series_lazy("no_such_dir/run")


def test_c36_stem_missing_dir_convert_writes_nothing(tmp_path, monkeypatch):
    """Bug report 2026-10-11 #12 (c36): convert() of a mistyped stem raises, no d3plot written."""
    from vtk2d3plot.api import convert

    _write_vtk(tmp_path / "runA000.vtk", 0.0)
    _write_vtk(tmp_path / "runA001.vtk", 1.0)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        convert("no_such_dir/run", tmp_path / "out" / "x.d3plot")
    assert not (tmp_path / "out" / "x.d3plot").exists()


def test_c36_bare_stem_in_cwd_still_works(tmp_path, monkeypatch):
    """Bug report 2026-10-11 #12 (c36): a bare stem 'run' keeps meaning ./runA*.vtk."""
    _write_vtk(tmp_path / "runA000.vtk", 0.0)
    _write_vtk(tmp_path / "runA001.vtk", 1.0)
    monkeypatch.chdir(tmp_path)
    ser = load_series("run")
    assert [pathlib.Path(s.path).name for s in ser.states] == ["runA000.vtk", "runA001.vtk"]
    assert ser.times.tolist() == [0.0, 1.0]
    lazy = load_series_lazy("run")
    assert [p.name for p in lazy.paths] == ["runA000.vtk", "runA001.vtk"]
    assert lazy.times.tolist() == [0.0, 1.0]


# ── #15 (c17): PVD manifest timesteps ───────────────────────────────────────


def _pvd_series_without_times(tmp_path):
    # Document order deliberately not sorted; file names say nothing about time.
    _write_vtu(tmp_path / "run_a.vtu", z=0.0)
    _write_vtu(tmp_path / "run_b.vtu", z=0.25)
    _write_vtu(tmp_path / "run_c.vtu", z=0.5)
    return _write_pvd(
        tmp_path / "run.pvd", [(0.5, "run_b.vtu"), (0.0, "run_a.vtu"), (1.0, "run_c.vtu")]
    )


def test_c17_pvd_times_used_eager(tmp_path):
    """Bug report 2026-10-11 #15 (c17): VTUs without any time take the PVD timesteps."""
    pvd = _pvd_series_without_times(tmp_path)
    ser = load_series(pvd)
    assert ser.times.tolist() == [0.0, 0.5, 1.0]
    assert [pathlib.Path(s.path).name for s in ser.states] == [
        "run_a.vtu",
        "run_b.vtu",
        "run_c.vtu",
    ]
    assert [s.time for s in ser.states] == [0.0, 0.5, 1.0]
    assert [float(s.points[0, 2]) for s in ser.states] == [0.0, 0.25, 0.5]
    assert ser.warnings == []


def test_c17_pvd_times_used_lazy(tmp_path):
    """Bug report 2026-10-11 #15 (c17): the lazy loader also takes the PVD timesteps."""
    pvd = _pvd_series_without_times(tmp_path)
    lazy = load_series_lazy(pvd)
    assert lazy.times.tolist() == [0.0, 0.5, 1.0]
    assert [p.name for p in lazy.paths] == ["run_a.vtu", "run_b.vtu", "run_c.vtu"]
    assert lazy.state0.time == 0.0
    got = [(i, s.time, float(s.points[0, 2])) for i, s in lazy.iter_states()]
    assert got == [(0, 0.0, 0.0), (1, 0.5, 0.25), (2, 1.0, 0.5)]


def test_c17_pvd_convert_timesteps(tmp_path):
    """Bug report 2026-10-11 #15 (c17): convert(run.pvd) writes the manifest times."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    pvd = _pvd_series_without_times(tmp_path)
    rep = convert(pvd, tmp_path / "pvd.d3plot")
    assert rep.n_states == 3
    d = D3plot(str(tmp_path / "pvd.d3plot"))
    assert np.allclose(d.arrays[ArrayType.global_timesteps], [0.0, 0.5, 1.0], rtol=0, atol=1e-12)
    # absolute coordinates per state follow the manifest order (z = 0, 0.25, 0.5)
    z = d.arrays[ArrayType.node_displacement][:, 0, 2]
    assert np.allclose(z, [0.0, 0.25, 0.5], rtol=0, atol=1e-12)


def test_c17_pvd_manifest_wins_over_file_time_with_warning(tmp_path):
    """Bug report 2026-10-11 #15 (c17): manifest time wins on disagreement, and says so."""
    _write_vtu(tmp_path / "s0.vtu", z=0.0, time_point_array=0.0)
    _write_vtu(tmp_path / "s1.vtu", z=0.1, time_point_array=7.0)  # disagrees with 0.5
    pvd = _write_pvd(tmp_path / "m.pvd", [(0.0, "s0.vtu"), (0.5, "s1.vtu")])
    for ser in (load_series(pvd), load_series_lazy(pvd)):
        assert ser.times.tolist() == [0.0, 0.5]
        assert len(ser.warnings) == 1
        w = ser.warnings[0]
        assert "s1.vtu" in w and "7" in w and "0.5" in w and "PVD" in w


# Series-level warnings reach ConversionReport only once api.convert() forwards
# series.warnings (api.py is outside cluster E; cross-cluster request). Strict
# xfail: when api.py is fixed these XPASS and fail the suite, so remove the marker.
_NEEDS_API_FORWARD = pytest.mark.xfail(
    strict=True,
    reason="api.convert() does not forward series.warnings yet (cross-cluster request to api.py)",
)


@_NEEDS_API_FORWARD
def test_c17_pvd_override_warning_reaches_report(tmp_path):
    """Bug report 2026-10-11 #15 (c17): the manifest override is in convert().warnings."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    _write_vtu(tmp_path / "s0.vtu", z=0.0, time_point_array=0.0)
    _write_vtu(tmp_path / "s1.vtu", z=0.1, time_point_array=7.0)  # disagrees with 0.5
    pvd = _write_pvd(tmp_path / "m.pvd", [(0.0, "s0.vtu"), (0.5, "s1.vtu")])
    rep = convert(pvd, tmp_path / "m.d3plot")
    d = D3plot(str(tmp_path / "m.d3plot"))
    assert np.allclose(d.arrays[ArrayType.global_timesteps], [0.0, 0.5], rtol=0, atol=1e-12)
    assert (
        "PVD manifest timestep 0.5 used for s1.vtu; it overrides the file's own time 7.0"
        in rep.warnings
    )


def test_c17_multipart_pvd_clear_error(tmp_path):
    """Bug report 2026-10-11 #15 (c17): a multi-part PVD (part=0/1 per timestep) is refused clearly."""
    from vtk2d3plot.series import parse_pvd

    _write_vtu(tmp_path / "p0_t0.vtu", z=0.0)
    _write_vtu(tmp_path / "p1_t0.vtu", z=0.0)
    pvd = tmp_path / "mp.pvd"
    pvd.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="0.1">\n'
        "  <Collection>\n"
        '    <DataSet timestep="0" part="0" file="p0_t0.vtu"/>\n'
        '    <DataSet timestep="0" part="1" file="p1_t0.vtu"/>\n'
        "  </Collection>\n"
        "</VTKFile>\n",
        encoding="utf-8",
    )
    pattern = r"mp\.pvd is a multi-part collection \(parts 0, 1\).*not supported"
    with pytest.raises(ValueError, match=pattern):
        parse_pvd(pvd)
    with pytest.raises(ValueError, match=pattern):
        load_series(pvd)
    with pytest.raises(ValueError, match=pattern):
        load_series_lazy(pvd)


# ── #19 (c24): identity arrays in all states or in none ─────────────────────


def test_c24_id_change_caught_when_state0_lacks_ids(tmp_path):
    """Bug report 2026-10-11 #19 (c24): state0 without NODE_ID, then 100..103, then 150..153."""
    _write_vtk(tmp_path / "runA000.vtk", 0.0)
    _write_vtk(tmp_path / "runA001.vtk", 1.0, node_ids=[100, 101, 102, 103])
    _write_vtk(tmp_path / "runA002.vtk", 2.0, node_ids=[150, 151, 152, 153])
    with pytest.raises(TopologyError, match=r"NODE_ID.*runA001\.vtk"):
        load_series(tmp_path)


def test_c24_id_missing_in_later_state(tmp_path):
    """Bug report 2026-10-11 #19 (c24): ELEMENT_ID in state0 but missing in state1 is an error."""
    _write_vtk(tmp_path / "runA000.vtk", 0.0, element_ids=[7])
    _write_vtk(tmp_path / "runA001.vtk", 1.0)
    with pytest.raises(TopologyError, match=r"ELEMENT_ID.*runA001\.vtk"):
        load_series(tmp_path)


def test_c24_id_mismatch_between_later_states(tmp_path):
    """Bug report 2026-10-11 #19 (c24): state2 != state1 is caught, not only != state0."""
    from vtk2d3plot.api import convert

    _write_vtk(tmp_path / "runA000.vtk", 0.0, node_ids=[1, 2, 3, 4])
    _write_vtk(tmp_path / "runA001.vtk", 1.0, node_ids=[1, 2, 3, 4])
    _write_vtk(tmp_path / "runA002.vtk", 2.0, node_ids=[4, 3, 2, 1])
    with pytest.raises(TopologyError, match=r"NODE_ID.*runA002\.vtk"):
        convert(tmp_path, tmp_path / "out.d3plot")
    assert not (tmp_path / "out.d3plot").exists()


def test_c24_consistent_ids_load(tmp_path):
    """Bug report 2026-10-11 #19 (c24): identical IDs in every state still load."""
    for i in range(3):
        _write_vtk(tmp_path / f"runA00{i}.vtk", float(i), node_ids=[5, 6, 7, 8], element_ids=[9])
    ser = load_series(tmp_path)
    assert ser.times.tolist() == [0.0, 1.0, 2.0]
    assert ser.states[2].point_data["NODE_ID"].tolist() == [5, 6, 7, 8]


# ── #27 (c02+c25): header-only T01 csv ──────────────────────────────────────

T01_HEADER = "# pyradioss time history (T01 equivalent)\nTIME,IE,KE,HE,CE,EN,DE,EW,ERR%,MASS\n"


def test_c25_parse_header_only_t01(tmp_path):
    """Bug report 2026-10-11 #27 (c25): header-only T01 gives no columns, not empty columns."""
    p = tmp_path / "BOXIMPT01.csv"
    p.write_text(T01_HEADER, encoding="utf-8")
    t01 = parse_t01_csv(p)
    assert t01.columns == {}
    assert t01.times.shape == (0,)


def test_c02_convert_with_header_only_t01(tmp_path):
    """Bug report 2026-10-11 #27 (c02): conversion succeeds without T01 globals, and warns."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    for i in range(3):
        shutil.copy(PORT_DIR / f"BOXIMPA00{i}.vtk", tmp_path)
    (tmp_path / "BOXIMPT01.csv").write_text(T01_HEADER, encoding="utf-8")

    rep = convert(tmp_path, tmp_path / "box.d3plot")
    assert rep.n_states == 3

    ser = load_series(tmp_path)
    assert ser.t01_data is not None and ser.t01_data.columns == {}
    assert any("BOXIMPT01.csv" in w and "no data rows" in w for w in ser.warnings)

    d = D3plot(str(tmp_path / "box.d3plot"))
    assert np.allclose(d.arrays[ArrayType.global_timesteps], ser.times, rtol=1e-6)
    # same as a run without any T01 file: no global energies are written
    assert d.arrays.get(ArrayType.global_internal_energy) is None
    assert d.arrays.get(ArrayType.global_kinetic_energy) is None
    assert d.arrays.get(ArrayType.global_total_energy) is None


@_NEEDS_API_FORWARD
def test_c02_header_only_t01_warning_reaches_report(tmp_path):
    """Bug report 2026-10-11 #27 (c02): 'no data rows' is in convert().warnings."""
    from vtk2d3plot.api import convert

    for i in range(3):
        shutil.copy(PORT_DIR / f"BOXIMPA00{i}.vtk", tmp_path)
    (tmp_path / "BOXIMPT01.csv").write_text(T01_HEADER, encoding="utf-8")
    rep = convert(tmp_path, tmp_path / "box.d3plot")
    assert (
        "T01 file BOXIMPT01.csv has no data rows; "
        "global energies from it (IE, KE, EN) are not written" in rep.warnings
    )


# ── #31 (c33): NaN time ─────────────────────────────────────────────────────


def test_c33_nan_time_clear_error(tmp_path):
    """Bug report 2026-10-11 #31 (c33): FIELD TIME 0, nan, 2 gives the clear ValueError."""
    _write_vtk(tmp_path / "runA000.vtk", 0.0)
    _write_vtk(tmp_path / "runA001.vtk", float("nan"))
    _write_vtk(tmp_path / "runA002.vtk", 2.0)
    pattern = r"Times are not strictly increasing.*runA000\.vtk \(time 0\.0\) is followed by file runA001\.vtk \(time nan\)"
    with pytest.raises(ValueError, match=pattern):
        load_series(tmp_path)
    with pytest.raises(ValueError, match=pattern):
        load_series_lazy(tmp_path)


def test_c33_single_state_nan_time(tmp_path):
    """Bug report 2026-10-11 #31 (c33): a lone NaN time (no difference to check) is refused."""
    _write_vtk(tmp_path / "runA000.vtk", float("nan"))
    pattern = r"Time is not finite: file runA000\.vtk \(time nan\)"
    with pytest.raises(ValueError, match=pattern):
        load_series(tmp_path)
    with pytest.raises(ValueError, match=pattern):
        load_series_lazy(tmp_path)


def test_c33_inf_last_time(tmp_path):
    """Bug report 2026-10-11 #31 (c33): +inf as last time passes the difference test but is refused."""
    _write_vtk(tmp_path / "runA000.vtk", 0.0)
    _write_vtk(tmp_path / "runA001.vtk", float("inf"))
    pattern = r"Time is not finite: file runA001\.vtk \(time inf\)"
    with pytest.raises(ValueError, match=pattern):
        load_series(tmp_path)
    with pytest.raises(ValueError, match=pattern):
        load_series_lazy(tmp_path)


# ── #32 (c35): upper-case extensions ────────────────────────────────────────


@pytest.fixture
def posix_case_glob(monkeypatch):
    """Make Path.glob case-sensitive by default, as on Linux/macOS (the report's platform).

    On Windows the default glob is case-insensitive, which hides the bug. A fix that
    passes ``case_sensitive=False`` explicitly, or does not use glob, is unaffected.
    """
    orig = pathlib.Path.glob

    def glob(self, pattern, *, case_sensitive=None, **kw):
        if case_sensitive is None:
            case_sensitive = True
        return orig(self, pattern, case_sensitive=case_sensitive, **kw)

    monkeypatch.setattr(pathlib.Path, "glob", glob)


def test_c35_upper_ext_directory_mode(tmp_path, posix_case_glob):
    """Bug report 2026-10-11 #32 (c35): runA2.VTK, runA10.VTK, runA1.vtk found and ordered numerically."""
    _write_vtk(tmp_path / "runA10.VTK", 10.0)
    _write_vtk(tmp_path / "runA2.VTK", 2.0)
    _write_vtk(tmp_path / "runA1.vtk", 1.0)
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    for ser in (load_series(tmp_path), load_series_lazy(tmp_path)):
        assert ser.times.tolist() == [1.0, 2.0, 10.0]
    lazy = load_series_lazy(tmp_path)
    assert [p.name for p in lazy.paths] == ["runA1.vtk", "runA2.VTK", "runA10.VTK"]


def test_c35_upper_ext_stem_mode(tmp_path, posix_case_glob):
    """Bug report 2026-10-11 #32 (c35): stem mode finds .VTU/.VTK and keeps one stem."""
    _write_vtu(tmp_path / "runA000.VTU", z=0.0, time_point_array=0.0)
    _write_vtu(tmp_path / "runA001.VTU", z=0.1, time_point_array=1.0)
    _write_vtk(tmp_path / "otherA000.VTK", 5.0)  # different stem, must be ignored
    ser = load_series(tmp_path / "run")
    assert [pathlib.Path(s.path).name for s in ser.states] == ["runA000.VTU", "runA001.VTU"]
    assert ser.times.tolist() == [0.0, 1.0]
    lazy = load_series_lazy(tmp_path / "run")
    assert [p.name for p in lazy.paths] == ["runA000.VTU", "runA001.VTU"]


def test_c35_upper_ext_multi_stem_still_rejected(tmp_path, posix_case_glob):
    """Bug report 2026-10-11 #32 (c35): two stems with upper-case extensions still raise."""
    _write_vtk(tmp_path / "run1A000.VTK", 0.0)
    _write_vtk(tmp_path / "run2A000.VTK", 0.0)
    with pytest.raises(ValueError, match=r"Multiple stems found: run1, run2"):
        load_series(tmp_path)


def test_c35_explicit_list_keeps_caller_order(tmp_path, posix_case_glob):
    """Bug report 2026-10-11 #32 (c35): explicit lists are not re-sorted."""
    a = _write_vtk(tmp_path / "runA000.VTK", 0.0)
    b = _write_vtk(tmp_path / "runA001.VTK", 1.0)
    assert load_series([a, b]).times.tolist() == [0.0, 1.0]
    with pytest.raises(
        ValueError, match=r"runA001\.VTK \(time 1\.0\) is followed by file runA000\.VTK"
    ):
        load_series([b, a])
