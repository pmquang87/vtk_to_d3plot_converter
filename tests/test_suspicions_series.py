"""Milestone 14, cluster B: series-loader leads from the 2026-10-11 bug report's
"Suspicions not checked" list (S02, S06, S07, S08, S09, S24) and the milestone-13
open point (e): both loaders report the first problem in file order.

All inputs are written by hand into tmp_path; port-dialect states are copied
from tests/data (read-only).
"""

import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest
from conftest import PORT_DIR, REPO

from vtk2d3plot.series import TopologyError, load_series, load_series_lazy, parse_t01_csv

pytestmark = pytest.mark.suspicions


# ── helpers ─────────────────────────────────────────────────────────────────


def _write_vtk(
    path,
    time,
    *,
    node_ids=None,
    extra_point=False,
    cell_type="9",
    title="hand-written state",
):
    """Legacy ASCII VTK: 4 nodes (5 with extra_point), 1 quad, optional FIELD TIME / NODE_ID.

    ``cell_type`` is written verbatim into CELL_TYPES, so a non-number makes the
    file unreadable (a parse error the lazy loader also hits: it reads cells).
    """
    lines = ["# vtk DataFile Version 3.0", title, "ASCII", "DATASET UNSTRUCTURED_GRID"]
    if time is not None:
        lines += ["FIELD FieldData 1", "TIME 1 1 double", repr(float(time))]
    pts = ["0 0 0", "1 0 0", "1 1 0", "0 1 0"] + (["2 2 0"] if extra_point else [])
    lines += [f"POINTS {len(pts)} float", *pts]
    lines += ["CELLS 1 5", "4 0 1 2 3", "CELL_TYPES 1", cell_type]
    if node_ids is not None:
        lines += [
            f"POINT_DATA {len(pts)}",
            "SCALARS NODE_ID int 1",
            "LOOKUP_TABLE default",
            " ".join(str(int(n)) for n in node_ids),
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _copy_port_states(dst, n):
    for i in range(n):
        shutil.copy(PORT_DIR / f"BOXIMPA{i:03d}.vtk", dst)


def _read_d3(path):
    from lasso.dyna import D3plot

    return D3plot(str(path))


# ── S02: T01 restart / duplicate rows ───────────────────────────────────────

# Segment 1 runs to t = 0.16 and has a plain duplicate at t = 0.04; the solver
# restarts at t = 0.06 (segment 2), whose last row is written twice.
_T01_RESTART = (
    "# pyradioss time history (T01 equivalent)\n"
    "TIME,IE,KE\n"
    "0.0,0,100\n"
    "0.04,99,999\n"  # superseded by the duplicate below
    "0.04,1,100\n"
    "0.08,2,100\n"  # superseded by the restart
    "0.12,3,100\n"  # superseded by the restart
    "0.16,4,100\n"  # superseded by the restart
    "0.06,50,200\n"
    "0.10,60,210\n"
    "0.14,70,220\n"
    "0.18,80,230\n"  # superseded by the duplicate below
    "0.18,90,240\n"
)
# The correct (later-segment-wins) time history, by hand.
_T01_KEPT_T = np.array([0.0, 0.04, 0.06, 0.10, 0.14, 0.18])
_T01_KEPT_IE = np.array([0.0, 1.0, 50.0, 60.0, 70.0, 90.0])
_T01_KEPT_KE = np.array([100.0, 100.0, 200.0, 210.0, 220.0, 240.0])
_T01_RESTART_WARNING = (
    "T01 file BOXIMPT01.csv: 5 data row(s) superseded by later rows with an equal or "
    "earlier TIME (solver restart or duplicate output); the later rows are used"
)


def test_s02_t01_restart_energies(tmp_path):
    """S02: a restart segment and duplicate rows in T01 give the later segment's energies."""
    from lasso.dyna import ArrayType

    from vtk2d3plot.api import convert

    _copy_port_states(tmp_path, 4)  # state times 0.00029 ... 0.150
    (tmp_path / "BOXIMPT01.csv").write_text(_T01_RESTART, encoding="utf-8")
    rep = convert(tmp_path, tmp_path / "box.d3plot")

    d = _read_d3(tmp_path / "box.d3plot")
    t = d.arrays[ArrayType.global_timesteps]
    assert t.shape == (4,)
    ie = np.interp(t, _T01_KEPT_T, _T01_KEPT_IE)
    ke = np.interp(t, _T01_KEPT_T, _T01_KEPT_KE)
    assert np.allclose(d.arrays[ArrayType.global_internal_energy], ie, rtol=1e-12, atol=0)
    assert np.allclose(d.arrays[ArrayType.global_kinetic_energy], ke, rtol=1e-12, atol=0)
    assert np.allclose(d.arrays[ArrayType.global_total_energy], ie + ke, rtol=1e-12, atol=0)
    assert _T01_RESTART_WARNING in rep.warnings


def test_s02_parse_t01_restart_strictly_increasing(tmp_path):
    """S02: parse_t01_csv returns strictly increasing times and counts superseded rows."""
    p = tmp_path / "runT01.csv"
    p.write_text(_T01_RESTART, encoding="utf-8")
    t01 = parse_t01_csv(p)
    assert t01.times.tolist() == _T01_KEPT_T.tolist()
    assert t01.columns["IE"].tolist() == _T01_KEPT_IE.tolist()
    assert t01.columns["KE"].tolist() == _T01_KEPT_KE.tolist()
    assert t01.superseded_rows == 5
    assert t01.skipped_lines == []


def test_s02_t01_nonfinite_time_row_skipped(tmp_path):
    """S02: a T01 row with a NaN or inf TIME cannot be ordered; it is skipped as malformed."""
    p = tmp_path / "runT01.csv"
    p.write_text("# c\nTIME,IE,KE\n0,1,2\nnan,5,5\n1,3,4\ninf,6,6\n2,5,6\n", encoding="utf-8")
    t01 = parse_t01_csv(p)
    assert t01.times.tolist() == [0.0, 1.0, 2.0]
    assert t01.columns["IE"].tolist() == [1.0, 3.0, 5.0]
    assert t01.skipped_lines == [4, 6]
    assert t01.superseded_rows == 0


def test_s02_clean_t01_has_no_restart_warning(tmp_path):
    """S02: the real (increasing) port T01 file gives no superseded rows and no warning."""
    from vtk2d3plot.api import convert

    _copy_port_states(tmp_path, 3)
    shutil.copy(PORT_DIR / "BOXIMPT01.csv", tmp_path)
    assert parse_t01_csv(tmp_path / "BOXIMPT01.csv").superseded_rows == 0
    rep = convert(tmp_path, tmp_path / "box.d3plot")
    assert not any("superseded" in w for w in rep.warnings)


# ── S06: other .vtk/.vtu files in a series folder ───────────────────────────

_S06_WARNING = (
    "Ignored 2 file(s) in the input folder without the <stem>A<number>.vtk/.vtu "
    "state name: extra.vtk, mesh.vtu"
)


def test_s06_ignored_files_reported(tmp_path):
    """S06: files without the state name are still dropped, but convert() says which."""
    _write_vtk(tmp_path / "runA000.vtk", 0.0)
    _write_vtk(tmp_path / "runA001.vtk", 1.0)
    _write_vtk(tmp_path / "extra.vtk", 5.0)
    _write_vtk(tmp_path / "mesh.vtu", 5.0)  # never read: name only
    from vtk2d3plot.api import convert

    rep = convert(tmp_path, tmp_path / "out.d3plot")
    assert rep.n_states == 2
    assert rep.warnings.count(_S06_WARNING) == 1, rep.warnings
    for ser in (load_series(tmp_path), load_series_lazy(tmp_path)):
        assert _S06_WARNING in ser.warnings


def test_s06_no_warning_without_extra_files(tmp_path):
    """S06: a clean series folder and an explicit list give no ignored-files warning."""
    a = _write_vtk(tmp_path / "runA000.vtk", 0.0)
    b = _write_vtk(tmp_path / "runA001.vtk", 1.0)
    _write_vtk(tmp_path / "extra.vtk", 5.0)
    assert not any("ignored" in w for w in load_series([a, b]).warnings)
    (tmp_path / "extra.vtk").unlink()
    assert not any("ignored" in w for w in load_series(tmp_path).warnings)


# ── S07: natural sort for names without A<digits> ───────────────────────────


def test_s07_natural_sort_directory(tmp_path):
    """S07: out_1, out_2, out_10 load in numeric order (string order puts out_10 second)."""
    from lasso.dyna import ArrayType

    from vtk2d3plot.api import convert

    for n in (10, 2, 1):
        _write_vtk(tmp_path / f"out_{n}.vtk", float(n))
    rep = convert(tmp_path, tmp_path / "out.d3plot")
    assert rep.n_states == 3
    ts = _read_d3(tmp_path / "out.d3plot").arrays[ArrayType.global_timesteps]
    assert ts.tolist() == [1.0, 2.0, 10.0]
    lazy = load_series_lazy(tmp_path)
    assert [p.name for p in lazy.paths] == ["out_1.vtk", "out_2.vtk", "out_10.vtk"]


def test_s07_natural_sort_several_digit_runs(tmp_path):
    """S07: every digit run compares as a number (case_2_step_9 before case_2_step_10)."""
    names = ["case_2_step_10.vtk", "case_2_step_9.vtk", "case_10_step_1.vtk"]
    for t, name in zip((2.0, 1.0, 3.0), names, strict=True):
        _write_vtk(tmp_path / name, t)
    ser = load_series(tmp_path)
    assert ser.times.tolist() == [1.0, 2.0, 3.0]
    assert [pathlib.Path(s.path).name for s in ser.states] == [
        "case_2_step_9.vtk",
        "case_2_step_10.vtk",
        "case_10_step_1.vtk",
    ]


def test_s07_natural_sort_ignores_case(tmp_path):
    """S07: text runs compare case-insensitively (out_1, Out_2, out_10)."""
    for t, name in ((1.0, "out_1.vtk"), (2.0, "Out_2.vtk"), (10.0, "out_10.vtk")):
        _write_vtk(tmp_path / name, t)
    assert load_series(tmp_path).times.tolist() == [1.0, 2.0, 10.0]


def _run_cli(args):
    return subprocess.run(
        [sys.executable, "-m", "vtk2d3plot.cli", *map(str, args)],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )


def test_s07_cli_glob_across_folders(tmp_path):
    """S07/S24: a CLI glob step_*/out.vtk orders the folders naturally; same names
    in different folders are different states."""
    from lasso.dyna import ArrayType

    for n in (10, 2, 1):
        (tmp_path / f"step_{n}").mkdir()
        _write_vtk(tmp_path / f"step_{n}" / "out.vtk", float(n))
    r = _run_cli([tmp_path / "step_*" / "out.vtk", "-o", tmp_path / "s.d3plot", "-q"])
    assert r.returncode == 0, r.stderr
    ts = _read_d3(tmp_path / "s.d3plot").arrays[ArrayType.global_timesteps]
    assert ts.tolist() == [1.0, 2.0, 10.0]


# ── S24: the same state twice (.vtk and .vtu, or A1 and A001) ───────────────


@pytest.mark.parametrize(
    ("first", "second"),
    [("runA1.vtk", "runA001.vtk"), ("runA001.vtk", "runA001.vtu")],
)
def test_s24_same_state_twice_clear_error(tmp_path, first, second):
    """S24: one state number in two files is refused with both names, not 'duplicate time'."""
    from vtk2d3plot.api import convert

    _write_vtk(tmp_path / "runA000.vtk", 0.0)
    _write_vtk(tmp_path / first, 1.0)
    _write_vtk(tmp_path / second, 1.0)
    _write_vtk(tmp_path / "runA002.vtk", 2.0)
    a, b = sorted([first, second])
    pattern = rf"{re.escape(a)} and {re.escape(b)} are the same state.*remove one"
    with pytest.raises(ValueError, match=pattern):
        load_series(tmp_path)
    with pytest.raises(ValueError, match=pattern):
        load_series_lazy(tmp_path)
    with pytest.raises(ValueError, match=pattern):
        load_series(tmp_path / "run")  # stem mode
    with pytest.raises(ValueError, match=pattern):
        convert(tmp_path, tmp_path / "out.d3plot")
    assert not (tmp_path / "out.d3plot").exists()


def test_s24_same_name_vtk_and_vtu_without_state_number(tmp_path):
    """S24: out_1.vtk and out_1.vtu (no A<number> names) are also the same state."""
    _write_vtk(tmp_path / "out_0.vtk", 0.0)
    _write_vtk(tmp_path / "out_1.vtk", 1.0)
    _write_vtk(tmp_path / "out_1.vtu", 1.0)
    with pytest.raises(ValueError, match=r"out_1\.vtk and out_1\.vtu are the same state"):
        load_series(tmp_path)


def test_s24_cli_glob_same_state_twice(tmp_path):
    """S24: a CLI glob that matches the same state twice stops with both names."""
    _write_vtk(tmp_path / "runA000.vtk", 0.0)
    _write_vtk(tmp_path / "runA1.vtk", 1.0)
    _write_vtk(tmp_path / "runA001.vtk", 1.0)
    r = _run_cli([tmp_path / "runA*.vtk", "-o", tmp_path / "out.d3plot", "-q"])
    assert r.returncode == 1
    assert "runA001.vtk and runA1.vtk are the same state" in r.stderr, r.stderr
    assert not (tmp_path / "out.d3plot").exists()


def test_s24_cli_glob_same_state_in_two_folders(tmp_path):
    """S24: one state number in two folders names the files with their folders."""
    for d, n in (("r1", 0), ("r1", 1), ("r2", 0)):
        (tmp_path / d).mkdir(exist_ok=True)
        _write_vtk(tmp_path / d / f"runA00{n}.vtk", float(n))
    r = _run_cli([tmp_path / "r*" / "runA*.vtk", "-o", tmp_path / "out.d3plot", "-q"])
    assert r.returncode == 1
    a, b = tmp_path / "r1" / "runA000.vtk", tmp_path / "r2" / "runA000.vtk"
    assert f"{a} and {b} are the same state" in r.stderr, r.stderr


def test_s24_explicit_list_not_affected(tmp_path):
    """S24: an explicit file list is the user's choice and is not de-duplicated."""
    a = _write_vtk(tmp_path / "runA000.vtk", 0.0)
    b = _write_vtk(tmp_path / "runA001.vtk", 1.0)
    c = _write_vtk(tmp_path / "runA1.vtk", 2.0)
    assert load_series([a, b, c]).times.tolist() == [0.0, 1.0, 2.0]


# ── S08: multi-part PVD (probe pins) ────────────────────────────────────────


def _write_pvd_rows(path, rows):
    body = "".join(f"    <DataSet {r}/>\n" for r in rows)
    path.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n  <Collection>\n'
        f"{body}  </Collection>\n</VTKFile>\n",
        encoding="utf-8",
    )
    return path


def test_s08_pvd_single_part_value_loads(tmp_path):
    """S08: one part value on every DataSet is a normal series, not a multi-part one."""
    _write_vtk(tmp_path / "a.vtk", None)
    _write_vtk(tmp_path / "b.vtk", None)
    pvd = _write_pvd_rows(
        tmp_path / "s.pvd",
        ['timestep="0" part="3" file="a.vtk"', 'timestep="1" part="3" file="b.vtk"'],
    )
    for loader in (load_series, load_series_lazy):
        assert loader(pvd).times.tolist() == [0.0, 1.0]


def test_s08_pvd_duplicate_timestep_names_both_files(tmp_path):
    """S08: two DataSets at one timestep (no part attribute) name both files in the error."""
    _write_vtk(tmp_path / "a.vtk", None)
    _write_vtk(tmp_path / "b.vtk", None)
    pvd = _write_pvd_rows(
        tmp_path / "d.pvd", ['timestep="0" file="a.vtk"', 'timestep="0" file="b.vtk"']
    )
    pattern = r"file a\.vtk \(time 0\.0\) is followed by file b\.vtk \(time 0\.0\)"
    for loader in (load_series, load_series_lazy):
        with pytest.raises(ValueError, match=pattern):
            loader(pvd)


# ── S09: malformed T01 rows (probe pins) ────────────────────────────────────


@pytest.mark.parametrize(
    "bad_row",
    ["1,3", "1,3,4,9", "1,x,4", ""],
    ids=["short", "long", "non-numeric", "blank"],
)
def test_s09_bad_t01_row_does_not_abort(tmp_path, bad_row):
    """S09: one bad T01 row is skipped (a blank line silently), the rest is used."""
    p = tmp_path / "runT01.csv"
    p.write_text(f"# c\nTIME,IE,KE\n0,1,2\n{bad_row}\n2,5,6\n", encoding="utf-8")
    t01 = parse_t01_csv(p)
    assert t01.times.tolist() == [0.0, 2.0]
    assert t01.columns["IE"].tolist() == [1.0, 5.0]
    assert t01.skipped_lines == ([] if bad_row == "" else [4])


def test_s09_ragged_header_does_not_abort(tmp_path):
    """S09: a header with a field count no row matches gives no data, and a warning."""
    from vtk2d3plot.api import convert

    _copy_port_states(tmp_path, 2)
    (tmp_path / "BOXIMPT01.csv").write_text("# c\nTIME,IE,KE,\n0,1,2\n1,3,4\n", encoding="utf-8")
    rep = convert(tmp_path, tmp_path / "box.d3plot")
    assert rep.n_states == 2
    assert any("BOXIMPT01.csv" in w and "no data rows" in w for w in rep.warnings)


# ── (e): both loaders report the first problem in file order ────────────────
#
# Each case: three files runA000..runA002, two problems in different files.
# Problems per file, checked in this order: time missing, time not after the
# previous time (or not finite), topology vs state 0, identity vs state 0.
# A parse error is raised while reading the file, i.e. before its checks.
# Out of scope: parse errors in arrays the lazy loader skips (metadata_only
# does not parse POINTS coordinates or result arrays), which only load_series sees.

_DECREASING_1 = r"file runA000\.vtk \(time 0\.0\) is followed by file runA001\.vtk \(time -1\.0\)"
_DECREASING_2 = r"file runA001\.vtk \(time 1\.0\) is followed by file runA002\.vtk \(time 0\.5\)"


def _case_files(tmp_path, spec):
    """spec: per file a dict of _write_vtk keyword overrides plus 'time'."""
    for i, kw in enumerate(spec):
        kw = dict(kw)
        t = kw.pop("time", float(i))
        _write_vtk(tmp_path / f"runA00{i}.vtk", t, **kw)


_ORDER_CASES = {
    "time1_identity2": (
        [{}, {"time": -1.0}, {"node_ids": [4, 3, 2, 1]}],
        ValueError,
        _DECREASING_1,
    ),
    "identity1_time2": (
        [{}, {"node_ids": [4, 3, 2, 1]}, {"time": 0.5}],
        TopologyError,
        r"point_data NODE_ID missing in runA000\.vtk but present in runA001\.vtk",
    ),
    "time1_topology2": (
        [{}, {"time": 0.0}, {"extra_point": True}],
        ValueError,
        r"file runA000\.vtk \(time 0\.0\) is followed by file runA001\.vtk \(time 0\.0\)",
    ),
    "topology1_time2": (
        [{}, {"extra_point": True}, {"time": 0.5}],
        TopologyError,
        r"Points count mismatch in runA001\.vtk",
    ),
    "topology1_missing_time2": (
        [{}, {"extra_point": True}, {"time": None}],
        TopologyError,
        r"Points count mismatch in runA001\.vtk",
    ),
    "missing_time1_topology2": (
        [{}, {"time": None}, {"extra_point": True}],
        ValueError,
        r"Time is None for file runA001\.vtk",
    ),
    "nan_time1_identity2": (
        [{}, {"time": float("nan")}, {"node_ids": [1, 2, 3, 4]}],
        ValueError,
        r"runA000\.vtk \(time 0\.0\) is followed by file runA001\.vtk \(time nan\)",
    ),
    "inf_time1_topology2": (
        [{}, {"time": float("inf")}, {"extra_point": True}],
        ValueError,
        r"Time is not finite: file runA001\.vtk \(time inf\)",
    ),
    "time2_parse_error1_is_first": (
        [{}, {"cell_type": "x"}, {"time": 0.5}],
        None,  # parse error type belongs to the reader; only the file is pinned
        r"runA001\.vtk",
    ),
    "time1_parse_error2": (
        [{}, {"time": -1.0}, {"cell_type": "x"}],
        ValueError,
        _DECREASING_1,
    ),
    "identity2_time2_same_file": (
        [{}, {}, {"time": 0.5, "node_ids": [1, 2, 3, 4]}],
        ValueError,
        _DECREASING_2,
    ),
}


@pytest.mark.parametrize("case", list(_ORDER_CASES), ids=list(_ORDER_CASES))
def test_e_loaders_report_same_first_problem(tmp_path, case):
    """(e): load_series and load_series_lazy raise the same first problem in file order."""
    spec, exc_type, pattern = _ORDER_CASES[case]
    _case_files(tmp_path, spec)
    got = []
    for loader in (load_series, load_series_lazy):
        with pytest.raises(Exception) as ei:  # type and message compared below
            loader(tmp_path)
        got.append((type(ei.value), str(ei.value)))
    assert got[0] == got[1], f"eager {got[0]!r} != lazy {got[1]!r}"
    if exc_type is not None:
        assert got[0][0] is exc_type
    assert re.search(pattern, got[0][1]), got[0][1]
