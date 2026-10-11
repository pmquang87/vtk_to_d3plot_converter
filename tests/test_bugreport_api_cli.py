"""Bug report 2026-10-11, cluster F: api.convert() output handling and the CLI.

Covers report items #10 (c05), #11 (c42), #28 (c04), #36 (c11), #37 (c43)
and #38 (c10).
"""

import os
import posixpath
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import PORT_DIR

pytestmark = pytest.mark.bugreport

N_PORT = 21  # states in the box_beam_impact port fixture
PORT_T0 = 2.902412789e-04  # time of BOXIMPA000.vtk (title line)


def _run_cli(args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "vtk2d3plot.cli", *[str(a) for a in args]],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )


def _snapshot(folder: Path) -> dict[str, bytes]:
    """Name -> content of every file below *folder* (relative names)."""
    return {
        str(p.relative_to(folder)): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()
    }


def _port_times() -> list[float]:
    from vtk2d3plot.vtk_reader import read_vtk

    return [read_vtk(PORT_DIR / f"BOXIMPA{i:03d}.vtk").time for i in range(N_PORT)]


# ----------------------------------------------------------------------
# #10 (c05): case-different family deleted on a case-sensitive FS
# ----------------------------------------------------------------------


def _fs_is_case_sensitive(folder: Path) -> bool:
    probe = folder / "case_probe_x"
    probe.write_text("x")
    try:
        return not (folder / "CASE_PROBE_X").exists()
    finally:
        probe.unlink()


def test_c05_case_family_survives_on_case_sensitive_fs(tmp_path, monkeypatch):
    """Bug report 2026-10-11 #10 (c05).

    With POSIX name comparison (simulated by patching the normcase the api
    uses), files of the family ``Box.d3plot*`` are a different family than
    ``box.d3plot`` and must survive writing ``box.d3plot``. On a
    case-insensitive FS ``Box.d3plot``/``Box.d3plot01`` cannot coexist with the
    new ``box.d3plot``/``box.d3plot01``, so the members used here lie outside
    the 21-state range; on a case-sensitive FS the report's exact pair is
    added too.
    """
    from vtk2d3plot import api

    monkeypatch.setattr(api, "_normcase", posixpath.normcase, raising=False)
    other = {"Box.d3plot50": b"other family 50", "Box.d3plot99": b"other family 99"}
    if _fs_is_case_sensitive(tmp_path):
        other.update({"Box.d3plot": b"other geometry", "Box.d3plot01": b"other state 1"})
    for name, data in other.items():
        (tmp_path / name).write_bytes(data)

    api.convert(PORT_DIR, tmp_path / "box.d3plot")

    for name, data in other.items():
        assert (tmp_path / name).is_file(), f"{name} of the other family was deleted"
        assert (tmp_path / name).read_bytes() == data
    assert (tmp_path / "box.d3plot").is_file()
    assert (tmp_path / "box.d3plot21").is_file()


@pytest.mark.skipif(sys.platform != "win32", reason="case-insensitive FS rule (Windows)")
def test_c05_case_family_still_replaced_on_windows(tmp_path):
    """Bug report 2026-10-11 #10 (c05): on Windows the case-different name IS the same family."""
    from vtk2d3plot.api import convert

    stale = tmp_path / "BOX.D3PLOT99"
    stale.write_bytes(b"stale")
    convert(PORT_DIR, tmp_path / "box.d3plot")
    assert not stale.exists()


# ----------------------------------------------------------------------
# #11 (c42): output path is an existing directory
# ----------------------------------------------------------------------


def test_c42_output_is_dir_api(tmp_path):
    """Bug report 2026-10-11 #11 (c42): convert() raises before touching anything."""
    from vtk2d3plot.api import convert

    out_dir = tmp_path / "out_dir"
    results = out_dir / "results"
    results.mkdir(parents=True)
    (results / "keep.txt").write_bytes(b"inside")
    (out_dir / "results07").write_bytes(b"neighbour state-like file")
    (out_dir / "other.txt").write_bytes(b"neighbour")
    before = _snapshot(out_dir)

    with pytest.raises(IsADirectoryError, match="existing directory"):
        convert(PORT_DIR, results)

    assert _snapshot(out_dir) == before
    assert sorted(p.name for p in out_dir.iterdir()) == ["other.txt", "results", "results07"]


def test_c42_output_is_dir_cli(tmp_path):
    """Bug report 2026-10-11 #11 (c42): the CLI prints a clean error and exits 1."""
    results = tmp_path / "results"
    results.mkdir()
    before = _snapshot(tmp_path)
    r = _run_cli([PORT_DIR, "-o", results, "-q"])
    assert r.returncode == 1
    assert r.stderr.startswith("ERROR: output path is an existing directory")
    assert "Traceback" not in r.stderr
    assert _snapshot(tmp_path) == before
    assert [p.name for p in tmp_path.iterdir()] == ["results"]


def test_c42_state_file_name_is_dir(tmp_path):
    """Bug report 2026-10-11 #11 (c42), same trap one level down: a directory named
    like a state file of the family must not swallow that state file."""
    from vtk2d3plot.api import convert

    blocker = tmp_path / "box.d3plot05"
    blocker.mkdir()
    (blocker / "keep.txt").write_bytes(b"inside")
    before = _snapshot(tmp_path)

    with pytest.raises(IsADirectoryError, match="box.d3plot05"):
        convert(PORT_DIR, tmp_path / "box.d3plot")

    assert _snapshot(tmp_path) == before
    assert [p.name for p in tmp_path.iterdir()] == ["box.d3plot05"]


# ----------------------------------------------------------------------
# #28 (c04): CLI glob expansion not sorted
# ----------------------------------------------------------------------


def _unpadded_copy(dst: Path) -> None:
    """Port fixture with names BOXIMPA0..BOXIMPA20 (no zero padding):
    file-system (lexicographic) order is A0, A1, A10, ..., A19, A2, A20, A3..."""
    dst.mkdir()
    for i in range(N_PORT):
        shutil.copy(PORT_DIR / f"BOXIMPA{i:03d}.vtk", dst / f"BOXIMPA{i}.vtk")


def test_c04_cli_glob_numeric_order(tmp_path):
    """Bug report 2026-10-11 #28 (c04): a quoted glob gives the same result as the directory."""
    from lasso.dyna import ArrayType, D3plot

    src = tmp_path / "vtk"
    _unpadded_copy(src)
    out = tmp_path / "glob.d3plot"
    r = _run_cli([str(src / "*.vtk"), "-o", out, "-q"])
    assert r.returncode == 0, r.stderr
    ts = D3plot(str(out)).arrays[ArrayType.global_timesteps]
    assert ts.tolist() == pytest.approx(_port_times(), rel=1e-12)
    assert ts[0] == pytest.approx(PORT_T0, rel=1e-12)


def test_c04_cli_glob_keeps_pattern_order(tmp_path):
    """Bug report 2026-10-11 #28 (c04): matches are sorted per pattern; the
    patterns themselves keep the order the user gave."""
    from lasso.dyna import ArrayType, D3plot

    src = tmp_path / "vtk"
    _unpadded_copy(src)
    pats = [str(src / "BOXIMPA?.vtk"), str(src / "BOXIMPA1?.vtk"), str(src / "BOXIMPA2?.vtk")]
    out = tmp_path / "ok.d3plot"
    r = _run_cli([*pats, "-o", out, "-q"])
    assert r.returncode == 0, r.stderr
    ts = D3plot(str(out)).arrays[ArrayType.global_timesteps]
    assert ts.tolist() == pytest.approx(_port_times(), rel=1e-12)

    # reversed pattern order: user order is kept, so times go backwards -> error
    r = _run_cli([*reversed(pats), "-o", tmp_path / "bad.d3plot", "-q"])
    assert r.returncode == 1
    assert "BOXIMPA20.vtk" in r.stderr and "BOXIMPA10.vtk" in r.stderr
    assert "not strictly increasing" in r.stderr


# ----------------------------------------------------------------------
# #36 (c11): default title for a relative input
# ----------------------------------------------------------------------


def _title(path: Path) -> str:
    from lasso.dyna import D3plot

    return D3plot(str(path)).header.title


@pytest.mark.parametrize(
    ("cwd_rel", "arg"),
    [
        ("model_x", "vtk"),
        (os.path.join("model_x", "vtk"), "."),
        (os.path.join("model_x", "vtk", "deeper"), ".."),
    ],
)
def test_c11_relative_input_title(tmp_path, cwd_rel, arg):
    """Bug report 2026-10-11 #36 (c11): relative input gives the same default title as absolute."""
    vtk = tmp_path / "model_x" / "vtk"
    (vtk / "deeper").mkdir(parents=True)
    for i in range(3):
        shutil.copy(PORT_DIR / f"BOXIMPA{i:03d}.vtk", vtk)

    r_abs = _run_cli([vtk, "-o", tmp_path / "abs.d3plot", "-q"])
    assert r_abs.returncode == 0, r_abs.stderr
    assert _title(tmp_path / "abs.d3plot") == "model_x"

    r_rel = _run_cli([arg, "-o", tmp_path / "rel.d3plot", "-q"], cwd=tmp_path / cwd_rel)
    assert r_rel.returncode == 0, r_rel.stderr
    assert _title(tmp_path / "rel.d3plot") == "model_x"


# ----------------------------------------------------------------------
# #37 (c43): one file per state, the promises must say so
# ----------------------------------------------------------------------


def _family_names(folder: Path, base: str) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.name.startswith(base))


def test_c43_cli_message_matches_files(tmp_path):
    """Bug report 2026-10-11 #37 (c43): the CLI message names the base file and
    every state file, and that is exactly what is on disk."""
    out = tmp_path / "out.d3plot"
    r = _run_cli([PORT_DIR, "-o", out])
    assert r.returncode == 0, r.stderr
    wrote = [ln for ln in r.stdout.splitlines() if ln.startswith("Wrote")]
    assert wrote == [f"Wrote {out} + {N_PORT} state files {out}01..{out}{N_PORT}"]
    expected = ["out.d3plot"] + [f"out.d3plot{i:02d}" for i in range(1, N_PORT + 1)]
    assert _family_names(tmp_path, "out.d3plot") == expected


def test_c43_report_output_files_and_rerun(tmp_path):
    """Bug report 2026-10-11 #37 (c43): ConversionReport.output_files lists the
    written family; a re-run with fewer states leaves no stale state files."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    out = tmp_path / "box.d3plot"
    rep = convert(PORT_DIR, out)
    names21 = ["box.d3plot"] + [f"box.d3plot{i:02d}" for i in range(1, N_PORT + 1)]
    assert [Path(f).name for f in rep.output_files] == names21
    assert all(Path(f).parent == tmp_path for f in rep.output_files)
    assert _family_names(tmp_path, "box.d3plot") == names21
    assert rep.to_dict()["output_files"] == rep.output_files

    src5 = tmp_path / "five"
    src5.mkdir()
    for i in range(5):
        shutil.copy(PORT_DIR / f"BOXIMPA{i:03d}.vtk", src5)
    rep5 = convert(src5, out)
    names5 = ["box.d3plot"] + [f"box.d3plot{i:02d}" for i in range(1, 6)]
    assert [Path(f).name for f in rep5.output_files] == names5
    assert _family_names(tmp_path, "box.d3plot") == names5
    ts = D3plot(str(out)).arrays[ArrayType.global_timesteps]
    assert ts.tolist() == pytest.approx(_port_times()[:5], rel=1e-12)


def test_c43_cli_help_states_one_file_per_state():
    """Bug report 2026-10-11 #37 (c43): the -o help no longer promises a two-file family."""
    r = _run_cli(["--help"])
    assert r.returncode == 0
    flat = " ".join(r.stdout.split())
    assert "one state file per state" in flat
    assert "<name>.d3plot + <name>.d3plot01)" not in flat


# ----------------------------------------------------------------------
# #38 (c10): family regex matches Unicode digits
# ----------------------------------------------------------------------


def test_c10_unicode_digit_file_survives(tmp_path):
    """Bug report 2026-10-11 #38 (c10): 'box.d3plot' + full-width digits is not a family member."""
    from vtk2d3plot.api import convert

    foreign = tmp_path / "box.d3plot０１"
    foreign.write_bytes(b"user file")
    arabic = tmp_path / "box.d3plot٣"  # ARABIC-INDIC DIGIT THREE
    arabic.write_bytes(b"user file 2")
    stale = tmp_path / "box.d3plot77"
    stale.write_bytes(b"stale ascii member")

    convert(PORT_DIR, tmp_path / "box.d3plot")

    assert foreign.read_bytes() == b"user file"
    assert arabic.read_bytes() == b"user file 2"
    assert not stale.exists()  # ASCII-digit members are still cleaned up
