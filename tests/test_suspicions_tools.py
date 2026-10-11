"""Bug report 2026-10-11, "Suspicions not checked", cluster E: developer tools.

Leads S15-S17 (golden_manifest.py), S18 / S27 (lspp_check.py), S20 / S28
(reference verifiers), S21 / S22 (reference d3plot builders).
Report: docs/reviews/2026-10-11-external-bug-report.md, "Suspicions not checked".
"""

from __future__ import annotations

import hashlib
import locale
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from conftest import GOLDEN_W3_D3, GOLDEN_ZUG_D3, GOLDEN_ZUG_VTK, PORT_DIR, REPO

pytestmark = pytest.mark.suspicions

TOOLS = REPO / "tools"
W3_D3 = GOLDEN_W3_D3 / "W3_SETUP_Ball_impact__ton-mm-s.d3plot"
ZUG_D3 = GOLDEN_ZUG_D3 / "zug_test3_RS.d3plot"


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO / "src"), env.get("PYTHONPATH", "")) if p
    )
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(args, **kw):
    return subprocess.run(
        [sys.executable, *map(str, args)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_env(),
        check=False,
        **kw,
    )


# ================================================================
# S15 / S16 / S17: tools/golden_manifest.py
# ================================================================


def _small_tree(root: Path, names=("golden/a.txt", "golden/b.txt", "port_dialect/c.txt")) -> Path:
    for i, name in enumerate(names):
        f = root / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(f"fixture {i}\n".encode())
    (root / "golden").mkdir(parents=True, exist_ok=True)
    return root


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("exc", [OSError, KeyboardInterrupt])
def test_s15_generate_failure_keeps_old_manifest(tmp_path, monkeypatch, capsys, exc):
    """S15: an error part-way through `generate` must leave the old manifest intact."""
    import tools.golden_manifest as gm

    root = _small_tree(tmp_path / "data")
    assert gm.main(["generate", "--root", str(root)]) == 0
    man = root / "golden" / gm.MANIFEST_NAME
    old = man.read_bytes()
    assert len([ln for ln in old.splitlines() if not ln.startswith(b"#")]) == 3
    listing = sorted(p.name for p in (root / "golden").iterdir())

    real = gm.hash_and_size
    calls = []

    def flaky(path):
        calls.append(path)
        if len(calls) == 2:
            raise exc("simulated failure on the 2nd file")
        return real(path)

    monkeypatch.setattr(gm, "hash_and_size", flaky)
    with pytest.raises(exc):
        gm.main(["generate", "--root", str(root)])
    assert len(calls) == 2
    assert man.read_bytes() == old  # not truncated, not partial
    assert sorted(p.name for p in (root / "golden").iterdir()) == listing  # no temp left
    monkeypatch.setattr(gm, "hash_and_size", real)
    capsys.readouterr()
    assert gm.main(["verify", "--root", str(root)]) == 0
    assert "OK: all 3 golden files match manifest" in capsys.readouterr().out


DS_NAME = "golden/a  b.txt"  # a path with a double space


@pytest.mark.parametrize("legacy", [True, False], ids=["legacy", "current"])
def test_s16_verify_double_space_path(tmp_path, capsys, legacy):
    """S16: `<sha>  <path>` (legacy) and `<sha>  <size>  <path>` lines whose path holds
    a double space must parse exactly - no crash, the right file checked."""
    import tools.golden_manifest as gm

    root = _small_tree(tmp_path / "data", names=(DS_NAME, "port_dialect/c.txt"))
    lines = ["# hand-written manifest\n"]
    for name in (DS_NAME, "port_dialect/c.txt"):
        data = (root / name).read_bytes()
        mid = "" if legacy else f"{len(data)}  "
        lines.append(f"{_sha(data)}  {mid}{name}\n")
    man = tmp_path / "m.sha256"
    man.write_text("".join(lines), encoding="utf-8", newline="\n")
    rc = gm.main(["verify", "--root", str(root), "--manifest", str(man)])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "OK: all 2 golden files match manifest" in out
    # a content change in the double-space file is reported under its exact path
    (root / DS_NAME).write_bytes(b"fixture X\n")  # same size
    rc = gm.main(["verify", "--root", str(root), "--manifest", str(man)])
    out = capsys.readouterr().out
    assert rc == 1
    assert f"  CHANGED: {DS_NAME}\n" in out
    assert "MISSING" not in out and "NEW" not in out
    assert "FAILED: 1 mismatches found." in out


def test_s16_verify_tolerates_stray_outer_whitespace(tmp_path, capsys):
    """S16 (verifier round 1): stray spaces around a hand-edited line are ignored, as
    before the fix - only inner double spaces belong to the path."""
    import tools.golden_manifest as gm

    root = _small_tree(tmp_path / "data", names=(DS_NAME, "port_dialect/c.txt"))
    lines = []
    for name in (DS_NAME, "port_dialect/c.txt"):
        data = (root / name).read_bytes()
        lines.append(f"  {_sha(data)}  {len(data)}  {name}   \n")
    man = tmp_path / "m.sha256"
    man.write_text("".join(lines), encoding="utf-8", newline="\n")
    assert gm.main(["verify", "--root", str(root), "--manifest", str(man)]) == 0
    assert "OK: all 2 golden files match manifest" in capsys.readouterr().out


def test_s16_generate_verify_roundtrip_double_space(tmp_path, capsys):
    """S16: generate + verify round-trip on a tree with a double-space file name."""
    import tools.golden_manifest as gm

    root = _small_tree(tmp_path / "data", names=(DS_NAME, "golden/x  y  z.txt"))
    assert gm.main(["generate", "--root", str(root)]) == 0
    assert gm.main(["verify", "--root", str(root)]) == 0
    assert "OK: all 2 golden files match manifest" in capsys.readouterr().out


def test_s17_changed_file_counted_once(tmp_path, capsys):
    """S17: a file whose content AND size changed is one mismatch (both lines printed)."""
    import tools.golden_manifest as gm

    root = _small_tree(tmp_path / "data")
    assert gm.main(["generate", "--root", str(root)]) == 0
    (root / "golden" / "b.txt").write_bytes(b"fixture 1 - longer now\n")
    capsys.readouterr()
    rc = gm.main(["verify", "--root", str(root)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "  CHANGED: golden/b.txt\n" in out
    assert "  SIZE: golden/b.txt (expected 10, got 23)\n" in out
    assert "FAILED: 1 mismatches found." in out
    # a second, independent change counts as a second mismatch
    (root / "port_dialect" / "c.txt").write_bytes(b"fixture 9\n")
    assert gm.main(["verify", "--root", str(root)]) == 1
    assert "FAILED: 2 mismatches found." in capsys.readouterr().out


# ================================================================
# S18 / S27: tools/lspp_check.py
# ================================================================


def _stub_lspp(tmp_path: Path, monkeypatch, stdout_lines: tuple[str, ...] = ()) -> Path:
    """A fake LS-PrePost that copies its command file to <stub dir>/recorded.cfile.

    ``stdout_lines`` are echoed to stdout (where the real LS-PrePost prints
    "d3plot files are 64-bit ieee format").
    """
    sdir = tmp_path / "stub"
    sdir.mkdir()
    bat = sdir / "fake_lspp.bat"
    echoes = "".join(f"echo {ln}\r\n" for ln in stdout_lines)
    # LS-PrePost is called as `exe c=<cfile>`; cmd splits at '=', so the path is %2.
    bat.write_text(
        "@echo off\r\n"
        'copy /y "%2" "%~dp0recorded.cfile" >nul\r\n'
        + echoes
        + "echo Finished reading model> lspost.msg\r\n"
        "echo Total number of states = 11>> lspost.msg\r\n",
        encoding="ascii",
    )
    monkeypatch.setenv("LSPP_EXE", str(bat))
    return sdir / "recorded.cfile"


def _cfile_lines(recorded: Path) -> list[str]:
    return recorded.read_text(encoding=locale.getencoding()).splitlines()


@pytest.mark.skipif(os.name != "nt", reason="stub LS-PrePost is a .bat file")
def test_s18_unreadable_state_count_is_reported(tmp_path, monkeypatch):
    """S18: when lasso cannot count the states, the fringe falls back to state 1 -
    and the result must say so (notes + detail), with the usual keys."""
    from tools.lspp_check import check_d3plot

    # The W3 golden is a 64-bit family: the stub prints the line real LS-PrePost prints.
    recorded = _stub_lspp(tmp_path, monkeypatch, (" d3plot files are 64-bit ieee format",))
    bad = tmp_path / "bad.d3plot"
    bad.write_bytes(b"\x07" * 4096)  # not a d3plot: lasso raises
    r = check_d3plot(bad, fringe="vonmises")
    assert "state 1;" in _cfile_lines(recorded)
    notes = r["notes"]
    assert len(notes) == 1, notes
    assert "state count" in notes[0] and "state 1" in notes[0] and "not the last" in notes[0]
    assert notes[0] in r["detail"]
    # keys stay stable: same set as a normal result and as the 'no such file' result
    ok = check_d3plot(W3_D3, fringe="vonmises")
    assert ok["notes"] == [] and "state 11;" in _cfile_lines(recorded)
    assert set(r) == set(ok) == set(check_d3plot(tmp_path / "nope.d3plot"))


def _port_family(tmp_path: Path, f64: bool) -> Path:
    from vtk2d3plot import convert

    out = tmp_path / ("f64" if f64 else "f32") / "box.d3plot"
    out.parent.mkdir(parents=True)
    convert(PORT_DIR, out, f64=f64)
    return out


@pytest.mark.skipif(os.name != "nt", reason="stub LS-PrePost is a .bat file")
@pytest.mark.parametrize(
    ("f64", "stdout_lines", "expected", "n_notes"),
    [
        (False, (), 4, 0),  # LS-PrePost 4.13 prints nothing for a 32-bit family
        (True, (" d3plot files are 64-bit ieee format",), 8, 0),
        (True, (), 8, 1),  # header says 64-bit, LS-PrePost said nothing: noted
    ],
    ids=["f32", "f64", "f64-silent"],
)
def test_s27_wordsize_stub(tmp_path, monkeypatch, f64, stdout_lines, expected, n_notes):
    """S27: the word size is 4 for a 32-bit family (LS-PrePost prints no line for it)."""
    from tools.lspp_check import check_d3plot

    d3 = _port_family(tmp_path, f64)
    _stub_lspp(tmp_path, monkeypatch, stdout_lines)
    r = check_d3plot(d3)
    assert r["loaded"] is True
    assert r["wordsize"] == expected
    assert len(r["notes"]) == n_notes, r["notes"]


@pytest.mark.lspp
@pytest.mark.parametrize(("f64", "expected"), [(False, 4), (True, 8)], ids=["f32", "f64"])
def test_s27_wordsize_lsprepost(tmp_path, monkeypatch, f64, expected):
    """S27: real LS-PrePost 4.13 on a 32-bit and a 64-bit family gives wordsize 4 / 8."""
    from tools.lspp_check import check_d3plot

    monkeypatch.delenv("LSPP_EXE", raising=False)
    d3 = _port_family(tmp_path, f64)
    r = check_d3plot(d3)
    assert r["loaded"], r["detail"]
    assert r["n_states"] == 21
    assert r["wordsize"] == expected
    assert r["notes"] == []


# ================================================================
# S20 / S28: tools/reference/verify_zug.py, verify_byid.py
# ================================================================

_ZERO_RE = re.compile(r"^\s+zero-ID (\S+\.vtk) ids=\[(\d+)\] max\|dxyz\|=(\S+)", re.MULTILINE)


def test_s20_verify_zug_compares_zero_id_node():
    """S20: the renumbered zero-ID node is compared too, and reported on its own line."""
    r = _run([TOOLS / "reference" / "verify_zug.py", ZUG_D3, GOLDEN_ZUG_VTK])
    assert r.returncode == 0, r.stdout + r.stderr[-2000:]
    zero = _ZERO_RE.findall(r.stdout)
    assert [(n, int(i)) for n, i, _ in zero] == [
        ("zug_test3_RSA001.vtk", 11659),
        ("zug_test3_RSA002.vtk", 11659),
        ("zug_test3_RSA003.vtk", 11659),
    ]
    # anim_to_vtk ASCII precision vs the float32 golden: below 1e-3
    assert max(float(v) for *_, v in zero) < 1.0e-3


def test_s20_verify_zug_zero_id_node_wrong_position(tmp_path):
    """S20: a moved zero-ID node is caught (its error is large), the others stay small."""
    vdir = tmp_path / "vtk"
    shutil.copytree(GOLDEN_ZUG_VTK, vdir)
    for f in sorted(vdir.glob("*.vtk")):
        lines = f.read_text().splitlines(True)
        k = next(i for i, ln in enumerate(lines) if ln.startswith("SCALARS NODE_ID")) + 2
        ids = [int(lines[k + j]) for j in range(987)]
        z = ids.index(0)
        p = next(i for i, ln in enumerate(lines) if ln.startswith("POINTS")) + 1
        flat = " ".join(lines[p : p + 987]).split()
        assert len(flat) >= 3 * (z + 1)
        # Move the zero-ID point by +5 in x (rewrite the POINTS block one point per line).
        pts = [flat[3 * j : 3 * j + 3] for j in range(987)]
        pts[z][0] = repr(float(pts[z][0]) + 5.0)
        end = p
        n_tok = 0
        while n_tok < 3 * 987:
            n_tok += len(lines[end].split())
            end += 1
        lines[p:end] = [" ".join(pt) + "\n" for pt in pts]
        f.write_text("".join(lines))
    r = _run([TOOLS / "reference" / "verify_zug.py", ZUG_D3, vdir])
    zero = _ZERO_RE.findall(r.stdout)
    assert len(zero) == 3, r.stdout + r.stderr[-2000:]
    assert all(4.9 < float(v) < 5.1 for *_, v in zero)
    st = re.findall(r"^state\s+\d+ \S+\.vtk .*max\|dxyz\|=(\S+)", r.stdout, re.MULTILINE)
    assert len(st) == 3 and max(float(v) for v in st) < 1.0e-3


@pytest.mark.parametrize("tool", ["verify_zug.py", "verify_byid.py"])
def test_s28_verifiers_other_stem_is_an_error(tmp_path, tool):
    """S28: a .vtk of another series in the folder gives ERROR + exit 2, not a traceback."""
    vdir = tmp_path / "vtk"
    shutil.copytree(GOLDEN_ZUG_VTK, vdir)
    shutil.copy(vdir / "zug_test3_RSA002.vtk", vdir / "copyA002.vtk")
    r = _run([TOOLS / "reference" / tool, ZUG_D3, vdir])
    assert r.returncode == 2, r.stdout + r.stderr[-2000:]
    assert "Traceback" not in r.stderr
    assert "ERROR: Multiple stems found: copy, zug_test3_RS" in r.stdout


# ================================================================
# S21 / S22: tools/reference/make_d3plot.py, make_mixed.py
# ================================================================


@pytest.mark.parametrize(
    ("script", "stem"), [("make_d3plot.py", "cube"), ("make_mixed.py", "mixed")]
)
def test_s21_s22_reference_output_folder_argument(tmp_path, script, stem):
    """S21/S22: the output folder is a command-line argument (no hard-coded user path)."""
    out = tmp_path / "out"
    r = _run([TOOLS / "reference" / script, out])
    assert r.returncode == 0, r.stdout + r.stderr[-2000:]
    assert "OVERALL: PASS" in r.stdout
    assert (out / f"{stem}.d3plot").is_file() and (out / f"{stem}.d3plot01").is_file()
    assert f"writing -> {out / f'{stem}.d3plot'}" in r.stdout


@pytest.mark.parametrize(
    ("script", "stem"), [("make_d3plot.py", "cube"), ("make_mixed.py", "mixed")]
)
def test_s21_s22_reference_default_is_fresh_temp_folder(tmp_path, script, stem):
    """S21/S22: without an argument the script writes into a fresh temporary folder."""
    r = _run([TOOLS / "reference" / script], cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr[-2000:]
    m = re.search(r"^writing -> (.+)$", r.stdout, re.MULTILINE)
    assert m, r.stdout
    written = Path(m.group(1).strip())
    fresh = written.parent.name.startswith(f"{Path(script).stem}_")
    try:
        assert written.name == f"{stem}.d3plot" and written.is_file()
        assert "c8098c03" not in str(written)  # not the old session's scratch path
        assert fresh, written  # a fresh mkdtemp folder named after the script
        assert written.parent.parent.resolve() == Path(tempfile.gettempdir()).resolve()
        assert sorted(p.name for p in written.parent.iterdir()) == [
            f"{stem}.d3plot",
            f"{stem}.d3plot01",
        ]
    finally:
        if fresh:  # never delete a folder this test did not see created
            shutil.rmtree(written.parent, ignore_errors=True)
