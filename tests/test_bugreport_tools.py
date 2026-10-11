"""Bug report 2026-10-11, cluster G: golden manifest, reference verifiers, lspp_check.

Report: docs/reviews/2026-10-11-external-bug-report.md, sections 21-25 and 40.
"""

from __future__ import annotations

import hashlib
import locale
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import (
    DATA,
    GOLDEN_W3_D3,
    GOLDEN_W3_VTK,
    GOLDEN_ZUG_D3,
    GOLDEN_ZUG_VTK,
    PORT_DIR,
    REPO,
)

pytestmark = pytest.mark.bugreport

TOOLS = REPO / "tools"
W3_STEM = "W3_SETUP_Ball_impact__ton-mm-s"
W3_D3 = GOLDEN_W3_D3 / f"{W3_STEM}.d3plot"
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
# #21 (c08) / #22 (c39): golden manifest
# ================================================================


def _is_text_fixture(f: Path, data: bytes) -> bool:
    # Independent of the tool: git's own heuristic (no NUL) plus the d3plot family.
    return b"\x00" not in data and not re.search(r"\.d3plot\d*$", f.name)


def _fake_repo(dst: Path, eol: str | None) -> Path:
    """Copy tools/golden_manifest.py and tests/data/{golden,port_dialect} to ``dst``.

    ``eol`` = "lf" / "crlf" rewrites every text fixture with that line ending;
    None keeps the bytes as checked out. Returns the copied data root.
    """
    (dst / "tools").mkdir(parents=True)
    shutil.copy(TOOLS / "golden_manifest.py", dst / "tools" / "golden_manifest.py")
    data = dst / "tests" / "data"
    for sub in ("golden", "port_dialect"):
        shutil.copytree(DATA / sub, data / sub)
    if eol is not None:
        for f in data.rglob("*"):
            if not f.is_file() or f.name == "MANIFEST.sha256":
                continue
            raw = f.read_bytes()
            if not _is_text_fixture(f, raw):
                continue
            lf = raw.replace(b"\r\n", b"\n")
            f.write_bytes(lf if eol == "lf" else lf.replace(b"\n", b"\r\n"))
    return data


def _verify(repo: Path, *extra):
    return _run([repo / "tools" / "golden_manifest.py", "verify", *extra], cwd=str(repo))


def test_c08_manifest_independent_of_line_endings(tmp_path):
    """Bug report 2026-10-11 #21 (c08): manifest verify must pass on LF and CRLF checkouts."""
    results = {}
    for eol in ("lf", "crlf"):
        repo = tmp_path / eol
        data = _fake_repo(repo, eol)
        vtk = data / "golden" / "zug_test3_RS" / "vtk" / "zug_test3_RSA002.vtk"
        assert (b"\r\n" in vtk.read_bytes()) == (eol == "crlf")
        results[eol] = _verify(repo)
    for eol, r in results.items():
        assert r.returncode == 0, f"{eol}: {r.stdout[-2000:]}"
        assert "OK: all 88 golden files match manifest" in r.stdout, eol


def test_c08_manifest_detects_one_byte_change(tmp_path):
    """Bug report 2026-10-11 #21 (c08): a one-byte content change must still fail verify."""
    repo = tmp_path / "lf"
    data = _fake_repo(repo, "lf")
    vtk = data / "golden" / "zug_test3_RS" / "vtk" / "zug_test3_RSA002.vtk"
    raw = bytearray(vtk.read_bytes())
    pos = raw.index(b"2.00014")
    raw[pos] = ord("3")  # TIME 2.00014 -> 3.00014, same size
    vtk.write_bytes(bytes(raw))
    d3 = data / "golden" / "zug_test3_RS" / "d3plot" / "zug_test3_RS.d3plot01"
    raw = bytearray(d3.read_bytes())
    raw[-1] ^= 0x01
    d3.write_bytes(bytes(raw))
    r = _verify(repo)
    assert r.returncode == 1, r.stdout
    assert "CHANGED: golden/zug_test3_RS/vtk/zug_test3_RSA002.vtk" in r.stdout
    assert "CHANGED: golden/zug_test3_RS/d3plot/zug_test3_RS.d3plot01" in r.stdout
    assert "FAILED: 2 mismatches found." in r.stdout


def test_c08_manifest_root_option(tmp_path):
    """Bug report 2026-10-11 #21 (c08): --root / --manifest select the tree that is verified.

    The real tool (whose default root is the real tests/data) is run on a CRLF
    copy. The copy is then made to differ from the real tree, so a tool that
    ignored --root (or --manifest) would still print OK and fail this test.
    """
    data = _fake_repo(tmp_path / "crlf", "crlf")
    tool = TOOLS / "golden_manifest.py"
    r = _run([tool, "verify", "--root", data])
    assert r.returncode == 0, r.stdout[-2000:]
    assert "OK: all 88 golden files match manifest" in r.stdout
    # one byte changed in the COPY only: --root must see it
    vtk = data / "golden" / "zug_test3_RS" / "vtk" / "zug_test3_RSA002.vtk"
    orig = vtk.read_bytes()
    raw = bytearray(orig)
    raw[raw.index(b"2.00014")] = ord("3")
    vtk.write_bytes(bytes(raw))
    r = _run([tool, "verify", "--root", data])
    assert r.returncode == 1, r.stdout[-2000:]
    assert "CHANGED: golden/zug_test3_RS/vtk/zug_test3_RSA002.vtk" in r.stdout
    assert "FAILED: 1 mismatches found." in r.stdout
    # the real tree is untouched and still verifies without --root
    r = _run([tool, "verify"])
    assert r.returncode == 0, r.stdout[-2000:]
    # --manifest: a manifest outside the tree, cut to 10 entries -> 78 NEW files
    lines = (data / "golden" / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines(True)
    short = tmp_path / "short.sha256"
    short.write_text("".join([ln for ln in lines if not ln.startswith("#")][:10]), "utf-8")
    vtk.write_bytes(orig)
    r = _run([tool, "verify", "--root", data, "--manifest", short])
    assert r.returncode == 1, r.stdout[-2000:]
    assert "FAILED: 78 mismatches found." in r.stdout


def test_c39_manifest_new_file_fails(tmp_path):
    """Bug report 2026-10-11 #22 (c39): a file not in the manifest must fail verify."""
    repo = tmp_path / "r"
    data = _fake_repo(repo, None)
    vdir = data / "golden" / "zug_test3_RS" / "vtk"
    shutil.copy(vdir / "zug_test3_RSA003.vtk", vdir / "zug_test3_RSA999.vtk")
    r = _verify(repo)
    assert r.returncode == 1, r.stdout
    assert "golden/zug_test3_RS/vtk/zug_test3_RSA999.vtk" in r.stdout
    assert "FAILED: 1 mismatches found." in r.stdout


def test_c39_manifest_truncated_fails(tmp_path):
    """Bug report 2026-10-11 #22 (c39): a manifest cut to 10 entries must fail verify."""
    repo = tmp_path / "r"
    data = _fake_repo(repo, None)
    man = data / "golden" / "MANIFEST.sha256"
    lines = man.read_text(encoding="utf-8").splitlines(keepends=True)
    entries = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
    assert len(entries) == 88
    header = [ln for ln in lines if ln.startswith("#")]
    man.write_text("".join(header + entries[:10]), encoding="utf-8")
    r = _verify(repo)
    assert r.returncode == 1, r.stdout
    assert "NEW (78 files not in manifest)" in r.stdout
    assert "FAILED: 78 mismatches found." in r.stdout


# ================================================================
# #23 (c21) / #24 (c22+c41): reference verifiers
# ================================================================

_STATE_RE = re.compile(r"^state\s+(\d+) (\S+\.vtk) .*max\|dxyz\|=(\S+)", re.MULTILINE)


def _states(stdout: str) -> list[tuple[int, str, float]]:
    return [(int(i), name, float(v)) for i, name, v in _STATE_RE.findall(stdout)]


def test_c21_verify_byid_zero_node_id(tmp_path):
    """Bug report 2026-10-11 #23 (c21): verify_byid runs on zug (one zero NODE_ID)."""
    r = _run([TOOLS / "reference" / "verify_byid.py", ZUG_D3, GOLDEN_ZUG_VTK])
    assert r.returncode == 0, r.stdout + r.stderr[-2000:]
    assert "zero NODE_IDs renumbered: 1" in r.stdout
    # The renumbered ID (max+1 = 11659) is exactly the Vortex golden's ID set.
    assert "same id set        : True" in r.stdout
    st = _states(r.stdout)
    assert [(i, n) for i, n, _ in st] == [
        (0, "zug_test3_RSA001.vtk"),
        (1, "zug_test3_RSA002.vtk"),
        (2, "zug_test3_RSA003.vtk"),
    ]
    # ASCII print precision of anim_to_vtk vs the float32 golden: about 5e-4.
    assert max(v for *_, v in st) < 1.0e-3
    assert max(v for *_, v in st) > 1.0e-5


def _renamed_copy(src_dir: Path, stem: str, numbers: dict[int, int], dst: Path) -> Path:
    dst.mkdir(parents=True)
    for old, new in numbers.items():
        shutil.copy(src_dir / f"{stem}A{old:03d}.vtk", dst / f"{stem}A{new}.vtk")
    return dst


def _convert_cli(vdir: Path, out: Path) -> None:
    r = _run(["-m", "vtk2d3plot.cli", vdir, "-o", out])
    assert r.returncode == 0, r.stderr[-2000:]


def test_c22_verify_zug_unpadded_names(tmp_path):
    """Bug report 2026-10-11 #24 (c22): verify_zug pairs A8/A9/A10 in numeric order."""
    vdir = _renamed_copy(GOLDEN_ZUG_VTK, "zug_test3_RS", {1: 8, 2: 9, 3: 10}, tmp_path / "vtk")
    out = tmp_path / "zug.d3plot"
    _convert_cli(vdir, out)
    r = _run([TOOLS / "reference" / "verify_zug.py", out, vdir])
    assert r.returncode == 0, r.stdout + r.stderr[-2000:]
    st = _states(r.stdout)
    assert [(i, n) for i, n, _ in st] == [
        (0, "zug_test3_RSA8.vtk"),
        (1, "zug_test3_RSA9.vtk"),
        (2, "zug_test3_RSA10.vtk"),
    ]
    assert [v for *_, v in st] == [0.0, 0.0, 0.0]


def test_c41_verify_byid_unpadded_names(tmp_path):
    """Bug report 2026-10-11 #24 (c41): verify_byid pairs A1..A11 in numeric order."""
    vdir = _renamed_copy(GOLDEN_W3_VTK, W3_STEM, {i: i for i in range(1, 12)}, tmp_path / "vtk")
    out = tmp_path / "w3.d3plot"
    _convert_cli(vdir, out)
    r = _run([TOOLS / "reference" / "verify_byid.py", out, vdir])
    assert r.returncode == 0, r.stdout + r.stderr[-2000:]
    st = _states(r.stdout)
    assert [(i, n) for i, n, _ in st] == [(i, f"{W3_STEM}A{i + 1}.vtk") for i in range(11)]
    assert [v for *_, v in st] == [0.0] * 11


@pytest.mark.parametrize("tool", ["verify_zug.py", "verify_byid.py"])
def test_c22_verifiers_fail_loudly_on_time_mismatch(tmp_path, tool):
    """Bug report 2026-10-11 #24 (c22+c41): a VTK/state pairing whose times differ exits 2."""
    # A001 <-> A003 swapped: numeric order no longer follows time.
    vdir = _renamed_copy(GOLDEN_ZUG_VTK, "zug_test3_RS", {1: 3, 2: 2, 3: 1}, tmp_path / "vtk")
    r = _run([TOOLS / "reference" / tool, ZUG_D3, vdir])
    assert r.returncode == 2, r.stdout + r.stderr[-2000:]
    assert r.stdout.count("ERROR: time mismatch") == 2
    assert "ERROR: 2 state(s) could not be paired" in r.stdout


@pytest.mark.parametrize("tool", ["verify_zug.py", "verify_byid.py"])
def test_c22_verifiers_fail_on_count_mismatch(tmp_path, tool):
    """Bug report 2026-10-11 #24 (c22+c41): fewer VTK files than d3plot states exits 2."""
    vdir = _renamed_copy(GOLDEN_ZUG_VTK, "zug_test3_RS", {1: 1, 2: 2}, tmp_path / "vtk")
    r = _run([TOOLS / "reference" / tool, ZUG_D3, vdir])
    assert r.returncode == 2, r.stdout + r.stderr[-2000:]
    assert "ERROR: 2 VTK files but 3 d3plot states" in r.stdout


PORT_STEM = "BOXIMP"
_TVTK_RE = re.compile(r"^state\s+\d+ \S+\.vtk t_vtk=(\S+)", re.MULTILINE)


def _port_d3plot(tmp_path: Path) -> Path:
    out = tmp_path / "box.d3plot"
    _convert_cli(PORT_DIR, out)
    return out


def _port_title_times() -> list[float]:
    times = []
    for i in range(21):
        title = (PORT_DIR / f"{PORT_STEM}A{i:03d}.vtk").read_text().splitlines()[1]
        times.append(float(title.split("t=")[1]))
    return times


def test_c22_verify_byid_port_title_time(tmp_path):
    """Bug report 2026-10-11 #24 (c22): port files (time only in the title) are time-checked."""
    out = _port_d3plot(tmp_path)
    r = _run([TOOLS / "reference" / "verify_byid.py", out, PORT_DIR])
    assert r.returncode == 0, r.stdout + r.stderr[-2000:]
    assert [float(v) for v in _TVTK_RE.findall(r.stdout)] == _port_title_times()
    assert "not time-checked" not in r.stdout
    # A001 <-> A003 swapped: the title times now disagree with the d3plot states.
    vdir = _renamed_copy(PORT_DIR, PORT_STEM, {i: i for i in range(21)}, tmp_path / "sw")
    a1, a3 = vdir / f"{PORT_STEM}A1.vtk", vdir / f"{PORT_STEM}A3.vtk"
    b1, b3 = a1.read_bytes(), a3.read_bytes()
    a1.write_bytes(b3)
    a3.write_bytes(b1)
    r = _run([TOOLS / "reference" / "verify_byid.py", out, vdir])
    assert r.returncode == 2, r.stdout + r.stderr[-2000:]
    assert r.stdout.count("ERROR: time mismatch") == 2
    assert "ERROR: 2 state(s) could not be paired" in r.stdout


def test_c22_verify_byid_warns_without_time(tmp_path):
    """Bug report 2026-10-11 #24 (c22): no TIME in the VTK -> a WARNING, not a silent skip."""
    out = _port_d3plot(tmp_path)
    vdir = tmp_path / "notime"
    vdir.mkdir()
    for i in range(21):
        name = f"{PORT_STEM}A{i:03d}.vtk"
        lines = (PORT_DIR / name).read_text().splitlines(True)
        lines[1] = "pyradioss state\n"
        (vdir / name).write_text("".join(lines))
    r = _run([TOOLS / "reference" / "verify_byid.py", out, vdir])
    assert r.returncode == 0, r.stdout + r.stderr[-2000:]
    assert _TVTK_RE.findall(r.stdout) == ["None"] * 21
    assert "WARNING: 21 state(s) have no TIME in the VTK file - pairing not time-checked" in (
        r.stdout
    )


def test_c22_verify_zug_node_id_changes_between_states(tmp_path):
    """Bug report 2026-10-11 #24 (c22): verify_zug checks the NODE_ID array of every state."""
    vdir = _renamed_copy(GOLDEN_ZUG_VTK, "zug_test3_RS", {1: 1, 2: 2, 3: 3}, tmp_path / "vtk")
    f = vdir / "zug_test3_RSA2.vtk"
    lines = f.read_text().splitlines(True)
    k = next(i for i, ln in enumerate(lines) if ln.startswith("SCALARS NODE_ID")) + 2
    assert lines[k : k + 2] == ["10876\n", "11332\n"]
    lines[k], lines[k + 1] = lines[k + 1], lines[k]  # two node IDs swapped in state 2 only
    f.write_text("".join(lines))
    r = _run([TOOLS / "reference" / "verify_zug.py", ZUG_D3, vdir])
    assert r.returncode == 2, r.stdout + r.stderr[-2000:]
    assert "ERROR: NODE_ID array of zug_test3_RSA2.vtk differs from zug_test3_RSA1.vtk" in (
        r.stdout
    )
    assert [n for _, n, _ in _states(r.stdout)] == ["zug_test3_RSA1.vtk", "zug_test3_RSA3.vtk"]


def test_c22_verify_zug_needs_node_ids(tmp_path):
    """Bug report 2026-10-11 #24 (c22): verify_zug on VTK without NODE_ID exits 2 with ERROR."""
    out = _port_d3plot(tmp_path)
    r = _run([TOOLS / "reference" / "verify_zug.py", out, PORT_DIR])
    assert r.returncode == 2, r.stdout + r.stderr[-2000:]
    assert "ERROR: BOXIMPA000.vtk has no NODE_ID array" in r.stdout
    assert "Traceback" not in r.stderr


# ================================================================
# #25 (c20) / #40 (c40): lspp_check
# ================================================================


def _stub_lspp(tmp_path: Path, monkeypatch) -> Path:
    """A fake LS-PrePost that copies its command file to <stub dir>/recorded.cfile."""
    sdir = tmp_path / "stub"
    sdir.mkdir()
    bat = sdir / "fake_lspp.bat"
    # LS-PrePost is called as `exe c=<cfile>`; cmd splits at '=', so the path is %2.
    bat.write_text(
        "@echo off\r\n"
        'copy /y "%2" "%~dp0recorded.cfile" >nul\r\n'
        "echo Finished reading model> lspost.msg\r\n"
        "echo Total number of states = 11>> lspost.msg\r\n",
        encoding="ascii",
    )
    monkeypatch.setenv("LSPP_EXE", str(bat))
    return sdir / "recorded.cfile"


def _cfile_lines(recorded: Path) -> list[str]:
    return recorded.read_text(encoding=locale.getencoding()).splitlines()


@pytest.mark.skipif(os.name != "nt", reason="stub LS-PrePost is a .bat file")
def test_c20_state_without_fringe_stub(tmp_path, monkeypatch):
    """Bug report 2026-10-11 #25 (c20): `state N;` is written without --fringe."""
    from tools.lspp_check import check_d3plot

    recorded = _stub_lspp(tmp_path, monkeypatch)
    r = check_d3plot(W3_D3, png=tmp_path / "x.png", state=5)
    assert r["loaded"] is True and r["n_states"] == 11
    lines = _cfile_lines(recorded)
    assert lines[:3] == [f'open d3plot "{W3_D3.resolve()}"', "ac", "state 5;"]
    assert not any(ln.startswith("fringe") for ln in lines)
    assert lines[-1] == "exit"
    # CLI path, same effect
    recorded.unlink()
    p = _run([TOOLS / "lspp_check.py", W3_D3, "--png", tmp_path / "y.png", "--state", "7"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert "state 7;" in _cfile_lines(recorded)
    # with a fringe and no state, the last state (11) is still chosen
    recorded.unlink()
    check_d3plot(W3_D3, fringe="vonmises")
    assert "state 11;" in _cfile_lines(recorded)


@pytest.mark.lspp
def test_c20_state_without_fringe_lsprepost(tmp_path, monkeypatch):
    """Bug report 2026-10-11 #25 (c20): real LS-PrePost renders the requested state."""
    from tools.lspp_check import check_d3plot

    monkeypatch.delenv("LSPP_EXE", raising=False)
    digests = {}
    for tag, st in (("s1", 1), ("s1_again", 1), ("s11", 11)):
        png = tmp_path / f"{tag}.png"
        r = check_d3plot(W3_D3, png=png, state=st)
        assert r["loaded"], r["detail"]
        assert png.stat().st_size > 5000
        digests[tag] = hashlib.sha256(png.read_bytes()).hexdigest()
    assert digests["s1"] == digests["s1_again"]  # rendering is deterministic
    assert digests["s1"] != digests["s11"]  # ... so state 11 really is another state


def _parse_quoted(line: str) -> str:
    return line.split('"')[1]


@pytest.mark.skipif(os.name != "nt", reason="stub LS-PrePost is a .bat file")
def test_c40_non_ascii_paths_stub(tmp_path, monkeypatch):
    """Bug report 2026-10-11 #40 (c40): a 'ü' in the d3plot/PNG folder must not crash."""
    from tools.lspp_check import check_d3plot

    recorded = _stub_lspp(tmp_path, monkeypatch)
    d = tmp_path / "Prüfung"
    d.mkdir()
    d3 = d / W3_D3.name
    shutil.copy(W3_D3, d3)
    r = check_d3plot(d3, png=d / "x.png")
    assert r["loaded"] is True
    lines = _cfile_lines(recorded)
    opened = _parse_quoted(lines[0])
    printed = _parse_quoted(next(ln for ln in lines if ln.startswith("print png")))
    # The command file names the very same files (long name or 8.3 short form).
    assert os.path.samefile(opened, d3)
    assert Path(opened).name == d3.name  # family files <name>01.. must stay findable
    assert os.path.samefile(Path(printed).parent, d)
    assert Path(printed).name == "x.png"
    assert r["path_notes"] and "Prüfung" in r["detail"]


@pytest.mark.skipif(os.name != "nt", reason="stub LS-PrePost is a .bat file")
def test_c40_unencodable_path_stub(tmp_path, monkeypatch):
    """Bug report 2026-10-11 #40 (c40): a path outside the ANSI code page gives a clear detail."""
    from tools.lspp_check import check_d3plot

    recorded = _stub_lspp(tmp_path, monkeypatch)
    d = tmp_path / "Ωmega"  # Greek capital omega: not in cp1252
    d.mkdir()
    d3 = d / W3_D3.name
    shutil.copy(W3_D3, d3)
    r = check_d3plot(d3)
    opened = _parse_quoted(_cfile_lines(recorded)[0])
    if Path(opened).exists():  # an ASCII 8.3 short folder was available
        assert os.path.samefile(opened, d3)
    else:
        assert "?mega" in opened
        assert "cannot be written to the command file" in r["detail"]


@pytest.mark.lspp
def test_c40_non_ascii_path_lsprepost(tmp_path, monkeypatch):
    """Bug report 2026-10-11 #40 (c40): real LS-PrePost loads and renders from a 'ü' folder."""
    from tools.lspp_check import check_d3plot

    monkeypatch.delenv("LSPP_EXE", raising=False)
    d = tmp_path / "Prüfung"
    d.mkdir()
    for f in GOLDEN_W3_D3.iterdir():
        shutil.copy(f, d / f.name)
    png = d / "w3.png"
    r = check_d3plot(d / W3_D3.name, fringe="vonmises", png=png)
    assert r["loaded"], r["detail"]
    assert r["n_states"] == 11
    assert png.stat().st_size > 5000


@pytest.mark.skipif(os.name != "nt", reason="stub LS-PrePost is a .bat file")
def test_c40_ansi_cfile_when_no_short_path(tmp_path, monkeypatch):
    """Bug report 2026-10-11 #40 (c40): no 8.3 short folder -> the cfile carries the
    'ü' in the ANSI code page (not ASCII) and the result detail warns about it."""
    import tools.lspp_check as lc

    enc = locale.getencoding()
    try:
        "ü".encode(enc)
    except UnicodeEncodeError:
        pytest.skip(f"ANSI code page {enc} cannot encode 'ü'")
    recorded = _stub_lspp(tmp_path, monkeypatch)
    monkeypatch.setattr(lc, "_short_path", lambda p: None)
    d = tmp_path / "Prüfung"
    d.mkdir()
    d3 = d / W3_D3.name
    shutil.copy(W3_D3, d3)
    r = lc.check_d3plot(d3, png=d / "x.png", state=2)
    raw = recorded.read_bytes()
    assert f'open d3plot "{d3.resolve()}"'.encode(enc) in raw
    assert "Prüfung".encode(enc) in raw and not raw.isascii()
    assert r["loaded"] is True  # the stub "loaded" it; real LS-PrePost 4.13 may not
    assert len(r["path_notes"]) == 2  # d3plot and PNG path
    assert "LS-PrePost may fail on it" in r["detail"]
    assert f"written in {enc}" in r["detail"]


@pytest.mark.skipif(os.name != "nt", reason="stub LS-PrePost is a .bat file")
def test_c40_missing_file_result_shape(tmp_path, monkeypatch):
    """Bug report 2026-10-11 #40 (c40): the 'no such file' result has the same keys
    (cfile, path_notes) as a normal result."""
    from tools.lspp_check import check_d3plot

    recorded = _stub_lspp(tmp_path, monkeypatch)
    ok = check_d3plot(W3_D3)
    missing = check_d3plot(tmp_path / "nope.d3plot")
    assert missing["loaded"] is False
    assert missing["cfile"] is None and missing["path_notes"] == []
    assert missing["detail"].startswith("no such file:")
    assert set(missing) == set(ok)
    assert recorded.exists()  # only from the first call
