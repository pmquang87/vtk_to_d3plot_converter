"""Bug report 2026-10-11, "Suspicions not checked", cluster C: public API and CLI.

Leads: S11 (partial family after a failed move), S13 + (d) (warning codes from
substring tests), S14 (user ``.bak`` files deleted), S25 (progress reported
twice), S26 (a glob matching nothing dropped silently) and (c) (default title
for glob / file-list / single-file / stem / .pvd inputs).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import GOLDEN_W3_VTK, PORT_DIR, REPO

pytestmark = pytest.mark.suspicions

N_PORT = 21  # states in the box_beam_impact port fixture


def _run_cli(args, cwd=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO / "src"), env.get("PYTHONPATH", "")) if p
    )
    return subprocess.run(
        [sys.executable, "-m", "vtk2d3plot.cli", *[str(a) for a in args]],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
        check=False,
    )


def _snapshot(folder: Path) -> dict[str, bytes | None]:
    """Relative name -> content of every entry below *folder* (None for folders)."""
    return {
        str(p.relative_to(folder)): (p.read_bytes() if p.is_file() else None)
        for p in sorted(folder.rglob("*"))
    }


def _port_subset(dst: Path, n: int = 3, stem: str = "BOXIMP") -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        shutil.copy(PORT_DIR / f"BOXIMPA{i:03d}.vtk", dst / f"{stem}A{i:03d}.vtk")
    return dst


def _title(path: Path) -> str:
    from lasso.dyna import D3plot

    return D3plot(str(path)).header.title.strip()


# ── synthetic official-dialect VTK writer (hex + quad) ────────────────

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
    ],
    dtype=float,
)
_CELLS = [(12, [0, 1, 2, 3, 4, 5, 6, 7]), (9, [8, 9, 10, 11])]


def _write_vtk(path: Path, *, time: float, cell_scalars: dict[str, list[float]]) -> None:
    """One official-dialect legacy VTK state: a hex (part 1) and a quad (part 2)."""
    n_pts, n_cells = len(_POINTS), len(_CELLS)
    lines = [
        "# vtk DataFile Version 3.0",
        "suspicions cluster C synthetic state",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        "FIELD FieldData 1",
        "TIME 1 1 double",
        f"{time!r}",
        f"POINTS {n_pts} float",
    ]
    lines += [" ".join(repr(float(v)) for v in row) for row in _POINTS]
    size = sum(len(c) + 1 for _, c in _CELLS)
    lines.append(f"CELLS {n_cells} {size}")
    lines += [" ".join(str(v) for v in [len(c), *c]) for _, c in _CELLS]
    lines.append(f"CELL_TYPES {n_cells}")
    lines += [str(t) for t, _ in _CELLS]
    lines.append(f"POINT_DATA {n_pts}")
    lines += ["SCALARS NODE_ID int 1", "LOOKUP_TABLE default"]
    lines += [str(i + 1) for i in range(n_pts)]
    lines.append("VECTORS Displacement float")
    lines += ["0.0 0.0 0.0"] * n_pts
    lines.append(f"CELL_DATA {n_cells}")
    lines += ["SCALARS ELEMENT_ID int 1", "LOOKUP_TABLE default", "100", "101"]
    lines += ["SCALARS PART_ID int 1", "LOOKUP_TABLE default", "1", "2"]
    for name, vals in cell_scalars.items():
        lines += [f"SCALARS {name} float 1", "LOOKUP_TABLE default"]
        lines += [repr(float(v)) for v in vals]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _synthetic_series(folder: Path, per_state: list[dict[str, list[float]]]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i, scalars in enumerate(per_state):
        _write_vtk(folder / f"synA{i:03d}.vtk", time=0.001 * (i + 1), cell_scalars=scalars)
    return folder


def _codes(rep) -> dict[str, str]:
    """message -> code value of a ConversionReport."""
    assert [w.message for w in rep.structured_warnings] == rep.warnings
    return {w.message: w.code.value for w in rep.structured_warnings}


def _code_of(rep, prefix: str) -> str:
    hits = [(m, c) for m, c in _codes(rep).items() if m.startswith(prefix)]
    assert len(hits) == 1, (prefix, rep.warnings)
    return hits[0][1]


# ======================================================================
# S11: a failed move must not leave a partial family behind
# ======================================================================


def _fail_move_to(monkeypatch, final_name: str) -> None:
    """Make shutil.move raise when it moves a file onto *final_name*."""
    real_move = shutil.move

    def fake_move(src, dst, *args, **kwargs):
        if Path(dst).name == final_name:
            raise OSError(f"injected failure moving onto {final_name}")
        return real_move(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "move", fake_move)


def test_s11_failed_move_without_old_family_leaves_nothing(tmp_path, monkeypatch):
    """S11: no family existed; a move onto the LAST family name fails. The folder
    must look exactly as before: no geometry file, no state files, no temp files."""
    from vtk2d3plot import convert

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "notes.txt").write_bytes(b"user file")
    before = _snapshot(out_dir)

    _fail_move_to(monkeypatch, f"box.d3plot{N_PORT}")
    with pytest.raises(OSError, match="injected failure"):
        convert(PORT_DIR, out_dir / "box.d3plot")

    assert _snapshot(out_dir) == before


def test_s11_failed_move_restores_old_family(tmp_path, monkeypatch):
    """S11: a family existed (incl. a stale high member); after a failed move the
    old family is back byte for byte and nothing else is left."""
    from vtk2d3plot import convert

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    for name in ("box.d3plot", "box.d3plot01", "box.d3plot30"):
        (out_dir / name).write_bytes(f"old {name}".encode())
    (out_dir / "notes.txt").write_bytes(b"user file")
    before = _snapshot(out_dir)

    _fail_move_to(monkeypatch, f"box.d3plot{N_PORT}")
    with pytest.raises(OSError, match="injected failure"):
        convert(PORT_DIR, out_dir / "box.d3plot")

    assert _snapshot(out_dir) == before


def test_s11_failure_while_backing_up_restores_everything(tmp_path, monkeypatch):
    """S11: a move of an OLD family file into the backup folder fails (2nd one)."""
    from vtk2d3plot import convert

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    for name in ("box.d3plot", "box.d3plot01", "box.d3plot02"):
        (out_dir / name).write_bytes(f"old {name}".encode())
    before = _snapshot(out_dir)
    real_move = shutil.move
    calls = []

    def fake_move(src, dst, *args, **kwargs):
        if Path(src).parent == out_dir:  # an old member leaving the output folder
            calls.append(src)
            if len(calls) == 2:
                raise OSError("injected failure while backing up")
        return real_move(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "move", fake_move)
    with pytest.raises(OSError, match="injected failure"):
        convert(PORT_DIR, out_dir / "box.d3plot")
    assert _snapshot(out_dir) == before


def test_s11_failure_inside_writer_changes_nothing(tmp_path, monkeypatch):
    """S11: lasso fails while writing; the old family and the folder are untouched,
    and an output folder this run created is removed again."""
    from lasso.dyna import D3plot

    from vtk2d3plot import convert

    def broken_write(self, *args, **kwargs):
        raise RuntimeError("injected writer failure")

    monkeypatch.setattr(D3plot, "write_d3plot", broken_write)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "box.d3plot").write_bytes(b"old geometry")
    before = _snapshot(tmp_path)
    with pytest.raises(RuntimeError, match="injected writer failure"):
        convert(PORT_DIR, out_dir / "box.d3plot")
    with pytest.raises(RuntimeError, match="injected writer failure"):
        convert(PORT_DIR, tmp_path / "new1" / "new2" / "box.d3plot")
    assert _snapshot(tmp_path) == before


def test_s11_restore_failure_keeps_backup_and_says_where(tmp_path, monkeypatch):
    """S11: when an old family file cannot be moved back, it stays in the private
    folder (byte-identical) and the exception names that folder and file."""
    from vtk2d3plot import convert

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    old = {n: f"old {n}".encode() for n in ("box.d3plot", "box.d3plot01", "box.d3plot02")}
    for name, data in old.items():
        (out_dir / name).write_bytes(data)
    real_move = shutil.move

    def fake_move(src, dst, *args, **kwargs):
        s, d = Path(src), Path(dst)
        if d.parent == out_dir and d.name == f"box.d3plot{N_PORT}":
            raise OSError("injected failure moving onto the last state file")
        if d == out_dir / "box.d3plot01" and s.parent.name == "old":
            raise OSError("injected failure moving the backup back")
        return real_move(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "move", fake_move)
    with pytest.raises(OSError, match="last state file") as info:
        convert(PORT_DIR, out_dir / "box.d3plot")

    kept = [p for p in out_dir.iterdir() if p.is_dir()]
    assert len(kept) == 1 and kept[0].name.startswith("_vtk2d3plot_"), kept
    assert (kept[0] / "old" / "box.d3plot01").read_bytes() == old["box.d3plot01"]
    notes = " ".join(getattr(info.value, "__notes__", []))
    assert str(kept[0] / "old") in notes and "box.d3plot01" in notes, notes
    files = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    assert files == ["box.d3plot", "box.d3plot02"]
    assert all((out_dir / n).read_bytes() == old[n] for n in files)


def test_s11_unremovable_new_file_is_named(tmp_path, monkeypatch):
    """S11: a new family file that cannot be removed during the roll-back is named
    in an exception note (nothing is left behind silently)."""
    from vtk2d3plot import convert

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _fail_move_to(monkeypatch, "box.d3plot03")
    real_unlink = Path.unlink

    def fake_unlink(self, *args, **kwargs):
        if self.name == "box.d3plot01":
            raise PermissionError("injected: file is locked")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fake_unlink)
    with pytest.raises(OSError, match="injected failure") as info:
        convert(PORT_DIR, out_dir / "box.d3plot")
    notes = " ".join(getattr(info.value, "__notes__", []))
    assert str(out_dir / "box.d3plot01") in notes, notes
    assert sorted(p.name for p in out_dir.iterdir()) == ["box.d3plot01"]


def test_s11_read_only_old_member_leaves_no_scratch(tmp_path):
    """S11/S14: a read-only old family member is replaced and the private scratch
    folder is still removed after a successful run."""
    import stat

    from vtk2d3plot import convert

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stale = out_dir / "box.d3plot30"
    stale.write_bytes(b"old read-only state")
    stale.chmod(stat.S_IREAD)
    try:
        convert(PORT_DIR, out_dir / "box.d3plot")
        family = ["box.d3plot"] + [f"box.d3plot{i:02d}" for i in range(1, N_PORT + 1)]
        assert sorted(p.name for p in out_dir.iterdir()) == family
    finally:
        for p in out_dir.rglob("*"):
            p.chmod(stat.S_IWRITE | stat.S_IREAD)


# ======================================================================
# S14: backups must never use a name that can be a user's file
# ======================================================================


def _family_and_user_bak(out_dir: Path) -> None:
    out_dir.mkdir()
    for name in ("box.d3plot", "box.d3plot01", "box.d3plot40"):
        (out_dir / name).write_bytes(f"old {name}".encode())
    for name in ("box.d3plot.bak", "box.d3plot01.bak", "box.d3plot40.bak"):
        (out_dir / name).write_bytes(f"USER {name}".encode())
    (out_dir / "notes.txt").write_bytes(b"user file")


def test_s14_user_bak_files_survive_success(tmp_path):
    """S14: '<family file>.bak' files belong to the user; a successful run keeps
    them unchanged and leaves no backup or temp entry of its own."""
    from vtk2d3plot import convert

    out_dir = tmp_path / "out"
    _family_and_user_bak(out_dir)
    user = {n: (out_dir / n).read_bytes() for n in ("box.d3plot.bak", "box.d3plot01.bak")}
    user["box.d3plot40.bak"] = (out_dir / "box.d3plot40.bak").read_bytes()
    user["notes.txt"] = b"user file"

    rep = convert(PORT_DIR, out_dir / "box.d3plot")

    family = ["box.d3plot"] + [f"box.d3plot{i:02d}" for i in range(1, N_PORT + 1)]
    assert sorted(p.name for p in out_dir.iterdir()) == sorted(family + list(user))
    for name, data in user.items():
        assert (out_dir / name).read_bytes() == data, name
    assert [Path(f).name for f in rep.output_files] == family


def test_s14_user_bak_files_survive_failure(tmp_path, monkeypatch):
    """S14: after a failed run the user's '.bak' files and the old family are
    unchanged and nothing else is left."""
    from vtk2d3plot import convert

    out_dir = tmp_path / "out"
    _family_and_user_bak(out_dir)
    before = _snapshot(out_dir)

    _fail_move_to(monkeypatch, "box.d3plot05")
    with pytest.raises(OSError, match="injected failure"):
        convert(PORT_DIR, out_dir / "box.d3plot")

    assert _snapshot(out_dir) == before


# ======================================================================
# S13 + (d): warning codes by message template, not by substring
# ======================================================================


def test_s13_erosion_status_source_is_not_forced_alive():
    """S13: 'element is_alive from EROSION_STATUS ...' only names the source of
    the deletion flags; nothing is forced alive (W3 golden, real warning)."""
    import tempfile

    from vtk2d3plot import convert

    with tempfile.TemporaryDirectory() as td:
        rep = convert(GOLDEN_W3_VTK, Path(td) / "w3.d3plot", title="w3")
    assert _code_of(rep, "element is_alive from EROSION_STATUS") == "ALIVE_SOURCE"


def test_s13_unmapped_membrane_name_is_unmapped(tmp_path):
    """S13: an unmapped array called 'X_membrane' made 'unmapped arrays ignored: ...'
    a STRAIN_MEMBRANE_IGNORED warning."""
    from vtk2d3plot import convert

    src = _synthetic_series(tmp_path / "src", [{"X_membrane": [1.0, 2.0]}] * 2)
    rep = convert(src, tmp_path / "o.d3plot")
    assert _code_of(rep, "unmapped arrays ignored: X_membrane") == "UNMAPPED_ARRAYS"


def test_s13_history_variable_zero_fill_is_zero_fill(tmp_path):
    """S13: 'element_solid_history_variables: zero-filled at states [...]' is a
    zero-fill, not the 'internal energy mapped to history variable' note."""
    from vtk2d3plot import convert

    e = {"3DELEM_Specific_Energy": [5.0, 0.0]}
    src = _synthetic_series(tmp_path / "src", [e, {}, e])
    rep = convert(src, tmp_path / "o.d3plot")
    assert _code_of(rep, "element_solid_history_variables: zero-filled at states [1]") == (
        "ZERO_FILL"
    )
    assert _code_of(rep, "solid internal energy mapped to element_solid_history_variables") == (
        "SOLID_INTERNAL_ENERGY_HISTVAR"
    )


def test_s13_alive_one_fill_is_not_forced_alive(tmp_path):
    """S13: 'element_shell_is_alive: one-filled (all-alive) at states [...] (source
    EROSION_STATUS array missing ...)' is a fill of a missing state, not the
    never-alive force-alive rule."""
    from vtk2d3plot import convert

    alive = {"EROSION_STATUS": [1.0, 1.0]}
    src = _synthetic_series(tmp_path / "src", [alive, {}, alive])
    rep = convert(src, tmp_path / "o.d3plot")
    assert _code_of(rep, "element_shell_is_alive: one-filled (all-alive) at states [1]") == (
        "ALIVE_FILL"
    )
    assert _code_of(rep, "element_solid_is_alive: one-filled (all-alive) at states [1]") == (
        "ALIVE_FILL"
    )


def test_s13_adversarial_user_text_in_messages(tmp_path):
    """(d): user-controlled text inside a message (array names, file names, titles,
    part names) never decides the code. Every name here contains words the old
    substring tests keyed on."""
    from vtk2d3plot import convert

    # file stem 'synthetic_erosion' (PVD file names, T01 file name) and a PVD
    # manifest whose timesteps differ from the files' own times
    src = tmp_path / "src"
    stem = "synthetic_erosion_local_frame"
    _synthetic_series(src, [{"erosion_forced_alive": [1.0, 1.0]}] * 2)
    for i in range(2):
        (src / f"synA{i:03d}.vtk").rename(src / f"{stem}A{i:03d}.vtk")
    (src / f"{stem}T01.csv").write_text("# comment\nTIME,IE,KE\n", encoding="utf-8")
    pvd = src / "series.pvd"
    pvd.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n'
        f'<DataSet timestep="0.5" file="{stem}A000.vtk"/>\n'
        f'<DataSet timestep="0.6" file="{stem}A001.vtk"/>\n'
        "</Collection>\n</VTKFile>\n",
        encoding="utf-8",
    )
    rep = convert(
        pvd,
        tmp_path / "o.d3plot",
        title="Erosion synthetic Größe – forced alive, zero-filled",
        part_names={1: "Größe synthetic membrane ignored", 2: "local frame erosion", 7: "x"},
    )
    expected = {
        "PVD manifest timestep 0.5 used for": "PVD_TIME_OVERRIDE",
        "PVD manifest timestep 0.6 used for": "PVD_TIME_OVERRIDE",
        f"T01 file {stem}T01.csv has no data rows": "T01_GLOBALS",
        "unmapped arrays ignored: erosion_forced_alive": "UNMAPPED_ARRAYS",
        "title: transliterated": "TITLE_SANITIZED",
        "title: replaced remaining non-ASCII": "TITLE_SANITIZED",
        "part 1 title: transliterated": "TITLE_SANITIZED",
        "part_names entries for part IDs not in the d3plot were ignored: 7": "PART_NAMES_UNUSED",
    }
    for prefix, code in expected.items():
        assert _code_of(rep, prefix) == code, (prefix, rep.warnings)


# Every warning template the package emits (one sample each), with user text
# chosen to contain the words of OTHER templates. A sample whose user part is
# fixed by the code (array names chosen by the builder) is the real text.
_ADV = "synthetic erosion forced alive local frame membrane ignored zero-filled title truncated"
TEMPLATE_SAMPLES = [
    ("No DISPLACEMENT array found; using state 0 coordinates as undeformed geometry", "NO_DISPLACEMENT"),
    ("solid stress tensor is synthetic-uniaxial from VONM", "SYNTHETIC_STRESS"),
    ("solid stress tensor is synthetic-uniaxial from 3DELEM_Von_Mises", "SYNTHETIC_STRESS"),
    ("shell stress tensor is synthetic-uniaxial from 2DELEM_Von_Mises", "SYNTHETIC_STRESS"),
    ("stress tensor is synthetic-uniaxial from VONM", "SYNTHETIC_STRESS"),
    ("solid internal energy mapped to element_solid_history_variables (index 0)", "SOLID_INTERNAL_ENERGY_HISTVAR"),
    ("shell stress in element-local coordinates (from anim_to_vtk); von Mises / principal invariants are exact, but raw component values differ from global frame", "SHELL_LOCAL_FRAME"),
    ("shell stress from 2DELEM_Stress_(membrane) — single layer, element-local coordinates; von Mises / principal invariants are exact", "SHELL_LOCAL_FRAME"),
    ("only 2DELEM_Stress_(upper) found (lower missing from all states); mapped to layer 1, element-local coordinates", "SHELL_LOCAL_FRAME"),
    ("only 2DELEM_Stress_(lower) found (upper missing from all states); mapped to layer 0, element-local coordinates", "SHELL_LOCAL_FRAME"),
    ("shell strain in element-local coordinates (from anim_to_vtk); principal invariants are exact, but raw component values differ from global frame", "SHELL_LOCAL_FRAME"),
    ("only 2DELEM_Strain_(upper) found (lower missing from all states); duplicated to both layers. shell strain in element-local coordinates (from anim_to_vtk); ...", "SHELL_LOCAL_FRAME"),
    ("only 2DELEM_Strain_(lower) found (upper missing from all states); duplicated to both layers. shell strain in element-local coordinates (from anim_to_vtk); ...", "SHELL_LOCAL_FRAME"),
    ("shell strain from 2DELEM_Strain_(membrane) — single layer duplicated to both layers. shell strain in element-local coordinates (from anim_to_vtk); ...", "SHELL_LOCAL_FRAME"),
    ("shell stress rotated from the element frame to the global frame", "SHELL_ROTATED"),
    ("shell stress from 2DELEM_Stress_(membrane) — single layer, rotated from the element frame to the global frame", "SHELL_ROTATED"),
    ("only 2DELEM_Stress_(upper) found (lower missing from all states); mapped to layer 1, rotated from the element frame to the global frame", "SHELL_ROTATED"),
    ("shell stress rotated with the OpenRadioss local frame, rebuilt from each state's deformed VTK coordinates (quads: ...); accuracy is limited by the VTK coordinate print precision", "SHELL_ROTATED"),
    ("rotate_shell_stress ignored: the shell stress is synthetic-uniaxial (no element frame to rotate from)", "ROTATION_IGNORED"),
    # merged from cluster D (suspicions (b) shell strain rotation, S01 layer wording)
    ("shell strain rotated from the element frame to the global frame", "SHELL_ROTATED"),
    ("only 2DELEM_Strain_(upper) found (lower missing from all states); duplicated to both layers. shell strain rotated from the element frame to the global frame", "SHELL_ROTATED"),
    ("only 2DELEM_Strain_(lower) found (upper missing from all states); duplicated to both layers. shell strain rotated from the element frame to the global frame", "SHELL_ROTATED"),
    ("shell strain from 2DELEM_Strain_(membrane) — single layer duplicated to both layers. shell strain rotated from the element frame to the global frame", "SHELL_ROTATED"),
    ("shell strain rotated with the OpenRadioss local frame, rebuilt from each state's deformed VTK coordinates (quads: ...); accuracy is limited by the VTK coordinate print precision; the strain is rotated as a tensor (shear stays tensorial)", "SHELL_ROTATED"),
    ("shell stress and strain rotated with the OpenRadioss local frame, rebuilt from each state's deformed VTK coordinates (quads: ...); the strain is rotated as a tensor (shear stays tensorial)", "SHELL_ROTATED"),
    ("rotate_shell_stress: the shell stress is synthetic-uniaxial (no element frame to rotate from) and is written unchanged; only the shell strain is rotated", "ROTATION_IGNORED"),
    ("3 shells have a degenerate local frame (zero-area or collapsed geometry) in at least one state (first: state 0); their rotated shell stress and strain are unreliable (zeroed or partly lost)", "DEGENERATE_SHELL_FRAMES"),
    ("shell element results have differing integration layer counts (stress=1, eps=1, strain=2) — reconciled to 2 layers (d3plot MAXINT constraint): shell stress from VONM copied to every layer; plastic strain from EPSP copied to every layer", "LAYER_MISMATCH"),
    ("3 shells have a degenerate local frame (zero-area or collapsed geometry) in at least one state (first: state 0); their rotated shell stress is unreliable (zeroed or partly lost)", "DEGENERATE_SHELL_FRAMES"),
    ("only 2DELEM_Plastic_Strain_Upper found (Lower missing from all states); mapped to layer 1", "SINGLE_LAYER_SOURCE"),
    ("only 2DELEM_Plastic_Strain_Lower found (Upper missing from all states); mapped to layer 0", "SINGLE_LAYER_SOURCE"),
    ("2DELEM_Strain_(membrane) ignored because upper/lower layers were also found", "STRAIN_MEMBRANE_IGNORED"),
    ("shell element results have differing integration layer counts (stress=1, eps=1, strain=2) — reconciled to 2 layers (d3plot MAXINT constraint)", "LAYER_MISMATCH"),
    ("both OFF and EROSION_STATUS present; using OFF (port dialect takes priority)", "ALIVE_SOURCE"),
    ("element is_alive from EROSION_STATUS (1=alive, 0=eroded); deleted elements will be hidden in LS-PrePost", "ALIVE_SOURCE"),
    ("EROSION_STATUS: 288 shell elements never alive in any state (parts [1, 4]); forced alive (rigid/non-integrated)", "EROSION_FORCED_ALIVE"),
    ("EROSION_STATUS: 2 solid elements return to 1 after being eroded (states [3]); kept eroded (no resurrection)", "EROSION_NO_RESURRECTION"),
    ("element_solid_history_variables: zero-filled at states [1, 2] (source array missing in those states)", "ZERO_FILL"),
    ("node_velocity: zero-filled at states [0] (source array missing in those states)", "ZERO_FILL"),
    ("element_shell_stress: 2DELEM_Stress_(upper) zero-filled at states [2] (source array missing in those states)", "ZERO_FILL"),
    ("element_shell_effective_plastic_strain: 2DELEM_Plastic_Strain_Lower zero-filled at states [0, 1] (source array missing in those states)", "ZERO_FILL"),
    ("element_solid_strain: zero-filled at all states (no solid strain source array; required because element strain sets the global d3plot ISTRN flag)", "ZERO_FILL"),
    ("element_solid_effective_plastic_strain: zero-filled at all states (no solid plastic-strain source array; ...)", "ZERO_FILL"),
    ("element_shell_is_alive: one-filled (all-alive) at states [1] (source EROSION_STATUS array missing in those states)", "ALIVE_FILL"),
    ("element_beam_is_alive: one-filled (all-alive) at states [1] (source OFF array missing in those states)", "ALIVE_FILL"),
    ("Masked 10 cells referencing zero-ID nodes (phantom geometry, matching Vortex behavior)", "PHANTOM_CELLS_MASKED"),
    ("known but unsupported arrays (no d3plot equivalent): Contact_Forces (absmax=1)", "KNOWN_UNSUPPORTED"),
    (f"unmapped arrays ignored: {_ADV.replace(' ', '_')} (absmax=1), X_membrane (absmax=2)", "UNMAPPED_ARRAYS"),
    (f"cell arrays '{_ADV}', '{_ADV.upper()}' differ only in case at states [0]; using 'x' (exact name), the others are ignored", "CASE_COLLISION"),
    (f"point arrays 'a', 'A' differ only in case at states [0]; using 'a' ({_ADV}), the others are ignored", "CASE_COLLISION"),
    ("Skipped 4 unsupported cells (VTK types: 1); these are not included in the d3plot", "SKIPPED_CELLS"),
    ("Fixed 2 tetras with negative volume by swapping nodes to match LS-DYNA winding order.", "INVERTED_TETRAS"),
    (f"title: transliterated non-ASCII chars ('{_ADV} ä' → '{_ADV} ae')", "TITLE_SANITIZED"),
    ("title: replaced remaining non-ASCII chars with '?'", "TITLE_SANITIZED"),
    ("title truncated from 99 to 72 chars (d3plot header: 72 readable chars)", "TITLE_SANITIZED"),
    (f"part 12 title: transliterated non-ASCII chars ('{_ADV} ä' → '{_ADV} ae')", "TITLE_SANITIZED"),
    ("part 12 title truncated from 99 to 72 chars (d3plot header: 72 readable chars)", "TITLE_SANITIZED"),
    ("part_names entries for part IDs not in the d3plot were ignored: 7, 9", "PART_NAMES_UNUSED"),
    (f"PVD manifest timestep 0.5 used for {_ADV}A000.vtk; it overrides the file's own time 0.4", "PVD_TIME_OVERRIDE"),
    (f"T01 file {_ADV}T01.csv: 2 malformed data row(s) skipped (wrong field count or non-numeric value; lines 7, 9)", "T01_GLOBALS"),
    (f"T01 file {_ADV}T01.csv has no data rows; global energies from it (internal, kinetic, total) are not written", "T01_GLOBALS"),
    ("T01 CSV has no data rows; global energies (internal, kinetic, total) not written", "T01_GLOBALS"),
    ("T01 CSV lacks the IE and/or KE column; global_total_energy not written", "T01_GLOBALS"),
    ("global_total_energy = IE+KE+HE+CE+DE from T01 (LS-DYNA convention: kinetic + internal + hourglass + contact + damping; EN numerical dissipation and EW external work excluded); T01 columns not written: none", "T01_GLOBALS"),
    # merged from cluster B (S02, S06)
    (f"T01 file {_ADV}T01.csv: 3 data row(s) superseded by later rows with an equal or earlier TIME (solver restart or duplicate output); the later rows are used", "T01_GLOBALS"),
    (f"Ignored 2 file(s) in the input folder without the <stem>A<number>.vtk/.vtu state name: {_ADV}.vtk, b.vtu", "IGNORED_INPUT_FILES"),
]  # fmt: skip


@pytest.mark.parametrize(("message", "code"), TEMPLATE_SAMPLES, ids=range(len(TEMPLATE_SAMPLES)))
def test_s13_template_table(message, code):
    """(d): every template gets its deliberate code and exactly one table row matches."""
    from vtk2d3plot.api import _WARNING_TEMPLATES, WarningCode, classify_warning

    assert classify_warning(message) is WarningCode(code)
    rows = [c for pattern, c in _WARNING_TEMPLATES if pattern.match(message)]
    assert rows == [WarningCode(code)], rows


@pytest.mark.parametrize("message", ["", "some new warning about synthetic erosion", _ADV])
def test_s13_unknown_message_is_other(message):
    """(d): a message that matches no template falls back to OTHER."""
    from vtk2d3plot.api import WarningCode, classify_warning

    assert classify_warning(message) is WarningCode.OTHER


def test_s13_real_fixture_warnings_all_have_a_code(tmp_path):
    """(d): the warnings of real conversions (rotated, with T01) never fall back to OTHER."""
    from conftest import GOLDEN_ZUG_VTK, TENSILE_DIR

    from vtk2d3plot import convert

    for i, src in enumerate((GOLDEN_ZUG_VTK, TENSILE_DIR, PORT_DIR)):
        rep = convert(src, tmp_path / f"o{i}.d3plot", rotate_shell_stress=True)
        others = [m for m, c in _codes(rep).items() if c == "OTHER"]
        assert not others, others


# ======================================================================
# S25: each state is reported once
# ======================================================================


def test_s25_progress_once_per_state(tmp_path):
    """S25: convert(progress=...) calls the callback once per state, in order."""
    from vtk2d3plot import convert

    calls: list[tuple[int, int]] = []
    convert(PORT_DIR, tmp_path / "box.d3plot", progress=lambda i, n: calls.append((i, n)))
    assert calls == [(i, N_PORT) for i in range(N_PORT)]


def test_s25_cli_prints_each_state_once(tmp_path):
    """S25: the CLI prints one 'Processing state i/N' line per state."""
    r = _run_cli([PORT_DIR, "-o", tmp_path / "box.d3plot"])
    assert r.returncode == 0, r.stderr
    lines = [ln.strip() for ln in r.stdout.splitlines() if "Processing state" in ln]
    assert lines == [f"Processing state {i}/{N_PORT}..." for i in range(1, N_PORT + 1)]


# ======================================================================
# S26: a glob that matches nothing is an error
# ======================================================================


def test_s26_empty_glob_next_to_matching_input(tmp_path):
    """S26: one pattern matches, the other does not -> exit 1 naming the pattern."""
    src = _port_subset(tmp_path / "src")
    empty = str(tmp_path / "missing" / "*.vtk")
    out = tmp_path / "o.d3plot"
    r = _run_cli([str(src / "*.vtk"), empty, "-o", out, "-q"])
    assert r.returncode == 1, r.stdout
    assert f"pattern '{empty}' matched no files" in r.stderr, r.stderr
    assert "Traceback" not in r.stderr
    assert not out.exists()


def test_s26_empty_glob_alone(tmp_path):
    """S26: a lone pattern that matches nothing gives the same error."""
    empty = str(tmp_path / "*.vtk")
    r = _run_cli([empty, "-o", tmp_path / "o.d3plot", "-q"])
    assert r.returncode == 1
    assert f"pattern '{empty}' matched no files" in r.stderr, r.stderr
    assert "Traceback" not in r.stderr


# ======================================================================
# (c): default title for every input form
# ======================================================================


def _pvd(folder: Path, names: list[str]) -> Path:
    from vtk2d3plot.vtk_reader import read_vtk

    rows = "".join(
        f'<DataSet timestep="{read_vtk(folder / n).time!r}" file="{n}"/>\n' for n in names
    )
    p = folder / "series.pvd"
    p.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n'
        f"{rows}</Collection>\n</VTKFile>\n",
        encoding="utf-8",
    )
    return p


@pytest.mark.parametrize("folder_rel", ["model_y/vtk", "case_z"])
@pytest.mark.parametrize("form", ["dir", "glob", "files", "single", "stem", "pvd"])
def test_c_default_title_every_input_form(tmp_path, form, folder_rel):
    """(c): without --title every input form gets the folder name as title
    ('vtk' -> its parent), like a directory input."""
    folder = _port_subset(tmp_path / folder_rel)
    names = sorted(p.name for p in folder.glob("*.vtk"))
    args = {
        "dir": [folder],
        "glob": [str(folder / "*.vtk")],
        "files": [folder / n for n in names],
        "single": [folder / names[0]],
        "stem": [folder / "BOXIMP"],
        "pvd": [_pvd(folder, names)] if form == "pvd" else [],
    }[form]
    out = tmp_path / "o.d3plot"
    r = _run_cli([*args, "-o", out, "-q"])
    assert r.returncode == 0, r.stderr
    assert _title(out) == ("model_y" if folder_rel == "model_y/vtk" else "case_z")


def test_c_files_from_two_folders_no_default_title(tmp_path):
    """(c): files from different folders share no folder -> no default title."""
    a = _port_subset(tmp_path / "a", n=1)
    b = tmp_path / "b"
    b.mkdir()
    shutil.copy(PORT_DIR / "BOXIMPA001.vtk", b)
    out = tmp_path / "o.d3plot"
    r = _run_cli([a / "BOXIMPA000.vtk", b / "BOXIMPA001.vtk", "-o", out, "-q"])
    assert r.returncode == 0, r.stderr
    assert _title(out) == ""


def test_c_explicit_title_wins(tmp_path):
    """(c): --title still overrides the folder name."""
    folder = _port_subset(tmp_path / "model_y" / "vtk")
    out = tmp_path / "o.d3plot"
    r = _run_cli([str(folder / "*.vtk"), "-o", out, "-q", "--title", "Mine"])
    assert r.returncode == 0, r.stderr
    assert _title(out) == "Mine"
