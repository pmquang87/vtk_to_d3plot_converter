"""Milestone 1 — VTK reader + series loader tests.

Tests from docs/plan.md M1.1 plus two bonus robustness tests
(multi-stem error, non-monotonic time error) per Q&A decisions.
"""

import shutil

import numpy as np
import pytest
from conftest import GOLDEN_W3_VTK, PORT_DIR

pytestmark = pytest.mark.milestone1


# ── reader: port dialect ────────────────────────────────────────────────────


def test_read_port_state0():
    from vtk2d3plot.vtk_reader import read_vtk

    s = read_vtk(PORT_DIR / "BOXIMPA000.vtk")
    assert s.points.shape == (220, 3)
    assert s.time == pytest.approx(2.902412789e-04)
    assert len(s.cells) == 200 and set(s.cell_types) == {9}
    assert s.point_data["DISPLACEMENT"].shape == (220, 3)
    assert s.cell_data["OFF"].shape == (200,)


# ── reader: official dialect ────────────────────────────────────────────────


def test_read_official_state():
    from vtk2d3plot.vtk_reader import read_vtk

    s = read_vtk(min(GOLDEN_W3_VTK.glob("*.vtk")))
    assert s.time is not None  # from FIELD TIME
    assert "NODE_ID" in s.point_data
    assert "PART_ID" in s.cell_data and "ELEMENT_ID" in s.cell_data
    assert any(k.startswith("3DELEM_Stress") for k in s.cell_data)
    t = next(v for k, v in s.cell_data.items() if k.startswith("3DELEM_Stress"))
    assert t.ndim == 3 and t.shape[1:] == (3, 3)  # TENSORS as (n,3,3)


# ── series loader ───────────────────────────────────────────────────────────


def test_series_port():
    from vtk2d3plot.series import load_series

    ser = load_series(PORT_DIR)
    assert ser.dialect == "port" and len(ser.states) == 21
    assert np.all(np.diff(ser.times) > 0)


# ── cross-check against meshio ──────────────────────────────────────────────


def test_reader_matches_meshio():
    meshio = pytest.importorskip("meshio")
    from vtk2d3plot.vtk_reader import read_vtk

    p = PORT_DIR / "BOXIMPA000.vtk"
    ours, ref = read_vtk(p), meshio.read(p)
    assert np.allclose(ours.points, ref.points)


# ── bonus: multi-stem directory raises ──────────────────────────────────────


def test_series_multi_stem_error(tmp_path):
    """A directory with two different VTK run stems must raise, not merge."""
    from vtk2d3plot.series import load_series

    # Create two dummy VTK files with different stems
    for name in ("run1A000.vtk", "run2A000.vtk"):
        p = tmp_path / name
        p.write_text(
            "# vtk DataFile Version 3.0\n"
            "dummy\n"
            "ASCII\n"
            "DATASET UNSTRUCTURED_GRID\n"
            "POINTS 0 float\n"
            "CELLS 0 0\n"
            "CELL_TYPES 0\n",
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match=r"run1.*run2|run2.*run1"):
        load_series(tmp_path)


# ── bonus: non-monotonic time raises ────────────────────────────────────────


def test_series_non_monotonic_time_error(tmp_path):
    """Swapped time ordering must be caught even when filenames sort correctly."""
    from vtk2d3plot.series import load_series

    # Copy two port-dialect files and rename so numeric order disagrees with time
    src0 = PORT_DIR / "BOXIMPA000.vtk"
    src1 = PORT_DIR / "BOXIMPA020.vtk"  # t=20 > t=0
    # Put the later-time file under a *lower* numeric suffix
    shutil.copy(src1, tmp_path / "TESTA000.vtk")
    shutil.copy(src0, tmp_path / "TESTA001.vtk")
    with pytest.raises(Exception, match=r"(?i)monoton|decreasing|order"):
        load_series(tmp_path)
