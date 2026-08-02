"""Milestone 5 — CLI + end-to-end integration + robustness."""

import shutil
import subprocess

import pytest
from conftest import PORT_DIR, REPO, TENSILE_DIR, VENV_PY

pytestmark = pytest.mark.milestone5


def test_cli_smoke(tmp_path):
    r = subprocess.run(
        [
            str(VENV_PY),
            "-m",
            "vtk2d3plot.cli",
            str(PORT_DIR),
            "-o",
            str(tmp_path / "box.d3plot"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "box.d3plot").exists() and (tmp_path / "box.d3plot01").exists()


def test_tensile_vtk_only_converts(tmp_path):
    # the run vortex-radioss CRASHES on — from VTK we must succeed
    from vtk2d3plot.api import convert

    rep = convert(TENSILE_DIR, tmp_path / "tensile.d3plot")
    assert rep.n_states == 20


@pytest.mark.pyradioss
def test_pyradioss_run_and_convert(tmp_path):
    # full loop: run the Python solver on a bundled example deck, convert its VTK
    pytest.importorskip("pyradioss")
    for f in (REPO / "tests" / "data" / "port_dialect" / "box_beam_impact").glob(
        "BOXIMP_000*.rad"
    ):
        shutil.copy(f, tmp_path)
    r1 = subprocess.run(
        [str(VENV_PY), "-m", "pyradioss.starter", "-i", "BOXIMP_0000.rad"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert r1.returncode == 0, r1.stderr[-2000:]
    r2 = subprocess.run(
        [str(VENV_PY), "-m", "pyradioss.engine", "-i", "BOXIMP_0001.rad"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert r2.returncode == 0, r2.stderr[-2000:]
    vtks = sorted(tmp_path.glob("BOXIMPA*.vtk"))
    assert vtks, "solver produced no VTK states"
    from vtk2d3plot.api import convert

    rep = convert(tmp_path, tmp_path / "box.d3plot")
    assert rep.n_states == len(vtks)
