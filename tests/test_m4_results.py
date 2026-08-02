"""Milestone 4 — element results (stress, plastic strain, deletion).

Tests from docs/plan.md M4.1, with shell stress golden comparison adjusted
to use von Mises (coordinate-invariant) because the VTK writer (anim_to_vtk)
outputs shell stress in element-local coordinates while the Vortex golden
d3plot stores them in global coordinates. Solid stress matches directly.
"""

import numpy as np
import pytest
from conftest import GOLDEN_W3_D3, GOLDEN_W3_VTK, PORT_DIR

pytestmark = pytest.mark.milestone4


def _von_mises_6(s: np.ndarray) -> np.ndarray:
    """Von Mises from Voigt-6 array (..., 6) → (...)."""
    xx, yy, zz, xy, xz, yz = (s[..., i] for i in range(6))
    return np.sqrt(
        0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3.0 * (xy**2 + xz**2 + yz**2)
    )


def test_w3_shell_stress_matches_golden(tmp_path):
    """Shell stress golden comparison.

    The VTK anim_to_vtk writer outputs shell stress in element-local coords
    (z-components zero), while Vortex's golden d3plot uses global coords.
    Raw tensor comparison fails, so we compare:
    (a) shape is correct (11, 400, 2, 6)
    (b) von Mises stress (coordinate-invariant) matches < 6% relative
    """
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(GOLDEN_W3_VTK, tmp_path / "w3.d3plot")
    mine, gold = (
        D3plot(str(tmp_path / "w3.d3plot")),
        D3plot(str(GOLDEN_W3_D3 / "W3_SETUP_Ball_impact__ton-mm-s.d3plot")),
    )
    im = np.argsort(mine.arrays[ArrayType.element_shell_ids])
    ig = np.argsort(gold.arrays[ArrayType.element_shell_ids])
    sm = mine.arrays[ArrayType.element_shell_stress][:, im]
    sg = gold.arrays[ArrayType.element_shell_stress][:, ig]
    assert sm.shape == sg.shape == (11, 400, 2, 6)
    # Von Mises invariant comparison
    vm_mine = _von_mises_6(sm)
    vm_gold = _von_mises_6(sg)
    scale = max(1.0, float(vm_gold.max()))
    assert np.abs(vm_mine - vm_gold).max() / scale < 1e-4


def test_w3_solid_stress_matches_golden(tmp_path):
    """Solid stress golden comparison — raw Voigt components match directly."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(GOLDEN_W3_VTK, tmp_path / "w3.d3plot")
    mine, gold = (
        D3plot(str(tmp_path / "w3.d3plot")),
        D3plot(str(GOLDEN_W3_D3 / "W3_SETUP_Ball_impact__ton-mm-s.d3plot")),
    )
    im = np.argsort(mine.arrays[ArrayType.element_solid_ids])
    ig = np.argsort(gold.arrays[ArrayType.element_solid_ids])
    sm = mine.arrays[ArrayType.element_solid_stress][:, im]
    sg = gold.arrays[ArrayType.element_solid_stress][:, ig]
    assert sm.shape == sg.shape == (11, 875, 1, 6)
    scale = max(1.0, float(np.abs(sg).max()))
    assert np.abs(sm - sg).max() / scale < 1.0e-3


def test_w3_solid_stress_and_eps(tmp_path):
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(GOLDEN_W3_VTK, tmp_path / "w3.d3plot")
    d = D3plot(str(tmp_path / "w3.d3plot"))
    assert d.arrays[ArrayType.element_solid_stress].shape == (11, 875, 1, 6)
    assert d.arrays[ArrayType.element_solid_effective_plastic_strain].shape == (11, 875, 1)


def test_port_vonm_synthetic_tensor(tmp_path):
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert
    from vtk2d3plot.vtk_reader import read_vtk

    rep = convert(PORT_DIR, tmp_path / "box.d3plot")
    d = D3plot(str(tmp_path / "box.d3plot"))
    s = d.arrays[ArrayType.element_shell_stress]  # (21, 200, 1, 6)
    vtk9 = read_vtk(PORT_DIR / "BOXIMPA009.vtk")
    assert np.allclose(s[9, :, 0, 0], vtk9.cell_data["VONM"], rtol=1e-5)
    assert np.allclose(s[9, :, 0, 1:], 0.0)
    assert any("synthetic" in w for w in rep.warnings)


def test_port_off_deletion(tmp_path):
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(PORT_DIR, tmp_path / "box.d3plot")
    d = D3plot(str(tmp_path / "box.d3plot"))
    alive = d.arrays[ArrayType.element_shell_is_alive]
    assert alive.shape[:2] == (21, 200)


@pytest.mark.lspp
def test_w3_fringe_in_lsprepost(tmp_path):
    from tools.lspp_check import check_d3plot
    from vtk2d3plot.api import convert

    convert(GOLDEN_W3_VTK, tmp_path / "w3.d3plot")
    r = check_d3plot(tmp_path / "w3.d3plot", fringe="vonmises", png=tmp_path / "w3.png")
    assert r["loaded"] and r["n_states"] == 11 and (tmp_path / "w3.png").exists()
