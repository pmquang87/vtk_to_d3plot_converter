import numpy as np
import pytest

pytestmark = pytest.mark.milestone11


def test_tensile_has_shell_strain(tmp_path):
    """TENSILE fixture: shell strain tensor must be written."""
    from conftest import TENSILE_DIR
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(TENSILE_DIR, tmp_path / "tensile.d3plot")
    dp = D3plot(str(tmp_path / "tensile.d3plot"))
    assert ArrayType.element_shell_strain in dp.arrays
    strain = dp.arrays[ArrayType.element_shell_strain]
    # Must have real data (not all zeros) in at least one state
    assert strain.shape[1] == 202  # 202 shells in TENSILE
    assert np.abs(strain).max() > 0


def test_tensile_has_shell_thickness(tmp_path):
    """TENSILE fixture: shell thickness must be written."""
    from conftest import TENSILE_DIR
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(TENSILE_DIR, tmp_path / "tensile.d3plot")
    dp = D3plot(str(tmp_path / "tensile.d3plot"))
    assert ArrayType.element_shell_thickness in dp.arrays
    thick = dp.arrays[ArrayType.element_shell_thickness]
    assert thick.shape[1] == 202
    # Initial thickness is 1.7 mm
    assert np.isclose(thick[0].max(), 1.7, atol=0.01)


def test_tensile_thickness_necking(tmp_path):
    """TENSILE: thickness decreases over time (necking)."""
    from conftest import TENSILE_DIR
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(TENSILE_DIR, tmp_path / "tensile.d3plot")
    dp = D3plot(str(tmp_path / "tensile.d3plot"))
    thick = dp.arrays[ArrayType.element_shell_thickness]
    # Last state min thickness < initial max (necking happened)
    assert thick[-1].min() < thick[0].max()


def test_w3_has_acceleration(tmp_path):
    """W3 fixture: acceleration must be written."""
    from conftest import W3_DIR
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(W3_DIR, tmp_path / "w3.d3plot")
    dp = D3plot(str(tmp_path / "w3.d3plot"))
    assert ArrayType.node_acceleration in dp.arrays
    accel = dp.arrays[ArrayType.node_acceleration]
    assert np.abs(accel).max() > 0


def test_tensile_strain_no_longer_unmapped(tmp_path):
    """TENSILE: strain and thickness must NOT appear in unmapped warnings."""
    from conftest import TENSILE_DIR

    from vtk2d3plot.api import convert

    rep = convert(TENSILE_DIR, tmp_path / "tensile.d3plot")
    for w in rep.warnings:
        if "unmapped" in w.lower():
            assert "2DELEM_Strain" not in w, f"Strain should be mapped: {w}"
            assert "2DELEM_Thickness" not in w, f"Thickness should be mapped: {w}"


def test_w3_acceleration_no_longer_unmapped(tmp_path):
    """W3: Acceleration must NOT appear in unmapped warnings."""
    from conftest import W3_DIR

    from vtk2d3plot.api import convert

    rep = convert(W3_DIR, tmp_path / "w3.d3plot")
    for w in rep.warnings:
        if "unmapped" in w.lower():
            assert "Acceleration" not in w, f"Acceleration should be mapped: {w}"
