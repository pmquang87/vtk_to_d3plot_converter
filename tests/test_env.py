"""Environment sanity - must stay green from day zero.

If anything here fails, the workspace setup is broken; fix the environment,
do not adapt the tests.
"""

import numpy as np
import pytest
from conftest import (
    GOLDEN_W3_D3,
    GOLDEN_W3_VTK,
    GOLDEN_ZUG_D3,
    GOLDEN_ZUG_VTK,
    PORT_DIR,
    TENSILE_DIR,
    VENV_PY,
)


def test_python_and_numpy():
    assert VENV_PY.is_file()
    assert np.zeros(3).dtype == np.float64


def test_lasso_pinned():
    import lasso
    from lasso.dyna import ArrayType, D3plot  # noqa: F401

    # the pin the whole project is verified against
    assert getattr(lasso, "__version__", "2.0.4") == "2.0.4"


def test_fixtures_present():
    assert len(list(PORT_DIR.glob("BOXIMPA*.vtk"))) == 21
    assert len(list(GOLDEN_W3_VTK.glob("*.vtk"))) == 11
    assert len(list(GOLDEN_ZUG_VTK.glob("*.vtk"))) == 3
    assert len(list(TENSILE_DIR.glob("*.vtk"))) == 20
    assert (GOLDEN_W3_D3 / "W3_SETUP_Ball_impact__ton-mm-s.d3plot").is_file()
    assert (GOLDEN_ZUG_D3 / "zug_test3_RS.d3plot").is_file()


def test_golden_d3plot_readable_by_lasso():
    from lasso.dyna import ArrayType, D3plot

    d = D3plot(str(GOLDEN_W3_D3 / "W3_SETUP_Ball_impact__ton-mm-s.d3plot"))
    assert d.arrays[ArrayType.node_coordinates].shape == (1417, 3)
    assert d.arrays[ArrayType.node_displacement].shape == (11, 1417, 3)
    assert d.arrays[ArrayType.element_shell_stress].shape == (11, 400, 2, 6)


def test_lasso_write_readback_smoke(tmp_path):
    """The core capability everything rests on: write a d3plot from scratch."""
    from lasso.dyna import ArrayType, D3plot

    coords = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=np.float32,
    )
    d3 = D3plot()
    a = d3.arrays
    a[ArrayType.node_coordinates] = coords
    a[ArrayType.node_ids] = np.arange(1, 9, dtype=np.int32)
    a[ArrayType.element_solid_node_indexes] = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
    a[ArrayType.element_solid_part_indexes] = np.array([0], dtype=np.int32)
    a[ArrayType.element_solid_ids] = np.array([1], dtype=np.int32)
    a[ArrayType.part_ids] = np.array([77], dtype=np.int32)
    a[ArrayType.global_timesteps] = np.array([0.0, 1.0], dtype=np.float32)
    disp = np.stack([coords, coords + [0.1, 0, 0]]).astype(np.float32)
    a[ArrayType.node_displacement] = disp

    out = tmp_path / "smoke.d3plot"
    d3.write_d3plot(str(out))
    r = D3plot(str(out))
    assert np.allclose(r.arrays[ArrayType.node_displacement], disp, atol=1e-6)
    assert (tmp_path / "smoke.d3plot01").is_file()


@pytest.mark.lspp
def test_lspp_check_tool_on_golden():
    from tools.lspp_check import check_d3plot

    r = check_d3plot(GOLDEN_W3_D3 / "W3_SETUP_Ball_impact__ton-mm-s.d3plot")
    assert r["loaded"], r
    assert r["n_states"] == 11
