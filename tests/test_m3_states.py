"""Milestone 3 — kinematic states roundtrip + golden-pair comparison.

Tests from docs/plan.md M3.1, copied faithfully.
"""

import numpy as np
import pytest
from conftest import GOLDEN_W3_D3, GOLDEN_W3_VTK, GOLDEN_ZUG_D3, GOLDEN_ZUG_VTK, PORT_DIR

pytestmark = pytest.mark.milestone3


def test_port_states_roundtrip(tmp_path):
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(PORT_DIR, tmp_path / "box.d3plot")
    d = D3plot(str(tmp_path / "box.d3plot"))
    assert len(d.arrays[ArrayType.global_timesteps]) == 21
    assert d.arrays[ArrayType.node_displacement].shape == (21, 220, 3)
    assert d.arrays[ArrayType.node_velocity].shape == (21, 220, 3)


def _byid_max_coord_diff(mine_path, golden_path):
    # join on node ID, compare per-state absolute coordinates
    from lasso.dyna import ArrayType, D3plot

    a, b = D3plot(str(mine_path)), D3plot(str(golden_path))
    ia = np.argsort(a.arrays[ArrayType.node_ids])
    ib = np.argsort(b.arrays[ArrayType.node_ids])
    assert np.array_equal(a.arrays[ArrayType.node_ids][ia], b.arrays[ArrayType.node_ids][ib])
    da = a.arrays[ArrayType.node_displacement][:, ia, :]
    db = b.arrays[ArrayType.node_displacement][:, ib, :]
    assert da.shape == db.shape
    return float(np.abs(da - db).max())


def test_w3_matches_vortex_golden(tmp_path):
    from vtk2d3plot.api import convert

    convert(GOLDEN_W3_VTK, tmp_path / "w3.d3plot")
    assert (
        _byid_max_coord_diff(
            tmp_path / "w3.d3plot",
            GOLDEN_W3_D3 / "W3_SETUP_Ball_impact__ton-mm-s.d3plot",
        )
        < 1.0e-2
    )  # ASCII precision bound


def test_zug_matches_vortex_golden(tmp_path):
    # the node-renumbering trap: passes ONLY with by-ID join and zero-ID policy max+1
    from vtk2d3plot.api import convert

    convert(GOLDEN_ZUG_VTK, tmp_path / "zug.d3plot")
    assert (
        _byid_max_coord_diff(
            tmp_path / "zug.d3plot",
            GOLDEN_ZUG_D3 / "zug_test3_RS.d3plot",
        )
        < 1.0e-2
    )
