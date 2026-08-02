"""Milestone 2 — topology mapping + geometry-only d3plot tests.

Tests from docs/plan.md M2.1, copied faithfully.
"""

import numpy as np
import pytest
from conftest import GOLDEN_W3_VTK, PORT_DIR

pytestmark = pytest.mark.milestone2


def test_map_port_topology():
    from vtk2d3plot.mapping import map_topology
    from vtk2d3plot.vtk_reader import read_vtk

    m = map_topology(read_vtk(PORT_DIR / "BOXIMPA000.vtk"))
    assert m.shell_conn.shape == (200, 4)
    assert m.node_ids[0] == 1 and m.part_ids.tolist() == [1]


def test_map_official_topology_w3():
    from vtk2d3plot.mapping import map_topology
    from vtk2d3plot.vtk_reader import read_vtk

    m = map_topology(read_vtk(min(GOLDEN_W3_VTK.glob("*.vtk"))))
    assert m.solid_conn.shape[0] == 875 and m.shell_conn.shape[0] == 400
    assert m.part_ids.shape[0] == 2
    assert 0 not in m.node_ids  # zero-ID renumbering


def test_geometry_roundtrip(tmp_path):
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    _rep = convert(PORT_DIR, tmp_path / "box.d3plot")
    d = D3plot(str(tmp_path / "box.d3plot"))
    assert d.arrays[ArrayType.node_coordinates].shape == (220, 3)
    x0 = d.arrays[ArrayType.node_coordinates]
    assert np.allclose(x0[0], [0.0, 0.0, 0.5], atol=1e-6)  # recovered undeformed z=0.5


@pytest.mark.lspp
def test_geometry_loads_in_lsprepost(tmp_path):
    from tools.lspp_check import check_d3plot
    from vtk2d3plot.api import convert

    convert(PORT_DIR, tmp_path / "box.d3plot")
    r = check_d3plot(tmp_path / "box.d3plot")
    assert r["loaded"] and r["n_states"] == 21
