"""Milestone 1 contract - RED on purpose until M1 is implemented.

The Antigravity agent extends this file with the full M1 test set specified in
docs/plan.md (test_read_port_state0, test_read_official_state, test_series_port,
test_reader_matches_meshio).
"""

import pytest

pytestmark = pytest.mark.milestone1


def test_m1_contract():
    """M1 delivers vtk_reader.read_vtk and series.load_series."""
    from vtk2d3plot.series import load_series  # noqa: F401
    from vtk2d3plot.vtk_reader import read_vtk  # noqa: F401
