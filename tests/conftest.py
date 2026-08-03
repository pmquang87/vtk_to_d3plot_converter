"""Shared fixtures/paths for the vtk2d3plot test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VENV_PY = Path(sys.executable)

# repo root importable so `from tools.lspp_check import check_d3plot` works
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATA = REPO / "tests" / "data"
PORT_DIR = DATA / "port_dialect" / "box_beam_impact"
GOLDEN_W3 = DATA / "golden" / "W3_SETUP_Ball_impact__ton-mm-s"
GOLDEN_W3_VTK = GOLDEN_W3 / "vtk"
GOLDEN_W3_D3 = GOLDEN_W3 / "d3plot"
GOLDEN_ZUG = DATA / "golden" / "zug_test3_RS"
GOLDEN_ZUG_VTK = GOLDEN_ZUG / "vtk"
GOLDEN_ZUG_D3 = GOLDEN_ZUG / "d3plot"
TENSILE_DIR = DATA / "golden" / "TENSILE_vtk_only"
W3_DIR = DATA / "golden" / "W3_SETUP_Ball_impact__ton-mm-s" / "vtk"


def lspp_available() -> bool:
    from tools.lspp_check import find_lsprepost

    return find_lsprepost() is not None


def pyradioss_available() -> bool:
    try:
        import pyradioss  # noqa: F401

        return True
    except ImportError:
        return False


def pytest_collection_modifyitems(config, items):
    skip_lspp = pytest.mark.skip(reason="LS-PrePost not installed (set LSPP_EXE)")
    skip_pyr = pytest.mark.skip(reason="pyradioss not importable in this venv")
    has_lspp = lspp_available()
    has_pyr = pyradioss_available()
    for item in items:
        if "lspp" in item.keywords and not has_lspp:
            item.add_marker(skip_lspp)
        if "pyradioss" in item.keywords and not has_pyr:
            item.add_marker(skip_pyr)
