from pathlib import Path

import pytest
from lasso.dyna import ArrayType

from vtk2d3plot.builder import build_d3plot
from vtk2d3plot.mapping import map_topology
from vtk2d3plot.series import find_t01_csv, load_series, load_series_lazy, parse_t01_csv

FIXTURE_DIR = Path(__file__).parent / "data/port_dialect/box_beam_impact"


def test_find_t01_csv():
    vtk_path = FIXTURE_DIR / "BOXIMPA000.vtk"
    if not vtk_path.exists():
        pytest.skip(f"Fixture not found: {vtk_path}")
    t01_path = find_t01_csv([vtk_path])
    assert t01_path is not None
    assert t01_path.name == "BOXIMPT01.csv"
    assert t01_path.exists()


def test_parse_t01_csv():
    t01_path = FIXTURE_DIR / "BOXIMPT01.csv"
    if not t01_path.exists():
        pytest.skip("Fixture not found")
    t01_data = parse_t01_csv(t01_path)
    assert len(t01_data.times) > 0
    assert "IE" in t01_data.columns
    assert "KE" in t01_data.columns


def test_t01_auto_detection():
    vtk_path = FIXTURE_DIR / "BOXIMPA000.vtk"
    if not vtk_path.exists():
        pytest.skip("Fixture not found")

    series = load_series(vtk_path)
    assert series.t01_data is not None

    lazy_series = load_series_lazy(vtk_path)
    assert lazy_series.t01_data is not None


def test_t01_builder_integration():
    vtk_path = FIXTURE_DIR / "BOXIMPA000.vtk"
    if not vtk_path.exists():
        pytest.skip("Fixture not found")

    series = load_series(vtk_path)
    model = map_topology(series.states[0])

    d3, _ = build_d3plot(series, model)

    ie = d3.arrays.get(ArrayType.global_internal_energy)
    ke = d3.arrays.get(ArrayType.global_kinetic_energy)
    en = d3.arrays.get(ArrayType.global_total_energy)

    assert ie is not None
    assert ke is not None
    assert len(ie) == len(series.times)
    assert len(ke) == len(series.times)

    if en is not None:
        assert len(en) == len(series.times)
