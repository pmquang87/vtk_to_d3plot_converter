import numpy as np
from conftest import GOLDEN_ZUG_VTK


def test_element_id_renumbering(tmp_path):
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(GOLDEN_ZUG_VTK, tmp_path / "zug.d3plot")
    d = D3plot(str(tmp_path / "zug.d3plot"))

    beam_ids = d.arrays[ArrayType.element_beam_ids]
    shell_ids = d.arrays[ArrayType.element_shell_ids]

    # Assert all > 0
    assert np.all(beam_ids > 0)
    assert np.all(shell_ids > 0)

    # Assert unique
    assert len(np.unique(beam_ids)) == len(beam_ids)
    assert len(np.unique(shell_ids)) == len(shell_ids)


def test_m5_title_assigned(tmp_path):
    from lasso.dyna import D3plot

    from vtk2d3plot.api import convert

    convert(GOLDEN_ZUG_VTK, tmp_path / "zug.d3plot", title="Test Title 123")
    d = D3plot(str(tmp_path / "zug.d3plot"))
    assert d.header.title.strip() == "Test Title 123"


def test_m2_tensile_membrane_stress(tmp_path):
    from conftest import TENSILE_DIR
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    rep = convert(TENSILE_DIR, tmp_path / "tensile.d3plot")
    d = D3plot(str(tmp_path / "tensile.d3plot"))
    assert ArrayType.element_shell_stress in d.arrays
    assert np.abs(d.arrays[ArrayType.element_shell_stress]).max() > 0
    assert any("membrane" in w.lower() for w in rep.warnings)


def test_m6_beam_alive(tmp_path):
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    mock_dir = tmp_path / "mock_beams"
    mock_dir.mkdir()
    vtk_content = """# vtk DataFile Version 3.0
pyradioss state t=0.0
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 2 float
0 0 0 1 0 0
CELLS 1 3
2 0 1
CELL_TYPES 1
3
CELL_DATA 1
SCALARS OFF float 1
LOOKUP_TABLE default
0.0
POINT_DATA 2
VECTORS DISPLACEMENT float
0 0 0 0 0 0
"""
    (mock_dir / "mockA000.vtk").write_text(vtk_content)

    convert(mock_dir, tmp_path / "mock.d3plot")
    d = D3plot(str(tmp_path / "mock.d3plot"))

    assert ArrayType.element_beam_is_alive in d.arrays
    alive = d.arrays[ArrayType.element_beam_is_alive]
    assert alive.shape == (1, 1)
    assert alive[0, 0] == 0.0
