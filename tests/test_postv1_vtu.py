import numpy as np
import pytest

from vtk2d3plot.series import parse_pvd
from vtk2d3plot.vtk_reader import read_vtk

VTU_CONTENT = """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="4" NumberOfCells="1">
      <Points>
        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii">
          0.0 0.0 0.0
          1.0 0.0 0.0
          0.0 1.0 0.0
          0.0 0.0 1.0
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">
          0 1 2 3
        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
          4
        </DataArray>
        <DataArray type="Int32" Name="types" format="ascii">
          10
        </DataArray>
      </Cells>
      <PointData>
        <DataArray type="Float32" Name="TIME" format="ascii">
          1.23
        </DataArray>
        <DataArray type="Float32" Name="Temperature" format="ascii">
          300 310 320 330
        </DataArray>
      </PointData>
      <CellData>
        <DataArray type="Float32" Name="Stress" NumberOfComponents="6" format="ascii">
          1.0 2.0 3.0 4.0 5.0 6.0
        </DataArray>
      </CellData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

PVD_CONTENT = """<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1">
  <Collection>
    <DataSet timestep="0.0" group="" part="0" file="test_0.vtu"/>
    <DataSet timestep="1.23" group="" part="0" file="test_1.vtu"/>
  </Collection>
</VTKFile>
"""


def test_read_vtu(tmp_path):
    vtu_path = tmp_path / "test.vtu"
    vtu_path.write_text(VTU_CONTENT)

    state = read_vtk(vtu_path)

    assert state.time == pytest.approx(1.23)
    assert state.points.shape == (4, 3)
    assert np.allclose(state.points[1], [1.0, 0.0, 0.0])

    assert len(state.cell_types) == 1
    assert state.cell_types[0] == 10

    assert len(state.cell_offsets) == 2
    assert state.cell_offsets[1] == 4

    assert len(state.cell_connectivity) == 4

    assert "Temperature" in state.point_data
    assert len(state.point_data["Temperature"]) == 4

    assert "Stress" in state.cell_data
    assert state.cell_data["Stress"].shape == (1, 6)


def test_parse_pvd(tmp_path):
    pvd_path = tmp_path / "test.pvd"
    pvd_path.write_text(PVD_CONTENT)

    entries = parse_pvd(pvd_path)
    assert len(entries) == 2
    assert entries[0][0] == 0.0
    assert entries[0][1].name == "test_0.vtu"
    assert entries[1][0] == 1.23
    assert entries[1][1].name == "test_1.vtu"
