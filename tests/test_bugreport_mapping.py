"""Bug report 2026-10-11, cluster C: topology mapping and the lasso SPH patch.

Covers report items #4 (c19, wedge orientation), #20 (c37, parts of skipped
cells in the part table) and #39 (c38, lasso SPH write patch).
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.bugreport


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# trilinear hex natural coordinates of LS-DYNA nodes N1..N8
_HEX_NAT = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=float,
)


def _hex_jacobian_det(xyz8: np.ndarray, xi: np.ndarray) -> float:
    """det(dX/dxi) of the trilinear 8-node hex at natural point xi."""
    dn = np.empty((8, 3))
    for a in range(8):
        s = _HEX_NAT[a]
        f = 1.0 + s * xi
        dn[a, 0] = s[0] * f[1] * f[2] / 8.0
        dn[a, 1] = f[0] * s[1] * f[2] / 8.0
        dn[a, 2] = f[0] * f[1] * s[2] / 8.0
    jac = xyz8.T @ dn  # (3 coords, 3 natural dirs)
    return float(np.linalg.det(jac))


def _write_vtk(
    path,
    points,
    cells,
    cell_types,
    *,
    time=0.0,
    part_ids=None,
    node_ids=None,
):
    """Write a minimal official-dialect legacy ASCII VTK file."""
    points = np.asarray(points, dtype=float)
    n_pts = len(points)
    n_cells = len(cells)
    size = sum(len(c) + 1 for c in cells)
    lines = [
        "# vtk DataFile Version 3.0",
        "synthetic bugreport cluster C",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        "FIELD FieldData 1",
        "TIME 1 1 double",
        f"{time:.6e}",
        f"POINTS {n_pts} float",
    ]
    lines += [f"{p[0]:.6e} {p[1]:.6e} {p[2]:.6e}" for p in points]
    lines.append(f"CELLS {n_cells} {size}")
    lines += [" ".join(str(v) for v in [len(c), *c]) for c in cells]
    lines.append(f"CELL_TYPES {n_cells}")
    lines += [str(t) for t in cell_types]
    lines.append(f"POINT_DATA {n_pts}")
    lines.append("SCALARS NODE_ID int 1")
    lines.append("LOOKUP_TABLE default")
    nid = node_ids if node_ids is not None else range(1, n_pts + 1)
    lines += [str(int(v)) for v in nid]
    lines.append("VECTORS Displacement float")
    lines += ["0 0 0"] * n_pts
    lines.append(f"CELL_DATA {n_cells}")
    lines.append("SCALARS ELEMENT_ID int 1")
    lines.append("LOOKUP_TABLE default")
    lines += [str(100 + i) for i in range(n_cells)]
    if part_ids is not None:
        lines.append("SCALARS PART_ID int 1")
        lines.append("LOOKUP_TABLE default")
        lines += [str(int(p)) for p in part_ids]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


# report geometry: base triangle at z=0, top triangle at z=-1 (VTK order)
_WEDGE_PTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
        [0.0, 1.0, -1.0],
    ]
)


def _read_vtk_state(tmp_path, points, cells, cell_types, **kw):
    from vtk2d3plot.vtk_reader import read_vtk

    p = tmp_path / "runA000.vtk"
    _write_vtk(p, points, cells, cell_types, **kw)
    return read_vtk(p)


# ---------------------------------------------------------------------------
# #4 (c19): wedge written inside-out
# ---------------------------------------------------------------------------


def test_c19_wedge_map_topology_connectivity(tmp_path):
    """Bug report 2026-10-11 #4 (c19): VTK_WEDGE must map to [0,1,4,3,2,2,5,5]."""
    from vtk2d3plot.mapping import map_topology

    st = _read_vtk_state(tmp_path, _WEDGE_PTS, [[0, 1, 2, 3, 4, 5]], [13])
    m = map_topology(st)
    assert m.solid_conn.tolist() == [[0, 1, 4, 3, 2, 2, 5, 5]]


def test_c19_wedge_positive_jacobian(tmp_path):
    """Bug report 2026-10-11 #4 (c19): mapped wedge has a positive Jacobian.

    Evaluated with trilinear hex shape functions at the element centre and at
    the four non-collapsed corners N1..N4 (N5=N6 and N7=N8 are collapsed).
    """
    from vtk2d3plot.mapping import map_topology

    st = _read_vtk_state(tmp_path, _WEDGE_PTS, [[0, 1, 2, 3, 4, 5]], [13])
    m = map_topology(st)
    xyz8 = st.points[m.solid_conn[0]]
    det_c = _hex_jacobian_det(xyz8, np.zeros(3))
    assert det_c == pytest.approx(1.0 / 16.0, rel=1e-12)
    for k in range(4):
        det_k = _hex_jacobian_det(xyz8, _HEX_NAT[k])
        assert det_k == pytest.approx(1.0 / 8.0, rel=1e-12), (k, det_k)


def test_c19_wedge_convert_readback(tmp_path):
    """Bug report 2026-10-11 #4 (c19): convert + lasso read-back of the wedge connectivity."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot import convert

    vdir = tmp_path / "vtk"
    vdir.mkdir()
    _write_vtk(vdir / "runA000.vtk", _WEDGE_PTS, [[0, 1, 2, 3, 4, 5]], [13], part_ids=[1])
    _write_vtk(vdir / "runA001.vtk", _WEDGE_PTS, [[0, 1, 2, 3, 4, 5]], [13], time=1.0, part_ids=[1])
    out = tmp_path / "wedge.d3plot"
    rep = convert(vdir, out)
    assert rep.n_solids == 1
    d = D3plot(str(out))
    conn = d.arrays[ArrayType.element_solid_node_indexes]
    assert conn.tolist() == [[0, 1, 4, 3, 2, 2, 5, 5]]
    xyz8 = d.arrays[ArrayType.node_coordinates][conn[0]].astype(float)
    assert _hex_jacobian_det(xyz8, np.zeros(3)) > 0
    assert _hex_jacobian_det(xyz8, _HEX_NAT[0]) == pytest.approx(1.0 / 8.0, rel=1e-6)


def test_c19_tetra_pyramid_orientation_positive(tmp_path):
    """Bug report 2026-10-11 #4 (c19) side check: right-handed VTK tetra and
    pyramid map to degenerate hexes with a positive Jacobian (no change needed)."""
    from vtk2d3plot.mapping import map_topology

    pts = np.array(
        [
            # tetra: (0,1,2) normal (+z) points toward node 3
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            # pyramid: base (4..7) normal (+z) points toward apex 8
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.5, 0.5, 1.0],
        ]
    )
    st = _read_vtk_state(tmp_path, pts, [[0, 1, 2, 3], [4, 5, 6, 7, 8]], [10, 14])
    m = map_topology(st)
    assert m.n_inverted_tetras == 0
    assert m.solid_conn.tolist() == [[0, 1, 2, 3, 3, 3, 3, 3], [4, 5, 6, 7, 8, 8, 8, 8]]
    tet = st.points[m.solid_conn[0]]
    pyr = st.points[m.solid_conn[1]]
    assert _hex_jacobian_det(tet, np.zeros(3)) > 0
    assert _hex_jacobian_det(pyr, np.zeros(3)) > 0
    for k in range(4):  # pyramid base corners are not collapsed
        assert _hex_jacobian_det(pyr, _HEX_NAT[k]) > 0


# ---------------------------------------------------------------------------
# #20 (c37): parts of skipped cells in the part table
# ---------------------------------------------------------------------------

_QUAD_VERTEX_PTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [5.0, 5.0, 5.0],
    ]
)


def test_c37_skipped_vertex_part_not_in_map(tmp_path):
    """Bug report 2026-10-11 #20 (c37): map_topology drops the part of a skipped VTK_VERTEX."""
    from vtk2d3plot.mapping import map_topology

    st = _read_vtk_state(tmp_path, _QUAD_VERTEX_PTS, [[4], [0, 1, 2, 3]], [1, 9], part_ids=[7, 3])
    m = map_topology(st)
    assert m.n_skipped_cells == 1
    assert m.part_ids.tolist() == [3]
    assert m.shell_part_idx.tolist() == [0]
    assert m.shell_ids.tolist() == [101]


def test_c37_phantom_masked_part_not_in_map(tmp_path):
    """Bug report 2026-10-11 #20 (c37): the part of a phantom-masked cell is dropped too."""
    from vtk2d3plot.mapping import map_topology

    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    # node index 5 has NODE_ID 0 -> the second quad (part 9) is phantom-masked
    st = _read_vtk_state(
        tmp_path,
        pts,
        [[1, 4, 5, 2], [0, 1, 2, 3]],
        [9, 9],
        part_ids=[9, 4],
        node_ids=[1, 2, 3, 4, 5, 0],
    )
    m = map_topology(st)
    assert m.n_phantom_masked == 1
    assert m.part_ids.tolist() == [4]
    assert m.shell_part_idx.tolist() == [0]


def test_c37_convert_part_table_readback(tmp_path):
    """Bug report 2026-10-11 #20 (c37): convert writes only parts of kept cells
    (part_ids, part titles incl. --part-names, nmmat via n_parts, report n_parts)."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot import convert

    vdir = tmp_path / "vtk"
    vdir.mkdir()
    for i in range(2):
        _write_vtk(
            vdir / f"runA00{i}.vtk",
            _QUAD_VERTEX_PTS,
            [[4], [0, 1, 2, 3]],
            [1, 9],
            time=float(i),
            part_ids=[7, 3],
        )
    out = tmp_path / "q.d3plot"
    rep = convert(vdir, out, part_names={7: "SPH_PART", 3: "SHELL_PART"})
    assert rep.n_parts == 1
    assert rep.n_shells == 1
    assert any("Skipped 1 unsupported cells (VTK types: 1)" in w for w in rep.warnings)
    d = D3plot(str(out))
    assert d.arrays[ArrayType.part_ids].tolist() == [3]
    assert d.header.n_parts == 1
    assert d.arrays[ArrayType.element_shell_part_indexes].tolist() == [0]
    titles = [t.decode("utf-8").strip("\x00").strip() for t in d.arrays[ArrayType.part_titles]]
    assert titles == ["SHELL_PART"]
    assert d.arrays[ArrayType.part_titles_ids].tolist() == [3]


def test_c37_unused_part_names_warn(tmp_path):
    """Bug report 2026-10-11 #20 (c37): part_names entries for parts that are not
    in the d3plot (here part 7, whose only cell is a skipped VTK_VERTEX) are
    reported in one warning instead of being ignored silently."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot import convert

    vdir = tmp_path / "vtk"
    vdir.mkdir()
    for i in range(2):
        _write_vtk(
            vdir / f"runA00{i}.vtk",
            _QUAD_VERTEX_PTS,
            [[4], [0, 1, 2, 3]],
            [1, 9],
            time=float(i),
            part_ids=[7, 3],
        )
    out = tmp_path / "q.d3plot"
    rep = convert(vdir, out, part_names={3: "A", 7: "B"})
    msg = "part_names entries for part IDs not in the d3plot were ignored: 7"
    assert [w for w in rep.warnings if "part_names entries" in w] == [msg]
    d = D3plot(str(out))
    assert d.arrays[ArrayType.part_ids].tolist() == [3]
    titles = [t.decode("utf-8").strip("\x00").strip() for t in d.arrays[ArrayType.part_titles]]
    assert titles == ["A"]
    # several unused IDs: sorted numerically (7 before 12)
    rep = convert(vdir, out, part_names={12: "C", 3: "A", 7: "B"})
    msg = "part_names entries for part IDs not in the d3plot were ignored: 7, 12"
    assert [w for w in rep.warnings if "part_names entries" in w] == [msg]
    # all names used -> no such warning
    rep = convert(vdir, out, part_names={3: "A"})
    assert not [w for w in rep.warnings if "part_names entries" in w]


# ---------------------------------------------------------------------------
# #39 (c38): lasso SPH patch injects zero SPH flags
# ---------------------------------------------------------------------------


def _sph_d3plot(n_particles: int, *, extra: bool = False):
    from lasso.dyna import ArrayType, D3plot

    import vtk2d3plot  # noqa: F401  (applies the lasso patch)

    n_nodes = n_particles + 1
    d3 = D3plot()
    x = np.arange(n_nodes * 3, dtype=np.float32).reshape(n_nodes, 3)
    d3.arrays[ArrayType.node_coordinates] = x
    d3.arrays[ArrayType.node_ids] = np.arange(11, 11 + n_nodes, dtype=np.int32)
    d3.arrays[ArrayType.sph_node_indexes] = np.arange(1, n_particles + 1, dtype=np.int32)
    d3.arrays[ArrayType.sph_node_material_index] = np.zeros(n_particles, dtype=np.int32)
    d3.arrays[ArrayType.part_ids] = np.array([1], dtype=np.int32)
    d3.arrays[ArrayType.global_timesteps] = np.array([0.0, 0.5], dtype=np.float32)
    disp = np.stack([x, x + 0.25]).astype(np.float32)
    d3.arrays[ArrayType.node_displacement] = disp
    dens = ((np.arange(2 * n_particles).reshape(2, n_particles) + 1.5) * 1000).astype(np.float32)
    d3.arrays[ArrayType.sph_density] = dens
    if extra:
        d3.arrays[ArrayType.sph_radius] = np.full((2, n_particles), 0.125, dtype=np.float32)
        d3.arrays[ArrayType.sph_pressure] = dens * 0.5
        d3.arrays[ArrayType.sph_stress] = (
            np.arange(2 * n_particles * 6, dtype=np.float32).reshape(2, n_particles, 6) - 7.0
        )
        d3.arrays[ArrayType.sph_effective_plastic_strain] = dens * 1e-4
        d3.arrays[ArrayType.sph_internal_energy] = dens * 2.0
        d3.arrays[ArrayType.sph_n_neighbors] = np.full((2, n_particles), 12.0, dtype=np.float32)
        d3.arrays[ArrayType.sph_strain] = (
            np.arange(2 * n_particles * 6, dtype=np.float32).reshape(2, n_particles, 6) * 1e-3
        )
        d3.arrays[ArrayType.sph_mass] = np.full((2, n_particles), 0.75, dtype=np.float32)
        deleted = np.zeros((2, n_particles), dtype=bool)
        deleted[1, -1] = True
        d3.arrays[ArrayType.sph_deletion] = deleted
    return d3


@pytest.mark.parametrize("n_particles", [1, 3])
def test_c38_sph_density_roundtrip(tmp_path, n_particles):
    """Bug report 2026-10-11 #39 (c38): SPH node indexes + density for 1 and 3
    particles are written and read back with exact values."""
    from lasso.dyna import ArrayType, D3plot

    d3 = _sph_d3plot(n_particles)
    out = tmp_path / "sph.d3plot"
    d3.write_d3plot(str(out))
    r = D3plot(str(out))
    assert r.header.n_sph_nodes == n_particles
    assert r.arrays[ArrayType.global_timesteps].tolist() == [0.0, 0.5]
    assert r.arrays[ArrayType.node_ids].tolist() == d3.arrays[ArrayType.node_ids].tolist()
    assert np.array_equal(r.arrays[ArrayType.sph_node_indexes], np.arange(1, n_particles + 1))
    assert np.array_equal(r.arrays[ArrayType.sph_node_material_index], np.zeros(n_particles))
    assert np.array_equal(r.arrays[ArrayType.sph_density], d3.arrays[ArrayType.sph_density])
    assert np.array_equal(
        r.arrays[ArrayType.node_displacement], d3.arrays[ArrayType.node_displacement]
    )
    assert not r.arrays[ArrayType.sph_deletion].any()


def test_c38_sph_all_state_arrays_roundtrip(tmp_path):
    """Bug report 2026-10-11 #39 (c38): every SPH state array lasso can write
    (deletion, radius, pressure, stress, eps, density, internal energy,
    neighbours, strain, mass) round-trips exactly for 3 particles.

    The geometry arrays after the SPH flag block (node IDs, SPH node / material
    list) are checked too: with a wrong isphfg(1) the reader skips the wrong
    number of geometry words, while the state arrays can still read back."""
    from lasso.dyna import ArrayType, D3plot

    d3 = _sph_d3plot(3, extra=True)
    out = tmp_path / "sph_all.d3plot"
    d3.write_d3plot(str(out))
    r = D3plot(str(out))
    assert r.header.n_sph_nodes == 3
    assert r.arrays[ArrayType.global_timesteps].tolist() == [0.0, 0.5]
    assert r.arrays[ArrayType.node_ids].tolist() == [11, 12, 13, 14]
    assert r.arrays[ArrayType.sph_node_indexes].tolist() == [1, 2, 3]
    assert r.arrays[ArrayType.sph_node_material_index].tolist() == [0, 0, 0]
    assert np.array_equal(
        r.arrays[ArrayType.node_coordinates], d3.arrays[ArrayType.node_coordinates]
    )
    for at in (
        ArrayType.sph_radius,
        ArrayType.sph_pressure,
        ArrayType.sph_stress,
        ArrayType.sph_effective_plastic_strain,
        ArrayType.sph_density,
        ArrayType.sph_internal_energy,
        ArrayType.sph_n_neighbors,
        ArrayType.sph_strain,
        ArrayType.sph_mass,
        ArrayType.sph_deletion,
        ArrayType.node_displacement,
    ):
        assert np.array_equal(r.arrays[at], d3.arrays[at]), at


def test_c38_sph_missing_material_index_raises(tmp_path):
    """Bug report 2026-10-11 #39 (c38): without sph_node_material_index the
    patched writer fails loudly instead of writing a corrupt file."""
    from lasso.dyna import ArrayType

    d3 = _sph_d3plot(3)
    del d3.arrays[ArrayType.sph_node_material_index]
    with pytest.raises(ValueError, match="sph_node_material_index"):
        d3.write_d3plot(str(tmp_path / "bad.d3plot"))
