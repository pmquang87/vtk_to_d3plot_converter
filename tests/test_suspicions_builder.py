"""Suspicions round, cluster D: builder, topology mapping, shell rotation.

Leads from docs/reviews/2026-10-11-external-bug-report.md ("Suspicions not
checked") plus two coordinator leads:

* (a) solid shear slots of the VTK 3x3 tensors (pins only, no defect found:
  anim_to_vtk.cpp writes ANIM yz at (0,2) and zx at (1,2); the current port
  writer does the same).
* (b) ``rotate_shell_stress`` must rotate the shell strain with the same frames.
* S01 single-layer shell stress sources must fill every layer once a second
  layer is forced by strain or plastic strain.
* S12 VTK lines: the d3plot orientation node and the two null words are 0.

All checks go through ``vtk2d3plot.convert`` and a lasso read-back of values.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import GOLDEN_ZUG_D3, GOLDEN_ZUG_VTK, PORT_DIR

pytestmark = pytest.mark.suspicions

ZUG_GOLD = GOLDEN_ZUG_D3 / "zug_test3_RS.d3plot"


# ---------------------------------------------------------------- helpers


def _read(path):
    from lasso.dyna import D3plot

    return D3plot(str(path))


def _write_vtk(
    path,
    *,
    title,
    time_val,
    points,
    cells,
    field_time=True,
    ids=True,
    point_vectors=None,
    cell_scalars=None,
    cell_tensors=None,
):
    """Write one legacy ASCII VTK state.

    cells: list of (vtk_type, [0-based point indices]); cell_tensors values are
    lists of 3x3 nested lists (one per cell). Numbers are written with repr()
    so the converter sees them at full double precision.
    """
    point_vectors = point_vectors or {}
    cell_scalars = cell_scalars or {}
    cell_tensors = cell_tensors or {}
    n_pts, n_cells = len(points), len(cells)
    L = ["# vtk DataFile Version 3.0", title, "ASCII", "DATASET UNSTRUCTURED_GRID"]
    if field_time:
        L += ["FIELD FieldData 1", "TIME 1 1 double", repr(float(time_val))]
    L.append(f"POINTS {n_pts} double")
    L += [" ".join(repr(float(c)) for c in p) for p in points]
    L.append(f"CELLS {n_cells} {sum(len(c) + 1 for _, c in cells)}")
    L += [" ".join(str(x) for x in [len(c), *c]) for _, c in cells]
    L.append(f"CELL_TYPES {n_cells}")
    L += [str(t) for t, _ in cells]
    L.append(f"POINT_DATA {n_pts}")
    if ids:
        L += ["SCALARS NODE_ID int 1", "LOOKUP_TABLE default"]
        L += [str(100 + i) for i in range(n_pts)]
    for name, vecs in point_vectors.items():
        L.append(f"VECTORS {name} double")
        L += [" ".join(repr(float(c)) for c in v) for v in vecs]
    L.append(f"CELL_DATA {n_cells}")
    if ids:
        L += ["SCALARS ELEMENT_ID int 1", "LOOKUP_TABLE default"]
        L += [str(500 + i) for i in range(n_cells)]
        L += ["SCALARS PART_ID int 1", "LOOKUP_TABLE default"]
        L += ["1"] * n_cells
    for name, vals in cell_scalars.items():
        L += [f"SCALARS {name} double 1", "LOOKUP_TABLE default"]
        L += [repr(float(v)) for v in vals]
    for name, ts in cell_tensors.items():
        L.append(f"TENSORS {name} double")
        for t in ts:
            L += [" ".join(repr(float(t[r][c])) for c in range(3)) for r in range(3)]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _plane(xx, yy, xy):
    """anim_to_vtk 2D tensor rows [x xy 0 / xy y 0 / 0 0 0] (anim_to_vtk.cpp:1366-1375)."""
    return [[xx, xy, 0.0], [xy, yy, 0.0], [0.0, 0.0, 0.0]]


def _voigt(t):
    """3x3 symmetric tensor -> d3plot slots [xx, yy, zz, xy, yz, zx]."""
    t = np.asarray(t, dtype=float)
    return np.array([t[0, 0], t[1, 1], t[2, 2], t[0, 1], t[1, 2], t[2, 0]])


def _clskew3(p1, p2, p3, p4):
    """OpenRadioss CLSKEW3 quad frame (columns e1, e2, e3), written out here."""
    r = p2 + p3 - p1 - p4
    s = p3 + p4 - p1 - p2
    e3 = np.cross(r, s)
    e3 /= np.linalg.norm(e3)
    e1 = r * np.linalg.norm(s) / np.linalg.norm(r) + np.cross(s, e3)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(e3, e1), e3], axis=-1)


def _c3coor3(p1, p2, p3):
    """OpenRadioss c3coor3 triangle frame: e1 = unit(N1->N2)."""
    e1 = (p2 - p1) / np.linalg.norm(p2 - p1)
    e3 = np.cross(p2 - p1, p3 - p1)
    e3 /= np.linalg.norm(e3)
    return np.stack([e1, np.cross(e3, e1), e3], axis=-1)


def _rot(axis, angle):
    """Rodrigues rotation matrix."""
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _eig(v):
    """Principal values of Voigt [xx, yy, zz, xy, yz, zx] tensors (..., 6)."""
    v = np.asarray(v, dtype=float)
    S = np.zeros(v.shape[:-1] + (3, 3))
    S[..., 0, 0], S[..., 1, 1], S[..., 2, 2] = v[..., 0], v[..., 1], v[..., 2]
    S[..., 0, 1] = S[..., 1, 0] = v[..., 3]
    S[..., 1, 2] = S[..., 2, 1] = v[..., 4]
    S[..., 2, 0] = S[..., 0, 2] = v[..., 5]
    return np.linalg.eigvalsh(S)


# One warped quad and one triangle, rigidly rotated by a different angle per
# state, so every state has its own element frames and none is axis-aligned.
_QUAD0 = np.array([[0.0, 0.0, 0.0], [2.0, 0.1, 0.0], [2.3, 1.2, 0.05], [-0.1, 0.9, 0.0]])
_TRI0 = np.array([[3.0, 0.0, 0.0], [4.5, 0.3, 0.0], [3.4, 1.4, 0.0]])
# local element-frame strain (tensorial shear, as anim_to_vtk writes it):
# [element][layer] = (exx, eyy, exy); layer 0 = lower, 1 = upper
_EPS_LOCAL = {
    0: ((1.0e-3, -4.0e-4, 2.5e-4), (2.0e-3, 5.0e-4, -3.0e-4)),
    1: ((7.0e-4, 1.0e-4, 6.0e-4), (-2.0e-4, 9.0e-4, 1.0e-4)),
}
_SIG_LOCAL = {
    0: ((120.0, -40.0, 25.0), (200.0, 50.0, -30.0)),
    1: ((70.0, 10.0, 60.0), (-20.0, 90.0, 10.0)),
}
_STRAIN_NAMES = ("2DELEM_Strain_(lower)", "2DELEM_Strain_(upper)")
_STRESS_NAMES = ("2DELEM_Stress_(lower)", "2DELEM_Stress_(upper)")


def _state_points(k):
    Q = _rot([1.0, 2.0, 3.0], 0.4 + 0.5 * k)
    shift = np.array([0.3 * k, -0.2, 1.0])
    return np.vstack([_QUAD0, _TRI0]) @ Q.T + shift


def _write_tilted_series(tmp_path, *, stress="tensor", strain=True, n_states=2):
    """Quad + triangle series with known element-frame tensors (scaled by 1+k)."""
    cells = [(9, [0, 1, 2, 3]), (5, [4, 5, 6])]
    paths = []
    for k in range(n_states):
        f = 1.0 + k
        tens = {}
        scal = {}
        if strain:
            for lay, name in enumerate(_STRAIN_NAMES):
                tens[name] = [_plane(*(f * np.array(_EPS_LOCAL[e][lay]))) for e in (0, 1)]
        if stress == "tensor":
            for lay, name in enumerate(_STRESS_NAMES):
                tens[name] = [_plane(*(f * np.array(_SIG_LOCAL[e][lay]))) for e in (0, 1)]
        elif stress == "von_mises":
            scal["2DELEM_Von_Mises"] = [111.0 * f, 222.0 * f]
        p = tmp_path / f"tiltA{k:03d}.vtk"
        _write_vtk(
            p,
            title="tilted quad + triangle",
            time_val=0.1 * (k + 1),
            points=_state_points(k),
            cells=cells,
            cell_scalars=scal,
            cell_tensors=tens,
        )
        paths.append(p)
    return paths


def _expected_global(table, k):
    """R t R^T per element/layer for state k, (2 elements, 2 layers, 6)."""
    P = _state_points(k)
    frames = (_clskew3(*P[0:4]), _c3coor3(*P[4:7]))
    out = np.zeros((2, 2, 6))
    for e in (0, 1):
        for lay in (0, 1):
            t = np.array(_plane(*((1.0 + k) * np.array(table[e][lay]))))
            out[e, lay] = _voigt(frames[e] @ t @ frames[e].T)
    return out


def _local_voigt(table, k):
    out = np.zeros((2, 2, 6))
    for e in (0, 1):
        for lay in (0, 1):
            out[e, lay] = _voigt(_plane(*((1.0 + k) * np.array(table[e][lay]))))
    return out


# ------------------------------------------- (a) solid shear slots (pins)


def test_a_pin_anim_to_vtk_solid_shear_slots(tmp_path):
    """(a) pin, passes on the base: anim_to_vtk writes every 3D tensor (stress,
    strain) of ANIM components v = [xx, yy, zz, xy, yz, zx] as rows
    [v0 v3 v4 / v3 v1 v5 / v4 v5 v2] (OpenCourant/Tools anim_to_vtk.cpp:1473-1475;
    ANIM order from engine srota6.F:191-196, written by tensor6.F:3365-3373), so
    VTK (0,2) holds yz and (1,2) holds zx. With xy=12, yz=23, zx=31 the d3plot
    slots [xx, yy, zz, xy, yz, zx] must read back [11, 22, 33, 12, 23, 31]."""
    from lasso.dyna import ArrayType as A

    from vtk2d3plot import convert

    v = [11.0, 22.0, 33.0, 12.0, 23.0, 31.0]
    rows = [[v[0], v[3], v[4]], [v[3], v[1], v[5]], [v[4], v[5], v[2]]]
    hexp = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    for k in range(2):
        _write_vtk(
            tmp_path / f"solA{k:03d}.vtk",
            title="anim_to_vtk solid",
            time_val=0.1 * (k + 1),
            points=hexp,
            cells=[(12, list(range(8)))],
            cell_tensors={"3DELEM_Stress": [rows], "3DELEM_Strain": [rows]},
        )
    convert(sorted(tmp_path.glob("solA*.vtk")), tmp_path / "sol.d3plot")
    d = _read(tmp_path / "sol.d3plot")
    np.testing.assert_array_equal(d.arrays[A.element_solid_stress][:, 0, 0], [v, v])
    np.testing.assert_array_equal(d.arrays[A.element_solid_strain][:, 0, 0], [v, v])


# pyfemsolver/output/anim_vtk.py _VOIGT9 at OpenRadioss_Python origin/main
# (56f5df0c, 2026-10-10; same as 1fd484b3, 2026-08-02): rows = sig[:, _VOIGT9]
_PORT_VOIGT9 = [0, 3, 4, 3, 1, 5, 4, 5, 2]


@pytest.mark.parametrize("title", ["pyfemsolver state t={t:.9E}", "pyradioss state t={t:.9E}"])
def test_a_pin_port_writer_solid_shear_slots(tmp_path, title):
    """(a) pin, passes on the base: the port writer fills the 3x3 from its solid
    sig Voigt [xx, yy, zz, xy, yz, zx] as sig[_VOIGT9] with the placement of the
    current upstream writer (yz at (0,2), zx at (1,2), like anim_to_vtk). With
    xy=1, yz=2, zx=3 the d3plot shear slots must read back [1, 2, 3]. (Port
    versions 145780ab..56f5df0c, 2026-09-14..2026-10-10, wrote zx at (0,2); their
    files cannot be told apart by content - see the cluster D report.)"""
    from lasso.dyna import ArrayType as A

    from vtk2d3plot import convert

    sig = np.array([10.0, 20.0, 30.0, 1.0, 2.0, 3.0])
    rows = sig[_PORT_VOIGT9].reshape(3, 3).tolist()
    hexp = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    for k in range(2):
        t = 0.1 * (k + 1)
        _write_vtk(
            tmp_path / f"portA{k:03d}.vtk",
            title=title.format(t=t),
            time_val=t,
            field_time=title.startswith("pyfemsolver"),
            points=hexp,
            cells=[(12, list(range(8)))],
            point_vectors={"DISPLACEMENT": [(0, 0, 0)] * 8, "VELOCITY": [(0, 0, 0)] * 8},
            cell_scalars={"VONM": [99.0], "EPSP": [0.0], "OFF": [1.0]},
            cell_tensors={"3DELEM_Stress": [rows]},
        )
    convert(sorted(tmp_path.glob("portA*.vtk")), tmp_path / "port.d3plot")
    s = _read(tmp_path / "port.d3plot").arrays[A.element_solid_stress]
    np.testing.assert_array_equal(s[:, 0, 0], [sig, sig])


# ------------------------------------------- (b) rotate the shell strain


def test_b_rotated_shell_strain_is_R_eps_RT(tmp_path):
    """(b) with rotate_shell_stress=True the shell strain is rotated with the
    same per-state frames as the stress: read back = R eps R^T (quad: CLSKEW3,
    triangle: N1->N2), both layers, every state; principal values unchanged.
    Base: the strain stayed in the element frame (equal to the local tensor)."""
    from lasso.dyna import ArrayType as A

    from vtk2d3plot import convert

    paths = _write_tilted_series(tmp_path)
    convert(paths, tmp_path / "rot.d3plot", rotate_shell_stress=True)
    d = _read(tmp_path / "rot.d3plot")
    strain = d.arrays[A.element_shell_strain]
    stress = d.arrays[A.element_shell_stress]
    assert strain.shape == (2, 2, 2, 6)
    for k in range(2):
        np.testing.assert_allclose(strain[k], _expected_global(_EPS_LOCAL, k), rtol=0, atol=1e-15)
        # stress keeps being rotated with the same frames (guard)
        np.testing.assert_allclose(stress[k], _expected_global(_SIG_LOCAL, k), rtol=0, atol=1e-10)
        np.testing.assert_allclose(_eig(strain[k]), _eig(_local_voigt(_EPS_LOCAL, k)), atol=1e-15)
    # the rotation produced out-of-plane components (the frames are tilted)
    assert np.abs(strain[..., [2, 4, 5]]).max() > 1e-5


def test_b_unrotated_shell_strain_unchanged(tmp_path):
    """(b) guard, passes on the base: without the flag the strain is written in
    the element frame exactly as the VTK holds it (tensorial shear unchanged)."""
    from lasso.dyna import ArrayType as A

    from vtk2d3plot import convert

    paths = _write_tilted_series(tmp_path)
    convert(paths, tmp_path / "plain.d3plot")
    strain = _read(tmp_path / "plain.d3plot").arrays[A.element_shell_strain]
    for k in range(2):
        np.testing.assert_array_equal(strain[k], _local_voigt(_EPS_LOCAL, k))


@pytest.mark.parametrize("stress", ["von_mises", None])
def test_b_strain_rotated_without_tensor_stress(tmp_path, stress):
    """(b) the strain is an element-frame tensor even when the shell stress is
    synthetic-uniaxial (2DELEM_Von_Mises) or absent: the flag rotates it, and
    the synthetic stress stays unchanged. Base: strain left local, and with no
    shell stress the flag did nothing at all, silently."""
    from lasso.dyna import ArrayType as A

    from vtk2d3plot import convert

    paths = _write_tilted_series(tmp_path, stress=stress)
    rep = convert(paths, tmp_path / "rot.d3plot", rotate_shell_stress=True)
    d = _read(tmp_path / "rot.d3plot")
    strain = d.arrays[A.element_shell_strain]
    for k in range(2):
        np.testing.assert_allclose(strain[k], _expected_global(_EPS_LOCAL, k), rtol=0, atol=1e-15)
    frame = [w for w in rep.warnings if "CLSKEW3" in w]
    assert len(frame) == 1 and frame[0].startswith("shell strain rotated with the OpenRadioss")
    assert not [w for w in rep.warnings if w.startswith("rotate_shell_stress ignored")]
    if stress == "von_mises":
        s = d.arrays[A.element_shell_stress]
        np.testing.assert_array_equal(s[1, :, :, 0], [[222.0, 222.0], [444.0, 444.0]])
        np.testing.assert_array_equal(s[..., 1:], 0.0)
        assert (
            "rotate_shell_stress: the shell stress is synthetic-uniaxial (no element frame "
            "to rotate from) and is written unchanged; only the shell strain is rotated"
        ) in rep.warnings


def test_b_zug_rotated_strain_matches_golden(tmp_path):
    """(b) zug_test3_RS carries 2-layer shell strain and the Vortex golden holds it
    in the global frame. Joined by shell ID, all 3 states, both layers:
    base (strain left in the element frame) 1.48 * max|eps|; rotated 2.1e-4,
    the VTK coordinate print precision (the stress residual is 1.6e-4).
    Doubling the shear (engineering hypothesis) gives 0.17, so the golden holds
    tensorial shear like the VTK."""
    from lasso.dyna import ArrayType as A

    from vtk2d3plot import convert

    rep = convert(GOLDEN_ZUG_VTK, tmp_path / "zug.d3plot", rotate_shell_stress=True)
    m, g = _read(tmp_path / "zug.d3plot"), _read(ZUG_GOLD)
    im = np.argsort(m.arrays[A.element_shell_ids])
    ig = np.argsort(g.arrays[A.element_shell_ids])
    assert np.array_equal(m.arrays[A.element_shell_ids][im], g.arrays[A.element_shell_ids][ig])
    em = m.arrays[A.element_shell_strain][:, im].astype(np.float64)
    eg = g.arrays[A.element_shell_strain][:, ig].astype(np.float64)
    assert em.shape == eg.shape == (3, 864, 2, 6)
    scale = np.abs(eg).max()
    assert scale == pytest.approx(0.0303265, rel=1e-5)
    rel = np.abs(em - eg).max() / scale
    assert rel < 5e-4, f"rel err {rel:.3e}"
    # warnings: no element-local claim for the strain, no "not rotated"
    assert not [w for w in rep.warnings if "shell strain" in w and "element-local" in w]
    assert not [w for w in rep.warnings if "not rotated" in w]
    assert "shell strain rotated from the element frame to the global frame" in rep.warnings
    frame = [w for w in rep.warnings if "CLSKEW3" in w]
    assert len(frame) == 1
    assert frame[0].startswith("shell stress and strain rotated with the OpenRadioss local frame")


@pytest.mark.parametrize(
    ("names", "subject"),
    [
        (_STRESS_NAMES + _STRAIN_NAMES, "shell stress and strain are"),
        (_STRAIN_NAMES, "shell strain is"),
    ],
    ids=["stress_and_strain", "strain_only"],
)
def test_b_degenerate_warning_names_strain(tmp_path, names, subject):
    """(b) a zero-area shell spoils the rotated strain too; the count warning
    must name what was rotated (stress and strain, or strain alone)."""
    from vtk2d3plot import convert

    cells = [(9, [0, 1, 2, 3]), (9, [4, 5, 6, 7])]
    good = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    flat = [(5, 0, 0), (6, 0, 0), (7, 0, 0), (8, 0, 0)]
    for k in range(2):
        _write_vtk(
            tmp_path / f"degA{k:03d}.vtk",
            title="degenerate",
            time_val=0.1 * (k + 1),
            points=good + flat,
            cells=cells,
            cell_tensors={n: [_plane(100.0, 50.0, 10.0)] * 2 for n in names},
        )
    rep = convert(
        sorted(tmp_path.glob("degA*.vtk")), tmp_path / "deg.d3plot", rotate_shell_stress=True
    )
    bad = [w for w in rep.warnings if "degenerate local frame" in w]
    expected = (
        "1 shells have a degenerate local frame (zero-area or collapsed geometry) in at "
        f"least one state (first: state 0); their rotated {subject} "
        "unreliable (zeroed or partly lost)"
    )
    assert bad == [expected]


# ------------------------------------------- S01 single-layer stress in every layer

_FLAT_QUADS_PTS = [(float(x), float(y), 0.0) for y in (0, 1) for x in (0, 1, 2, 3)]
_FLAT_QUADS = [(9, [i, i + 1, i + 5, i + 4]) for i in range(3)]


def _write_flat_series(tmp_path, cell_scalars_fn=None, cell_tensors_fn=None, n_states=2):
    paths = []
    for k in range(n_states):
        p = tmp_path / f"flatA{k:03d}.vtk"
        _write_vtk(
            p,
            title="three flat quads",
            time_val=0.1 * (k + 1),
            points=_FLAT_QUADS_PTS,
            cells=_FLAT_QUADS,
            cell_scalars=cell_scalars_fn(k) if cell_scalars_fn else None,
            cell_tensors=cell_tensors_fn(k) if cell_tensors_fn else None,
        )
        paths.append(p)
    return paths


def _membrane(k, q):
    return _plane(100.0 * (k + 1) + q, 50.0 - q, 10.0 + q)


def test_s01_membrane_stress_fills_every_layer(tmp_path):
    """S01: shell strain forces 2 layers; a 2DELEM_Stress_(membrane) value is
    constant through the thickness, so it must be in layer 1 too (like the
    duplicated membrane strain). Base: layer 1 all zero."""
    from lasso.dyna import ArrayType as A

    from vtk2d3plot import convert

    paths = _write_flat_series(
        tmp_path,
        cell_tensors_fn=lambda k: {
            "2DELEM_Stress_(membrane)": [_membrane(k, q) for q in range(3)],
            "2DELEM_Strain_(membrane)": [_plane(1e-3, 2e-4, 3e-5)] * 3,
        },
    )
    rep = convert(paths, tmp_path / "mem.d3plot")
    s = _read(tmp_path / "mem.d3plot").arrays[A.element_shell_stress]
    assert s.shape == (2, 3, 2, 6)
    for k in range(2):
        exp = np.array([_voigt(_membrane(k, q)) for q in range(3)])
        np.testing.assert_array_equal(s[k, :, 0], exp)
        np.testing.assert_array_equal(s[k, :, 1], exp)
    rec = [w for w in rep.warnings if "reconciled" in w]
    expected = (
        "shell element results have differing integration layer counts (stress=1, eps=0, "
        "strain=2) — reconciled to 2 layers (d3plot MAXINT constraint): shell stress "
        "from 2DELEM_Stress_(membrane) copied to every layer"
    )
    assert rec == [expected]


def test_s01_von_mises_synthetic_fills_every_layer(tmp_path):
    """S01: 2-layer plastic strain forces 2 layers; the synthetic-uniaxial
    stress from 2DELEM_Von_Mises must be in layer 1 too. Base: layer 1 zero."""
    from lasso.dyna import ArrayType as A

    from vtk2d3plot import convert

    paths = _write_flat_series(
        tmp_path,
        cell_scalars_fn=lambda k: {
            "2DELEM_Von_Mises": [10.0 * (k + 1) + q for q in range(3)],
            "2DELEM_Plastic_Strain_Lower": [0.01, 0.02, 0.03],
            "2DELEM_Plastic_Strain_Upper": [0.04, 0.05, 0.06],
        },
    )
    rep = convert(paths, tmp_path / "vm.d3plot")
    s = _read(tmp_path / "vm.d3plot").arrays[A.element_shell_stress]
    assert s.shape == (2, 3, 2, 6)
    for k in range(2):
        vm = [10.0 * (k + 1) + q for q in range(3)]
        np.testing.assert_array_equal(s[k, :, 0, 0], vm)
        np.testing.assert_array_equal(s[k, :, 1, 0], vm)
    np.testing.assert_array_equal(s[..., 1:], 0.0)
    assert any(
        "shell stress from 2DELEM_Von_Mises copied to every layer" in w for w in rep.warnings
    )


def test_s01_port_vonm_fills_every_layer(tmp_path):
    """S01: port-dialect VONM (a max over layers, like EPSP, which is already
    replicated) with per-layer plastic strain forcing 2 layers: layer 1 must
    carry VONM too. Base: layer 1 zero."""
    from lasso.dyna import ArrayType as A

    from vtk2d3plot import convert

    for k in range(2):
        t = 0.1 * (k + 1)
        _write_vtk(
            tmp_path / f"BOXA{k:03d}.vtk",
            title=f"pyradioss state t={t:.9E}",
            time_val=t,
            field_time=False,
            ids=False,
            points=_FLAT_QUADS_PTS,
            cells=_FLAT_QUADS,
            point_vectors={"DISPLACEMENT": [(0, 0, 0)] * 8, "VELOCITY": [(0, 0, 0)] * 8},
            cell_scalars={
                "VONM": [5.0 * (k + 1), 6.0, 7.0],
                "EPSP": [0.0, 0.1, 0.2],
                "OFF": [1.0, 1.0, 1.0],
                "2DELEM_Plastic_Strain_Lower": [0.0, 0.05, 0.1],
                "2DELEM_Plastic_Strain_Upper": [0.0, 0.1, 0.2],
            },
        )
    rep = convert(sorted(tmp_path.glob("BOXA*.vtk")), tmp_path / "box.d3plot")
    s = _read(tmp_path / "box.d3plot").arrays[A.element_shell_stress]
    assert s.shape == (2, 3, 2, 6)
    np.testing.assert_array_equal(s[1, :, 0, 0], [10.0, 6.0, 7.0])
    np.testing.assert_array_equal(s[1, :, 1, 0], [10.0, 6.0, 7.0])
    assert any("shell stress from VONM copied to every layer" in w for w in rep.warnings)


def test_s01_upper_only_stress_stays_in_its_layer(tmp_path):
    """S01 guard (milestone-8 P1b rule), passes on the base: an upper-surface-only
    stress is not a through-thickness value; it stays in layer 1, layer 0 zero.
    The reconciliation warning says so."""
    from lasso.dyna import ArrayType as A

    from vtk2d3plot import convert

    paths = _write_flat_series(
        tmp_path,
        cell_tensors_fn=lambda k: {
            "2DELEM_Stress_(upper)": [_membrane(k, q) for q in range(3)],
            "2DELEM_Strain_(membrane)": [_plane(1e-3, 2e-4, 3e-5)] * 3,
        },
    )
    rep = convert(paths, tmp_path / "up.d3plot")
    s = _read(tmp_path / "up.d3plot").arrays[A.element_shell_stress]
    np.testing.assert_array_equal(s[:, :, 0], 0.0)
    np.testing.assert_array_equal(s[1, :, 1], [_voigt(_membrane(1, q)) for q in range(3)])
    rec = [w for w in rep.warnings if "reconciled" in w]
    assert len(rec) == 1
    assert rec[0].endswith(
        "shell stress from 2DELEM_Stress_(upper) in layer 1 only, other layers zero"
    )


@pytest.mark.parametrize(
    ("scalars", "tensor", "detail"),
    [
        (
            {"2DELEM_Plastic_Strain": [0.1, 0.2, 0.3]},
            "2DELEM_Stress_(membrane)",
            (
                "shell stress from 2DELEM_Stress_(membrane) copied to every layer; "
                "plastic strain from 2DELEM_Plastic_Strain copied to every layer"
            ),
        ),
        (
            {"2DELEM_Plastic_Strain_Upper": [0.1, 0.2, 0.3]},
            "2DELEM_Stress_(lower)",
            (
                "shell stress from 2DELEM_Stress_(lower) in layer 0 only, other layers zero; "
                "plastic strain from 2DELEM_Plastic_Strain_Upper in layer 1 only, "
                "other layers zero"
            ),
        ),
        (
            {"2DELEM_Plastic_Strain_Lower": [0.1, 0.2, 0.3]},
            "2DELEM_Stress_(upper)",
            (
                "shell stress from 2DELEM_Stress_(upper) in layer 1 only, other layers zero; "
                "plastic strain from 2DELEM_Plastic_Strain_Lower in layer 0 only, "
                "other layers zero"
            ),
        ),
    ],
    ids=["membrane_single_eps", "lower_only_upper_eps", "upper_only_lower_eps"],
)
def test_s01_reconcile_warning_names_each_family(tmp_path, scalars, tensor, detail):
    """S01: the reconciliation warning says, per single-layer family, whether the
    value was copied to every layer or sits in one layer only; the values agree."""
    from lasso.dyna import ArrayType as A

    from vtk2d3plot import convert

    paths = _write_flat_series(
        tmp_path,
        cell_scalars_fn=lambda k: scalars,
        cell_tensors_fn=lambda k: {
            tensor: [_membrane(k, q) for q in range(3)],
            "2DELEM_Strain_(membrane)": [_plane(1e-3, 2e-4, 3e-5)] * 3,
        },
    )
    rep = convert(paths, tmp_path / "rec.d3plot")
    rec = [w for w in rep.warnings if "reconciled" in w]
    head = (
        "shell element results have differing integration layer counts (stress=1, eps=1, "
        "strain=2) — reconciled to 2 layers (d3plot MAXINT constraint): "
    )
    assert rec == [head + detail]
    d = _read(tmp_path / "rec.d3plot")
    s = d.arrays[A.element_shell_stress][1]
    e = d.arrays[A.element_shell_effective_plastic_strain][1]
    sv = np.array([_voigt(_membrane(1, q)) for q in range(3)])
    ev = np.array(next(iter(scalars.values())))
    layer_s = {"(membrane)": (0, 1), "(lower)": (0,), "(upper)": (1,)}[tensor[14:]]
    layer_e = {"Strain": (0, 1), "Upper": (1,), "Lower": (0,)}[next(iter(scalars)).split("_")[-1]]
    for lay in (0, 1):
        np.testing.assert_array_equal(s[:, lay], sv if lay in layer_s else 0.0)
        np.testing.assert_allclose(e[:, lay], ev if lay in layer_e else 0.0, rtol=1e-15)


def test_s01_port_fixture_unchanged(tmp_path):
    """S01 guard, passes on the base: the port fixture has one layer everywhere
    (VONM + EPSP), so no layer is added and no reconciliation warning appears."""
    from lasso.dyna import ArrayType as A

    from vtk2d3plot import convert

    rep = convert(PORT_DIR, tmp_path / "box.d3plot")
    s = _read(tmp_path / "box.d3plot").arrays[A.element_shell_stress]
    assert s.shape == (21, 200, 1, 6)
    assert not [w for w in rep.warnings if "reconciled" in w]


# ------------------------------------------- S12 beam orientation / null words


def test_s12_beam_orientation_and_null_words_are_zero(tmp_path):
    """S12: d3plot IX2 = N1, N2, orientation node, two null entries, material
    (LS-DYNA database manual Jan 2023 p. 18; the null entries may carry a beam
    type and section ratios, type = ix2(5) & 0x3F). A VTK line has no
    orientation node, so words 3-5 are 0, i.e. lasso index -1 (lasso writes
    index + 1). Base: N1 repeated in words 3-5 (an orientation node on the beam
    axis, and a node number where the manual expects null)."""
    from lasso.dyna import ArrayType as A

    from vtk2d3plot import convert

    pts = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0)]
    for k in range(2):
        _write_vtk(
            tmp_path / f"beamA{k:03d}.vtk",
            title="beams",
            time_val=0.1 * (k + 1),
            points=pts,
            cells=[(3, [0, 1]), (3, [1, 2]), (3, [3, 2])],
        )
    convert(sorted(tmp_path.glob("beamA*.vtk")), tmp_path / "beam.d3plot")
    b = _read(tmp_path / "beam.d3plot").arrays[A.element_beam_node_indexes]
    np.testing.assert_array_equal(b, [[0, 1, -1, -1, -1], [1, 2, -1, -1, -1], [3, 2, -1, -1, -1]])


@pytest.mark.lspp
def test_s12_beams_with_null_words_load_in_lsprepost(tmp_path):
    """S12 smoke: LS-PrePost loads a beam + shell model whose beam words 3-5 are 0."""
    from tools.lspp_check import check_d3plot
    from vtk2d3plot import convert

    pts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, -2), (1, 1, -2)]
    for k in range(2):
        _write_vtk(
            tmp_path / f"bsA{k:03d}.vtk",
            title="beams and shell",
            time_val=0.1 * (k + 1),
            points=pts,
            cells=[(9, [0, 1, 2, 3]), (3, [0, 4]), (3, [2, 5])],
        )
    convert(sorted(tmp_path.glob("bsA*.vtk")), tmp_path / "bs.d3plot")
    r = check_d3plot(tmp_path / "bs.d3plot")
    assert r["loaded"] and r["n_states"] == 2, r["detail"]
