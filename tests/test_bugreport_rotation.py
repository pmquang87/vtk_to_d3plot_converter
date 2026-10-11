"""Bug report 2026-10-11, cluster A: shell stress rotation (#1 c06, #2 c07).

d3plot shell stress slot order is [xx, yy, zz, xy, yz, zx] (LS-DYNA
database manual Jan 2023, p. 46: Sigma-x, -y, -z, -xy, -yz, -zx).
OpenRadioss 4-node shells use the symmetric local frame of CLSKEW3
(IREP=0): e3 = unit(r x s), e1 ~ r*|s|/|r| + s x e3, e2 = e3 x e1, with
r = x2+x3-x1-x4 and s = x3+x4-x1-x2 (pyfemsolver/elements/shell_qeph.py:148-172).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import GOLDEN_W3_D3, GOLDEN_W3_VTK, GOLDEN_ZUG_D3, GOLDEN_ZUG_VTK

pytestmark = pytest.mark.bugreport

W3_GOLD = GOLDEN_W3_D3 / "W3_SETUP_Ball_impact__ton-mm-s.d3plot"
ZUG_GOLD = GOLDEN_ZUG_D3 / "zug_test3_RS.d3plot"


def _rotated_vs_golden(vtk_dir, gold_path, out):
    """Convert with rotation; return (mine, gold) shell stress joined by shell ID."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot import convert

    convert(vtk_dir, out, rotate_shell_stress=True)
    m, g = D3plot(str(out)), D3plot(str(gold_path))
    im = np.argsort(m.arrays[ArrayType.element_shell_ids])
    ig = np.argsort(g.arrays[ArrayType.element_shell_ids])
    assert np.array_equal(
        m.arrays[ArrayType.element_shell_ids][im], g.arrays[ArrayType.element_shell_ids][ig]
    )
    sm = m.arrays[ArrayType.element_shell_stress][:, im].astype(np.float64)
    sg = g.arrays[ArrayType.element_shell_stress][:, ig].astype(np.float64)
    assert sm.shape == sg.shape
    return m, sm, sg


# ---------------------------------------------------------------- #1 (c06)


def test_c06_quad_frame_is_openradioss_symmetric_frame():
    """Bug report 2026-10-11 #1 (c06): quad e1/e2 lie symmetric about the
    bisector of r and s (OpenRadioss CLSKEW3), not along the diagonals.

    Parallelogram (0,0)-(2,0)-(3,1)-(1,1): r = (4,0,0) at 0 deg, s = (2,2,0)
    at 45 deg, bisector 22.5 deg -> e1 at -22.5 deg, e2 at +67.5 deg.
    The old diagonal frame gave e1 = unit(N3-N1) at 18.43 deg.
    """
    from vtk2d3plot.shell_rotation import compute_shell_local_frames

    pts = np.array([[0.0, 0, 0], [2.0, 0, 0], [3.0, 1, 0], [1.0, 1, 0]])
    R = compute_shell_local_frames(pts, np.array([[0, 1, 2, 3]]))
    c, s = np.cos(np.pi / 8), np.sin(np.pi / 8)
    np.testing.assert_allclose(R[0, :, 0], [c, -s, 0.0], atol=1e-14)
    np.testing.assert_allclose(R[0, :, 1], [s, c, 0.0], atol=1e-14)
    np.testing.assert_allclose(R[0, :, 2], [0.0, 0.0, 1.0], atol=1e-14)


def test_c06_quad_frame_tilted_warped_matches_clskew3_formula():
    """Bug report 2026-10-11 #1 (c06): warped quad in a tilted plane; the frame
    must equal CLSKEW3 e1 = unit(r*|s|/|r| + s x e3), e3 = unit(r x s)."""
    from vtk2d3plot.shell_rotation import compute_shell_local_frames

    rng = np.random.default_rng(7)
    base = np.array([[0.0, 0, 0], [1.3, 0.1, 0], [1.1, 0.9, 0], [-0.2, 1.2, 0]])
    pts = base + 0.05 * rng.standard_normal((4, 3))
    a = 0.6
    rot = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    pts = pts @ rot.T
    R = compute_shell_local_frames(pts, np.array([[0, 1, 2, 3]]))

    r = pts[1] + pts[2] - pts[0] - pts[3]
    s = pts[2] + pts[3] - pts[0] - pts[1]
    e3 = np.cross(r, s) / np.linalg.norm(np.cross(r, s))
    e1 = r * np.linalg.norm(s) / np.linalg.norm(r) + np.cross(s, e3)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    np.testing.assert_allclose(R[0], np.stack([e1, e2, e3], axis=-1), atol=1e-13)


def test_c06_w3_rotated_inplane_stress_matches_golden(tmp_path):
    """Bug report 2026-10-11 #1 (c06): W3 with rotate_shell_stress=True must
    reproduce the global-frame Vortex golden (by shell ID, all 11 states,
    both layers). Unfixed: in-plane error 0.2746 * max|sigma|."""
    _, sm, sg = _rotated_vs_golden(GOLDEN_W3_VTK, W3_GOLD, tmp_path / "w3.d3plot")
    assert sm.shape == (11, 400, 2, 6)
    scale = np.abs(sg).max()
    assert scale == pytest.approx(915.603, rel=1e-5)
    rel = np.abs(sm[..., [0, 1, 3]] - sg[..., [0, 1, 3]]).max() / scale
    assert rel < 1e-5, f"in-plane rel err {rel:.3e}"
    # all six slots, including the transverse shear slots and szz
    rel6 = np.abs(sm - sg).max() / scale
    assert rel6 < 1e-5, f"all-slot rel err {rel6:.3e}"


def test_c06_zug_rotated_inplane_stress_matches_golden(tmp_path):
    """Bug report 2026-10-11 #1 (c06): same check on zug_test3_RS (flat shells,
    3 states, 2 layers). Unfixed: error 0.608 * max|sigma|.

    Fixed: 1.6e-4 * max|sigma|. That residual is the VTK print precision of
    POINTS (6 significant digits, e.g. "225.182" on 2.4 mm elements -> frame
    angle error ~1.6e-4 rad); the same rotation with the golden binary
    coordinates reproduces the golden to 3.5e-6. Hence the 5e-4 bound."""
    _, sm, sg = _rotated_vs_golden(GOLDEN_ZUG_VTK, ZUG_GOLD, tmp_path / "zug.d3plot")
    assert sm.shape == (3, 864, 2, 6)
    scale = np.abs(sg).max()
    assert scale == pytest.approx(0.622875, rel=1e-5)
    rel = np.abs(sm - sg).max() / scale
    assert rel < 5e-4, f"rel err {rel:.3e}"


def test_c06_triangle_frame_unchanged_n1n2():
    """Bug report 2026-10-11 #1 (c06), guard: 3-node shells keep the OpenRadioss
    c3coor3 frame e1 = unit(N2-N1), e3 = unit((N2-N1) x (N3-N1)) - the quad fix
    must not touch it (pyfemsolver/elements/shell_tri3.py:95-101)."""
    from vtk2d3plot.shell_rotation import compute_shell_local_frames

    pts = np.array([[1.0, 2.0, 0.5], [3.0, 2.5, 1.0], [1.5, 4.0, 2.0]])
    R = compute_shell_local_frames(pts, np.array([[0, 1, 2, 2]]))
    v12, v13 = pts[1] - pts[0], pts[2] - pts[0]
    e1 = v12 / np.linalg.norm(v12)
    e3 = np.cross(v12, v13) / np.linalg.norm(np.cross(v12, v13))
    np.testing.assert_allclose(R[0], np.stack([e1, np.cross(e3, e1), e3], axis=-1), atol=1e-14)


# ---------------------------------------------------------------- #2 (c07)


def test_c07_rotate_uses_d3plot_shear_slots():
    """Bug report 2026-10-11 #2 (c07): rotate_stress_to_global reads and writes
    slots 4/5 as d3plot [yz, zx]. Local frame e1 = +y, e2 = -x, e3 = z; a pure
    local yz = 1 (slot 4) becomes global zx = -1 (slot 5), yz = 0.
    Unfixed: [0, 0, 0, 0, 0, 1] (slot 4 read as xz, written as xz)."""
    from vtk2d3plot.shell_rotation import rotate_stress_to_global

    R = np.array([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    local = np.array([[[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]])
    np.testing.assert_allclose(
        rotate_stress_to_global(local, R), [[[0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]], atol=1e-15
    )
    # local zx = 1 (slot 5) -> S_g = e3 e1^T + e1 e3^T -> global yz = 1 (slot 4)
    local = np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]])
    np.testing.assert_allclose(
        rotate_stress_to_global(local, R), [[[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]], atol=1e-15
    )


def test_c07_w3_rotated_stress_is_traction_free_on_midplane(tmp_path):
    """Bug report 2026-10-11 #2 (c07): plane stress rotated to global must give
    S.n = 0 with S read in d3plot order [xx,yy,zz,xy,yz,zx] and n the shell
    normal from the deformed geometry of each state. Independent of the
    in-plane frame (n = unit(r x s) = unit(v13 x v24) for any in-plane e1).
    Unfixed: max|S.n| = 300 against max|S| = 1130."""
    from lasso.dyna import ArrayType

    m, _, _ = _rotated_vs_golden(GOLDEN_W3_VTK, W3_GOLD, tmp_path / "w3.d3plot")
    # re-read unsorted arrays for the geometry join (connectivity is by index)
    st = m.arrays[ArrayType.element_shell_stress].astype(np.float64)
    conn = m.arrays[ArrayType.element_shell_node_indexes]
    X = m.arrays[ArrayType.node_displacement].astype(np.float64)  # absolute coords
    worst = 0.0
    for i in range(st.shape[0]):
        P = X[i][conn]
        n = np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        v = st[i]
        S = np.zeros(v.shape[:-1] + (3, 3))
        S[..., 0, 0], S[..., 1, 1], S[..., 2, 2] = v[..., 0], v[..., 1], v[..., 2]
        S[..., 0, 1] = S[..., 1, 0] = v[..., 3]
        S[..., 1, 2] = S[..., 2, 1] = v[..., 4]
        S[..., 2, 0] = S[..., 0, 2] = v[..., 5]
        worst = max(worst, float(np.abs(np.einsum("elij,ej->eli", S, n)).max()))
    assert np.abs(st).max() > 900.0
    assert worst < 1e-5 * np.abs(st).max(), f"max|S.n| = {worst:.4g}"
