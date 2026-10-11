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


def test_c06_degenerate_frames_are_flagged():
    """Bug report 2026-10-11 #1 (c06), verifier follow-up: shells whose frame
    cannot be built (collapsed quad, zero-area triangle) get zero axes and their
    rotated stress is wrong (zeroed or partly lost). degenerate_shell_frames
    must flag exactly those shells so the caller can warn instead of failing
    silently."""
    from vtk2d3plot.shell_rotation import (
        compute_shell_local_frames,
        degenerate_shell_frames,
        rotate_stress_to_global,
    )

    pts = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [2.0, 0.0, 0.0],  # 4 (collinear with 0, 1)
        ]
    )
    conn = np.array(
        [
            [0, 1, 2, 3],  # good quad
            [0, 0, 0, 0],  # quad collapsed to a point
            [0, 1, 2, 2],  # good triangle
            [0, 1, 4, 4],  # zero-area triangle (collinear)
        ]
    )
    R = compute_shell_local_frames(pts, conn)
    np.testing.assert_array_equal(degenerate_shell_frames(R), [False, True, False, True])
    # the flagged shells really get wrong stress under rotation: the collapsed
    # quad has R = 0 (all zero), the collinear triangle keeps only e1 = +x, so
    # only local xx survives (yy and xy are lost)
    local = np.tile(np.array([100.0, 50.0, 0.0, 10.0, 0.0, 0.0]), (4, 1, 1))
    g = rotate_stress_to_global(local, R)
    np.testing.assert_array_equal(g[1], 0.0)
    np.testing.assert_allclose(g[3, 0], [100.0, 0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(g[0, 0], [100.0, 50.0, 0.0, 10.0, 0.0, 0.0], atol=1e-12)
    assert degenerate_shell_frames(np.empty((0, 3, 3))).shape == (0,)


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


# ------------------------------------------- integration: builder warnings


def test_c06_rotation_warnings_name_the_real_frame(tmp_path):
    """Bug report 2026-10-11 #1 (c06), builder side: with rotation the report must
    not claim a 'diagonal-based local frame' (the old, wrong frame) nor keep the
    'element-local coordinates' wording; it names the OpenRadioss frame once."""
    from vtk2d3plot import convert

    rep = convert(GOLDEN_W3_VTK, tmp_path / "w3.d3plot", rotate_shell_stress=True)
    text = "\n".join(rep.warnings)
    assert "diagonal" not in text
    assert "shell stress in rotated" not in text
    assert not [w for w in rep.warnings if w.startswith("shell stress") and "element-local" in w]
    assert "shell stress rotated from the element frame to the global frame" in rep.warnings
    frame = [w for w in rep.warnings if "CLSKEW3" in w]
    assert len(frame) == 1 and "N1->N2" in frame[0]
    assert not [w for w in rep.warnings if "degenerate local frame" in w]


def test_c06_rotation_skips_synthetic_uniaxial_stress(tmp_path):
    """Bug report 2026-10-11 #1 (c06), builder side: the port fixture's shell stress
    is synthetic-uniaxial [VONM, 0, 0, 0, 0, 0] with no element frame. Rotating it
    would invent components, so it is written unchanged and the report says so."""
    from conftest import PORT_DIR
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot import convert

    rep = convert(PORT_DIR, tmp_path / "rot.d3plot", rotate_shell_stress=True)
    convert(PORT_DIR, tmp_path / "plain.d3plot")
    rot = D3plot(str(tmp_path / "rot.d3plot")).arrays[ArrayType.element_shell_stress]
    plain = D3plot(str(tmp_path / "plain.d3plot")).arrays[ArrayType.element_shell_stress]
    assert np.abs(plain[..., 0]).max() > 0.0
    np.testing.assert_array_equal(rot, plain)
    assert any(w.startswith("rotate_shell_stress ignored") for w in rep.warnings)
    assert not [w for w in rep.warnings if "CLSKEW3" in w]


def _write_two_quad_state(path, time_val):
    """Official-dialect state: one good quad and one zero-area (collinear) quad,
    both with 2-layer shell stress tensors (local xx = 100, yy = 50, xy = 10)."""
    pts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    pts += [(5, 0, 0), (6, 0, 0), (7, 0, 0), (8, 0, 0)]
    lines = [
        "# vtk DataFile Version 3.0",
        "two quads",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        "FIELD FieldData 1",
        "TIME 1 1 double",
        repr(float(time_val)),
        "POINTS 8 float",
        *[f"{x} {y} {z}" for x, y, z in pts],
        "CELLS 2 10",
        "4 0 1 2 3",
        "4 4 5 6 7",
        "CELL_TYPES 2",
        "9",
        "9",
        "POINT_DATA 8",
        "SCALARS NODE_ID int 1",
        "LOOKUP_TABLE default",
        "1 2 3 4 5 6 7 8",
        "CELL_DATA 2",
        "SCALARS ELEMENT_ID int 1",
        "LOOKUP_TABLE default",
        "1 2",
        "SCALARS PART_ID int 1",
        "LOOKUP_TABLE default",
        "1 1",
    ]
    for name in ("2DELEM_Stress_(lower)", "2DELEM_Stress_(upper)"):
        lines.append(f"TENSORS {name} float")
        lines += ["100 10 0", "10 50 0", "0 0 0"] * 2
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_c06_rotation_warns_about_degenerate_shells(tmp_path):
    """Bug report 2026-10-11 #1 (c06), builder side: a shell whose local frame cannot
    be built (here a zero-area quad, r = 0) gets wrong rotated stress; the report
    must count it instead of failing silently."""
    from vtk2d3plot import convert

    for k in range(2):
        _write_two_quad_state(tmp_path / f"runA{k:03d}.vtk", 0.1 * (k + 1))
    rep = convert(
        sorted(tmp_path.glob("runA*.vtk")), tmp_path / "out.d3plot", rotate_shell_stress=True
    )
    bad = [w for w in rep.warnings if "degenerate local frame" in w]
    expected = (
        "1 shells have a degenerate local frame (zero-area or collapsed geometry) in at "
        "least one state (first: state 0); their rotated shell stress is unreliable "
        "(zeroed or partly lost)"
    )
    assert bad == [expected]
