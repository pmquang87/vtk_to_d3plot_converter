"""Element-local -> global rotation of shell stress (``rotate_shell_stress``).

anim_to_vtk writes shell stress in the OpenRadioss element frame. This module
rebuilds that frame from the deformed nodal coordinates and rotates the
tensors to the global frame that d3plot expects.

Voigt slot convention (used for input AND output): the d3plot / builder
order ``[xx, yy, zz, xy, yz, zx]`` (LS-DYNA database manual, Jan 2023, p. 46:
Sigma-x, -y, -z, -xy, -yz, -zx). Slot 4 is the (1,2) tensor entry, slot 5 the
(2,0) entry.
"""

import numpy as np

_EPS = 1e-12


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n < _EPS, 1.0, n)


def compute_shell_local_frames(points: np.ndarray, shell_conn: np.ndarray) -> np.ndarray:
    """
    Compute the OpenRadioss local frame of each shell from nodal coordinates.

    4-node shells: the symmetric frame of OpenRadioss CLSKEW3 (IREP=0), as
    ported in pyfemsolver/elements/shell_qeph.py:148-172 and shell_qbat.py:185-225:
        r  = x2 + x3 - x1 - x4,   s = x3 + x4 - x1 - x2
        e3 = unit(r x s)
        e1 = unit(r * |s|/|r| + s x e3)
        e2 = e3 x e1
    e1 and e2 lie symmetric about the bisector of r and s (each 45 deg from it).

    3-node shells (connectivity [n1, n2, n3, n3]): OpenRadioss c3coor3 frame
    (pyfemsolver/elements/shell_tri3.py:95-101): e1 = unit(x2 - x1),
    e3 = unit((x2 - x1) x (x3 - x1)), e2 = e3 x e1.

    Args:
        points: (n_nodes, 3) array of deformed coordinates
        shell_conn: (n_shells, 4) array of zero-based node indices

    Returns:
        R: (n_shells, 3, 3) rotation matrices mapping local to global. R = [e1, e2, e3]
        as columns.
    """
    if len(shell_conn) == 0:
        return np.empty((0, 3, 3), dtype=points.dtype)

    N1 = points[shell_conn[:, 0]]
    N2 = points[shell_conn[:, 1]]
    N3 = points[shell_conn[:, 2]]
    N4 = points[shell_conn[:, 3]]

    is_triangle = shell_conn[:, 2] == shell_conn[:, 3]
    is_quad = ~is_triangle

    e1 = np.zeros_like(N1)
    e3 = np.zeros_like(N1)

    # Quads: CLSKEW3 symmetric frame
    if np.any(is_quad):
        r = N2[is_quad] + N3[is_quad] - N1[is_quad] - N4[is_quad]
        s = N3[is_quad] + N4[is_quad] - N1[is_quad] - N2[is_quad]
        e3_q = _unit(np.cross(r, s))
        r_len = np.linalg.norm(r, axis=1, keepdims=True)
        s_len = np.linalg.norm(s, axis=1, keepdims=True)
        ratio = np.where(r_len < _EPS, 1.0, s_len / np.where(r_len < _EPS, 1.0, r_len))
        e1[is_quad] = _unit(r * ratio + np.cross(s, e3_q))
        e3[is_quad] = e3_q

    # Triangles: c3coor3 frame, e1 along N1->N2
    if np.any(is_triangle):
        v12 = N2[is_triangle] - N1[is_triangle]
        v13 = N3[is_triangle] - N1[is_triangle]
        e1[is_triangle] = _unit(v12)
        e3[is_triangle] = _unit(np.cross(v12, v13))

    e2 = np.cross(e3, e1)

    R = np.stack([e1, e2, e3], axis=-1)
    return R


def degenerate_shell_frames(R: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Flag shells whose local frame could not be built (zero-area or collapsed
    geometry: r x s = 0 for quads, (x2 - x1) x (x3 - x1) = 0 for triangles).

    compute_shell_local_frames returns zero axes for such shells (all three for
    a collapsed quad; e2 and e3 for a zero-area triangle), so
    rotate_stress_to_global zeroes or partly drops their stress. Callers must
    warn about the True entries (count) instead of failing silently.

    Args:
        R: (n_shells, 3, 3) frames from compute_shell_local_frames
        tol: allowed deviation of det(R) from +1

    Returns:
        (n_shells,) bool, True where R is not a proper rotation.
    """
    if len(R) == 0:
        return np.zeros(0, dtype=bool)
    return np.abs(np.linalg.det(R.astype(np.float64)) - 1.0) > tol


def rotate_stress_to_global(local_stress_voigt: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Rotate local stress tensors to the global frame: S_g = R S_l R^T.

    Args:
        local_stress_voigt: (n_shells, n_layers, 6) Voigt stress in the d3plot /
            builder slot order [xx, yy, zz, xy, yz, zx]
        R: (n_shells, 3, 3) rotation matrices (columns e1, e2, e3)

    Returns:
        global_stress_voigt: (n_shells, n_layers, 6) rotated stress, same slot
        order [xx, yy, zz, xy, yz, zx]
    """
    if len(local_stress_voigt) == 0:
        return local_stress_voigt

    n_shells, n_layers, _ = local_stress_voigt.shape

    # Reconstruct 3x3 tensor
    S_local = np.zeros((n_shells, n_layers, 3, 3), dtype=local_stress_voigt.dtype)
    S_local[:, :, 0, 0] = local_stress_voigt[:, :, 0]
    S_local[:, :, 1, 1] = local_stress_voigt[:, :, 1]
    S_local[:, :, 2, 2] = local_stress_voigt[:, :, 2]

    S_local[:, :, 0, 1] = local_stress_voigt[:, :, 3]  # xy
    S_local[:, :, 1, 0] = local_stress_voigt[:, :, 3]

    S_local[:, :, 1, 2] = local_stress_voigt[:, :, 4]  # yz
    S_local[:, :, 2, 1] = local_stress_voigt[:, :, 4]

    S_local[:, :, 2, 0] = local_stress_voigt[:, :, 5]  # zx
    S_local[:, :, 0, 2] = local_stress_voigt[:, :, 5]

    R_exp = np.expand_dims(R, axis=1)  # (n_shells, 1, 3, 3)
    R_T_exp = R_exp.transpose(0, 1, 3, 2)

    S_global = np.matmul(np.matmul(R_exp, S_local), R_T_exp)

    global_stress_voigt = np.empty_like(local_stress_voigt)
    global_stress_voigt[:, :, 0] = S_global[:, :, 0, 0]
    global_stress_voigt[:, :, 1] = S_global[:, :, 1, 1]
    global_stress_voigt[:, :, 2] = S_global[:, :, 2, 2]
    global_stress_voigt[:, :, 3] = S_global[:, :, 0, 1]
    global_stress_voigt[:, :, 4] = S_global[:, :, 1, 2]
    global_stress_voigt[:, :, 5] = S_global[:, :, 2, 0]

    return global_stress_voigt
