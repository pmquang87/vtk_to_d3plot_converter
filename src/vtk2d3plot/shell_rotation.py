import numpy as np


def compute_shell_local_frames(points: np.ndarray, shell_conn: np.ndarray) -> np.ndarray:
    """
    Compute local coordinate frames for shell elements based on nodal coordinates.

    Args:
        points: (n_nodes, 3) array of deformed coordinates
        shell_conn: (n_shells, 4) array of zero-based node indices

    Returns:
        R: (n_shells, 3, 3) rotation matrices mapping local to global. R = [e1, e2, e3].
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
    e2_raw = np.zeros_like(N1)

    # Quads
    v13 = N3[is_quad] - N1[is_quad]
    v24 = N4[is_quad] - N2[is_quad]
    e1[is_quad] = v13
    e2_raw[is_quad] = v24

    # Triangles
    v12 = N2[is_triangle] - N1[is_triangle]
    v13_tri = N3[is_triangle] - N1[is_triangle]
    e1[is_triangle] = v12
    e2_raw[is_triangle] = v13_tri

    # Normalize e1
    e1_norm = np.linalg.norm(e1, axis=1, keepdims=True)
    e1_norm = np.where(e1_norm < 1e-12, 1.0, e1_norm)
    e1 = e1 / e1_norm

    e3 = np.cross(e1, e2_raw)
    e3_norm = np.linalg.norm(e3, axis=1, keepdims=True)
    e3_norm = np.where(e3_norm < 1e-12, 1.0, e3_norm)
    e3 = e3 / e3_norm

    e2 = np.cross(e3, e1)

    R = np.stack([e1, e2, e3], axis=-1)
    return R


def rotate_stress_to_global(local_stress_voigt: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Rotate local stress tensors to global frame.

    Args:
        local_stress_voigt: (n_shells, n_layers, 6) voigt stress [xx, yy, zz, xy, xz, yz]
        R: (n_shells, 3, 3) rotation matrices

    Returns:
        global_stress_voigt: (n_shells, n_layers, 6) rotated stress
    """
    if len(local_stress_voigt) == 0:
        return local_stress_voigt

    n_shells, n_layers, _ = local_stress_voigt.shape

    # Reconstruct 3x3 tensor
    S_local = np.zeros((n_shells, n_layers, 3, 3), dtype=local_stress_voigt.dtype)
    S_local[:, :, 0, 0] = local_stress_voigt[:, :, 0]
    S_local[:, :, 1, 1] = local_stress_voigt[:, :, 1]
    S_local[:, :, 2, 2] = local_stress_voigt[:, :, 2]

    S_local[:, :, 0, 1] = local_stress_voigt[:, :, 3]
    S_local[:, :, 1, 0] = local_stress_voigt[:, :, 3]

    S_local[:, :, 0, 2] = local_stress_voigt[:, :, 4]
    S_local[:, :, 2, 0] = local_stress_voigt[:, :, 4]

    S_local[:, :, 1, 2] = local_stress_voigt[:, :, 5]
    S_local[:, :, 2, 1] = local_stress_voigt[:, :, 5]

    R_exp = np.expand_dims(R, axis=1)  # (n_shells, 1, 3, 3)
    R_T_exp = R_exp.transpose(0, 1, 3, 2)

    S_global = np.matmul(np.matmul(R_exp, S_local), R_T_exp)

    global_stress_voigt = np.empty_like(local_stress_voigt)
    global_stress_voigt[:, :, 0] = S_global[:, :, 0, 0]
    global_stress_voigt[:, :, 1] = S_global[:, :, 1, 1]
    global_stress_voigt[:, :, 2] = S_global[:, :, 2, 2]
    global_stress_voigt[:, :, 3] = S_global[:, :, 0, 1]
    global_stress_voigt[:, :, 4] = S_global[:, :, 0, 2]
    global_stress_voigt[:, :, 5] = S_global[:, :, 1, 2]

    return global_stress_voigt
