import numpy as np

from vtk2d3plot.shell_rotation import compute_shell_local_frames, rotate_stress_to_global


def test_compute_shell_local_frames_identity():
    # A single quad lying on XY plane
    # 3 -- 2 (1,1,0)
    # |    |
    # 0 -- 1 (1,0,0)
    # diagonal 0->2 is (1,1,0). Normalized e1 = (1/sqrt(2), 1/sqrt(2), 0)
    # Wait, the convention is:
    # N1, N2, N3, N4
    # e1 = normalize(N3-N1)
    # e2_raw = N4-N2
    # e3 = normalize(cross(e1, e2_raw))
    # e2 = cross(e3, e1)
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    shell_conn = np.array([[0, 1, 2, 3]])

    R = compute_shell_local_frames(points, shell_conn)
    assert R.shape == (1, 3, 3)

    # N3-N1 = [1, 1, 0]
    # e1 = [1/sqrt(2), 1/sqrt(2), 0]
    e1_exp = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0])

    # N4-N2 = [-1, 1, 0]
    # cross(e1, e2_raw) = cross([1, 1, 0], [-1, 1, 0]) -> [0, 0, 2]
    # e3 = [0, 0, 1]
    e3_exp = np.array([0.0, 0.0, 1.0])

    # e2 = cross(e3, e1) = [-1/sqrt(2), 1/sqrt(2), 0]
    e2_exp = np.array([-1 / np.sqrt(2), 1 / np.sqrt(2), 0])

    np.testing.assert_allclose(R[0, :, 0], e1_exp)
    np.testing.assert_allclose(R[0, :, 1], e2_exp)
    np.testing.assert_allclose(R[0, :, 2], e3_exp)


def test_rotate_stress_to_global_identity():
    # If R is identity, stress should not change
    R = np.eye(3)[np.newaxis, ...]

    # voigt [xx, yy, zz, xy, xz, yz]
    local_stress = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
    global_stress = rotate_stress_to_global(local_stress, R)

    np.testing.assert_allclose(global_stress, local_stress)


def test_rotate_stress_to_global_45deg():
    # 45 deg around Z
    c = np.cos(np.pi / 4)
    s = np.sin(np.pi / 4)

    R = np.array([[[c, -s, 0], [s, c, 0], [0, 0, 1]]])

    # Uniaxial stress in X in local frame
    local_stress = np.array([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])

    global_stress = rotate_stress_to_global(local_stress, R)

    # S_global = R @ diag(1,0,0) @ R.T
    # [c -s 0] [1 0 0] [ c s 0]
    # [s  c 0] [0 0 0] [-s c 0]
    # [0  0 1] [0 0 0] [ 0 0 1]
    # = [c 0 0] [ c s 0] = [c^2  cs 0]
    #   [s 0 0] [-s c 0]   [cs  s^2 0]
    #   [0 0 0] [ 0 0 1]   [ 0    0 0]
    # c^2 = 0.5, s^2 = 0.5, cs = 0.5

    exp_global_stress = np.array([[[0.5, 0.5, 0.0, 0.5, 0.0, 0.0]]])
    np.testing.assert_allclose(global_stress, exp_global_stress, atol=1e-15)


def test_compute_shell_local_frames_triangle():
    # Triangle N1, N2, N3
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    shell_conn = np.array([[0, 1, 2, 2]])

    R = compute_shell_local_frames(points, shell_conn)

    # e1 = normalize(N2-N1) = [1, 0, 0]
    e1_exp = np.array([1.0, 0.0, 0.0])

    # e2_raw = N3-N1 = [0, 1, 0]
    # e3 = normalize(cross(e1, e2_raw)) = [0, 0, 1]
    e3_exp = np.array([0.0, 0.0, 1.0])

    # e2 = cross(e3, e1) = [0, 1, 0]
    e2_exp = np.array([0.0, 1.0, 0.0])

    np.testing.assert_allclose(R[0, :, 0], e1_exp)
    np.testing.assert_allclose(R[0, :, 1], e2_exp)
    np.testing.assert_allclose(R[0, :, 2], e3_exp)
