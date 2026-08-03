import numpy as np
import pytest
from lasso.dyna import ArrayType, D3plot

pytestmark = pytest.mark.milestone11


def test_lasso_solid_stress_shape_mismatch(tmp_path):
    coords = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=np.float32,
    )
    timesteps = np.array([0.0, 1.0], dtype=np.float32)
    disp = np.zeros((2, 8, 3), dtype=np.float32)

    d3 = D3plot()
    d3.arrays[ArrayType.node_coordinates] = coords
    d3.arrays[ArrayType.node_ids] = np.arange(1, 9, dtype=np.int32)
    d3.arrays[ArrayType.element_solid_node_indexes] = np.array(
        [[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32
    )
    d3.arrays[ArrayType.element_solid_part_indexes] = np.array([0], dtype=np.int32)
    d3.arrays[ArrayType.part_ids] = np.array([1], dtype=np.int32)
    d3.arrays[ArrayType.global_timesteps] = timesteps
    d3.arrays[ArrayType.node_displacement] = disp

    # Lasso 2.0.4 strictly checks array dimensions and WILL crash with ValueError
    # if the shape is wrong (e.g., missing components dimension or wrong size).
    d3.arrays[ArrayType.element_solid_stress] = np.ones((1, 1, 1, 6), dtype=np.float32) * 5.0

    out = tmp_path / "test.d3plot"
    with pytest.raises(ValueError):
        d3.write_d3plot(str(out))


def test_lasso_part_energy_nmmat(tmp_path):
    d3 = D3plot()
    d3.arrays[ArrayType.node_coordinates] = np.zeros((8, 3), dtype=np.float32)
    d3.arrays[ArrayType.node_ids] = np.arange(1, 9, dtype=np.int32)
    d3.arrays[ArrayType.element_solid_node_indexes] = np.array(
        [[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32
    )
    d3.arrays[ArrayType.element_solid_part_indexes] = np.array([0], dtype=np.int32)
    d3.arrays[ArrayType.part_ids] = np.array([10], dtype=np.int32)
    d3.arrays[ArrayType.global_timesteps] = np.array([0.0], dtype=np.float32)
    d3.arrays[ArrayType.node_displacement] = np.zeros((1, 8, 3), dtype=np.float32)

    # n_parts header field defaults to 0.
    # To set part energy, we need to explicitly set n_parts in header
    d3.arrays[ArrayType.part_internal_energy] = np.array([[10.0]], dtype=np.float32)

    d3.header.n_parts = 1  # MUST MATCH n_parts

    out = tmp_path / "test2.d3plot"
    d3.write_d3plot(str(out))

    r = D3plot(str(out))
    assert r.header.n_parts == 1
    assert r.arrays[ArrayType.part_internal_energy].shape == (1, 1)
    assert np.allclose(r.arrays[ArrayType.part_internal_energy], [[10.0]])


def test_lasso_node_displacement_absolute(tmp_path):
    d3 = D3plot()
    coords = np.array([[10.0, 0, 0]], dtype=np.float32)
    d3.arrays[ArrayType.node_coordinates] = coords
    d3.arrays[ArrayType.global_timesteps] = np.array([0.0, 1.0], dtype=np.float32)

    # We want a relative displacement of +2.0 in X at t=1.0.
    # But lasso node_displacement is ABSOLUTE.
    # So we must write 10.0 + 2.0 = 12.0
    disp = np.zeros((2, 1, 3), dtype=np.float32)
    disp[0] = coords
    disp[1] = coords + np.array([[2.0, 0, 0]])

    d3.arrays[ArrayType.node_displacement] = disp

    out = tmp_path / "test3.d3plot"
    d3.write_d3plot(str(out))

    r = D3plot(str(out))
    # It reads back the absolute coordinates
    assert np.allclose(r.arrays[ArrayType.node_displacement][1], [[12.0, 0, 0]])
