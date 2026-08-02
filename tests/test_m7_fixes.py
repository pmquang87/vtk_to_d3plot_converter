"""Milestone 7 — fix round: N1–N4 new defects, M1/M7/M8 remedies, regression tests.

Each test is written to FAIL before the corresponding fix (test-first).
"""

import subprocess
import time

import numpy as np
import pytest
from conftest import GOLDEN_W3_VTK, GOLDEN_ZUG_VTK, PORT_DIR, VENV_PY

pytestmark = pytest.mark.milestone7


# ── Helpers: synthetic VTK file generators ───────────────────────────


def _write_vtk_official_2layer(
    path, *, time_val, n_shells, stress_lower, stress_upper, include_lower=True, include_upper=True
):
    """Write a minimal official-dialect VTK with shell stress layers.

    stress_lower/upper: (n_shells, 3, 3) or None.
    """
    lines = [
        "# vtk DataFile Version 3.0",
        "EnSight Gold transient geometry",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        "",
        "FIELD FieldData 1",
        "TIME 1 1 double",
        f"{time_val}",
        "",
        f"POINTS {n_shells * 4} float",
    ]
    # Generate n_shells quads, each with 4 unique nodes
    for si in range(n_shells):
        base = si * 2.0
        lines.append(f"{base} 0 0 {base + 1} 0 0 {base + 1} 1 0 {base} 1 0")

    total_cells = n_shells
    total_ints = n_shells * 5  # 4 nodes + count per cell
    lines.append("")
    lines.append(f"CELLS {total_cells} {total_ints}")
    for si in range(n_shells):
        b = si * 4
        lines.append(f"4 {b} {b + 1} {b + 2} {b + 3}")

    lines.append("")
    lines.append(f"CELL_TYPES {total_cells}")
    for _ in range(n_shells):
        lines.append("9")

    n_points = n_shells * 4
    lines.append("")
    lines.append(f"POINT_DATA {n_points}")
    lines.append("SCALARS NODE_ID int 1")
    lines.append("LOOKUP_TABLE default")
    for i in range(1, n_points + 1):
        lines.append(str(i))
    lines.append("VECTORS Displacement float")
    for _ in range(n_points):
        lines.append("0 0 0")

    lines.append("")
    lines.append(f"CELL_DATA {total_cells}")
    lines.append("SCALARS ELEMENT_ID int 1")
    lines.append("LOOKUP_TABLE default")
    for i in range(1, total_cells + 1):
        lines.append(str(i))
    lines.append("SCALARS PART_ID int 1")
    lines.append("LOOKUP_TABLE default")
    for _ in range(total_cells):
        lines.append("1")

    if include_lower and stress_lower is not None:
        lines.append("TENSORS 2DELEM_Stress_(lower) float")
        for si in range(n_shells):
            t = stress_lower[si]
            for row in range(3):
                lines.append(f"{t[row, 0]} {t[row, 1]} {t[row, 2]}")
            lines.append("")

    if include_upper and stress_upper is not None:
        lines.append("TENSORS 2DELEM_Stress_(upper) float")
        for si in range(n_shells):
            t = stress_upper[si]
            for row in range(3):
                lines.append(f"{t[row, 0]} {t[row, 1]} {t[row, 2]}")
            lines.append("")

    path.write_text("\n".join(lines))


def _write_vtk_port(
    path,
    *,
    time_val,
    n_quads,
    n_hexes,
    vonm_quads=None,
    vonm_hexes=None,
    include_vonm=True,
    include_disp=True,
):
    """Write a minimal port-dialect VTK with optional hex + quad cells."""
    n_points_quad = n_quads * 4
    n_points_hex = n_hexes * 8
    n_points = n_points_quad + n_points_hex
    n_cells = n_quads + n_hexes

    lines = [
        "# vtk DataFile Version 3.0",
        f"pyradioss state t={time_val:.9E}",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {n_points} double",
    ]

    # Hex nodes first (3D → 2D → 1D in port dialect)
    for hi in range(n_hexes):
        bx = hi * 2.0
        for dz in [0, 1]:
            for dy in [0, 1]:
                for dx in [0, 1]:
                    lines.append(f"{bx + dx} {dy} {dz}")

    # Quad nodes
    for qi in range(n_quads):
        bx = (n_hexes + qi) * 2.0
        for dy in [0, 1]:
            for dx in [0, 1]:
                lines.append(f"{bx + dx} {dy} 0")

    total_ints = n_hexes * 9 + n_quads * 5
    lines.append(f"CELLS {n_cells} {total_ints}")
    # Hexes first
    for hi in range(n_hexes):
        b = hi * 8
        lines.append(f"8 {b} {b + 1} {b + 3} {b + 2} {b + 4} {b + 5} {b + 7} {b + 6}")
    # Quads
    for qi in range(n_quads):
        b = n_points_hex + qi * 4
        lines.append(f"4 {b} {b + 1} {b + 3} {b + 2}")

    lines.append(f"CELL_TYPES {n_cells}")
    for _ in range(n_hexes):
        lines.append("12")
    for _ in range(n_quads):
        lines.append("9")

    if include_disp:
        lines.append(f"POINT_DATA {n_points}")
        lines.append("VECTORS DISPLACEMENT double")
        for _ in range(n_points):
            lines.append("0 0 0")

    if include_vonm:
        lines.append(f"CELL_DATA {n_cells}")
        lines.append("SCALARS VONM double 1")
        lines.append("LOOKUP_TABLE default")
        all_vonm = []
        if vonm_hexes is not None:
            all_vonm.extend(vonm_hexes)
        else:
            all_vonm.extend([100.0] * n_hexes)
        if vonm_quads is not None:
            all_vonm.extend(vonm_quads)
        else:
            all_vonm.extend([50.0] * n_quads)
        for v in all_vonm:
            lines.append(f"{v}")

    lines.append("")
    path.write_text("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════
# N1: Partial-layer silent zero-fill warning
# ═══════════════════════════════════════════════════════════════════════


def test_n1_partial_layer_fill_warning(tmp_path):
    """A state with only (lower) but not (upper) must emit a partial-fill warning."""
    d = tmp_path / "vtk"
    d.mkdir()

    n = 2  # 2 shell elements
    lo = np.array(
        [[[100, 0, 0], [0, 0, 0], [0, 0, 0]], [[200, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.float64
    )
    hi = np.array(
        [[[110, 0, 0], [0, 0, 0], [0, 0, 0]], [[210, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.float64
    )

    # State 0: both layers
    _write_vtk_official_2layer(
        d / "testA001.vtk", time_val=0.0, n_shells=n, stress_lower=lo, stress_upper=hi
    )
    # State 1: only lower
    _write_vtk_official_2layer(
        d / "testA002.vtk",
        time_val=1.0,
        n_shells=n,
        stress_lower=lo * 1.5,
        stress_upper=hi * 1.5,
        include_upper=False,
    )

    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    rep = convert(d, tmp_path / "out.d3plot")
    dp = D3plot(str(tmp_path / "out.d3plot"))

    stress = dp.arrays[ArrayType.element_shell_stress]
    # State 1, upper layer (layer 1) should be zero (missing)
    assert np.allclose(stress[1, :, 1, :], 0.0), "upper layer at state 1 should be zero"
    # State 0, upper layer should be nonzero
    assert np.abs(stress[0, :, 1, :]).max() > 0, "upper layer at state 0 should be nonzero"

    # Must warn about partial fill — specifically about upper being missing
    partial_warns = [
        w
        for w in rep.warnings
        if "upper" in w.lower() or "partial" in w.lower() or "2DELEM_Stress_(upper)" in w
    ]
    assert partial_warns, f"Expected a partial-fill warning about upper layer; got: {rep.warnings}"


# ═══════════════════════════════════════════════════════════════════════
# N3: --title > 40 chars must not crash; truncation to 36
# ═══════════════════════════════════════════════════════════════════════


def test_n3_title_over_40_chars(tmp_path):
    """A 50-char title must not crash; truncated to 36 with warning."""
    from lasso.dyna import D3plot

    from vtk2d3plot.api import convert

    title_50 = "A" * 50
    rep = convert(PORT_DIR, tmp_path / "box.d3plot", title=title_50)
    dp = D3plot(str(tmp_path / "box.d3plot"))

    # Must not crash (implicit: we got here)
    # Title should be truncated to 36 chars
    assert dp.header.title.strip() == "A" * 36
    # Warning about truncation
    assert any("truncat" in w.lower() for w in rep.warnings), (
        f"Expected truncation warning; got: {rep.warnings}"
    )


def test_n3_title_37_chars(tmp_path):
    """A 37-char title must be truncated to 36 with warning."""
    from lasso.dyna import D3plot

    from vtk2d3plot.api import convert

    title_37 = "B" * 37
    rep = convert(PORT_DIR, tmp_path / "box.d3plot", title=title_37)
    dp = D3plot(str(tmp_path / "box.d3plot"))
    assert dp.header.title.strip() == "B" * 36
    assert any("truncat" in w.lower() for w in rep.warnings)


def test_n3_title_36_exact(tmp_path):
    """A 36-char title must NOT be truncated; no warning."""
    from lasso.dyna import D3plot

    from vtk2d3plot.api import convert

    title_36 = "C" * 36
    rep = convert(PORT_DIR, tmp_path / "box.d3plot", title=title_36)
    dp = D3plot(str(tmp_path / "box.d3plot"))
    assert dp.header.title.strip() == "C" * 36
    assert not any("truncat" in w.lower() for w in rep.warnings), (
        f"No truncation warning expected for 36-char title; got: {rep.warnings}"
    )


# ═══════════════════════════════════════════════════════════════════════
# N4: Membrane stress + 2-layer eps must not crash
# ═══════════════════════════════════════════════════════════════════════


def test_n4_membrane_stress_2layer_eps(tmp_path):
    """Membrane stress (1-layer) + upper/lower eps (2-layer) must reconcile."""
    d = tmp_path / "vtk"
    d.mkdir()

    n = 2  # 2 shells
    n_points = n * 4
    n_cells = n

    mem_stress = np.zeros((n, 3, 3), dtype=np.float64)
    mem_stress[:, 0, 0] = [100, 200]  # σxx

    eps_lo = np.array([0.01, 0.02], dtype=np.float64)
    eps_hi = np.array([0.03, 0.04], dtype=np.float64)

    lines = [
        "# vtk DataFile Version 3.0",
        "EnSight Gold transient geometry",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        "",
        "FIELD FieldData 1",
        "TIME 1 1 double",
        "0.0",
        "",
        f"POINTS {n_points} float",
    ]
    for si in range(n):
        bx = si * 2.0
        lines.append(f"{bx} 0 0 {bx + 1} 0 0 {bx + 1} 1 0 {bx} 1 0")

    lines.append("")
    lines.append(f"CELLS {n_cells} {n_cells * 5}")
    for si in range(n):
        b = si * 4
        lines.append(f"4 {b} {b + 1} {b + 2} {b + 3}")
    lines.append("")
    lines.append(f"CELL_TYPES {n_cells}")
    for _ in range(n):
        lines.append("9")

    lines.append("")
    lines.append(f"POINT_DATA {n_points}")
    lines.append("SCALARS NODE_ID int 1")
    lines.append("LOOKUP_TABLE default")
    for i in range(1, n_points + 1):
        lines.append(str(i))
    lines.append("VECTORS Displacement float")
    for _ in range(n_points):
        lines.append("0 0 0")

    lines.append("")
    lines.append(f"CELL_DATA {n_cells}")
    lines.append("SCALARS ELEMENT_ID int 1")
    lines.append("LOOKUP_TABLE default")
    for i in range(1, n_cells + 1):
        lines.append(str(i))
    lines.append("SCALARS PART_ID int 1")
    lines.append("LOOKUP_TABLE default")
    for _ in range(n_cells):
        lines.append("1")

    # Membrane stress tensor
    lines.append("TENSORS 2DELEM_Stress_(membrane) float")
    for si in range(n):
        t = mem_stress[si]
        for row in range(3):
            lines.append(f"{t[row, 0]} {t[row, 1]} {t[row, 2]}")
        lines.append("")

    # Upper/Lower plastic strain
    lines.append("SCALARS 2DELEM_Plastic_Strain_Lower float 1")
    lines.append("LOOKUP_TABLE default")
    for v in eps_lo:
        lines.append(str(v))
    lines.append("SCALARS 2DELEM_Plastic_Strain_Upper float 1")
    lines.append("LOOKUP_TABLE default")
    for v in eps_hi:
        lines.append(str(v))

    (d / "testA001.vtk").write_text("\n".join(lines))

    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(d, tmp_path / "out.d3plot")
    dp = D3plot(str(tmp_path / "out.d3plot"))

    # Both stress and eps must have the same layer count
    stress = dp.arrays[ArrayType.element_shell_stress]
    eps = dp.arrays[ArrayType.element_shell_effective_plastic_strain]
    assert stress.shape[2] == eps.shape[2], (
        f"Stress layers {stress.shape[2]} != eps layers {eps.shape[2]}"
    )
    # eps should have 2 layers with the correct values
    assert eps.shape[2] >= 2
    assert np.allclose(eps[0, :, 0], eps_lo, atol=1e-5)
    assert np.allclose(eps[0, :, 1], eps_hi, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# N5: is_alive fill warning should say "one-filled", not "zero-filled"
# ═══════════════════════════════════════════════════════════════════════


def test_n5_alive_fill_warning_text(tmp_path):
    """is_alive missing-state warning must say 'one-filled', not 'zero-filled'."""
    d = tmp_path / "vtk"
    d.mkdir()

    # State 0: has OFF, state 1: no OFF
    vtk0 = """# vtk DataFile Version 3.0
pyradioss state t=0.0
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 4 double
0 0 0 1 0 0 1 1 0 0 1 0
CELLS 1 5
4 0 1 2 3
CELL_TYPES 1
9
POINT_DATA 4
VECTORS DISPLACEMENT double
0 0 0 0 0 0 0 0 0 0 0 0
CELL_DATA 1
SCALARS OFF double 1
LOOKUP_TABLE default
1.0
"""
    vtk1 = """# vtk DataFile Version 3.0
pyradioss state t=1.0
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 4 double
0 0 0 1 0 0 1 1 0 0 1 0
CELLS 1 5
4 0 1 2 3
CELL_TYPES 1
9
POINT_DATA 4
VECTORS DISPLACEMENT double
0 0 0 0 0 0 0 0 0 0 0 0
CELL_DATA 1
SCALARS VONM double 1
LOOKUP_TABLE default
50.0
"""
    (d / "testA000.vtk").write_text(vtk0)
    (d / "testA001.vtk").write_text(vtk1)

    from vtk2d3plot.api import convert

    rep = convert(d, tmp_path / "out.d3plot")
    alive_warns = [w for w in rep.warnings if "is_alive" in w]
    assert alive_warns, f"Expected is_alive warning; got: {rep.warnings}"
    for w in alive_warns:
        assert "one-filled" in w or "all-alive" in w.lower(), (
            f"is_alive warning should say 'one-filled' or 'all-alive', not 'zero-filled': {w}"
        )
        assert "zero-filled" not in w, f"is_alive warning must NOT say 'zero-filled': {w}"


# ═══════════════════════════════════════════════════════════════════════
# M1: Shell stress local-frame warning
# ═══════════════════════════════════════════════════════════════════════


def test_m1_shell_stress_local_frame_warning(tmp_path):
    """W3 conversion must warn about element-local coordinates in shell stress."""
    from vtk2d3plot.api import convert

    rep = convert(GOLDEN_W3_VTK, tmp_path / "w3.d3plot")
    local_warns = [w for w in rep.warnings if "local" in w.lower() or "element-local" in w.lower()]
    assert local_warns, f"Expected local-frame warning for shell stress; got: {rep.warnings}"


# ═══════════════════════════════════════════════════════════════════════
# M3: Union detection — array only in later states
# ═══════════════════════════════════════════════════════════════════════


def test_m3_union_detection(tmp_path):
    """VONM absent from state 0 but present in state 1 must still produce stress."""
    d = tmp_path / "vtk"
    d.mkdir()

    # State 0: no VONM
    _write_vtk_port(d / "testA000.vtk", time_val=0.0, n_quads=2, n_hexes=0, include_vonm=False)
    # State 1: has VONM
    _write_vtk_port(
        d / "testA001.vtk", time_val=1.0, n_quads=2, n_hexes=0, vonm_quads=[100.0, 200.0]
    )

    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(d, tmp_path / "out.d3plot")
    dp = D3plot(str(tmp_path / "out.d3plot"))

    assert ArrayType.element_shell_stress in dp.arrays, (
        "Shell stress should be present even if VONM missing from state 0"
    )
    stress = dp.arrays[ArrayType.element_shell_stress]
    # State 0: zero (missing VONM)
    assert np.allclose(stress[0], 0.0)
    # State 1: nonzero
    assert np.abs(stress[1]).max() > 0


# ═══════════════════════════════════════════════════════════════════════
# M7: Phantom zero-ID-node cell masking
# ═══════════════════════════════════════════════════════════════════════


def test_m7_zug_shell_count_matches_golden(tmp_path):
    """Zug conversion must produce 864 shells (matching golden), not 874."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(GOLDEN_ZUG_VTK, tmp_path / "zug.d3plot")
    dp = D3plot(str(tmp_path / "zug.d3plot"))
    n_shells = dp.arrays[ArrayType.element_shell_ids].shape[0]
    assert n_shells == 864, (
        f"Expected 864 shells (golden), got {n_shells} — phantom cells not masked"
    )


def test_m7_phantom_mask_warning(tmp_path):
    """Zug conversion must warn about masked phantom cells."""
    from vtk2d3plot.api import convert

    rep = convert(GOLDEN_ZUG_VTK, tmp_path / "zug.d3plot")
    phantom_warns = [
        w
        for w in rep.warnings
        if "phantom" in w.lower()
        or "zero-ID node" in w.lower()
        or "masked" in w.lower()
        or "zero-id" in w.lower()
    ]
    assert phantom_warns, f"Expected phantom-masking warning; got: {rep.warnings}"


# ═══════════════════════════════════════════════════════════════════════
# C1: Solid stress from port-dialect VONM (regression guard)
# ═══════════════════════════════════════════════════════════════════════


def test_c1_solid_port_vonm(tmp_path):
    """A port model with hex + quad carrying VONM must produce nonzero solid stress."""
    d = tmp_path / "vtk"
    d.mkdir()

    _write_vtk_port(
        d / "testA000.vtk",
        time_val=0.0,
        n_quads=2,
        n_hexes=1,
        vonm_hexes=[150.0],
        vonm_quads=[50.0, 75.0],
    )

    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    rep = convert(d, tmp_path / "out.d3plot")
    dp = D3plot(str(tmp_path / "out.d3plot"))

    # Solid stress must exist and be nonzero
    assert ArrayType.element_solid_stress in dp.arrays
    solid_stress = dp.arrays[ArrayType.element_solid_stress]
    assert solid_stress.shape == (1, 1, 1, 6)
    assert np.isclose(solid_stress[0, 0, 0, 0], 150.0, rtol=1e-5), (
        f"solid stress xx should be 150.0 (VONM), got {solid_stress[0, 0, 0, 0]}"
    )
    assert np.allclose(solid_stress[0, 0, 0, 1:], 0.0), "other stress components should be 0"
    assert any("synthetic" in w for w in rep.warnings)


# ═══════════════════════════════════════════════════════════════════════
# CLI --f64 roundtrip (verifies arrays are actually float64)
# ═══════════════════════════════════════════════════════════════════════


def test_cli_f64_roundtrip(tmp_path):
    """--f64 must produce wordsize=8 with float64 arrays."""
    r = subprocess.run(
        [
            str(VENV_PY),
            "-m",
            "vtk2d3plot.cli",
            str(PORT_DIR),
            "-o",
            str(tmp_path / "box.d3plot"),
            "--f64",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr

    from lasso.dyna import ArrayType, D3plot

    dp = D3plot(str(tmp_path / "box.d3plot"))
    assert dp.header.wordsize == 8
    assert dp.arrays[ArrayType.node_coordinates].dtype == np.float64
    assert dp.arrays[ArrayType.node_displacement].dtype == np.float64


# ═══════════════════════════════════════════════════════════════════════
# M8: Reader performance bound (timed W3 series load)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_m8_reader_performance_bound(tmp_path):
    """W3 VTK series load must complete within 15 seconds."""
    from vtk2d3plot.series import load_series

    t0 = time.perf_counter()
    _series = load_series(GOLDEN_W3_VTK)
    elapsed = time.perf_counter() - t0

    assert elapsed < 15.0, f"W3 series load took {elapsed:.1f}s — exceeds 15s performance bound"
