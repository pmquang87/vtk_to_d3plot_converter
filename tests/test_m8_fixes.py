"""Milestone 8 — fix round: P1/P1b/P2/P3 defects from M7 verification.

Each test is written to FAIL before the corresponding fix (test-first).
"""

import subprocess

import numpy as np
import pytest
from conftest import PORT_DIR, VENV_PY

pytestmark = pytest.mark.milestone8


# ── Helpers: synthetic VTK generators ────────────────────────────────


def _quad_geometry_block(n_shells):
    """Return (n_points, points_lines, cells_lines, cell_types_lines) for n quad shells."""
    n_points = n_shells * 4
    pts = []
    for si in range(n_shells):
        bx = si * 2.0
        pts.append(f"{bx} 0 0 {bx + 1} 0 0 {bx + 1} 1 0 {bx} 1 0")

    cells = []
    for si in range(n_shells):
        b = si * 4
        cells.append(f"4 {b} {b + 1} {b + 2} {b + 3}")

    ct = ["9"] * n_shells
    return n_points, pts, cells, ct


def _write_official_vtk(
    path,
    *,
    time_val,
    n_shells,
    # stress layers (tensor data)
    stress_lower=None,
    stress_upper=None,
    stress_membrane=None,
    # stress scalar
    vonmises_2d=None,
    # eps layers (scalar data)
    eps_lower=None,
    eps_upper=None,
    eps_single=None,
):
    """Write an official-dialect VTK with arbitrary combinations of stress/eps arrays."""
    n_points, pts, cell_lines, ct_lines = _quad_geometry_block(n_shells)
    n_cells = n_shells

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
        f"POINTS {n_points} float",
    ]
    lines.extend(pts)
    lines.append("")
    lines.append(f"CELLS {n_cells} {n_cells * 5}")
    lines.extend(cell_lines)
    lines.append("")
    lines.append(f"CELL_TYPES {n_cells}")
    lines.extend(ct_lines)

    # Point data
    lines.append("")
    lines.append(f"POINT_DATA {n_points}")
    lines.append("SCALARS NODE_ID int 1")
    lines.append("LOOKUP_TABLE default")
    for i in range(1, n_points + 1):
        lines.append(str(i))
    lines.append("VECTORS Displacement float")
    for _ in range(n_points):
        lines.append("0 0 0")

    # Cell data
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

    # Stress tensors
    if stress_lower is not None:
        lines.append("TENSORS 2DELEM_Stress_(lower) float")
        for si in range(n_shells):
            t = stress_lower[si]
            for row in range(3):
                lines.append(f"{t[row, 0]} {t[row, 1]} {t[row, 2]}")
            lines.append("")

    if stress_upper is not None:
        lines.append("TENSORS 2DELEM_Stress_(upper) float")
        for si in range(n_shells):
            t = stress_upper[si]
            for row in range(3):
                lines.append(f"{t[row, 0]} {t[row, 1]} {t[row, 2]}")
            lines.append("")

    if stress_membrane is not None:
        lines.append("TENSORS 2DELEM_Stress_(membrane) float")
        for si in range(n_shells):
            t = stress_membrane[si]
            for row in range(3):
                lines.append(f"{t[row, 0]} {t[row, 1]} {t[row, 2]}")
            lines.append("")

    # Stress scalar
    if vonmises_2d is not None:
        lines.append("SCALARS 2DELEM_Von_Mises float 1")
        lines.append("LOOKUP_TABLE default")
        for v in vonmises_2d:
            lines.append(str(v))

    # Eps layers
    if eps_lower is not None:
        lines.append("SCALARS 2DELEM_Plastic_Strain_Lower float 1")
        lines.append("LOOKUP_TABLE default")
        for v in eps_lower:
            lines.append(str(v))

    if eps_upper is not None:
        lines.append("SCALARS 2DELEM_Plastic_Strain_Upper float 1")
        lines.append("LOOKUP_TABLE default")
        for v in eps_upper:
            lines.append(str(v))

    if eps_single is not None:
        lines.append("SCALARS 2DELEM_Plastic_Strain float 1")
        lines.append("LOOKUP_TABLE default")
        for v in eps_single:
            lines.append(str(v))

    lines.append("")
    path.write_text("\n".join(lines))


def _write_port_vtk_mixed(
    path,
    *,
    time_val,
    n_quads,
    n_hexes=0,
    vonm=None,
    epsp=None,
    include_disp=True,
    # official-dialect arrays on the same cells
    eps_lower=None,
    eps_upper=None,
    stress_lower=None,
    stress_upper=None,
):
    """Write a port-dialect VTK with VONM/EPSP plus optional official eps/stress arrays."""
    n_points_hex = n_hexes * 8
    n_points_quad = n_quads * 4
    n_points = n_points_hex + n_points_quad
    n_cells = n_hexes + n_quads

    lines = [
        "# vtk DataFile Version 3.0",
        f"pyradioss state t={time_val:.9E}",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {n_points} double",
    ]

    for hi in range(n_hexes):
        bx = hi * 2.0
        for dz in [0, 1]:
            for dy in [0, 1]:
                for dx in [0, 1]:
                    lines.append(f"{bx + dx} {dy} {dz}")

    for qi in range(n_quads):
        bx = (n_hexes + qi) * 2.0
        for dy in [0, 1]:
            for dx in [0, 1]:
                lines.append(f"{bx + dx} {dy} 0")

    total_ints = n_hexes * 9 + n_quads * 5
    lines.append(f"CELLS {n_cells} {total_ints}")
    for hi in range(n_hexes):
        b = hi * 8
        lines.append(f"8 {b} {b + 1} {b + 3} {b + 2} {b + 4} {b + 5} {b + 7} {b + 6}")
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

    lines.append(f"CELL_DATA {n_cells}")

    if vonm is not None:
        lines.append("SCALARS VONM double 1")
        lines.append("LOOKUP_TABLE default")
        for v in vonm:
            lines.append(str(v))

    if epsp is not None:
        lines.append("SCALARS EPSP double 1")
        lines.append("LOOKUP_TABLE default")
        for v in epsp:
            lines.append(str(v))

    # Official-dialect arrays mixed in
    if stress_lower is not None:
        lines.append("TENSORS 2DELEM_Stress_(lower) float")
        for si in range(n_cells):
            t = stress_lower[si]
            for row in range(3):
                lines.append(f"{t[row, 0]} {t[row, 1]} {t[row, 2]}")
            lines.append("")

    if stress_upper is not None:
        lines.append("TENSORS 2DELEM_Stress_(upper) float")
        for si in range(n_cells):
            t = stress_upper[si]
            for row in range(3):
                lines.append(f"{t[row, 0]} {t[row, 1]} {t[row, 2]}")
            lines.append("")

    if eps_lower is not None:
        lines.append("SCALARS 2DELEM_Plastic_Strain_Lower float 1")
        lines.append("LOOKUP_TABLE default")
        for v in eps_lower:
            lines.append(str(v))

    if eps_upper is not None:
        lines.append("SCALARS 2DELEM_Plastic_Strain_Upper float 1")
        lines.append("LOOKUP_TABLE default")
        for v in eps_upper:
            lines.append(str(v))

    lines.append("")
    path.write_text("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════
# P1: Three remaining N4 crash sites
# ═══════════════════════════════════════════════════════════════════════


def test_p1_vonmises_official_2layer_eps(tmp_path):
    """2DELEM_Von_Mises(1-layer) + eps Upper/Lower(2-layer) must not crash."""
    d = tmp_path / "vtk"
    d.mkdir()
    n = 2
    _write_official_vtk(
        d / "testA001.vtk",
        time_val=0.0,
        n_shells=n,
        vonmises_2d=[100.0, 200.0],
        eps_lower=[0.01, 0.02],
        eps_upper=[0.03, 0.04],
    )

    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(d, tmp_path / "out.d3plot")
    dp = D3plot(str(tmp_path / "out.d3plot"))

    stress = dp.arrays[ArrayType.element_shell_stress]
    eps = dp.arrays[ArrayType.element_shell_effective_plastic_strain]
    # Both must have 2 layers (reconciled)
    assert stress.shape[2] == 2, f"stress layers={stress.shape[2]}, expected 2"
    assert eps.shape[2] == 2, f"eps layers={eps.shape[2]}, expected 2"
    # Stress layer 0 should have the vonmises value in xx slot
    assert np.isclose(stress[0, 0, 0, 0], 100.0, rtol=1e-4)


def test_p1_vonm_port_2layer_eps(tmp_path):
    """VONM(1-layer) + eps Upper/Lower(2-layer) must not crash."""
    d = tmp_path / "vtk"
    d.mkdir()
    n = 2
    _write_port_vtk_mixed(
        d / "testA001.vtk",
        time_val=0.0,
        n_quads=n,
        vonm=[100.0, 200.0],
        eps_lower=[0.01, 0.02],
        eps_upper=[0.03, 0.04],
    )

    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(d, tmp_path / "out.d3plot")
    dp = D3plot(str(tmp_path / "out.d3plot"))

    stress = dp.arrays[ArrayType.element_shell_stress]
    eps = dp.arrays[ArrayType.element_shell_effective_plastic_strain]
    assert stress.shape[2] == 2, f"stress layers={stress.shape[2]}, expected 2"
    assert eps.shape[2] == 2, f"eps layers={eps.shape[2]}, expected 2"


def test_p1_stress_2layer_epsp_port(tmp_path):
    """stress lower/upper(2-layer) + EPSP(1-layer) must not crash."""
    d = tmp_path / "vtk"
    d.mkdir()
    n = 2
    lo = np.zeros((n, 3, 3), dtype=np.float64)
    lo[:, 0, 0] = [100, 200]
    hi = np.zeros((n, 3, 3), dtype=np.float64)
    hi[:, 0, 0] = [110, 210]

    _write_port_vtk_mixed(
        d / "testA001.vtk",
        time_val=0.0,
        n_quads=n,
        epsp=[0.01, 0.02],
        stress_lower=lo,
        stress_upper=hi,
    )

    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(d, tmp_path / "out.d3plot")
    dp = D3plot(str(tmp_path / "out.d3plot"))

    stress = dp.arrays[ArrayType.element_shell_stress]
    eps = dp.arrays[ArrayType.element_shell_effective_plastic_strain]
    assert stress.shape[2] == 2, f"stress layers={stress.shape[2]}, expected 2"
    assert eps.shape[2] == 2, f"eps layers={eps.shape[2]}, expected 2"
    # eps should be replicated from EPSP into both layers
    assert np.isclose(eps[0, 0, 0], 0.01, atol=1e-5)
    assert np.isclose(eps[0, 0, 1], 0.01, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# P1b: Single-layer-only stress/eps silently dropped
# ═══════════════════════════════════════════════════════════════════════


def test_p1b_upper_only_stress_produces_output(tmp_path):
    """Series with only 2DELEM_Stress_(upper) must produce shell stress, not drop it."""
    d = tmp_path / "vtk"
    d.mkdir()
    n = 2
    hi = np.zeros((n, 3, 3), dtype=np.float64)
    hi[:, 0, 0] = [100, 200]

    # Only upper stress, no lower
    _write_official_vtk(
        d / "testA001.vtk",
        time_val=0.0,
        n_shells=n,
        stress_upper=hi,
    )

    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    rep = convert(d, tmp_path / "out.d3plot")
    dp = D3plot(str(tmp_path / "out.d3plot"))

    assert ArrayType.element_shell_stress in dp.arrays, (
        "Shell stress should NOT be silently dropped when only upper layer exists"
    )
    stress = dp.arrays[ArrayType.element_shell_stress]
    assert np.abs(stress).max() > 0, "Stress should contain nonzero values"
    # Should warn about missing lower layer
    upper_warns = [w for w in rep.warnings if "upper" in w.lower() or "lower" in w.lower()]
    assert upper_warns, f"Expected warning about single-layer stress; got: {rep.warnings}"


def test_p1b_lower_only_eps_produces_output(tmp_path):
    """Series with only 2DELEM_Plastic_Strain_Lower must produce shell eps."""
    d = tmp_path / "vtk"
    d.mkdir()
    n = 2

    # Only lower eps, no upper; plus membrane stress so we have layers
    mem = np.zeros((n, 3, 3), dtype=np.float64)
    mem[:, 0, 0] = [50, 60]

    _write_official_vtk(
        d / "testA001.vtk",
        time_val=0.0,
        n_shells=n,
        stress_membrane=mem,
        eps_lower=[0.01, 0.02],
    )

    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(d, tmp_path / "out.d3plot")
    dp = D3plot(str(tmp_path / "out.d3plot"))

    assert ArrayType.element_shell_effective_plastic_strain in dp.arrays, (
        "Shell eps should NOT be silently dropped when only lower layer exists"
    )
    eps = dp.arrays[ArrayType.element_shell_effective_plastic_strain]
    assert np.abs(eps).max() > 0, "Eps should contain nonzero values"


# ═══════════════════════════════════════════════════════════════════════
# P2: Non-ASCII title crashes
# ═══════════════════════════════════════════════════════════════════════


def test_p2_non_ascii_title_no_crash(tmp_path):
    """Non-ASCII title 'Größe' must not crash; should convert with warning."""
    from vtk2d3plot.api import convert

    rep = convert(PORT_DIR, tmp_path / "box.d3plot", title="Größe")
    from lasso.dyna import D3plot

    # Must not crash (implicit: we got here)
    # Verify readback contains transliterated content
    dp = D3plot(str(tmp_path / "box.d3plot"))
    readback = dp.header.title.strip()
    # 'Größe' should transliterate to 'Groesse'
    assert "Groesse" in readback, f"Expected 'Groesse' in title readback, got {readback!r}"
    # M9: non-ASCII chars are now transliterated, so we get a transliteration warning
    assert any("translit" in w.lower() or "truncat" in w.lower() for w in rep.warnings), (
        f"Expected transliteration or truncation warning for non-ASCII title; got: {rep.warnings}"
    )


def test_p2_non_ascii_long_title_truncation(tmp_path):
    """Long title with many non-ASCII chars (>36 bytes) must truncate on char boundary."""
    from lasso.dyna import D3plot

    from vtk2d3plot.api import convert

    # 'ä' is 2 bytes in UTF-8; 20 of them = 40 bytes > 36
    title = "ä" * 20
    rep = convert(PORT_DIR, tmp_path / "box.d3plot", title=title)
    dp = D3plot(str(tmp_path / "box.d3plot"))

    # The read-back title should fit within 36 chars (lasso reads 36)
    readback = dp.header.title.strip()
    assert len(readback) <= 36, f"Read-back title is {len(readback)} chars, expected ≤36"
    # The padded title must fit in 40 bytes
    assert len(readback.ljust(40).encode("utf-8")) <= 40, (
        f"Padded title is {len(readback.ljust(40).encode('utf-8'))} bytes, expected ≤40"
    )
    assert any("truncat" in w.lower() for w in rep.warnings)


# ═══════════════════════════════════════════════════════════════════════
# P3a: CLI --title test (mandated by contract)
# ═══════════════════════════════════════════════════════════════════════


def test_cli_title_truncation(tmp_path):
    """CLI --title >36 chars must truncate, not crash."""
    from lasso.dyna import D3plot

    title_50 = "A" * 50
    r = subprocess.run(
        [
            str(VENV_PY),
            "-m",
            "vtk2d3plot.cli",
            str(PORT_DIR),
            "-o",
            str(tmp_path / "box.d3plot"),
            "--title",
            title_50,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "truncat" in r.stdout.lower(), f"Expected truncation in output; got: {r.stdout}"
    dp = D3plot(str(tmp_path / "box.d3plot"))
    assert dp.header.title.strip() == "A" * 36


# ═══════════════════════════════════════════════════════════════════════
# P3b: N1-eps partial-fill test
# ═══════════════════════════════════════════════════════════════════════


def test_n1_eps_partial_layer_fill_warning(tmp_path):
    """Eps with only Lower at state 1 (Upper missing) must emit partial-fill warning."""
    d = tmp_path / "vtk"
    d.mkdir()
    n = 2

    # Also need stress so shell_layers > 0; use membrane
    mem = np.zeros((n, 3, 3), dtype=np.float64)
    mem[:, 0, 0] = [50, 60]

    # State 0: both eps layers
    _write_official_vtk(
        d / "testA001.vtk",
        time_val=0.0,
        n_shells=n,
        stress_membrane=mem,
        eps_lower=[0.01, 0.02],
        eps_upper=[0.03, 0.04],
    )
    # State 1: only lower eps (upper missing)
    _write_official_vtk(
        d / "testA002.vtk",
        time_val=1.0,
        n_shells=n,
        stress_membrane=mem * 1.5,
        eps_lower=[0.05, 0.06],
    )

    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    rep = convert(d, tmp_path / "out.d3plot")
    dp = D3plot(str(tmp_path / "out.d3plot"))

    eps = dp.arrays[ArrayType.element_shell_effective_plastic_strain]
    # State 1, upper layer (layer 1) should be zero (missing)
    assert np.allclose(eps[1, :, 1], 0.0), "Upper eps at state 1 should be zero"
    # State 0, upper layer should be nonzero
    assert np.abs(eps[0, :, 1]).max() > 0, "Upper eps at state 0 should be nonzero"

    # Must warn about partial fill for upper eps
    partial_warns = [
        w
        for w in rep.warnings
        if "Upper" in w or "upper" in w.lower() or "Plastic_Strain_Upper" in w
    ]
    assert partial_warns, f"Expected a partial-fill warning about upper eps; got: {rep.warnings}"
