"""Milestone 9 — fix round: R1/R2/P2 defects from M8 verification.

Each test is written to FAIL before the corresponding fix (test-first).
"""

import numpy as np
import pytest
from conftest import PORT_DIR

pytestmark = pytest.mark.milestone9


# ── Helpers: synthetic VTK generators (copied from test_m8_fixes.py) ──


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
    stress_lower=None,
    stress_upper=None,
    stress_membrane=None,
    vonmises_2d=None,
    eps_lower=None,
    eps_upper=None,
    eps_single=None,
    erosion_status=None,
    extra_scalars=None,
):
    """Write an official-dialect VTK with arbitrary combinations of arrays."""
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

    # EROSION_STATUS
    if erosion_status is not None:
        lines.append("SCALARS EROSION_STATUS int 1")
        lines.append("LOOKUP_TABLE default")
        for v in erosion_status:
            lines.append(str(int(v)))

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

    # Extra scalars (for unmapped-array test)
    if extra_scalars:
        for name, values in extra_scalars.items():
            lines.append(f"SCALARS {name} float 1")
            lines.append("LOOKUP_TABLE default")
            for v in values:
                lines.append(str(v))

    lines.append("")
    path.write_text("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════
# R1: Generic eps discarded when Upper exists without Lower
# ═══════════════════════════════════════════════════════════════════════


def test_r1_generic_eps_not_discarded_when_upper_present(tmp_path):
    """When 2DELEM_Plastic_Strain AND _Upper both exist, generic must NOT be discarded.

    The generic array should be used (via the eps_single branch), not the upper_only branch.
    This test FAILS on pre-M9 code where upper_only takes priority over generic.
    """
    d = tmp_path / "vtk"
    d.mkdir()
    n = 2

    # Membrane stress so we have a stress source
    mem = np.zeros((n, 3, 3), dtype=np.float64)
    mem[:, 0, 0] = [50, 60]

    # Both generic eps AND upper eps present — generic should win
    _write_official_vtk(
        d / "testA001.vtk",
        time_val=0.0,
        n_shells=n,
        stress_membrane=mem,
        eps_single=[0.10, 0.20],  # generic — should be used
        eps_upper=[0.77, 0.88],  # upper — should NOT override generic
    )

    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(d, tmp_path / "out.d3plot")
    dp = D3plot(str(tmp_path / "out.d3plot"))

    eps = dp.arrays[ArrayType.element_shell_effective_plastic_strain]
    # Generic eps should be replicated into all layers
    assert np.isclose(eps[0, 0, 0], 0.10, atol=1e-4), (
        f"Expected generic eps=0.10 in layer 0, got {eps[0, 0, 0]}"
    )
    # Should NOT contain the upper-only value
    assert not np.isclose(eps[0, 0, 0], 0.77, atol=0.01), (
        "Generic eps was discarded in favor of upper-only — R1 regression!"
    )


def test_r1_all_three_eps_arrays_generic_wins(tmp_path):
    """When generic + Upper + Lower all present, the 2-layer (Upper/Lower) branch wins."""
    d = tmp_path / "vtk"
    d.mkdir()
    n = 2

    mem = np.zeros((n, 3, 3), dtype=np.float64)
    mem[:, 0, 0] = [50, 60]

    _write_official_vtk(
        d / "testA001.vtk",
        time_val=0.0,
        n_shells=n,
        stress_membrane=mem,
        eps_single=[0.10, 0.20],
        eps_lower=[0.01, 0.02],
        eps_upper=[0.03, 0.04],
    )

    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(d, tmp_path / "out.d3plot")
    dp = D3plot(str(tmp_path / "out.d3plot"))

    eps = dp.arrays[ArrayType.element_shell_effective_plastic_strain]
    # 2-layer branch should win when both upper and lower exist
    assert eps.shape[2] == 2, f"Expected 2 layers, got {eps.shape[2]}"
    assert np.isclose(eps[0, 0, 0], 0.01, atol=1e-4), "Lower eps in layer 0"
    assert np.isclose(eps[0, 0, 1], 0.03, atol=1e-4), "Upper eps in layer 1"


# ═══════════════════════════════════════════════════════════════════════
# R2: Upper-only data must slot into the correct layer index
# ═══════════════════════════════════════════════════════════════════════


def test_r2_upper_only_stress_slots_into_layer1(tmp_path):
    """Upper-only stress with 2-layer eps must land in layer 1, not layer 0."""
    d = tmp_path / "vtk"
    d.mkdir()
    n = 2

    hi = np.zeros((n, 3, 3), dtype=np.float64)
    hi[:, 0, 0] = [100, 200]

    _write_official_vtk(
        d / "testA001.vtk",
        time_val=0.0,
        n_shells=n,
        stress_upper=hi,
        eps_lower=[0.01, 0.02],
        eps_upper=[0.03, 0.04],
    )

    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(d, tmp_path / "out.d3plot")
    dp = D3plot(str(tmp_path / "out.d3plot"))

    stress = dp.arrays[ArrayType.element_shell_stress]
    assert stress.shape[2] == 2, f"Expected 2 layers, got {stress.shape[2]}"
    # Upper stress should be in layer 1 (not layer 0)
    assert np.isclose(stress[0, 0, 1, 0], 100.0, rtol=1e-4), (
        f"Upper stress should be in layer 1, got layer1={stress[0, 0, 1, 0]}"
    )
    # Layer 0 should be zero (no lower source)
    assert np.allclose(stress[0, :, 0, :], 0.0), "Layer 0 should be zero when only upper exists"


def test_r2_upper_only_stress_single_layer_stays_layer0(tmp_path):
    """Upper-only stress with no eps (1-layer case) still works in layer 0."""
    d = tmp_path / "vtk"
    d.mkdir()
    n = 2

    hi = np.zeros((n, 3, 3), dtype=np.float64)
    hi[:, 0, 0] = [100, 200]

    _write_official_vtk(
        d / "testA001.vtk",
        time_val=0.0,
        n_shells=n,
        stress_upper=hi,
    )

    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(d, tmp_path / "out.d3plot")
    dp = D3plot(str(tmp_path / "out.d3plot"))

    stress = dp.arrays[ArrayType.element_shell_stress]
    # 1 layer → data must be in layer 0
    assert stress.shape[2] == 1, f"Expected 1 layer, got {stress.shape[2]}"
    assert np.isclose(stress[0, 0, 0, 0], 100.0, rtol=1e-4)


# ═══════════════════════════════════════════════════════════════════════
# P2: Transliteration-based title handling
# ═══════════════════════════════════════════════════════════════════════


def test_p2_transliteration_groesse(tmp_path):
    """'Größe' must transliterate to 'Groesse', not strip to 'Gr'."""
    from lasso.dyna import D3plot

    from vtk2d3plot.api import convert

    rep = convert(PORT_DIR, tmp_path / "box.d3plot", title="Größe")
    dp = D3plot(str(tmp_path / "box.d3plot"))

    readback = dp.header.title.strip()
    assert "Groesse" in readback, f"Expected 'Groesse' in title, got {readback!r}"
    assert any("translit" in w.lower() for w in rep.warnings), (
        f"Expected transliteration warning; got: {rep.warnings}"
    )


def test_p2_transliteration_full_german(tmp_path):
    """'Zugversuch Größe 3' must preserve most content via transliteration."""
    from lasso.dyna import D3plot

    from vtk2d3plot.api import convert

    convert(PORT_DIR, tmp_path / "box.d3plot", title="Zugversuch Größe 3")
    dp = D3plot(str(tmp_path / "box.d3plot"))

    readback = dp.header.title.strip()
    assert "Zugversuch" in readback, f"Lost 'Zugversuch', got {readback!r}"
    assert "Groesse" in readback, f"Lost 'Groesse', got {readback!r}"
    assert "3" in readback, f"Lost '3', got {readback!r}"


def test_p2_many_umlauts_transliteration(tmp_path):
    """20× 'ä' → 'ae'×20 = 40 chars → truncated to 36."""
    from lasso.dyna import D3plot

    from vtk2d3plot.api import convert

    title = "ä" * 20  # transliterates to 'ae'*20 = 40 chars
    convert(PORT_DIR, tmp_path / "box.d3plot", title=title)
    dp = D3plot(str(tmp_path / "box.d3plot"))

    readback = dp.header.title.strip()
    # Must be pure ASCII
    readback.encode("ascii")  # should not raise
    # Must fit in 36 readable chars
    assert len(readback) <= 36, f"Title too long: {len(readback)} chars"
    # Should contain 'ae' repeats, not empty
    assert len(readback) > 0, "Title should not be empty"
    assert "ae" in readback, f"Expected 'ae' pattern, got {readback!r}"


def test_p2_pure_ascii_still_works(tmp_path):
    """Pure ASCII title must still work and NOT emit transliteration warning."""
    from lasso.dyna import D3plot

    from vtk2d3plot.api import convert

    rep = convert(PORT_DIR, tmp_path / "box.d3plot", title="My Model 2026")
    dp = D3plot(str(tmp_path / "box.d3plot"))

    readback = dp.header.title.strip()
    assert readback == "My Model 2026"
    assert not any("translit" in w.lower() for w in rep.warnings)


def test_p2_surrogate_does_not_crash(tmp_path):
    """Unpaired surrogates in title must not crash."""
    from vtk2d3plot.api import convert

    # Create a title with a replacement character (simulating bad input)
    title = "test\ufffd"
    convert(PORT_DIR, tmp_path / "box.d3plot", title=title)
    # Just must not crash


def test_p2_empty_title_no_crash(tmp_path):
    """Empty title must not crash or produce spurious warnings."""
    from vtk2d3plot.api import convert

    rep = convert(PORT_DIR, tmp_path / "box.d3plot", title="")
    assert not any("translit" in w.lower() for w in rep.warnings)
    assert not any("truncat" in w.lower() for w in rep.warnings)


# ═══════════════════════════════════════════════════════════════════════
# Hardened P1b value assertions (mutation resilience)
# ═══════════════════════════════════════════════════════════════════════


def test_p1b_upper_only_stress_value_pinned(tmp_path):
    """Upper-only stress must preserve the exact input value, not just be nonzero."""
    d = tmp_path / "vtk"
    d.mkdir()
    n = 2
    hi = np.zeros((n, 3, 3), dtype=np.float64)
    hi[:, 0, 0] = [100, 200]

    _write_official_vtk(
        d / "testA001.vtk",
        time_val=0.0,
        n_shells=n,
        stress_upper=hi,
    )

    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(d, tmp_path / "out.d3plot")
    dp = D3plot(str(tmp_path / "out.d3plot"))

    stress = dp.arrays[ArrayType.element_shell_stress]
    # Find the layer containing the data (could be 0 or 1 depending on layer count)
    n_layers = stress.shape[2]
    if n_layers == 1:
        data_layer = 0
    else:
        data_layer = 1  # R2 fix: upper goes to layer 1

    assert np.isclose(stress[0, 0, data_layer, 0], 100.0, rtol=1e-4), (
        f"Expected stress xx=100.0, got {stress[0, 0, data_layer, 0]}"
    )
    assert np.isclose(stress[0, 1, data_layer, 0], 200.0, rtol=1e-4), (
        f"Expected stress xx=200.0, got {stress[0, 1, data_layer, 0]}"
    )
    # Off-diag must be zero
    assert np.allclose(stress[0, 0, data_layer, 1:], 0.0, atol=1e-6)


def test_p1b_lower_only_eps_value_pinned(tmp_path):
    """Lower-only eps must preserve the exact input value."""
    d = tmp_path / "vtk"
    d.mkdir()
    n = 2
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

    eps = dp.arrays[ArrayType.element_shell_effective_plastic_strain]
    assert np.isclose(eps[0, 0, 0], 0.01, atol=1e-5), f"Expected eps=0.01, got {eps[0, 0, 0]}"
    assert np.isclose(eps[0, 1, 0], 0.02, atol=1e-5), f"Expected eps=0.02, got {eps[0, 1, 0]}"


# ═══════════════════════════════════════════════════════════════════════
# M10: Real surrogate + bytes title tests
# ═══════════════════════════════════════════════════════════════════════


def test_p2_real_surrogate_does_not_crash(tmp_path):
    """Real unpaired surrogate in title must not crash."""
    from vtk2d3plot.api import convert

    # Create a string with a real unpaired surrogate
    "test" + "\ud800".encode("utf-16", "surrogatepass").decode("utf-16", "surrogatepass")
    # This will fail on encode — the _sanitize_title should handle it
    # Actually, Python 3 strings can't contain unpaired surrogates normally.
    # The guard in builder.py does: title.encode("utf-8", errors="replace").decode("utf-8")
    # which replaces surrogates with the replacement character.
    # The real test is: can we pass surrogatepass-decoded content without crashing?
    try:
        title_with_surr = "test\ud800"  # This may raise in some Python builds
    except Exception:  # noqa: BLE001
        pytest.skip("Platform does not support surrogate strings")
    convert(PORT_DIR, tmp_path / "box.d3plot", title=title_with_surr)
    # Must not crash


def test_p2_bytes_title_does_not_crash(tmp_path):
    """Bytes input for title must not crash."""
    from vtk2d3plot.builder import _sanitize_title

    title_bytes = b"Zugversuch Gr\xc3\xb6\xc3\x9fe"  # UTF-8 encoded "Größe"
    result, _warns = _sanitize_title(title_bytes)
    assert isinstance(result, str)
    assert "Groesse" in result, f"Expected transliterated result, got {result!r}"


def test_p2_bytes_latin1_title(tmp_path):
    """Latin-1 bytes input for title must not crash (graceful degradation)."""
    from vtk2d3plot.builder import _sanitize_title

    title_bytes = b"Gr\xf6\xdfe"  # Latin-1 encoded "Größe" (not valid UTF-8)
    result, _warns = _sanitize_title(title_bytes)
    assert isinstance(result, str)
    # Should not crash; content may be garbled but must be ASCII
    result.encode("ascii")  # must not raise


# ═══════════════════════════════════════════════════════════════════════
# M10: Stress-lower-only and eps-lower-only value pinning
# ═══════════════════════════════════════════════════════════════════════


def test_stress_lower_only_value_pinned(tmp_path):
    """Lower-only stress must preserve exact input values."""
    d = tmp_path / "vtk"
    d.mkdir()
    n = 2
    lo = np.zeros((n, 3, 3), dtype=np.float64)
    lo[:, 0, 0] = [100, 200]
    lo[0, 0, 1] = 30  # xy component for elem 0

    _write_official_vtk(
        d / "testA001.vtk",
        time_val=0.0,
        n_shells=n,
        stress_lower=lo,
    )

    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(d, tmp_path / "out.d3plot")
    dp = D3plot(str(tmp_path / "out.d3plot"))

    stress = dp.arrays[ArrayType.element_shell_stress]
    # Lower stress should be in layer 0
    assert np.isclose(stress[0, 0, 0, 0], 100.0, rtol=1e-4), (
        f"Expected xx=100.0, got {stress[0, 0, 0, 0]}"
    )
    assert np.isclose(stress[0, 1, 0, 0], 200.0, rtol=1e-4), (
        f"Expected xx=200.0, got {stress[0, 1, 0, 0]}"
    )
    assert np.isclose(stress[0, 0, 0, 3], 30.0, rtol=1e-4), (
        f"Expected xy=30.0, got {stress[0, 0, 0, 3]}"
    )
    # Off-diagonal zeros
    assert np.isclose(stress[0, 1, 0, 1], 0.0, atol=1e-6)


def test_eps_half_lower_only_value_pinned(tmp_path):
    """Lower-only eps with 2-layer stress must land in layer 0 with exact values."""
    d = tmp_path / "vtk"
    d.mkdir()
    n = 2
    lo = np.zeros((n, 3, 3), dtype=np.float64)
    lo[:, 0, 0] = [50, 60]
    hi = np.zeros((n, 3, 3), dtype=np.float64)
    hi[:, 0, 0] = [55, 65]

    _write_official_vtk(
        d / "testA001.vtk",
        time_val=0.0,
        n_shells=n,
        stress_lower=lo,
        stress_upper=hi,
        eps_lower=[0.011, 0.022],
    )

    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(d, tmp_path / "out.d3plot")
    dp = D3plot(str(tmp_path / "out.d3plot"))

    eps = dp.arrays[ArrayType.element_shell_effective_plastic_strain]
    assert eps.shape[2] == 2, f"Expected 2 layers, got {eps.shape[2]}"
    assert np.isclose(eps[0, 0, 0], 0.011, atol=1e-5), f"Expected 0.011, got {eps[0, 0, 0]}"
    assert np.isclose(eps[0, 1, 0], 0.022, atol=1e-5), f"Expected 0.022, got {eps[0, 1, 0]}"
    # Layer 1 should be zero (no upper eps)
    assert np.allclose(eps[0, :, 1], 0.0), "Layer 1 should be zero"
