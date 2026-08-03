import numpy as np
import pytest
from conftest import GOLDEN_ZUG_D3, GOLDEN_ZUG_VTK, PORT_DIR, TENSILE_DIR

pytestmark = pytest.mark.milestone10

# ── Helpers: synthetic VTK generators (copied from test_m9_fixes.py) ──


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
    off=None,
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

    # OFF
    if off is not None:
        lines.append("SCALARS OFF int 1")
        lines.append("LOOKUP_TABLE default")
        for v in off:
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


# ── Fix 1: C1 Erosion Latch (CRITICAL) ──


def test_erosion_latch_zug_all_alive(tmp_path):
    """Zug fixture: all 864 shells must be alive (matching Vortex golden)."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(GOLDEN_ZUG_VTK, tmp_path / "zug.d3plot")
    dp = D3plot(str(tmp_path / "zug.d3plot"))

    # Load Vortex golden for comparison
    gd = D3plot(str(GOLDEN_ZUG_D3 / "zug_test3_RS.d3plot"))
    golden_alive = gd.arrays[ArrayType.element_shell_is_alive]

    assert ArrayType.element_shell_is_alive in dp.arrays
    alive = dp.arrays[ArrayType.element_shell_is_alive]
    assert alive.shape == golden_alive.shape  # (3, 864)
    # Vortex golden: all 864 alive in all 3 states
    # Our output must match
    for si in range(alive.shape[0]):
        n_alive = int((alive[si] == 1.0).sum())
        assert n_alive == 864, f"State {si}: expected 864 alive, got {n_alive}"


def test_erosion_latch_tensile_real_death(tmp_path):
    """TENSILE: 3 shells die at states 15/16 via real 1→0 transitions."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(TENSILE_DIR, tmp_path / "tensile.d3plot")
    dp = D3plot(str(tmp_path / "tensile.d3plot"))

    alive = dp.arrays[ArrayType.element_shell_is_alive]
    # States 0-15 (indices 0-14): all 202 alive
    for si in range(15):
        n_dead = int((alive[si] == 0.0).sum())
        assert n_dead == 0, f"State {si}: expected 0 dead, got {n_dead}"
    # State 16 (index 15): 2 dead (shells 200, 201)
    n_dead_15 = int((alive[15] == 0.0).sum())
    assert n_dead_15 == 2, f"State 15: expected 2 dead, got {n_dead_15}"
    # State 17+ (index 16+): 3 dead (shells 199, 200, 201)
    for si in range(16, alive.shape[0]):
        n_dead = int((alive[si] == 0.0).sum())
        assert n_dead == 3, f"State {si}: expected 3 dead, got {n_dead}"


def test_erosion_latch_no_resurrection(tmp_path):
    """Synthetic: element going 1→0→1 must stay dead after first death."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    d = tmp_path / "vtk"
    d.mkdir()

    # Create 3-state VTK where element 1 goes alive→dead→alive
    # Shell 0 is alive (1,1,1), Shell 1 is resurrection (1,0,1)
    for i, status in enumerate([[1, 1], [1, 0], [1, 1]]):
        _write_official_vtk(
            d / f"testA00{i + 1}.vtk", time_val=float(i), n_shells=2, erosion_status=status
        )

    convert(d, tmp_path / "out.d3plot")
    dp = D3plot(str(tmp_path / "out.d3plot"))

    alive = dp.arrays[ArrayType.element_shell_is_alive]
    assert np.allclose(alive[:, 0], [1, 1, 1])
    assert np.allclose(alive[:, 1], [1, 0, 0])


def test_erosion_latch_warning_names_parts(tmp_path):
    """Latch warning must name count and part IDs."""
    from vtk2d3plot.api import convert

    rep = convert(GOLDEN_ZUG_VTK, tmp_path / "zug.d3plot")
    latch_warns = [w for w in rep.warnings if "never alive" in w.lower()]
    assert latch_warns, f"Expected latch warning; got: {rep.warnings}"
    # Must name the count (288) and part IDs
    assert "288" in latch_warns[0]


# ── Fix 2: T2 Unmapped absmax Over All States ──


def test_unmapped_absmax_accumulates(tmp_path):
    """Unmapped array absmax must reflect max across ALL states, not just first."""
    from vtk2d3plot.api import convert

    d = tmp_path / "vtk"
    d.mkdir()

    # State 0: CustomArray = [0.0, 0.0]
    _write_official_vtk(
        d / "testA001.vtk", time_val=0.0, n_shells=2, extra_scalars={"CustomArray": [0.0, 0.0]}
    )
    # State 1: CustomArray = [0.0, 99.99]
    _write_official_vtk(
        d / "testA002.vtk", time_val=1.0, n_shells=2, extra_scalars={"CustomArray": [0.0, 99.99]}
    )

    rep = convert(d, tmp_path / "out.d3plot")
    unmapped = [w for w in rep.warnings if "unmapped" in w.lower()]
    assert unmapped
    assert "CustomArray" in unmapped[0]
    import re

    m = re.search(r"CustomArray \(absmax=([0-9.e+\-]+)\)", unmapped[0])
    assert m, f"Could not parse absmax from warning: {unmapped[0]}"
    assert np.isclose(float(m.group(1)), 99.99)


def test_zug_specific_energy_mapped(tmp_path):
    """Zug 2DELEM_Specific_Energy is 0 at state 0 but nonzero later — it must be mapped to internal energy."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    rep = convert(GOLDEN_ZUG_VTK, tmp_path / "zug.d3plot")
    unmapped = [w for w in rep.warnings if "unmapped" in w.lower()]
    if unmapped:
        assert "2DELEM_Specific_Energy" not in unmapped[0], (
            "Specific energy should be mapped, not unmapped"
        )

    dp = D3plot(str(tmp_path / "zug.d3plot"))
    ie = dp.arrays[ArrayType.element_shell_internal_energy]
    # State 2 has 2015.09
    assert ie.shape[0] > 2
    assert ie.shape[1] > 0
    assert np.isclose(ie[2].max(), 2015.09, rtol=0.01), (
        f"Expected max IE ≈ 2015.09, got {ie[2].max()}"
    )


# ── Fix 3: Title Budget ──


def test_f64_title_budget_72(tmp_path):
    """f64 mode title budget must be 72, not 76."""
    from lasso.dyna import D3plot

    from vtk2d3plot.api import convert

    title_80 = "X" * 80
    rep = convert(PORT_DIR, tmp_path / "box.d3plot", title=title_80, f64=True)
    dp = D3plot(str(tmp_path / "box.d3plot"))
    readback = dp.header.title.strip()
    assert len(readback) == 72, f"Expected 72 chars, got {len(readback)}"
    assert any("truncat" in w.lower() for w in rep.warnings)


# ── Fix 4: Erosion + OFF Coexistence Warning ──


def test_off_erosion_coexistence_warning(tmp_path):
    """When both OFF and EROSION_STATUS present, must warn about priority."""
    from vtk2d3plot.api import convert

    d = tmp_path / "vtk"
    d.mkdir()

    _write_official_vtk(
        d / "testA001.vtk",
        time_val=0.0,
        n_shells=2,
        erosion_status=[1, 1],
        off=[1, 1],
    )

    rep = convert(d, tmp_path / "out.d3plot")
    coexist_warns = [w for w in rep.warnings if "both off and erosion_status present" in w.lower()]
    assert coexist_warns, f"Expected coexistence warning; got: {rep.warnings}"
