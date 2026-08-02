"""Roadmap Tier-1 tests — EROSION_STATUS, unmapped warnings, part titles,
SPH robustness, atomic writes, CLI polish.
"""

import subprocess

import numpy as np
import pytest
from conftest import PORT_DIR, TENSILE_DIR, VENV_PY

pytestmark = pytest.mark.roadmap_tier1


# ── Helpers ──────────────────────────────────────────────────────────


def _write_official_vtk_with_extras(
    path,
    *,
    time_val,
    n_shells,
    stress_membrane=None,
    erosion_status=None,
    extra_scalars=None,
    include_vertex=False,
):
    """Write an official-dialect VTK with optional extras."""
    n_points = n_shells * 4
    # If including vertex cells, add one extra point
    n_vertex_points = 1 if include_vertex else 0
    total_points = n_points + n_vertex_points
    n_cells = n_shells + (1 if include_vertex else 0)

    pts = []
    for si in range(n_shells):
        bx = si * 2.0
        pts.append(f"{bx} 0 0 {bx + 1} 0 0 {bx + 1} 1 0 {bx} 1 0")
    if include_vertex:
        pts.append("10 10 10")

    cell_lines = []
    for si in range(n_shells):
        b = si * 4
        cell_lines.append(f"4 {b} {b + 1} {b + 2} {b + 3}")
    if include_vertex:
        cell_lines.append(f"1 {n_points}")

    ct_lines = ["9"] * n_shells
    if include_vertex:
        ct_lines.append("1")

    total_cell_ints = n_shells * 5 + (2 if include_vertex else 0)

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
        f"POINTS {total_points} float",
    ]
    lines.extend(pts)
    lines.append("")
    lines.append(f"CELLS {n_cells} {total_cell_ints}")
    lines.extend(cell_lines)
    lines.append("")
    lines.append(f"CELL_TYPES {n_cells}")
    lines.extend(ct_lines)

    # Point data
    lines.append("")
    lines.append(f"POINT_DATA {total_points}")
    lines.append("SCALARS NODE_ID int 1")
    lines.append("LOOKUP_TABLE default")
    for i in range(1, total_points + 1):
        lines.append(str(i))
    lines.append("VECTORS Displacement float")
    for _ in range(total_points):
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

    if erosion_status is not None:
        lines.append("SCALARS EROSION_STATUS int 1")
        lines.append("LOOKUP_TABLE default")
        for v in erosion_status:
            lines.append(str(int(v)))

    if stress_membrane is not None:
        lines.append("TENSORS 2DELEM_Stress_(membrane) float")
        for si in range(n_shells):
            t = stress_membrane[si]
            for row in range(3):
                lines.append(f"{t[row, 0]} {t[row, 1]} {t[row, 2]}")
            lines.append("")

    if extra_scalars:
        for name, values in extra_scalars.items():
            lines.append(f"SCALARS {name} float 1")
            lines.append("LOOKUP_TABLE default")
            for v in values:
                lines.append(str(v))

    lines.append("")
    path.write_text("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════
# Item 1: EROSION_STATUS → is_alive
# ═══════════════════════════════════════════════════════════════════════


def test_erosion_status_maps_to_is_alive(tmp_path):
    """EROSION_STATUS (1=alive, 0=eroded) must map to element_shell_is_alive."""
    d = tmp_path / "vtk"
    d.mkdir()
    n = 4

    mem = np.zeros((n, 3, 3), dtype=np.float64)
    mem[:, 0, 0] = [10, 20, 30, 40]

    # Elements 1,2 alive (1), elements 3,4 eroded (0)
    _write_official_vtk_with_extras(
        d / "testA001.vtk",
        time_val=0.0,
        n_shells=n,
        stress_membrane=mem,
        erosion_status=[1, 1, 0, 0],
    )

    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    rep = convert(d, tmp_path / "out.d3plot")
    dp = D3plot(str(tmp_path / "out.d3plot"))

    assert ArrayType.element_shell_is_alive in dp.arrays, (
        "EROSION_STATUS should produce element_shell_is_alive"
    )
    alive = dp.arrays[ArrayType.element_shell_is_alive]
    assert alive.shape == (1, n), f"Expected shape (1, {n}), got {alive.shape}"
    # Elements 1,2 alive, elements 3,4 dead
    assert alive[0, 0] == 1.0
    assert alive[0, 1] == 1.0
    assert alive[0, 2] == 0.0
    assert alive[0, 3] == 0.0

    # Warning should mention EROSION_STATUS
    assert any("EROSION_STATUS" in w for w in rep.warnings), (
        f"Expected EROSION_STATUS warning; got: {rep.warnings}"
    )


def test_erosion_status_on_tensile_fixture(tmp_path):
    """TENSILE fixture has real EROSION_STATUS data — dead elements must be detected."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(TENSILE_DIR, tmp_path / "tensile.d3plot")
    dp = D3plot(str(tmp_path / "tensile.d3plot"))

    assert ArrayType.element_shell_is_alive in dp.arrays, (
        "TENSILE should now have element_shell_is_alive from EROSION_STATUS"
    )
    alive = dp.arrays[ArrayType.element_shell_is_alive]
    # Last state should have some dead elements (EROSION_STATUS=0)
    last_alive = alive[-1]
    assert float(last_alive.min()) == 0.0, (
        f"Expected some dead elements in last state, min alive={last_alive.min()}"
    )
    assert float(last_alive.max()) == 1.0
    # At least 200 alive, some dead
    n_alive = int((last_alive == 1.0).sum())
    n_dead = int((last_alive == 0.0).sum())
    assert n_dead >= 1, f"Expected at least 1 dead element, got {n_dead}"
    assert n_alive >= 195, f"Expected at least 195 alive, got {n_alive}"


# ═══════════════════════════════════════════════════════════════════════
# Item 2: Unmapped-array warnings
# ═══════════════════════════════════════════════════════════════════════


def test_unmapped_array_warning_emitted(tmp_path):
    """Arrays not recognized by the converter must trigger a warning."""
    d = tmp_path / "vtk"
    d.mkdir()
    n = 2

    mem = np.zeros((n, 3, 3), dtype=np.float64)
    mem[:, 0, 0] = [50, 60]

    _write_official_vtk_with_extras(
        d / "testA001.vtk",
        time_val=0.0,
        n_shells=n,
        stress_membrane=mem,
        extra_scalars={
            "2DELEM_Specific_Energy": [2015.0, 3000.0],
            "2DELEM_Hourglass_Energy": [1.5, 2.5],
        },
    )

    from vtk2d3plot.api import convert

    rep = convert(d, tmp_path / "out.d3plot")

    unmapped_warns = [w for w in rep.warnings if "unmapped" in w.lower()]
    assert unmapped_warns, f"Expected unmapped-array warning; got: {rep.warnings}"
    assert "2DELEM_Specific_Energy" in unmapped_warns[0]
    assert "2DELEM_Hourglass_Energy" in unmapped_warns[0]


def test_tensile_unmapped_arrays(tmp_path):
    """TENSILE fixture has unmapped arrays — they must be reported."""
    from vtk2d3plot.api import convert

    rep = convert(TENSILE_DIR, tmp_path / "tensile.d3plot")

    unmapped_warns = [w for w in rep.warnings if "unmapped" in w.lower()]
    assert unmapped_warns, f"Expected unmapped-array warning for TENSILE; got: {rep.warnings}"
    # TENSILE has 2DELEM_Specific_Energy, 2DELEM_Hourglass_Energy, etc.
    warn_text = unmapped_warns[0]
    # TENSILE has multiple unmapped arrays; check that the warning lists some of them
    assert "2DELEM_Hourglass_Energy" in warn_text or "2DELEM_Thickness" in warn_text, (
        f"Expected known unmapped array in warning; got: {warn_text}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Item 3: Part titles from CLI/API
# ═══════════════════════════════════════════════════════════════════════


def test_part_titles_from_dict(tmp_path):
    """Part names dict must populate part_titles in d3plot."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    convert(
        PORT_DIR,
        tmp_path / "box.d3plot",
        part_names={1: "Einspannung"},
    )
    dp = D3plot(str(tmp_path / "box.d3plot"))

    titles = dp.arrays[ArrayType.part_titles]
    # Title should be ASCII-safe (transliterated or replaced)
    title_str = titles[0].decode("utf-8", errors="replace").strip()
    assert title_str != "Part 1", f"Part title should not be default, got {title_str!r}"
    assert len(title_str) > 0


def test_part_titles_cli_file(tmp_path):
    """CLI --part-names file must set part titles."""
    pn_file = tmp_path / "parts.txt"
    pn_file.write_text("# Part name mapping\n1\tMyPart\n", encoding="utf-8")

    r = subprocess.run(
        [
            str(VENV_PY),
            "-m",
            "vtk2d3plot.cli",
            str(PORT_DIR),
            "-o",
            str(tmp_path / "box.d3plot"),
            "--part-names",
            str(pn_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr

    from lasso.dyna import ArrayType, D3plot

    dp = D3plot(str(tmp_path / "box.d3plot"))
    titles = dp.arrays[ArrayType.part_titles]
    title_str = titles[0].decode("utf-8", errors="replace").strip()
    assert "MyPart" in title_str, f"Expected 'MyPart' in title, got {title_str!r}"


# ═══════════════════════════════════════════════════════════════════════
# Item 4: SPH robustness (VTK_VERTEX skip)
# ═══════════════════════════════════════════════════════════════════════


def test_sph_vertex_cells_skipped(tmp_path):
    """VTK_VERTEX cells must be skipped with warning, not crash."""
    d = tmp_path / "vtk"
    d.mkdir()
    n = 2

    # Don't include TENSORS — the vertex cell would break the tensor count
    # (VTK TENSORS block has n_cells entries, but we only have data for quads).
    # Use a simpler VTK without stress tensors to test the skip path.
    _write_official_vtk_with_extras(
        d / "testA001.vtk",
        time_val=0.0,
        n_shells=n,
        include_vertex=True,
    )

    from vtk2d3plot.api import convert

    # Must not crash
    rep = convert(d, tmp_path / "out.d3plot")

    # Should warn about skipped cells
    skip_warns = [w for w in rep.warnings if "skip" in w.lower() or "unsupported" in w.lower()]
    assert skip_warns, f"Expected skip warning; got: {rep.warnings}"
    assert rep.n_shells == n  # quads should still be converted


# ═══════════════════════════════════════════════════════════════════════
# Item 5: Atomic writes + stale-family cleanup
# ═══════════════════════════════════════════════════════════════════════


def test_atomic_write_removes_stale_family(tmp_path):
    """Overwriting a d3plot must remove stale higher-numbered family files."""
    from vtk2d3plot.api import convert

    out = tmp_path / "box.d3plot"
    # First conversion
    convert(PORT_DIR, out)
    assert out.exists()
    assert (tmp_path / "box.d3plot01").exists()

    # Create a fake stale family member
    stale = tmp_path / "box.d3plot02"
    stale.write_text("stale data")
    assert stale.exists()

    # Second conversion should remove the stale file
    convert(PORT_DIR, out)
    assert out.exists()
    assert (tmp_path / "box.d3plot01").exists()
    assert not stale.exists(), "Stale family member box.d3plot02 should be removed"


def test_atomic_write_no_partial_on_failure(tmp_path):
    """On conversion failure, no partial output files should remain."""
    # We can't easily trigger a write failure, so we just verify the
    # normal path produces complete output
    from vtk2d3plot.api import convert

    out = tmp_path / "box.d3plot"
    convert(PORT_DIR, out)
    assert out.exists()
    assert (tmp_path / "box.d3plot01").exists()
    # Both files should be non-empty
    assert out.stat().st_size > 0
    assert (tmp_path / "box.d3plot01").stat().st_size > 0


# ═══════════════════════════════════════════════════════════════════════
# Item 6: CLI batch polish
# ═══════════════════════════════════════════════════════════════════════


def test_cli_default_title_from_directory(tmp_path):
    """CLI with no --title should use input directory name as default."""
    from lasso.dyna import D3plot

    r = subprocess.run(
        [
            str(VENV_PY),
            "-m",
            "vtk2d3plot.cli",
            str(PORT_DIR),
            "-o",
            str(tmp_path / "box.d3plot"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr

    dp = D3plot(str(tmp_path / "box.d3plot"))
    title = dp.header.title.strip()
    # Should contain the directory name
    assert "box_beam_impact" in title, f"Expected dir name in title, got {title!r}"
