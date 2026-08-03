import pytest
from conftest import PORT_DIR


def test_atomic_removes_1digit_stale(tmp_path):
    """Stale 1-digit family member (d3plot1) must be removed."""
    from vtk2d3plot.api import convert

    out = tmp_path / "box.d3plot"
    convert(PORT_DIR, out)
    # Create fake 1-digit stale
    stale = tmp_path / "box.d3plot1"
    stale.write_text("stale")
    convert(PORT_DIR, out)
    assert not stale.exists(), "1-digit stale should be removed"


def test_atomic_removes_3digit_stale(tmp_path):
    """Stale 3-digit family member (d3plot123) must be removed."""
    from vtk2d3plot.api import convert

    out = tmp_path / "box.d3plot"
    convert(PORT_DIR, out)
    stale = tmp_path / "box.d3plot123"
    stale.write_text("stale")
    convert(PORT_DIR, out)
    assert not stale.exists(), "3-digit stale should be removed"


def test_atomic_preserves_preexisting_on_failure(tmp_path):
    """On conversion failure, pre-existing family must survive."""
    from vtk2d3plot.api import convert

    out = tmp_path / "box.d3plot"
    # First successful conversion
    convert(PORT_DIR, out)
    assert out.exists()
    out01 = tmp_path / "box.d3plot01"
    assert out01.exists()
    orig_size = out.stat().st_size
    orig_size01 = out01.stat().st_size

    # Now trigger a failure by passing invalid input
    bad_dir = tmp_path / "empty_dir"
    bad_dir.mkdir()
    with pytest.raises((ValueError, FileNotFoundError, RuntimeError, Exception)):
        convert(bad_dir, out)

    # Pre-existing files must survive
    assert out.exists(), "Pre-existing d3plot should survive failure"
    assert out01.exists(), "Pre-existing d3plot01 should survive failure"
    assert out.stat().st_size == orig_size
    assert out01.stat().st_size == orig_size01


def test_atomic_brackets_in_name(tmp_path):
    """Output name with brackets must not delete wrong files."""
    from vtk2d3plot.api import convert

    out = tmp_path / "test[1].d3plot"
    convert(PORT_DIR, out)
    assert out.exists()
    # Create an unrelated file that could match if brackets were unescaped
    unrelated = tmp_path / "test1.d3plot"
    unrelated.write_text("unrelated")
    # Re-convert — should not delete the unrelated file
    convert(PORT_DIR, out)
    assert unrelated.exists(), "Unrelated file should not be deleted"
