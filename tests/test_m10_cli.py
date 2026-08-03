import subprocess

from conftest import PORT_DIR, VENV_PY


def test_cli_part_names_cp1252(tmp_path):
    """cp1252-encoded part-names file must not crash."""
    pn_file = tmp_path / "parts.txt"
    # Write cp1252-encoded text with German umlauts
    pn_file.write_bytes(b"1\tEinspannung Gr\xf6\xdfe\n")  # ö=0xf6, ß=0xdf in cp1252
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


def test_cli_part_names_utf8_bom(tmp_path):
    """UTF-8 BOM part-names file must not be silently ignored."""
    pn_file = tmp_path / "parts.txt"
    # UTF-8 BOM + content
    pn_file.write_bytes(b"\xef\xbb\xbf1\tMyPart\n")
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
    assert "MyPart" in title_str, f"BOM file content should not be ignored; got {title_str!r}"


def test_cli_vtk_subdir_title(tmp_path):
    """When input dir is '<model>/vtk/', title should use parent dir, not 'vtk'."""
    import shutil

    # Create <model>/vtk/ structure
    model_dir = tmp_path / "my_model" / "vtk"
    model_dir.mkdir(parents=True)
    # Copy VTK files into the vtk subdir
    for f in sorted(PORT_DIR.glob("*.vtk")):
        shutil.copy(f, model_dir)
    # Also copy .rad files if any (not needed for this test)

    r = subprocess.run(
        [str(VENV_PY), "-m", "vtk2d3plot.cli", str(model_dir), "-o", str(tmp_path / "out.d3plot")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    from lasso.dyna import D3plot

    dp = D3plot(str(tmp_path / "out.d3plot"))
    title = dp.header.title.strip()
    assert "my_model" in title, f"Expected 'my_model' in title, got {title!r}"
    assert title != "vtk", "Title should not be 'vtk'"


def test_cli_per_state_progress(tmp_path):
    """CLI non-quiet mode must print per-state progress lines."""
    r = subprocess.run(
        [str(VENV_PY), "-m", "vtk2d3plot.cli", str(PORT_DIR), "-o", str(tmp_path / "box.d3plot")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    # Must contain per-state progress lines like "Processing state 1/3..."
    assert "Processing state 1/" in r.stdout, (
        f"Expected per-state progress in stdout; got:\n{r.stdout}"
    )
