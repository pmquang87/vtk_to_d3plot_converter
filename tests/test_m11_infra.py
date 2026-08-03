import pytest

pytestmark = pytest.mark.milestone11


def test_conversion_report_json():
    from vtk2d3plot.api import ConversionReport

    rep = ConversionReport(
        output="test.d3plot",
        n_states=3,
        n_nodes=100,
        n_solids=0,
        n_shells=50,
        n_beams=0,
        n_parts=2,
        dialect="official",
        warnings=["test warning"],
    )
    d = rep.to_dict()
    assert d["n_states"] == 3
    assert d["warnings"] == ["test warning"]
    j = rep.to_json()
    import json

    parsed = json.loads(j)
    assert parsed == d


def test_cli_json_output(tmp_path):
    import json
    import subprocess

    from conftest import PORT_DIR, VENV_PY

    r = subprocess.run(
        [
            str(VENV_PY),
            "-m",
            "vtk2d3plot.cli",
            str(PORT_DIR),
            "-o",
            str(tmp_path / "box.d3plot"),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["n_states"] > 0
    assert "output" in data
