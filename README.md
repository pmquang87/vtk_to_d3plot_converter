# vtk2d3plot

Convert a time series of legacy ASCII VTK unstructured-grid files — as written
by [OpenRadioss_Python](https://github.com/pmquang87/OpenRadioss_Python)
(pyradioss) or by OpenRadioss's `anim_to_vtk` tool — into an LS-DYNA **d3plot**
family readable by LS-PrePost, HyperView, META, etc.

Built on the only maintained open-source d3plot writer,
[lasso-python](https://github.com/open-lasso-python/lasso-python), following
the adapter architecture proven by
[Vortex-Radioss](https://github.com/Vortex-CAE/Vortex-Radioss).

## Status

v1 complete (Milestones 1–5). Supports both VTK dialects (official `anim_to_vtk`
and pyradioss port), solid/shell/beam geometry, displacement, velocity, stress
tensors, plastic strain, and element deletion flags.

## Usage

### CLI

```powershell
# Basic conversion (directory of VTK files)
.venv\Scripts\python.exe -m vtk2d3plot.cli path\to\vtk\dir -o out\model.d3plot

# Explicit file list
.venv\Scripts\python.exe -m vtk2d3plot.cli file1.vtk file2.vtk -o out.d3plot

# Options
.venv\Scripts\python.exe -m vtk2d3plot.cli vtk_dir -o out.d3plot --f64 --title "My model" -q
```

| Flag | Description |
|------|-------------|
| `-o / --output` | Output d3plot path (required) |
| `--f64` | Write 64-bit output (default: 32-bit) |
| `--title` | Title string embedded in d3plot header |
| `-q / --quiet` | Suppress informational output |

### Python API

```python
from vtk2d3plot.api import convert

report = convert("path/to/vtk/dir", "output.d3plot", title="My model")
print(f"{report.n_states} states, {report.n_nodes} nodes")
```

Output is a family: `model.d3plot` (geometry) + `model.d3plot01` (states) —
always keep them together.

## Development

```powershell
.venv\Scripts\python.exe -m pytest -q                  # full suite
.venv\Scripts\python.exe -m pytest -q -m milestone1    # one milestone
.venv\Scripts\python.exe tools\lspp_check.py out\model.d3plot   # LS-PrePost ground truth
```

Golden test data in `tests/data/`: matched ANIM+VTK+d3plot triples generated
from real OpenRadioss runs (see `docs/research/local-verification-report.md`
for how they were made and verified).
