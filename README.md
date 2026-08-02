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

Greenfield. The implementation is executed milestone-by-milestone by coding
agents in Antigravity IDE — see `docs/plan.md` (the contract),
`docs/ANTIGRAVITY_SETUP.md` (one-time setup) and `AGENTS.md` (agent rules).

## Usage (target CLI, available after Milestone 5)

```powershell
.venv\Scripts\python.exe -m vtk2d3plot.cli path\to\vtk\dir -o out\model.d3plot
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
