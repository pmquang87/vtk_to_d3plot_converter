# vtk2d3plot — agent instructions

Python converter: legacy ASCII VTK time series (from pyradioss / OpenRadioss
anim_to_vtk) → LS-DYNA d3plot family (written via lasso-python).

## Environment (do not deviate)

- Windows 11, PowerShell. No WSL, no Git Bash, no Unix commands (ls/rm/cat/grep).
- Python 3.14.2 in `.venv`. NEVER bare `python` or `pip`. Always exact paths:
  - `.venv\Scripts\python.exe`
  - `.venv\Scripts\pip.exe`
- All dependencies are preinstalled (numpy, lasso-python 2.0.4, meshio, pytest,
  ruff, and `pyradioss` editable from `..\OpenRadioss_Python`). Do NOT create a
  venv, do NOT run pip install, unless the user explicitly asks.

## Commands

- Full test suite:   `.venv\Scripts\python.exe -m pytest -q`
- One milestone:     `.venv\Scripts\python.exe -m pytest -q -m milestone2`
- Skip slow/LS-PrePost: add `-m "not lspp and not slow"`
- Lint:              `.venv\Scripts\python.exe -m ruff check src tests tools`
- Format:            `.venv\Scripts\python.exe -m ruff format src tests tools`
- Convert (once CLI exists): `.venv\Scripts\python.exe -m vtk2d3plot.cli <dir> -o out.d3plot`
- Run the FEM solver on an example deck (in a scratch dir containing the .rad files):
  - `.venv\Scripts\pyradioss-starter.exe -i BOXIMP_0000.rad`
  - `.venv\Scripts\pyradioss-engine.exe  -i BOXIMP_0001.rad`   (writes BOXIMPA000.vtk ...)

## Terminal rules

- One flat command per call. Do NOT wrap whole commands in outer quotes.
- No `&&`, no pipes, no redirection — separate commands.
- Never start long-running/watch processes; ask the user instead.
- Windows backslash paths everywhere.
- The LS-PrePost exe path contains literal square brackets — always quote it:
  `"C:\Program Files\Ansys\LS-PrePost-2026R1[4.13]\lsprepost4.13.exe"`

## Project

- The full implementation plan is @docs/plan.md — read it before any work.
  Work ONE milestone at a time. Do not start milestone N+1.
- Input format details: @docs/vtk_input_spec.md (two VTK dialects!).
- d3plot format background: @docs/d3plot_format.md.
- What is already locally PROVEN (do not re-derive): @docs/research/local-verification-report.md.
- Verified working example code: `tools/reference/` — read `make_d3plot.py`
  and `make_mixed.py` before writing any lasso code; they are the template.
- Layout: `src/vtk2d3plot/` (package), `tests/` (+ `tests/data/` golden files),
  `tools/` (checkers), `docs/`.
- After each milestone: full test suite green → tick the checkbox in
  docs/plan.md → `git add -A` and `git commit`.

## Domain rules

- d3plot is a binary format; byte offsets and word counts are load-bearing.
  Never guess header fields or array shapes — cite docs/d3plot_format.md,
  the lasso 2.0.4 source in `.venv\Lib\site-packages\lasso\`, or ask.
- lasso quirks that WILL bite you (details in docs/plan.md "Global constraints"):
  node_displacement = ABSOLUTE coordinates; part_titles = bytes; solids have
  exactly 1 or 8 integration layers; the converter writes one file per state
  (`write_d3plot(..., single_file=False)`: `<out>` + `<out>01..NN`, keep them
  all together); the array_type.py docstring for solid eps shape is wrong;
  ISTRN is one global flag and solids need eff. plastic strain whenever they
  carry state data (builder zero-fills, see docs/plan.md).
- The d3plot stress slot order is [xx, yy, zz, xy, yz, zx] (LS-DYNA manual;
  an older plan.md note claiming [xy, xz, yz] was wrong).
- Verify every writer change three ways: (1) lasso read-back, (2) golden-pair
  by-ID comparison (NEVER compare by array index — nodes are re-ordered),
  (3) LS-PrePost via `tools/lspp_check.py` (see the lspp-check skill).
- Units and node/element/part IDs must round-trip exactly.
