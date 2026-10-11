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
| `-o / --output` | Output d3plot path (required); geometry file + one state file per state |
| `--f64` | Write 64-bit output (default) |
| `--f32` | Write 32-bit output (overrides `--f64`) |
| `--title` | Title string embedded in d3plot header |
| `-q / --quiet` | Suppress informational output |

### Python API

```python
from vtk2d3plot.api import convert

report = convert("path/to/vtk/dir", "output.d3plot", title="My model")
print(f"{report.n_states} states, {report.n_nodes} nodes")
```

Output is a family: `model.d3plot` (geometry) plus one state file per state,
`model.d3plot01`, `model.d3plot02`, ... (LS-DYNA convention; a 21-state series
writes 22 files). Always keep all of them together. The CLI prints the range,
e.g. `Wrote model.d3plot + 21 state files model.d3plot01..model.d3plot21`, and
`report.output_files` lists every file. Re-converting to the same path replaces
the whole family, including stale higher-numbered state files from a longer
earlier run. The output path must be a file path, not an existing directory.

Wildcards in CLI inputs (`"vtk\*.vtk"`) are expanded by the converter; the
matches of each pattern are sorted by state number (`<stem>A<number>.vtk`), the
same as for a directory input. A pattern that matches no file is an error
(exit code 1, the message names the pattern), also when other inputs match.

Without `--title` the CLI uses the name of the folder that holds the inputs as
the d3plot title, for every input form: a directory, a glob pattern, an
explicit file list, a single file, a stem (`<folder>\<stem>`) or a `.pvd`
manifest (its own folder). A folder called `vtk` gives its parent's name.
Inputs from several folders give no title.

The family is replaced via one private scratch folder `_vtk2d3plot_<random>`
next to the output, which is removed afterwards; no other file name is used,
so files that are not family members (such as `model.d3plot.bak`) are never
touched. If the conversion fails while the files are moved, no new family file
is left behind and the previous family is restored. A file that cannot be
removed or moved back (locked, for example) is named in the error message; an
old family file that cannot be moved back stays in the scratch folder.

`convert(..., progress=callback)` calls `callback(i, n)` once per state while
the states are loaded; the CLI prints it as `Processing state i/n...`.

#### Warning codes

`report.warnings` holds the warning texts; `report.structured_warnings` pairs
each text with a code (also in `--json`). The code comes from the fixed start
of the message (its template), never from array, file or part names inside
it; a message that matches no template gets `OTHER`.

| Code | Meaning |
|------|---------|
| `NO_DISPLACEMENT` | no displacement array; state 0 coordinates used as undeformed geometry |
| `SYNTHETIC_STRESS` | stress tensor built as uniaxial `[vm, 0, 0, 0, 0, 0]` from a von Mises scalar |
| `SHELL_LOCAL_FRAME` | shell stress or strain is in element-local coordinates |
| `SHELL_STRESS_ROTATED` | shell stress was rotated to the global frame (`--rotate-shell-stress`) |
| `ROTATION_IGNORED` | `--rotate-shell-stress` ignored (synthetic stress has no frame) |
| `DEGENERATE_SHELL_FRAMES` | some shells have no usable local frame; their rotated stress is unreliable |
| `SINGLE_LAYER_SOURCE` | only the upper or the lower plastic-strain layer exists; mapped to one layer |
| `STRAIN_MEMBRANE_IGNORED` | membrane strain ignored because upper/lower strain exists |
| `LAYER_MISMATCH` | shell results have different layer counts; reconciled |
| `SOLID_INTERNAL_ENERGY_HISTVAR` | solid internal energy written as solid history variable 0 |
| `ALIVE_SOURCE` | which array gives the element deletion flags (`EROSION_STATUS`, or `OFF` when both exist) |
| `EROSION_FORCED_ALIVE` | elements never alive in any state are shown alive (rigid / non-integrated) |
| `EROSION_NO_RESURRECTION` | elements that return to alive after erosion are kept eroded |
| `ALIVE_FILL` | deletion flags missing in some states; those states are all-alive |
| `ZERO_FILL` | a result array is missing in some states (or needed by the record layout); zero-filled |
| `SKIPPED_CELLS` | unsupported VTK cells were left out |
| `INVERTED_TETRAS` | tetras with negative volume were re-ordered |
| `PHANTOM_CELLS_MASKED` | cells that reference zero-ID nodes were masked |
| `KNOWN_UNSUPPORTED` | known arrays without a d3plot equivalent were not written |
| `UNMAPPED_ARRAYS` | unknown arrays were not written |
| `CASE_COLLISION` | arrays whose names differ only in case; one was used |
| `TITLE_SANITIZED` | header title or a part title was transliterated, replaced or truncated |
| `PART_NAMES_UNUSED` | part-name entries for part IDs that are not in the d3plot |
| `PVD_TIME_OVERRIDE` | a `.pvd` timestep overrides the time stored in the file |
| `IGNORED_INPUT_FILES` | a directory input held `.vtk/.vtu` files without the `<stem>A<number>` state name; they were not loaded |
| `T01_GLOBALS` | notes on the global energies from the `T01` CSV (composition, skipped rows, superseded restart rows, missing data) |
| `OTHER` | no template matched |

A directory input loads its `<stem>A<number>.vtk/.vtu` files in state-number
order and lists any other `.vtk/.vtu` files it ignores in the report warnings;
a folder without such names is loaded in natural name order (`out_2` before
`out_10`). The same state in two files (`runA1.vtk` and `runA001.vtk`, or
`runA001.vtk` and `runA001.vtu`) is an error. Global energies come from a
`<stem>T01.csv` next to the states; after a solver restart (a row whose TIME
is not after the previous row) the later rows win, with a report warning.

## Development

```powershell
.venv\Scripts\python.exe -m pytest -q                  # full suite
.venv\Scripts\python.exe -m pytest -q -m milestone1    # one milestone
.venv\Scripts\python.exe tools\lspp_check.py out\model.d3plot   # LS-PrePost ground truth
```

Golden test data in `tests/data/`: matched ANIM+VTK+d3plot triples generated
from real OpenRadioss runs (see `docs/research/local-verification-report.md`
for how they were made and verified).

Checksums of the golden data are in `tests/data/golden/MANIFEST.sha256`.
`tools\golden_manifest.py` hashes text fixtures with CRLF normalised to LF, so
the manifest verifies on LF and CRLF checkouts; files missing from the
manifest make `verify` fail. After any fixture change run
`.venv\Scripts\python.exe tools\golden_manifest.py generate` (options:
`--root <data dir>`, `--manifest <file>`).

The reference verifiers `tools\reference\verify_byid.py` and `verify_zug.py`
pair VTK files with d3plot states in the converter's numeric `A<number>` order,
renumber zero NODE_IDs like the converter, and exit with code 2 on a count,
state-time or NODE_ID mismatch.
