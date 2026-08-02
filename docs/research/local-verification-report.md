# Local verification report (what is already PROVEN on this machine)

Produced 2026-08-02 by an 8-agent research/verification workflow plus two
Gemini Deep Research reports (`gemini-deep-research-report-1.md`,
`gemini-deep-research-report-2-implementation-dossier.md`). Everything below
was executed and checked on THIS machine — coding agents should treat it as
ground truth and not re-derive it.

## 1. Tool landscape (research verdict, adversarially double-checked)

- **No existing VTK→d3plot converter exists**, open-source or commercial.
  Checked: GitHub global code search (authenticated), GitLab, Bitbucket,
  SourceForge, Codeberg, Zenodo, PyPI, conda; German/Chinese/Japanese
  channels; commercial post-processors (HyperView, META, GNS Animator, Oasys).
  Single exception: `squall321/KooD3plotReader` (C++, VtkToD3plotConverter) —
  0 stars, NO license (all-rights-reserved), unvalidated → legally unusable.
- **OpenRadioss has no native d3plot writer** (all 45 "d3plot" hits in its
  repo are LS-DYNA *input*-deck configs for dyna2rad; zero hits in
  engine/starter/tools; nothing in recent release notes).
- d3plot writers: **lasso-python is the only one** (BSD-3, active). qd-cae
  archived 2021; dynareadout read-only (upstream repo deleted); Ansys pyDyna
  writes keyword decks, not d3plot.

## 2. lasso-python 2.0.4 write path — locally proven

Environment: Python 3.14.2, lasso-python 2.0.4, numpy 2.5.1 (all cp314 wheels
exist). LS-PrePost 2026 R1 [4.13] at
`C:\Program Files\Ansys\LS-PrePost-2026R1[4.13]\lsprepost4.13.exe`.

- From-scratch cube (8 nodes / 1 hex / 3 states / stress + eps): written,
  **bit-exact round-trip** through lasso's reader, **loads in LS-PrePost
  batch**, which recomputed von Mises = 196.977 — exactly √38800 from the
  written tensor σ=[200,40,−20,0,0,0]. Displacements render correctly.
- Mixed model (solids + shells, 2 parts, non-contiguous node ids 101–108/
  201–204, element ids 5001/9001, 4 states, 3 shell layers): all preserved in
  LS-PrePost (`m 88` isolates the shell part; ids appear in fringe min/max).
- A 121-node/100-shell plate with 5 states: LS-PrePost fringe max 86.6025 =
  analytic √(100²−100·50+50²). Also verified: tetra-as-degenerate-hex, thick
  shells, beams, history variables, deletion flags, 32- and 64-bit output.
- The exact working scripts are in `tools/reference/make_d3plot.py` and
  `make_mixed.py`. Run them if anything seems off.

Hard API constraints discovered (all enforced in docs/plan.md):
`node_displacement` = ABSOLUTE coords; `part_titles` = bytes (+
`part_titles_ids` paired); solids 1|8 layers only; 2-file family output;
solid-eps docstring shape wrong (real: `(n_states, n_solids, n_layers)`);
64-bit needs itype+ftype+wordsize together; `D3plotHeader` has no
`n_timesteps` (use `len(arrays["timesteps"])`).

## 3. Vortex-Radioss — dissected and exercised locally

- 3 files, ~140 KB; **imports** lasso (not vendored); `readAndConvert` builds
  a ONE-state D3plot per A-file, writes to temp names, renames into the
  `<stem>.d3plot`, `.d3plot01`, ... family. Forces 64-bit. Shell results use
  2 integration points (lower/upper); solid eps always zeros (ANIM limitation);
  beams geometry-only; von Mises never written (LS-PrePost derives it).
- **Node handling to replicate in comparisons**: nodes re-sorted ascending by
  ID; zero IDs renumbered max+1 (`sequential()`, Anim_to_D3plot.py:240).
- Crashes on some ANIM files: `UnicodeDecodeError` in
  `RadiossReader.add_array` via lasso `read_text(encoding="utf8")` — strict
  decode of non-UTF-8 function-name blocks (reproducible on the TENSILE run).
  Hence Vortex is a reference, NOT a dependency.
- Not on PyPI — installed from git (`v1.2 @ a16ab0c`).

## 4. Golden test data (in `tests/data/`, generated + verified locally)

Three-way sets produced from real OpenRadioss runs on E:\ (read-only source),
via `anim_to_vtk_win64.exe` (ANIM→VTK) and Vortex-Radioss (ANIM→d3plot):

| Set | Content | Why it matters |
|---|---|---|
| `golden/W3_SETUP_Ball_impact__ton-mm-s` | 11 ANIM + 11 VTK + d3plot family (12 files, 64-bit) — 1417 nodes, 400 shells, 875 solids, 2 parts | PRIMARY. VTK↔d3plot agree: same node ids & order, times match, coords to ≤5e-3 abs (ASCII precision). LS-PrePost: "Total number of states = 11". |
| `golden/zug_test3_RS` | 3 ANIM + 3 VTK + d3plot family — 987 nodes, 864 shells, 342 beams, 8 parts | THE TRAP CASE: node order differs (sorted), one zero NODE_ID renumbered to 11659. Index-wise diff 2.5e2, by-ID diff 5e-4. Forces by-ID comparison. |
| `golden/TENSILE_vtk_only` | 20 VTK, NO d3plot | Vortex crashed on these ANIMs (UnicodeDecodeError). Our converter must succeed from the VTKs. |
| `port_dialect/box_beam_impact` | 21 VTK from pyradioss + decks + T01.csv | Port dialect fixture: 220 nodes, 200 shells, VONM/EPSP/OFF, time in title line. |

Comparison harness precedent: `tools/reference/verify_byid.py` (independent
ASCII-VTK parse + lasso d3plot read + by-ID join).

## 5. LS-PrePost batch verification — proven pattern

`lsprepost4.13.exe c=<cfile>`, d3plot path in the cfile with BACKSLASHES,
judge success from `lspost.msg` ("Finished reading model", no "Invalid
command") and "Total number of states = N". PNG export works with
`print png "<fwd/slash/path>" LANDSCAPE opaque VEC A4 dpi 150 enlisted "OGL1x1"`
after `fringe 9` / `pfringe` / `range actele;`. Implemented ready-to-use in
`tools/lspp_check.py`; Antigravity skill: `.agents/skills/lspp-check/`.

## 6. Open items deliberately deferred (see plan backlog)

EROSION_STATUS semantics (no-fail run had 3154 zeros at t=0 → 0 ≠ deleted);
wedge/pyramid degenerate node order (verified visually in M2); global variables
from T01.csv resampling; port shell-stress local→global rotation (upstream);
SPH.
