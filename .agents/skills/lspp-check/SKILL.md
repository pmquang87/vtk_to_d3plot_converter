---
name: lspp-check
description: Verify a converted d3plot by loading (and optionally fringing/rendering) it in LS-PrePost headless batch mode. Use after any change to the d3plot writer path, and whenever a test marked "lspp" fails.
---

# Checking a d3plot in LS-PrePost (headless, no GUI)

LS-PrePost is the ground-truth reader for d3plot. It is installed at:

```
C:\Program Files\Ansys\LS-PrePost-2026R1[4.13]\lsprepost4.13.exe
```

(Literal square brackets in the path — always quote it. A double-precision
build `...[4.13]DP\lsprepost4.13DP.exe` also exists.)

## Preferred route: the repo helper

```
.venv\Scripts\python.exe tools\lspp_check.py <path-to.d3plot>
.venv\Scripts\python.exe tools\lspp_check.py <path-to.d3plot> --fringe vonmises --png out.png
```

It returns/prints a JSON summary: `loaded` (bool), `n_states`, `wordsize`,
plus the raw `lspost.msg` path for diagnosis. Every result dict has the same
keys, including `cfile` and `path_notes`.

- `--state N` (1-based) is honoured with or without `--fringe`; without
  `--state`, `--fringe` shows the last state (bug report 2026-10-11 #25).
- Non-ASCII d3plot/PNG paths are written into the cfile via the folder's 8.3
  short path (the file name stays long so LS-PrePost finds the family files).
  If no short path exists, the path is written in the ANSI code page and
  `detail` / `path_notes` say LS-PrePost may fail; LS-PrePost 4.13 crashes
  (0xC0000409) on a cp1252 'ü' folder (bug report 2026-10-11 #40).

In Python:

```python
from tools.lspp_check import check_d3plot
r = check_d3plot("C:/abs/path/model.d3plot", fringe="vonmises", png="C:/abs/out.png")
assert r["loaded"] and r["n_states"] == 21
```

## What it does under the hood (raw knowledge, verified on this machine)

- Runs `lsprepost4.13.exe c=<cfile>` with a command file; waits; exit code 0.
- Inside the cfile the `open d3plot` path MUST use BACKSLASHES — a
  forward-slash path crashes the d3plot reader. `print png` paths may use
  forward slashes.
- Success is judged from `lspost.msg` in the working directory:
  `Finished reading model` present, no `Invalid command` lines, and the
  stdout/msg line `Total number of states = N`.
- Proven cfile pattern for a von-Mises fringe screenshot at the last state:

```
open d3plot "C:\abs\path\model.d3plot"
ac
state 3;
fringe 9
pfringe
range actele;
isometric x
print png "C:/abs/out.png" LANDSCAPE opaque VEC A4 dpi 150 enlisted "OGL1x1"
exit
```

  (`fringe 9` = von Mises stress, `fringe 7` = effective plastic strain;
  `m <part_id>` isolates a part; `ac` = autocenter.)

## Interpreting failures

- Exit 0 but `n_states` lower than expected (e.g. 1) → state files of the
  family are missing next to the geometry file, or were not copied along. The
  converter writes one file per state (`<name>.d3plot01`, `...02`, ...);
  LS-PrePost stops at the first missing number. Ship all `<name>.d3plot*`
  files together.
- Hang / no `lspost.msg` → almost always a forward-slash d3plot path in the
  cfile, or a non-quoted `[4.13]` exe path.
- Loads but geometry looks exploded/twisted in the PNG → element connectivity
  mapping error (wrong degenerate-node padding order). See the mapping table
  in docs/plan.md.
- `ID Section mismatch` style aborts → NARBS/ID arrays inconsistent with
  element counts; check node_ids/element ids arrays lengths.
