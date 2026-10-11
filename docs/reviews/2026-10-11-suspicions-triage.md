# Suspicions round (milestone 14): triage record

Input: the 28 leads in "Suspicions not checked" of
`docs/reviews/2026-10-11-external-bug-report.md`, plus 5 open points (a)-(e)
found while verifying milestone 13. Base: `fix/bug-report-2026-10-11` at 7e1f48a.

Every lead was checked against the current code first. A lead that reproduced
got a failing test through the public interface (`convert`, the CLI, or the
named function / tool `main`), was seen to fail, and was then fixed. A lead
that did not reproduce got no code change; the probe is recorded below.
Tests: `tests/test_suspicions_*.py`, marker `suspicions` (264 tests).

Verdicts: 24 fixed (S28 only in part), 4 already fixed by milestone 13,
open point (a) not reproducible (the code was right).

## Leads

| ID | Where (old line) | Lead | Verdict | Evidence on base / fix |
|---|---|---|---|---|
| S01 | builder.py:337 | single-layer shell stress fills only layer 0 when 2 layers are forced | FIXED | TENSILE layer 1 was all zero. Membrane / VONM / 2DELEM_Von_Mises stress is now copied to every layer, like plastic strain. Upper-only / lower-only stay in their own layer. LS-PrePost showed 0 at `intpt 2` before. |
| S02 | builder.py:1036 | T01 times not checked before `np.interp` | FIXED | Restart rows gave wrong energies. `parse_t01_csv` keeps TIME strictly increasing; a later row supersedes earlier rows at or after its time; one report warning. Non-finite TIME rows are skipped as malformed. |
| S03 | vtk_reader.py:216 | VTU `byte_order` ignored | ALREADY FIXED | 36 BigEndian probes (inline, raw / base64 appended; none / zlib / lzma; UInt32 / UInt64 headers) read correctly. Pins added. |
| S04 | vtk_reader.py:391 | VTK 8+ METADATA / INFORMATION blocks | FIXED | Bare `IndexError` inside FIELD; a component name `CELL_DATA` was read as a section. METADATA is now skipped by count, after every array kind, following vtkDataWriter / vtkDataReader. |
| S05 | vtk_reader.py:354 | truncated files raise StopIteration / IndexError / RuntimeError | FIXED | Every truncation point now gives a ValueError that names the file, also in the lazy iterator and for cut VTU base64. |
| S06 | series.py:235 | files without `A<digits>` dropped silently | FIXED | Directory input now lists ignored files in the report (`IGNORED_INPUT_FILES`). |
| S07 | series.py:233 | string sort (`out_10` before `out_2`) | FIXED | Natural sort; the time check no longer rejects such a series. |
| S08 | series.py:34 | multi-part PVD gives duplicate times | ALREADY FIXED | Milestone 13 refuses multi-part PVDs with a clear error; other cases name both files. Pins added. Merging parts is a possible feature, not done. |
| S09 | series.py:91 | ragged T01 row aborts the load | ALREADY FIXED | Short / long / non-numeric rows are skipped with a warning. Pins added. |
| S10 | series.py:168 | dialect label wrong; "unknown" unreachable | FIXED | A non-pyradioss title time gave "official". New `VtkState.time_source`; "official" needs a FIELD time (VTU FieldData time still counts; VTU PointData TIME alone is now "unknown"). |
| S11 | api.py:210 | failed rename leaves a partial family | FIXED | With no old family, a failed move left `box.d3plot` .. `box.d3plot20`. Roll-back now removes every new file and restores old ones. |
| S12 | mapping.py:44 | beam orientation node = N1 | FIXED | Manual p. 18: words 3-5 are orientation node and two null entries; the null entries can carry a beam type, so a repeated N1 is a wrong value there. Now -1 (written as word 0). LS-PrePost loads and renders both the same. Note: the Vortex golden also writes a real node there. |
| S13 | api.py:235 | warning codes depend on substring order | FIXED | 4 known mislabels reproduced. Codes now come from a table of message-start templates (`classify_warning`); no warning in any golden conversion falls to OTHER. 13 new codes (README). |
| S14 | api.py:166 | a user's `<file>.bak` is deleted | FIXED | Scratch and backup files live in a private `_vtk2d3plot_<random>` folder; user files are never touched. |
| S15 | golden_manifest.py:58 | `generate` truncates first | FIXED | Hash everything, then write a temp file and `os.replace`. |
| S16 | golden_manifest.py:82 | legacy line with a double space crashes | FIXED | Regex parse; paths may hold double spaces; a bad line counts as an error. |
| S17 | golden_manifest.py:99 | changed file counted twice | FIXED | One error per file. |
| S18 | lspp_check.py:85 | silent fallback to state 1 | FIXED | New `notes` key says so. |
| S19 | verify_byid.py:45 | port-dialect files crash | ALREADY FIXED | Port d3plot verifies with rc 0, with and without title times. |
| S20 | verify_zug.py:51 | zero-ID node never compared | FIXED | Zero-ID nodes are compared on their own line (zug: id 11659, 1.41e-4). |
| S21 | make_d3plot.py:11 | hard-coded output path | FIXED | Optional output folder argument; default is a fresh temp folder. |
| S22 | make_mixed.py:10 | hard-coded output path | FIXED | Same as S21. |
| S23 | vtk_reader.py:415 | legacy VTK 5.1 OFFSETS / CONNECTIVITY unsupported | FIXED | 5.1 files (hand-written and meshio) convert identically to 4.2. Offsets are validated. |
| S24 | series.py:235 | same state as .vtk and .vtu loaded twice | FIXED | Clear error naming both files (directory, stem and CLI glob). |
| S25 | api.py:104 | progress printed twice | FIXED | `convert` reports each state once (during loading). |
| S26 | cli.py:89 | unmatched glob dropped silently | FIXED | `ERROR: pattern '<p>' matched no files`, exit 1. |
| S27 | lspp_check.py:130 | 32-bit word-size probe never matches | FIXED | LS-PrePost 4.13 prints a line only for 64-bit files; the word size now falls back to the d3plot header. |
| S28 | verify_byid.py:40 | extra .vtk shifts the pairing | PARTLY | An extra `notes.vtk` is dropped, pairing is right (milestone 13). A file of another series crashed with a traceback; now `ERROR: Multiple stems found`, exit 2. |

## Open points from milestone 13

| ID | Point | Verdict | Evidence / fix |
|---|---|---|---|
| (a) | port-dialect solid shear slot placement | NOT REPRODUCIBLE | anim_to_vtk.cpp (OpenCourant/Tools @4e52942, L1473-1475) writes rows `[v0 v3 v4 / v3 v1 v5 / v4 v5 v2]` from the ANIM order x, y, z, xy, yz, zx (OpenRadioss srota6.F, tensor6.F), so (0,2) = yz. Vortex copies ANIM components 1:1 (golden is ground truth). pyfemsolver `origin/main` has `_VOIGT9 = [0,3,4,3,1,5,4,5,2]` (restored in 56f5df0c, 2026-10-10). The converter was right; only the docstring and plan.md were updated. |
| (b) | `--rotate-shell-stress` does not rotate shell strain | FIXED | Strain is rotated as a tensor with the same frames. The ANIM shell strain shear is tensorial (tensorc.F halves the engineering shear); the zug golden matches the rotated strain to 2.06e-4 (5e-6 with the golden's binary coordinates), doubled shear gives 0.166. |
| (c) | quoted CLI glob gives an empty title | FIXED | The default title comes from the folder that holds all inputs, for every input form. |
| (d) | warning codes mislabel some warnings | FIXED | See S13. |
| (e) | `load_series` and `load_series_lazy` report different first errors | FIXED | Both check each file in order: missing time, order, finite, topology, identity. Same type and message for 11 two-problem cases. |

## Golden diff (base vs head, all arrays joined by ID)

W3 and port: identical (plain and rotated, f32 and f64). Changes, all expected:

| Case | Arrays | Why |
|---|---|---|
| TENSILE | `element_shell_stress` layer 1 (now equal to layer 0) | S01 |
| TENSILE rotated | the above plus `element_shell_strain`; principal values equal to 2.7e-16 | S01, (b) |
| zug | `element_beam_node_indexes` columns 2-4 (now -1) | S12 |
| zug rotated | the above plus `element_shell_strain`; principal values equal to 4.6e-16 | S12, (b) |

LS-PrePost 4.13 loads every head conversion with the right state count and
fringes von Mises (W3 rotated, zug, zug rotated, TENSILE, TENSILE rotated f32,
port f32); the 32-bit files now report word size 4.

## Left open (not bugs in this repo, or out of scope)

- Port files written by pyfemsolver between 2026-09-14 (145780ab) and
  2026-10-10 (56f5df0c) have yz and zx swapped at the source. So does the
  local `OpenRadioss_Python` main branch, which has diverged from `origin/main`
  and lacks 56f5df0c. Such files cannot be told apart from correct ones; they
  must be re-exported (or a manual override flag added).
- OpenRadioss tensor6.F (8-point hexa, law >= 28) averages the solid strain
  shear without halving it, so `3DELEM_Strain` shear may be engineering strain
  for those elements.
- T01 still aborts a conversion on a missing TIME column or a non-UTF-8 byte.
- METADATA after sections the reader ignores (NORMALS, TEXTURE_COORDINATES,
  ...) is not skipped; it only matters if a component name equals a keyword.
- A `DATA n` INFORMATION line is read as a string vector only when the next
  `NAME` line (or the block end) follows its n items; VTK decides by its key
  registry, which a reader without VTK does not have.
