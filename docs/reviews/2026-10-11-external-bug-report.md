# Bug report: pmquang87/vtk_to_d3plot_converter

**Commit:** `af9cf52b8e33843da9e92e4174b0136870946e64` (2026-10-04, "Merge pull request #1 from pmquang87/feat/sph-patch-reader-performance")

**Result: 40 confirmed bugs: 4 High, 21 Medium, 15 Low.** They come from 43 claims that an independent prover demonstrated with a failing test. Three pairs described the same defect and are merged here (c02+c25, c12+c13, c22+c41). One of the 43, c01 (non-ASCII part names), was demonstrated by a fresh prover started after the run, because both of its provers in the run failed.

**How a bug got onto this list:**
1. One agent read every line of a slice of the code, and a second agent reread it.
2. A third agent got only the claim, not the reasoning, and wrote a failing test through the public interface (`vtk2d3plot.convert`, the CLI, or the named module function). It then checked a fix on a copy of the code.
3. I re-ran every test myself against the untouched checkout, where all 43 fail. I then applied each suggested fix to a fresh copy, and all 43 tests pass there.

**Evidence:** `bug-hunt-evidence/<id>/` holds, for each claim:
- the test;
- the prover's test output;
- the fix as a diff;
- my two re-runs, `my_rerun_original_code.txt` and `my_rerun_with_fix.txt`.

The tests use sandbox paths: the checkout was `/tmp/hunt/repo` and tests ran with `PYTHONPATH=/tmp/hunt/repo/src`. Change the path constants at the top of a test to run it elsewhere.

**Severity:**
- **High:** ordinary input gives a wrong result with no error, loses data, or never finishes.
- **Medium:** a crash on ordinary input, or a wrong result only on unusual input.
- **Low:** everything else.
- Between two labels, I chose the lower one.

---

## High

### 1. `--rotate-shell-stress` uses the wrong local frame for quad shells (c06)
- **Where:** `src/vtk2d3plot/shell_rotation.py:29-33`, `compute_shell_local_frames`. The quad axis e1 is the diagonal N1→N3, and e2 comes from the diagonal N2→N4.
- **Input:** `convert(tests/data/golden/W3_SETUP_Ball_impact__ton-mm-s/vtk, out, rotate_shell_stress=True)`, or the CLI flag `--rotate-shell-stress`.
- **What should happen:** the output is global-frame shell stress, as the CLI help and the builder warning promise. It should match the reference d3plot of the same run in the repo's test data. That file was produced by another converter (Vortex-Radioss) and is in the global frame.
- **What happens:** σxx, σyy and σxy differ from the reference by up to 27 % of max|σ| (0.2746, 0.2734 and 0.2724), with no warning.
  - anim_to_vtk writes shell stress in the OpenRadioss element frame. There, e1 and e2 lie symmetric about the bisector of r = x2+x3−x1−x4 and s = x3+x4−x1−x2.
  - Rotating the same input with that frame reproduces the reference to 2e-6.
- **Failing test:** `bug-hunt-evidence/c06/test_c06_shell_frame.py`. It converts W3 with rotation, matches shells by ID against the reference d3plot, and requires the in-plane components to agree within 1e-3·max|σ|.
- **Fix:** build the quad frame from r and s (`c06/fix.diff`). The repo's own `tests/test_postv1_shell_rotation.py::test_compute_shell_local_frames_identity` pins the diagonal frame and must be updated along with this fix.
  ```diff
  -    v13 = N3[is_quad] - N1[is_quad]
  -    v24 = N4[is_quad] - N2[is_quad]
  -    e1[is_quad] = v13
  -    e2_raw[is_quad] = v24
  +    r = N2[is_quad] + N3[is_quad] - N1[is_quad] - N4[is_quad]
  +    s = N3[is_quad] + N4[is_quad] - N1[is_quad] - N2[is_quad]
  +    n = unit(cross(r, s)); a = unit(unit(r) + unit(s)); b = cross(n, a)
  +    e1[is_quad] = a - b
  +    e2_raw[is_quad] = a + b          # (unit() with the existing <1e-12 guards; see diff)
  ```

### 2. `--rotate-shell-stress` swaps the transverse shear components (c07)
- **Where:**
  - The call site is `src/vtk2d3plot/builder.py:353`.
  - `rotate_stress_to_global` (`shell_rotation.py:82-86, 98-99`) reads and writes Voigt slots 4/5 as [xz, yz].
  - The builder's arrays are in d3plot order [xx, yy, zz, xy, yz, zx] (`docs/d3plot_format.md:41`, `_tensor_to_voigt`).
- **Input:** same as #1.
- **What should happen:** a correctly rotated plane-stress tensor has no traction on the shell mid-plane (S·n = 0). The reference d3plot meets this: max|S·n| = 8.8e-5.
- **What happens:** max|S·n| = 300 against max|S| = 1130, and slot 4 differs from the reference by up to 383 (reference max 191). Read as [xz, yz], the same output gives S·n = 8.5e-14. So the global xz and yz land in the slots the format defines as yz and zx.
- **Failing test:** `bug-hunt-evidence/c07/test_c07_shell_shear_slots.py`. The check does not depend on the in-plane frame, so it is independent of #1.
- **Fix** (`c07/fix_c07.diff`):
  ```diff
  -            shell_stress[i] = rotate_stress_to_global(shell_stress[i], R)
  +            swap = [0, 1, 2, 3, 5, 4]  # d3plot [yz, zx] <-> rotate_stress_to_global [xz, yz]
  +            shell_stress[i] = rotate_stress_to_global(shell_stress[i][..., swap], R)[..., swap]
  ```

### 3. The d3plot "total energy" is the T01 `EN` column, which is not total energy (c26)
- **Where:** `src/vtk2d3plot/builder.py:1039-1041`. The comment `# EN = total energy` leads to `global_total_energy`.
- **Input:** `convert(tests/data/port_dialect/box_beam_impact, out)`. The pyradioss T01 CSV is picked up automatically.
- **What should happen:** total energy ≈ KE+IE+HE+CE ≈ 0.156–0.163 for this impact run. KE alone is 0.156 at t = 0.
- **What happens:** total energy = [0.0, 1.6e-18, 6.2e-5, −1.4e-3, …, −8.0e-3], so it is zero at the start and negative later, with no warning. In `BOXIMPT01.csv`, EN is 0 in the first row and about −0.004 at t = 0.5, where IE+KE+HE+CE = 0.160.
- **Failing test:** `bug-hunt-evidence/c26/test_c26_total_energy.py`. It reads back `global_total_energy` and compares TE[0] with KE+IE.
- **Fix:** sum the energy columns that are present, as below (`c26/fix_c26.diff`). The repo does not document what pyradioss means by EN and DE, so the conservative alternative is to leave total energy out.
  ```diff
  -        if "EN" in t01.columns:
  -            en_resampled = np.interp(series.times, t01.times, t01.columns["EN"]).astype(ftype)
  +        te_cols = [c for c in ("IE", "KE", "HE", "CE", "DE") if c in t01.columns]
  +        if te_cols:
  +            te = np.sum([t01.columns[c] for c in te_cols], axis=0)
  +            en_resampled = np.interp(series.times, t01.times, te).astype(ftype)
  ```

### 4. Wedge (VTK 13) elements are written inside-out, a regression from a refactor (c19)
- **Where:** `src/vtk2d3plot/mapping.py:36`, `map_topology`: `13: [0, 1, 2, 2, 3, 4, 5, 5]`.
- **Input:** any mesh with a VTK_WEDGE. Example: points (0,0,0), (1,0,0), (0,1,0), (0,0,−1), (1,0,−1), (0,1,−1) in VTK order.
- **What should happen:** `docs/plan.md:193` gives `[0,1,4,3,2,2,5,5]`, marked "verified (M2 LS-PrePost render … clean prism, no twist)". That is LS-DYNA's pentahedron pattern (N5=N6, N7=N8), with a positive Jacobian.
- **What happens:**
  - The d3plot gets connectivity [0,1,2,2,3,4,5,5], with corner Jacobian −1, so the element is inverted.
  - `git log -L` shows the verified mapping from commit `7b8ba86`. Commit `cf782f2` ("Refactor: core processing optimizations…") changed it, with no doc or test update.
- **Failing test:** `bug-hunt-evidence/c19/test_c19_wedge.py`. It tests `map_topology`, and `convert` followed by a lasso read-back.
- **Fix:** restore `13: [0, 1, 4, 3, 2, 2, 5, 5]` (`c19/fix.diff`).

---

## Medium

### 5. Arrays are found ignoring case but read with exact case, so results come out zero or "all alive" (c12, c13)
- **Where:** `src/vtk2d3plot/builder.py`. The presence flags come from `_any_state_has`, which ignores case. Every gather then looks up the exact name:
  - result arrays, e.g. `"VONM" in state.cell_data` at line 344, and the same pattern at about 30 places in `_build_solid_results` / `_build_shell_results`;
  - OFF / EROSION_STATUS at lines 804, 811 and 818 (`alive_key in state.cell_data`).
  - The helpers that ignore case, `get_cell_key` and `_find_key`, already exist but are not used there.
- **Input:** cell arrays named `vonm` / `epsp` (e.g. VONM [5,7], [6,8]), or `off` / `Off`.
- **What should happen:** the same output as for `VONM` / `EPSP` / `OFF`.
- **What happens:**
  - Shell stress and plastic strain are all zero, with the warning "zero-filled … source array missing".
  - `is_alive` is all 1, so element deletion is lost.
  - The real arrays are listed as "unmapped arrays ignored".
- **Failing tests:** `bug-hunt-evidence/c13/test_c13_case_insensitive_gather.py` and `bug-hunt-evidence/c12/test_c12_off_case.py`.
- **Fix:** look keys up with `_find_key` / `get_cell_key` at every gather. `c12/fix_c12.diff` covers OFF/EROSION_STATUS. `c13/fix_c13.diff` changes all gathers mechanically.

### 6. The EROSION_STATUS latch kills elements that become active after state 0 (c14)
- **Where:** `builder.py:866-870`. `was_dead = alive_arr[si-1] == 0.0` latches any 0, including the zeros before an element is first alive.
- **Input:** EROSION_STATUS per state [0,1], [1,1], [1,0] for two shells. Element 0 reads 0, 1, 1.
- **What should happen:** element 0 is alive in states 1 and 2. Only a 1→0 change is a death, per the code comment and `docs/reviews/2026-08-03-m9-verification.md` ("an element is dead only after having been observed alive").
- **What happens:** element 0 gets `is_alive` [0, 0, 0] and is hidden in every state.
- **Failing test:** `bug-hunt-evidence/c14/test_c14_erosion_leading_zero.py`.
- **Fix:** latch only after the element has been seen alive (`c14/fix.diff`):
  ```diff
  +                seen_alive = alive_arr[0] == 1.0
                   for si in range(1, n_states):
  -                    was_dead = alive_arr[si - 1] == 0.0
  +                    was_dead = seen_alive & (alive_arr[si - 1] == 0.0)
                       alive_arr[si, was_dead] = 0.0
  +                    seen_alive |= alive_arr[si] == 1.0
  ```

### 7. Solid internal energy is silently lost when shells carry strain but solids do not (c15)
- **Where:** `builder.py:224-229` stores 3DELEM_Specific_Energy as one solid history variable. Lines `1112-1115` then add shell strain without solid strain.
  - Shell strain makes lasso set the global flag ISTRN=1.
  - lasso then computes NEIPH − 6·ISTRN = −5 and writes the history variable into an empty slice.
- **Input:** a series with one hex and one quad. Cell data: 3DELEM_Specific_Energy = 42 and 2DELEM_Strain_(lower/upper), with no 3DELEM_Strain.
- **What should happen:** 42.0 appears in every state's solid record, as the converter's own warning says: "solid internal energy mapped to element_solid_history_variables (index 0)".
- **What happens:**
  - The value 42.0 does not appear anywhere in the written files.
  - Reading back gives no history variables, and an all-zero solid strain array that the converter never wrote.
  - No warning is given.
- **Failing test:** `bug-hunt-evidence/c15/test_c15_solid_ie_lost.py`. It checks the written bytes.
- **Fix:** zero-fill solid strain in this case (`c15/fix_c15.diff`).

### 8. Without a solid plastic-strain array, solid history variables and solid strain shift by one word (c23)
- **Where:** `builder.py:1098`. `element_solid_effective_plastic_strain` is set only if 3DELEM_Plastic_Strain or EPSP exists.
  - lasso still writes the 7 basic solid words but sets IOSHL2=0.
  - Readers therefore do not skip the plastic-strain word.
- **Input:** a one-hex official-dialect series with 3DELEM_Stress, 3DELEM_Specific_Energy = 7, 8, 9 and 3DELEM_Strain (.01 … .06), and no plastic strain (e.g. an elastic run).
- **What should happen:** history variable [7, 8, 9]; strain [.01, .02, .03, .04, .05, .06].
- **What happens:**
  - The history variable reads back as [0, 0, 0].
  - Strain reads back as [7, .01, .02, .03, .04, .05]: εx is replaced by the energy and εzx is lost.
  - With 3DELEM_Plastic_Strain added, both read back correctly.
- **Failing test:** `bug-hunt-evidence/c23/test_c23_solid_histvar.py`. It uses a lasso read-back; LS-PrePost was not available to check.
- **Fix:** zero-fill solid plastic strain whenever solids have results (`c23/fix.diff`).

### 9. Non-ASCII part names corrupt the d3plot part table (c01)
- **Where:** `src/vtk2d3plot/builder.py:1073`, `build_d3plot`. Part names are encoded with `name.encode("utf-8")`, without the `_sanitize_title` clean-up the header title gets (lines 1128-1133).
  - lasso 2.0.4 pads each part title to 72 characters and only then UTF-8-encodes it.
  - Each multi-byte character therefore makes the record longer and shifts every later record.
  - This is a regression: commit `819a3df` (M9) cleaned part names, and commit `8fd2edf` went back to the raw name.
- **Input:** `convert(dir, out, part_names={1: "Größe", 2: "Alu"})` on a two-part mesh. The CLI's `--part-names` option, with a UTF-8 file of `id<TAB>name` lines, behaves the same.
- **What should happen:** reading the file back gives part IDs [1, 2], and part 2's title is "Alu".
- **What happens:**
  - `part_titles_ids` reads back as [1, 139296], and part 2's title is `b"\x00\x00Alu…"`.
  - 32-bit output is broken the same way.
  - Four accented letters, or a CJK name, on part 1 give the ID 9128910880.
  - ASCII names come back correctly.
- **Failing test:** `bug-hunt-evidence/c01/test_c01_part_titles.py`. It covers the API and the CLI, each with an ASCII control.
- **Fix:** clean part names the same way as the title (`c01/fix_c01.diff`). Like the title, this loses information: umlauts become ae/oe/ue/ss, other non-ASCII characters become '?', and a TITLE_SANITIZED warning is added.
  ```diff
  -            titles_list.append(name.encode("utf-8"))
  +            name, name_warns = _sanitize_title(name, wordsize=8)
  +            warn_list.extend(f"part {int(pid)} {w}" for w in name_warns)
  +            titles_list.append(name.encode("ascii"))
  ```

### 10. Output cleanup deletes another d3plot family whose name differs only in case (c05)
- **Where:** `src/vtk2d3plot/api.py:139`. The family regex is compiled with `re.IGNORECASE`. Matching files are backed up and then deleted after a successful write.
- **Input:** on Linux, a folder holding `Box.d3plot` and `Box.d3plot01`, then `convert(PORT_DIR, dir/'box.d3plot')`.
- **What should happen:** only the `box.d3plot*` family is replaced. The project's own `test_atomic_case_insensitive_cleanup` says a case-different name "just won't match" on Linux.
- **What happens:** `Box.d3plot` and `Box.d3plot01` are deleted.
- **Failing test:** `bug-hunt-evidence/c05/test_c05_case_family.py`.
- **Fix:** compare names with `os.path.normcase` instead of `re.IGNORECASE` (`c05/fix_c05.diff`).

### 11. An output path that is an existing directory leaves a broken family and reports success (c42)
- **Where:** `api.py:176-179`. `shutil.move(tmp, out.parent/out.name)` moves the file *into* the directory when one exists.
- **Input:** `mkdir out_dir/results`, then `convert(PORT_DIR, 'out_dir/results')`.
- **What should happen:** an error, or a d3plot at the reported path.
- **What happens:**
  - No exception is raised.
  - `results/` holds only `_vtk2d3plot_tmp_results`, the geometry file still under its temp name.
  - `results01` … `results21` are written next to the directory.
  - The report says the output is `out_dir/results`.
- **Failing test:** `bug-hunt-evidence/c42/test_c42_output_is_dir.py`.
- **Fix:** add `if out.is_dir(): raise IsADirectoryError(...)` at the start (`c42/fix_c42.diff`).

### 12. A stem input with a mistyped relative folder converts files from the current directory (c36)
- **Where:** `src/vtk2d3plot/series.py:123-124`, and `310-311` in `load_series_lazy`: `if not parent.exists() and not path.is_absolute(): parent = Path(".")`.
- **Input:** `load_series('no_such_dir/run')` or `convert('no_such_dir/run', …)`, run from a directory that contains `runA000.vtk` and `runA001.vtk`.
- **What should happen:** an error. `no_such_dir/run` means `no_such_dir/runA*.vtk`.
- **What happens:** `./runA000.vtk` and `./runA001.vtk` are loaded and converted, with no warning.
- **Failing test:** `bug-hunt-evidence/c36/test_c36_stem_missing_dir.py`.
- **Fix:** raise `FileNotFoundError` when the parent is not a directory (`c36/fix_c36.diff`).

### 13. Compressed VTU files, the default of VTK and meshio, are read as empty arrays (c03)
- **Where:** `src/vtk2d3plot/vtk_reader.py:187-233`, `read_vtu.parse_data_array`. It ignores `VTKFile@compressor` and takes the first word of the compression header (the block count, 1) as the byte count.
- **Input:** `meshio.write('x.vtu', mesh, compression='zlib')` (meshio's default), then `read_vtk('x.vtu')`.
- **What should happen:** the mesh's points, cells and arrays.
- **What happens:**
  - `read_vtk` returns points of shape (0, 3), no cells, empty arrays and time None, without raising.
  - `convert` and the CLI then stop with a misleading "Time is None for file …".
- **Failing test:** `bug-hunt-evidence/c03/test_c03_vtu_zlib.py`.
- **Fix:** parse the compression header and decompress block by block (`c03/fix.diff`, 66 lines; handles zlib and lzma, with UInt32 or UInt64 headers).

### 14. Appended base64 VTU data is decoded as one stream, which crashes (c16)
- **Where:** `vtk_reader.py:179` decodes the whole AppendedData in one `b64decode`. Lines `225-233` then use the offsets as positions in the decoded bytes.
  - VTK encodes each array as its own base64 block, and its offsets count encoded characters.
  - `b64decode` stops at the first block's `=` padding.
- **Input:** a VTU with `<AppendedData encoding="base64">` laid out as VTK writes it. meshio reads the test file correctly.
- **What should happen:** the 8 points, the hex connectivity and the point data.
- **What happens:** `struct.error: unpack requires a buffer of 4 bytes` at the second array.
- **Failing test:** `bug-hunt-evidence/c16/test_c16_appended_b64.py`.
- **Fix:** keep the text encoded and decode each array from its own offset (`c16/fix.diff`).

### 15. PVD timesteps are discarded, so a standard PVD series cannot be loaded (c17)
- **Where:** `series.py:113` (`paths = [e[1] for e in pvd_entries]`) and `series.py:301` in the lazy loader. `read_vtu` takes time only from a point array named TIME.
- **Input:** `run.pvd` with DataSet timesteps 0, 0.5 and 1.0, pointing to VTU files without a TIME point array. The time is either in FieldData TimeValue or absent.
- **What should happen:** times [0, 0.5, 1.0]. `docs/plan.md` says "PVD manifest parser added".
- **What happens:** `ValueError: Time is None for file run_0.vtu` from `load_series`, `load_series_lazy` and `convert`.
- **Failing test:** `bug-hunt-evidence/c17/test_c17_pvd_times.py`.
- **Fix:** keep the manifest times and use them when a file carries no time (`c17/fix_c17.diff`).

### 16. A VTU with several Pieces keeps only the last one (c29)
- **Where:** `vtk_reader.py:147-148`. `piece = elem` runs at every Piece end event, so each Piece overwrites the previous one.
- **Input:** a VTU with two Pieces of 4 points and 1 cell each.
- **What should happen:** 8 points and 2 cells. meshio reads 8 points.
- **What happens:** 4 points and 1 cell. The first Piece is dropped without notice, and `convert` reports n_nodes = 4.
- **Failing test:** `bug-hunt-evidence/c29/test_c29_multipiece.py`.
- **Fix:** collect every Piece and concatenate them, shifting connectivity and offsets (`c29/fix_c29.diff`).

### 17. FIELD arrays inside POINT_DATA / CELL_DATA are thrown away (c32)
- **Where:** `vtk_reader.py:387-403`. FIELD values are read only to look for TIME, then discarded.
- **Input:** a legacy file with `POINT_DATA 4 / FIELD FieldData 1 / Temperature 1 4 double …`, or any file from `meshio.write(..., file_format='vtk42', binary=False)`. meshio stores all point and cell data this way.
- **What should happen:** `point_data['Temperature']`, `cell_data['VONM']` and so on, as VTK's legacy reader and meshio return them.
- **What happens:** `point_data = {}` and `cell_data = {}`. Every result array is lost, with no warning.
- **Failing test:** `bug-hunt-evidence/c32/test_c32_field_in_point_cell_data.py`.
- **Fix:** store FIELD arrays in the current data dict (`c32/fix.diff`).

### 18. Int16 / UInt16 VTU arrays decode as float64 garbage (c30)
- **Where:** `vtk_reader.py:189-199`. `dt_map` has no `Int16` or `UInt16` entry, and `dt_map.get(..., np.float64)` falls back to float64 without a warning.
- **Input:** a PointData array with `type="Int16" format="binary"` holding [1, 2, 3, 4].
- **What should happen:** [1, 2, 3, 4]; meshio agrees.
- **What happens:** `[5.56e-309]`, a single garbage float.
- **Failing test:** `bug-hunt-evidence/c30/test_c30_int16.py`.
- **Fix:** add `"Int16": np.int16, "UInt16": np.uint16` (`c30/fix_c30.diff`).

### 19. The ID consistency check is skipped whenever state 0 lacks the ID array (c24)
- **Where:** `series.py:189-202`. NODE_ID, ELEMENT_ID and PART_ID are checked only if the array exists in state 0, and each state is compared only with state 0.
- **Input:** same topology in all three files; `runA000.vtk` has no NODE_ID, `runA001.vtk` has NODE_ID 100..103, and `runA002.vtk` has NODE_ID 150..153.
- **What should happen:** `TopologyError`. `docs/plan.md` decision 8 says differing identity arrays are a hard error.
- **What happens:** the series loads, and `convert` writes node IDs [1, 2, 3, 4] with no warning.
- **Failing test:** `bug-hunt-evidence/c24/test_c24_identity_check.py`.
- **Fix:** require the array to be present in both or neither (`c24/fix.diff`).

### 20. Parts of skipped cells still appear in the d3plot part table (c37)
- **Where:** `src/vtk2d3plot/mapping.py:132-142`. Part IDs are collected from PART_ID over all cells, including skipped cells (unsupported types such as VTK_VERTEX) and masked cells.
- **Input:** one VTK_VERTEX with PART_ID 7, and one quad with PART_ID 3.
- **What should happen:** part_ids [3], nmmat 1, n_parts 1. The converter itself warns that skipped cells "are not included in the d3plot".
- **What happens:** part_ids [7, 3], nmmat 2, n_parts 2, so the table has an empty part.
- **Failing test:** `bug-hunt-evidence/c37/test_c37_skipped_parts.py`.
- **Fix:** take the unique part IDs over kept cells only (`c37/fix_c37.diff`).

### 21. The golden-manifest check depends on line endings and fails on every LF checkout, including the ubuntu CI job (c08)
- **Where:**
  - `tools/golden_manifest.py:29-33` (`compute_sha256` hashes raw bytes), plus the raw `st_size` checks at lines 62 and 98.
  - The repo has no `.gitattributes`. The 58 text fixtures are stored with LF endings in git, while `MANIFEST.sha256` holds the hashes and sizes of their CRLF form.
- **Input:** `python tools/golden_manifest.py verify` on a Linux or macOS checkout. `.github/workflows/test.yml` runs on ubuntu-latest as well as Windows.
- **What should happen:** "OK" for unchanged fixtures.
- **What happens:**
  - `FAILED: 116 mismatches` (58 CHANGED + 58 SIZE), exit code 1.
  - So `tests/test_m12_fixes.py::test_golden_manifest` fails. It was the only failing test when I ran the suite here.
  - Whichever way the manifest is generated, it passes on only one of the two CI platforms.
- **Failing test:** `bug-hunt-evidence/c08/test_c08_manifest_eol.py`.
- **Fix:** hash and measure text files with CRLF normalised to LF, then regenerate the manifest (`c08/fix_c08.diff`). Alternatively, add `tests/data/** -text` to `.gitattributes` and regenerate.

### 22. Golden-manifest `verify` says OK when fixtures are added or the manifest is cut short (c39)
- **Where:** `tools/golden_manifest.py:106-112`. Files missing from the manifest are printed as NEW but never counted as errors.
- **Input:**
  - (a) add `zug_test3_RSA999.vtk` to a golden folder;
  - (b) truncate the manifest to 10 lines.
- **What should happen:** FAILED.
- **What happens:**
  - (a) "OK: all 88 golden files match manifest", exit code 0. Meanwhile `load_series` on that folder now fails.
  - (b) "OK: all 10 …", exit code 0.
- **Failing test:** `bug-hunt-evidence/c39/test_c39_manifest_new_files.py`.
- **Fix:** add `errors += len(new_files)` (`c39/fix.diff`).

### 23. `verify_byid.py` crashes on a zero NODE_ID, the very case the docs single out (c21)
- **Where:** `tools/reference/verify_byid.py:49-50`. `lut[int(v)]` is looked up with the raw VTK IDs, but the converter renumbers ID 0 to max+1.
- **Input:** `verify_byid.py tests/data/golden/zug_test3_RS/d3plot/zug_test3_RS.d3plot tests/data/golden/zug_test3_RS/vtk`.
- **What should happen:** a per-state comparison by ID, with differences around 5e-4.
- **What happens:** `KeyError: 0`, and no state is compared.
- **Failing test:** `bug-hunt-evidence/c21/test_c21_verify_byid_zero_id.py`.
- **Fix:** renumber zero IDs the same way the converter does (`c21/fix_c21.diff`).

### 24. The reference verifiers compare d3plot states with the wrong VTK files when names are unpadded or a run has 1000+ states (c22, c41)
- **Where:** `tools/reference/verify_zug.py:40` and `tools/reference/verify_byid.py:40` sort file names as strings with `sorted()`. The converter orders files by the number in the A(\d+) suffix.
- **Input:** the zug VTK files renamed `…A8/A9/A10.vtk`, converted with the CLI, then checked with each verifier.
- **What should happen:** state i is compared with the file it was converted from (max|dxyz| ≈ 0).
- **What happens:** state 0 is compared with A10. The false differences are 4.0 / 2.0 / 2.0 (verify_zug) and 20 / 10 / 10 (verify_byid). Nothing flags it: the times are printed but never checked.
- **Failing tests:** `bug-hunt-evidence/c22/test_c22_verify_zug_order.py` and `bug-hunt-evidence/c41/test_verify_byid_sort.py`.
- **Fix:** sort by the numeric suffix (`c22/fix.diff`, `c41/fix.diff`).

### 25. `lspp_check` ignores `--state` unless `--fringe` is also given (c20)
- **Where:** `tools/lspp_check.py:73-75`. The `state N;` command is only written inside `if fringe:`.
- **Input:** `check_d3plot(W3.d3plot, png='x.png', state=5)`, or `lspp_check.py f.d3plot --png x.png --state 5`.
- **What should happen:** a PNG of state 5.
- **What happens:** the command file has no state command, so the PNG shows LS-PrePost's default state, and the run is reported as successful.
- **Failing test:** `bug-hunt-evidence/c20/test_c20_lspp_state.py`. LS-PrePost is replaced, through `LSPP_EXE`, by a stub that records the command file.
- **Fix:** write `state N;` whenever a state is given (`c20/fix.diff`).

---

## Low

### 26. Solid strain combined with shells that have no results crashes the write (c09)
- **Where:** `builder.py:1114-1115`. Solid strain makes lasso set the global ISTRN=1, which means 12 strain words per shell, but no shell state array exists.
- **Input:** a series with a hex, a quad and a triangle (and a line). The only cell results are 3DELEM_Strain and 3DELEM_Stress; there are no 2DELEM_* arrays.
- **What should happen:** a d3plot. The plan's rule is to zero-fill or skip missing arrays.
- **What happens:** lasso raises `ValueError: … expected output data for only 2 shells and not 0`, and nothing is written.
- **Failing test:** `bug-hunt-evidence/c09/test_c09_solid_strain_shells_without_results.py`.
- **Fix:** zero-fill shell strain in this case (`c09/fix_c09.diff`).

### 27. A T01 CSV with only its header line aborts the conversion (c02, c25)
- **Where:** `series.py:88-95` (`parse_t01_csv` returns zero-length columns for a header-only file) and `builder.py:1032-1041` (`np.interp` on those columns).
- **Input:** the port-dialect VTK files plus a `BOXIMPT01.csv` holding only the comment and header lines, e.g. from a run that stopped before its first time-history output.
- **What should happen:** conversion without the T01 globals. The T01 file is optional and found automatically.
- **What happens:** `ValueError: array of sample points is empty`, and no d3plot is written.
- **Failing tests:** `bug-hunt-evidence/c25/test_c25_empty_t01.py` and `bug-hunt-evidence/c02/test_c02_empty_t01.py`.
- **Fix:** return `columns={}` when there are no rows (`c25/fix.diff`), or guard the interpolation in the builder (`c02/fix.diff`).

### 28. The CLI's glob expansion is not sorted (c04)
- **Where:** `src/vtk2d3plot/cli.py:89`, `expanded_inputs.extend(_glob.glob(inp))`. The resulting list keeps file-system order, and `load_series` requires increasing times.
- **Input:** `vtk2d3plot 'dir/*.vtk' -o out.d3plot` on Linux (quoted, so the CLI expands it), or file names without zero padding, or a run with 1000+ states.
- **What should happen:** the same result as passing `dir`.
- **What happens:** "Times are not strictly increasing … BOXIMPA018.vtk is followed by BOXIMPA013.vtk", exit code 1.
- **Failing test:** `bug-hunt-evidence/c04/test_c04_cli_glob_order.py`.
- **Fix:** sort each pattern's matches numerically (`c04/fix_c04.diff`).

### 29. The title time regex crashes on free text and matches inside words (c18)
- **Where:** `vtk_reader.py:356-358`. `re.search(r"t=([0-9.eE+\-]+)", title)` feeds an unguarded `float()`, and this runs before FIELD TIME is read.
- **Input:** the title `Result=extracted`, `state at t=end` or `crash test at t=1.5.` with a valid FIELD TIME of 0.5; or the title `pyradioss count=5 state` with no FIELD block.
- **What should happen:** time 0.5 in the first cases, None in the last.
- **What happens:** `ValueError: could not convert string to float: 'e'`, also through `convert`. The title `count=5` gives time 5.0.
- **Failing test:** `bug-hunt-evidence/c18/test_c18_title_regex.py`.
- **Fix:** require a word boundary and a well-formed number (`c18/fix_c18.diff`).

### 30. An empty title line breaks legacy VTK parsing (c31)
- **Where:** `vtk_reader.py:354`. The title is read from a generator that skips blank lines.
- **Input:** a legacy .vtk file whose line 2 is empty.
- **What should happen:** title ''.
- **What happens:** 'ASCII' is taken as the title, then `ValueError: Expected ASCII format, got: DATASET UNSTRUCTURED_GRID`.
- **Failing test:** `bug-hunt-evidence/c31/test_c31_empty_title.py`.
- **Fix:** read line 2 directly from the file object (`c31/fix_c31.diff`).

### 31. A NaN time raises IndexError instead of the intended clear ValueError (c33)
- **Where:** `series.py:157` and `356`. `np.where(diffs <= 0)[0][0]` finds nothing when the bad difference is NaN.
- **Input:** three states with FIELD TIME 0, nan and 2.
- **What should happen:** "Times are not strictly increasing … (time nan)".
- **What happens:** `IndexError: index 0 is out of bounds`; the CLI prints "ERROR: index 0 is out of bounds…".
- **Failing test:** `bug-hunt-evidence/c33/test_c33_nan_time.py`.
- **Fix:** `idx = int(np.flatnonzero(~(diffs > 0))[0])` (`c33/fix.diff`).

### 32. Upper-case `.VTK` / `.VTU` files are not found in directory or stem mode (c35)
- **Where:** `series.py:115, 127, 303, 313`. These use the case-sensitive `Path.glob("*.vtk")`, while the rest of the code accepts any case.
- **Input:** a folder with `runA000.VTK` and `runA001.VTK`.
- **What should happen:** 2 states. Single-file and explicit-list inputs with the same files work.
- **What happens:** `ValueError: No files found to load.`
- **Failing test:** `bug-hunt-evidence/c35/test_c35_upper_ext.py`.
- **Fix:** `glob(..., case_sensitive=False)`, which needs Python 3.12; the project already requires 3.12 (`c35/fix.diff`).

### 33. `load_series_lazy` never checks NODE_ID / ELEMENT_ID / PART_ID (c34)
- **Where:** `series.py:338`. Later states are read with `metadata_only=True`, which drops the ID arrays, and the lazy loader has no ID comparison at all.
- **Input:** two states, where state 1 has NODE_ID 4 3 2 1 instead of 1 2 3 4.
- **What should happen:** `TopologyError`, as `load_series` raises.
- **What happens:** the series is accepted.
- **Scope:** `convert` uses `load_series`, so only code that calls `load_series_lazy` directly is affected.
- **Failing test:** `bug-hunt-evidence/c34/test_c34_lazy_ids.py`.
- **Fix:** keep the ID arrays in metadata mode and compare them (`c34/fix.diff`).

### 34. Missing-state warnings for internal energy are dropped (c27)
- **Where:** `builder.py:756, 763`. The fill lists are unpacked into `_solid_history_vars_fill` and `_shell_internal_energy_fill`, which are never added to `_simple_fills` (lines 876-883).
- **Input:** 2DELEM_Specific_Energy (or 3DELEM_Specific_Energy) present only in state 0.
- **What should happen:** one "zero-filled at states [1]" warning, per design decision 2 in the plan.
- **What happens:** state 1 is zero-filled with no warning.
- **Failing test:** `bug-hunt-evidence/c27/test_c27_internal_energy_fill_warning.py`.
- **Fix:** keep both lists and add them to `_simple_fills` (`c27/fix.diff`).

### 35. Lower-case point arrays that were mapped are reported as "unmapped arrays ignored" (c28)
- **Where:** `builder.py:1011`. It checks against the exact-case set `_RECOGNIZED_POINT`, while `get_point_key` maps names ignoring case.
- **Input:** a point vector named `velocity`.
- **What should happen:** no "unmapped" warning for an array that was mapped.
- **What happens:** `node_velocity` is written correctly, yet the report says "unmapped arrays ignored: velocity (absmax=1)".
- **Failing test:** `bug-hunt-evidence/c28/test_c28_lowercase_velocity.py`.
- **Fix:** check the kinematic names ignoring case (`c28/fix.diff`).

### 36. The default title is `vtk` when the input is the relative path `vtk` (c11)
- **Where:** `cli.py:150-155`. `Path(args.inputs[0]).parent.name` is '' for a relative path. The same code gives an empty title for '.' and the title '..' for '..'.
- **Input:** `cd model_x && vtk2d3plot vtk -o out.d3plot`.
- **What should happen:** title `model_x`, per the code comment and the repo's `test_cli_vtk_subdir_title`.
- **What happens:** title `vtk`. The absolute path gives `model_x`.
- **Failing test:** `bug-hunt-evidence/c11/test_c11_relative_vtk_title.py`.
- **Fix:** `p = Path(os.path.abspath(args.inputs[0]))` (`c11/fix.diff`).

### 37. One file is written per state, not the promised two-file family (c43)
- **Where:** `api.py:153`, `write_d3plot(..., single_file=False)`. In lasso 2.0.4 this writes one file per state.
- **Input:** converting the 21-state port series.
- **What should happen:** two files holding all states. This is what the `convert` docstring, the CLI `-o` help, the CLI message "Wrote X and X01" and the README ("model.d3plot + model.d3plot01 — always keep them together") all say.
- **What happens:** 22 files are written, and the two named files hold 1 state. The data is complete on disk, but a user who keeps only the two reported files loses 20 states.
- **Failing test:** `bug-hunt-evidence/c43/test_c43_family.py`.
- **Fix:** `single_file=True` (`c43/fix.diff`; the prover notes that in this mode lasso pads only the end of the file). The other option is to correct the docs and the message.

### 38. The output-family regex `\d` also matches non-ASCII digits (c10)
- **Where:** `api.py:139`. In a str pattern, `\d*` matches any Unicode decimal digit, while lasso's own family rule is `[0-9]+`.
- **Input:** a user file `box.d3plot０１` (full-width digits) next to the output `box.d3plot`.
- **What should happen:** the file is left alone.
- **What happens:** it is backed up and deleted.
- **Note:** the claim as first worded was rejected as intended behaviour (see "Rejected" below). That claim was that ASCII suffixes such as `run10` get removed when writing `run1`.
- **Failing test:** `bug-hunt-evidence/c10/test_c10_unicode_digit_family.py`.
- **Fix:** use `[0-9]*` (`c10/fix_c10.diff`).

### 39. The lasso SPH patch always injects zero SPH flags (c38)
- **Where:** `src/vtk2d3plot/lasso_patch.py:20-22`, `getattr(self.header, "numsph", 0)`. lasso's `D3plotHeader` has no such attributes (they live in `raw_header` / `_sph_info`), so 0 is always used.
- **Input:** after `import vtk2d3plot`, which applies the patch globally, write any D3plot with SPH node indexes and SPH density for 1 or 3 particles.
- **What should happen:** the SPH state data is written; that is what the patch claims to fix.
- **What happens:**
  - With 1 particle, the SPH state data is silently dropped, and the file reads back as garbage timesteps.
  - With 3 particles, an uncaught `ValueError: could not broadcast … (3,) into (3,0)`.
- **Scope:** the converter itself never writes SPH, so this only affects other lasso use in the same Python process.
- **Failing test:** `bug-hunt-evidence/c38/test_c38_sph_patch.py`.
- **Fix:** derive the SPH flags from the arrays present and write the state block directly (`c38/fix_c38.diff`).

### 40. `lspp_check` crashes on non-ASCII paths (c40)
- **Where:** `tools/lspp_check.py:95`, `cfile.write_text(..., encoding="ascii")`.
- **Input:** a d3plot or PNG path containing `ü`.
- **What should happen:** the result dict, as the docstring promises.
- **What happens:** `UnicodeEncodeError`, before LS-PrePost even starts.
- **Failing test:** `bug-hunt-evidence/c40/test_c40_lspp_nonascii.py` (with a stub in place of LS-PrePost).
- **Fix:** write the file in the locale encoding with `errors="replace"` (`c40/fix.diff`).

---

## What was read
- **Files:** 16 files, 3,559 lines.
  - All of `src/vtk2d3plot/`: 9 files, 2,946 lines.
  - All of `tools/`: 7 files, 613 lines.
- **Not hunted:** tests, test data and docs. The docs were used as the specification.
- **Slices:**
  - 1: `builder.py` (1,140 lines).
  - 2: `vtk_reader.py` and `series.py` (887 lines).
  - 3: `api.py`, `cli.py`, `mapping.py`, `shell_rotation.py`, `lasso_patch.py` and `__init__.py` (919 lines).
  - 4: `tools/` (613 lines).
- **Coverage:** every slice was read in full by a first reader and again by a second reader. All 8 reported reading every line of every file in their slice, and their counts match the file lengths.
- **Suspicions:** 71 in total (reader / second reader per slice: 11/2, 20/4, 11/6, 15/2).
  - 43 were rated 3 or higher and all went to provers; none were cut.
  - 28 were rated 2 and were not checked (listed below).

## Rejected, and why
- **c10 as first worded** was the only claim a prover rejected. The claim: the cleanup regex deletes another family named output + digits, e.g. `run10` and `run1001` when writing `run1`. The prover rejected it because:
  - the project's tests require 1- and 3-digit stale family members to be removed (`test_atomic_removes_1digit_stale`, `test_atomic_removes_3digit_stale`);
  - the M9 review asks for whole-family cleanup;
  - lasso's own `write_d3plot` removes the same `<base>[0-9]+` files anyway.
- The second prover agreed with that and instead demonstrated the narrower Unicode-digit case, reported above as #38.
- No other claim was rejected.

## Suspicions not checked
- 28 suspicions, all rated 2, listed below. No prover looked at these, so treat them as leads, not bugs:
  - `builder.py:337`: once shell strain forces 2 layers, single-layer stress sources fill only layer 0, while plastic-strain and strain sources are copied into both layers.
  - `builder.py:1036`: `np.interp` is given T01 times without checking that they increase, so restarts or duplicate rows give silently wrong energies.
  - `vtk_reader.py:216`: the VTU `byte_order` attribute is ignored (the header is read little-endian, the data in native order).
  - `vtk_reader.py:391`: METADATA/INFORMATION blocks inside FIELD (written by VTK 8 and later) are parsed as array headers.
  - `vtk_reader.py:354`: truncated files raise a bare StopIteration or IndexError instead of ValueError, and a RuntimeError inside the lazy iterator.
  - `series.py:235`: once any file matches `A<digits>`, other .vtk/.vtu files in the folder are dropped without a warning.
  - `series.py:233`: names without `A<digits>` sort as strings, so `out_10` comes before `out_2` and the time check then rejects the series.
  - `series.py:34`: PVD DataSets for different parts at the same timestep become separate states, so loading fails on duplicate times.
  - `series.py:91`: a ragged row in the optional T01 CSV makes the whole load fail.
  - `series.py:168`: the dialect label is wrong in edge cases, and 'unknown' can never be reached.
  - `api.py:210`: if a rename fails when no family existed before, a partial family is left behind.
  - `mapping.py:44`: the beam orientation node is set to N1 (degenerate), not 0 as the comment says.
  - `api.py:235`: warning classification depends on substring order, so zero-fill warnings can get the wrong code.
  - `api.py:166`: an existing `<file>.bak` belonging to the user is deleted without a warning.
  - `golden_manifest.py:58`: `generate` truncates the manifest before hashing, so an error part-way leaves a partial manifest.
  - `golden_manifest.py:82`: a legacy-format line whose path contains a double space is parsed wrongly and crashes.
  - `golden_manifest.py:99`: a changed file is counted twice in the mismatch total.
  - `lspp_check.py:85`: a broad `except` around the state count falls back to state 1 without a warning.
  - `verify_byid.py:45`: port-dialect files (no NODE_ID or TIME) crash the script.
  - `verify_zug.py:51`: the renumbered zero-ID node, the case this check exists for, is never compared.
  - `make_d3plot.py:11`: hard-coded per-user Windows output path.
  - `make_mixed.py:10`: hard-coded per-user Windows output path.
  - `vtk_reader.py:415`: legacy VTK 5.1 files (OFFSETS/CONNECTIVITY layout, the default of VTK 9 and meshio) are not supported, although the header check accepts any version.
  - `series.py:235`: the same states present as both .vtk and .vtu are loaded twice, so loading fails on duplicate times.
  - `api.py:104`: the progress callback runs for both loading and building, so the CLI shows progress twice.
  - `cli.py:89`: a glob that matches nothing is dropped silently when another input matches.
  - `lspp_check.py:130`: the 32-bit word-size probe looks for a string LS-PrePost never prints, so it never detects 32-bit files.
  - `verify_byid.py:40`: one extra non-series .vtk file in the folder shifts the state pairing.

## Agents that failed
- **15 first provers** failed both of their attempts without answering: c01, c06, c07, c08, c15, c20, c21, c23, c24, c27, c30, c36, c37, c38, c40.
  - Each of these claims went to a fresh second prover.
  - All second provers answered except the one for c01, which also failed.
  - I was not told why any agent failed.
- No reader or second reader failed.
- The run used 67 agents: 4 readers, 4 second readers, 43 provers and 16 second provers.
- After the run, one more prover re-checked c01 and demonstrated it (#9). That makes 68 agents in all.

## Setup notes
- **Dependency:** the code needs `lasso-python==2.0.4`, which was not installed. I installed it, with numpy, pytest and meshio, into a sandbox virtualenv (Python 3.12) so the code could be run.
- **Repo untouched:** the repository itself was never modified; `git status` was clean after the run.
- **Baseline:** the project's own suite passes except `test_golden_manifest`, which is bug #21.
