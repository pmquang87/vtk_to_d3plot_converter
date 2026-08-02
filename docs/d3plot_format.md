# d3plot format notes (background — lasso-python does the binary work)

The converter never writes bytes itself; lasso-python 2.0.4 derives the
control block from the arrays you populate. This file exists so nobody has to
guess what lasso is doing underneath, and to anchor debugging when LS-PrePost
rejects a file.

## Authoritative documentation (URL-verified 2026-08-02)

| Source | URL | Note |
|---|---|---|
| "LS-DYNA Database Binary Output Files", Jan 2023 (newest public, 107 pp) | ftp://ftp.lstc.com/anonymous/outgoing/lsprepost/d3plot_database_manual/ | free, no login; also Sep 2017 + Apr 2020 revisions there |
| Older revisions (Mar 2014 – Jan 2018) | ftp://ftp.lstc.com/anonymous/outgoing/jday/manuals/ | nine revisions, all HTTP-200 verified |
| Apr 2020 PDF vendored in lasso repo | https://github.com/open-lasso-python/lasso-python (in-repo docs) | most robust non-LSTC mirror |
| dynasupport.com format pages | https://www.dynasupport.com/manuals/additional/ls-dyna-database-manual | currently UNREACHABLE (Akamai timeouts); recover via web.archive.org |
| Reference reader with spec commentary | VTK repo, `IO/LSDyna/vtkLSDynaReader.cxx` (3785 lines) | header comments document where the LSTC spec is ambiguous/wrong |
| Reference writer (the one we use) | lasso-python `src/lasso/dyna/d3plot.py` (9724 lines) | `write_d3plot` at :6169 + ~27 `_write_*` methods |
| Worked mapping example | Vortex-Radioss `vortex_radioss/animtod3plot/Anim_to_D3plot.py` | fills 36 ArrayType keys, then `write_d3plot(..., single_file=False)` |

## Structure in one page

Words are 4 bytes (32-bit) or 8 bytes (64-bit); the file is a typeless word
stream. Layout: TITLE (10 words) → 64-word CONTROL BLOCK → material section →
GEOMETRY (x,y,z for NUMNP nodes; then connectivity: solids 9·NEL8 [8 nodes +
material], beams 6·NEL2, shells 5·NEL4) → NARBS user-numbering section →
STATE blocks. Each state: TIME (1 word) → NGLBV global variables → nodal data
(order U, V, A as flagged by IU/IV/IA, each 3·NUMNP) → element data (solids
NEL8·NV3D, then beams NEL2·NV1D, then shells NEL4·NV2D) → element deletion
words. Files are padded in 512-word blocks; family splits to `<name>01`,
`<name>02`, ... The final file ends with the float marker **−999999.0**.

Key control words (indices per the manual): NDIM(15), NUMNP(16), NGLBV(18),
IT/IU/IV/IA(19–22), NEL8(23), NEL2(28), NEL4(31), NV3D(27), NV1D(29),
NV2D(32), MAXINT(36), NARBS(39), NMMAT(51ish per revision), IOSHL1–4.
Shell words/element: NV2D = MAXINT·(6·IOSHL1 + IOSHL2 + NEIPS) + 8·IOSHL3 +
4·IOSHL4 (+12 if ISTRN and strain present). ISTRN is *derived* from the NV2D
surplus, not stored. NARBS length = 16 + NUMNP + NEL8 + NEL2 + NEL4 (+NELT)
(+3·NMMAT when material-order arrays present); negative NSORT pointer signals
arbitrary numbering present.

Stress tensor component order everywhere: **σxx, σyy, σzz, σxy, σyz, σzx**.
Element shape is inferred by post-processors from repeated nodes in the
fixed-width connectivity (degenerate padding) — see the mapping table in
docs/plan.md.

## What lasso derives vs. what you control

- Derived from arrays: NUMNP, NEL*, NGLBV, IU/IV/IA flags, MAXINT (from the
  shell stress/eps layer dimension), NV*, NARBS content (from `node_ids`,
  `element_*_ids`, `part_ids`), IT (from node_temperature presence).
- You control: precision (header itype/ftype/**wordsize** — all three for
  64-bit), title, `single_file` (True → geometry + one state file `<name>01`;
  False → one state file per block), `block_size_bytes`.
- Not supported / broken in lasso 2.0.4: SPH state writing (issue #101),
  writing after modifying `node_temperature` (#65), rigid-body DCOMP
  compression (#71). Avoid all three.

## LS-PrePost failure signatures

- Forward-slash d3plot path in a cfile → reader crash (batch).
- Missing `<name>01` beside `<name>` → loads geometry, shows 1 state silently.
- NARBS/id array lengths inconsistent with element counts → "ID Section
  mismatch" or silent part-table corruption.
- Wrong degenerate padding → geometry renders twisted/exploded.
- float64 arrays written into a 32-bit-declared file → wordcount mismatch,
  garbage or "Wrong filetype" errors in readers.
