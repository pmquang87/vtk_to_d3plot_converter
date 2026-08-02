# M6 fix verification (2026-08-02, commit 58c0804)

Verified by an 11-agent workflow: 4 verifiers re-ran the ORIGINAL repro
scripts that proved each finding of `2026-08-02-m1-5-review.md`, plus a
fresh-eyes diff review; 7 new claims adversarially adjudicated (all 7
CONFIRMED). Suite state at verification: 32/32 green (incl. lspp + pyradioss).

## Scoreboard against the review findings

| ID | Status | Regression test | Notes |
|---|---|---|---|
| C1 port solids stress | **FIXED** (repro exact: [vonm,0,0,0,0,0], mixed case nonzero, warned) | **NO** — revert would go undetected | beam-scalar branch still missing (beams-only port model drops VONM/EPSP silently) |
| C2 zero ELEMENT_IDs | **FIXED** — zug beams now 1..342, byte-identical to the Vortex golden | YES (fails on pre-fix code — verified) | cross-block collisions possible (same as Vortex); policy inconsistent with no-ELEMENT_ID fallback |
| M1 shell local frame | **NOT FIXED** — not attempted (commit msg honest; plan checkbox is not) | no | cheapest close-out: warn on the (upper/lower) path + tighten VM tolerance to 1e-4 + record deviation |
| M2 membrane stress | **FIXED** (TENSILE now has stress + membrane warning; test exists) | YES | "warn on recognized-but-unmapped family" clause not implemented |
| M3 state-0 gate | **FIXED** (union over states, `_any_state_has`) | **NO** | |
| M4 zero-fill warnings | **PARTIAL** — whole-array-missing case warned & aggregated | NO | partial-LAYER holes remain (see N1) |
| M5 title | **FIXED core / REGRESSED edge** (test exists, API path) | YES | >40 chars aborts opaquely (N3); 37–40 chars round-trip truncated to 36 (lasso writes 10 words, reads 9) |
| M6 beam is_alive | **FIXED** (test verified to fail on pre-fix code) | YES | mock is 1-state/1-beam only; no lspp/mixed coverage |
| M7 phantom zero-ID shells | **NOT FIXED — worse**: C2 now hands the 10 phantom zug shells clean sequential IDs (11016–11025), removing the zero-ID red flag; ours 874 shells vs golden 864 (Vortex masks them out) | no | decide: mask like Vortex (drop cells referencing the zero-ID node) or keep + warn |
| M8 reader performance | **REGRESSED** — `np.fromstring(sep=' ')` is the SLOW legacy text parser: **2.6× slower** than the per-line loop it replaced, on the exact benchmarked 66 MB file | no perf test | correct fix: split the whole block once, `np.array(tokens, dtype=...)` |
| Minors | 4 fixed (file-naming in count errors, both stale doc lines, wedge line), 2 partial (out>01 mention, tetra/wedge fixtures), rest NOT fixed (--f64 test, dialect classifier, ncomp/symmetry checks, unmapped-array warnings, lspp wordsize, silent dropped files, tuple signature doc) | | |

## NEW defects introduced by 58c0804 (all adjudicated CONFIRMED, all major)

**N1. Partial-layer silent zero-fill (stress AND plastic strain).**
`builder.py:181` and `:261`: the fill-tracking fires only when BOTH
`(lower)` and `(upper)` are absent. A state carrying only one layer gets the
other layer silently zero — half-zero stress with no warning. Newly reachable
because M3's union detection now enables the 2-layer branch even when the
layers appear in different states.

**N2. M8 "performance fix" is a 2.6× performance regression.**
`vtk_reader.py:38/61`: `np.fromstring(s, sep=' ')` is NumPy's deprecated
slow text parser. Measured slower than the code it replaced on the
review's benchmark file. Also lost the offending-value context in
malformed-number errors. Fix: one `str.split()` over the whole numeric block
+ `np.array(tokens, dtype=...)`; add a timed sanity bound to the test suite.

**N3. `--title` > 40 chars aborts the conversion** with an opaque lasso
checksum RuntimeError (and can leave a truncated file). Titles 37–40 chars
read back truncated to 36 (lasso writes 10 title words but reads 9).
Fix: validate/truncate to 36 chars with a warning in builder.py.

**N4. Membrane stress + Upper/Lower plastic strain = crash.**
A file with 1-layer membrane stress and 2-layer plastic strain now dies in
lasso with a raw shape broadcast error (layer-count coupling between the M2
membrane branch and the eps replication logic). Fix: derive eps layer count
independently of the stress layer count (lasso allows them to differ? — no:
reconcile to one MAXINT; pick max and pad, with a warning).

**N5 (minor but misleading).** The aggregated fill warning labels the
is_alive arrays "zero-filled" when they are actually ONE-filled (all alive) —
the message states the opposite of what was written.

**N6 (minor).** `plan.md` checkbox claims "Fix all issues flagged in the
2026-08-02 code review" — untrue (M1, M7 open). Correct the record.

## Contract for the next fix round (M7)

1. N1–N4 (new majors), N5/N6 (records).
2. M1 minimum remedy, M7 decision (recommend: mask zero-ID-node cells like
   Vortex; reconciles shell count with the golden and kills the phantom
   geometry), M8 done properly with a perf bound.
3. The mandated missing regression tests: C1 solids-port fixture (commit the
   scratchpad repro as a test fixture), M3 union behavior, M4/N1 partial-fill
   warnings, CLI --title, --f64.
4. Leftover minors from the table above, opportunistically.

Acceptance: full suite green INCLUDING the new tests; re-run of
`repro_solid_port.py`, `beam_repro.py`, `m6_c2_zug_ids.py` equivalents as
committed tests; no perf regression vs the M5 baseline (time one W3 series
load in a test with a generous bound).
