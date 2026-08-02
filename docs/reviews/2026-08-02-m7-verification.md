# M7 fix verification (2026-08-02, commits 0c1389c + c621e1c)

Verified by a 14-agent workflow: 6 verifiers (finding clusters re-running the
original repro evidence, a test-rigor auditor running the new tests against
pre-fix worktrees, a fresh-eyes diff reviewer) + adversarial adjudication of
8 new major claims (6 CONFIRMED, 2 DOWNGRADED to minor); the 3 remaining
unique claims (missing tests / record) confirmed directly by the orchestrator.
Suite state at verification: 45 passed, 0 skipped (incl. lspp + pyradioss).

## Scoreboard against the M6-verification contract

| ID | Status | Regression test | Notes |
|---|---|---|---|
| N1 partial-layer fill | **FIXED** (stress + eps paths; warning names array/layer/states, aggregated) | stress YES (fails pre-fix) / **eps NO** | new gap: a series carrying only ONE layer in ALL states drops stress entirely, silently (P1) |
| N2 reader parser | **FIXED** (np.fromstring gone; 66 MB file: pre-fix 14.0 s → 5.0 s, 2.8×; error context restored) | perf test exists but **cannot detect the regression** (15 s bound vs 0.4 s measured; regressed code also passes) | this is a revert to the M5 strategy (M5 measured 5.1 s — statistically identical), not an improvement |
| N3 --title abort | **FIXED for ASCII** (35/36/37/40/41/60 all convert; >36 warns + truncates; pre-fix aborts closed) | YES, 3 tests (fail pre-fix) | **NOT fixed for non-ASCII** (P2): 'ä', 'Größe' still abort with the checksum RuntimeError and leave a stub file |
| N4 layer-count crash | **PARTIAL** — reconciliation (`n_shell_layers = max(...)` + warning) added, contract scenario A converts | YES for case A (fails pre-fix) | only 3 of 6 allocation sites honour it; 3 combinations still crash (P1) |
| N5 is_alive wording | **FIXED** ("one-filled (all-alive)"; read-back confirms values) | YES (fails pre-fix) | |
| N6 plan.md record | **FIXED in 0c1389c, RE-BROKEN by c621e1c** — the M7 tick claims "all missing regression tests" while CLI --title is absent and 3 N4 combos crash | | the untruthful-tick pattern repeated a third time |
| M1 minimum remedy | **MOSTLY DONE** — local-frame warning fires on real W3/zug conversions (test fails pre-fix); VM tolerance tightened to 1e-4 | YES | "record the deviation durably" NOT done; plan.md:169 still says "intentionally 6%" — now contradicts the code; raw-component deviation unchanged at 49.5 % |
| M7 phantom shells | **FIXED** — masks the 10 zero-ID-node cells; zug shells 864 == golden, shell/beam ids positionally equal, zero-ID node renumbered to 11659 exactly as Vortex; warning emitted; lspp_check ok | YES, 2 tests (fail pre-fix) | mask keys on NODE_ID==0 with no sanity check (0-based-ID files would lose node-0 cells; all-zero NODE_ID → empty d3plot, exit 0) — adjudicated minor, pathological inputs only |
| M8 performance | **DONE (as a revert)** — beats the regressed code 2.8×, equal to the M5 baseline; acceptance "no regression vs M5" met | bound test non-detecting (minor) | contract's letter (one whole-block split) not followed; still per-line split + list extend |

## Mandated regression tests (contract item 3)

| Test | Status |
|---|---|
| C1 solids-port fixture | **PARTIAL** — test_c1_solid_port_vonm exists, fails on pre-M6 code; but fixture includes shells (doesn't exercise the solids-only path), asserts stress only, and survives re-adding the original C1 gate (mutation-tested) |
| M3 union behavior | **DONE** |
| M4/N1 partial-fill warnings | **PARTIAL** — N1 stress path tested; N1 eps path, M4 whole-array-missing, N4 transpose all work but untested |
| CLI --title | **MISSING** — the only occurrence of "--title" in tests/ is a comment (test_m7_fixes.py:235); all title tests go through the API |
| --f64 | **DONE** — real CLI spawn, wordsize/itype/ftype verified, lspp_check ok (coverage, as asked; was never broken) |
| M8 perf bound | exists but hollow — see N2 row |

Acceptance clauses: full suite green incl. new tests YES (45/45, 0 skips);
repro-equivalents committed MOSTLY (C1 partial, beam_repro equivalent still
thin); no perf regression vs M5 YES (4.99 s vs 5.14 s median on the 66 MB
benchmark, statistically indistinguishable).

## NEW/REMAINING defects (P1/P2 adjudicated CONFIRMED major)

**P1. N4 is only 3/6 closed — three layer-count combinations still crash**
with the identical lasso ValueError the contract quoted. `builder.py:218`
(2DELEM_Von_Mises official branch), `:227` (port VONM branch) allocate
shell_stress with hardcoded 1 layer; `:292` (port EPSP branch) allocates
shell_eps with hardcoded 1 layer. Reproduced: VonMises(1)+eps Upper/Lower(2),
VONM(1)+eps Upper/Lower(2), stress lower/upper(2)+EPSP(1) all die at
lasso d3plot.py:9444. The reconcile warning fires, then the allocation
ignores `n_shell_layers`. Fix: use `n_shell_layers` at all three sites
(mirror builder.py:205/:283) + tests for all three combinations.

**P1b. Whole-series single-layer stress silently dropped.** `builder.py:99`:
`has_shell_stress_2layer` requires BOTH (lower) and (upper) to exist
somewhere in the series. A file carrying only `2DELEM_Stress_(upper)` in
every state produces NO shell stress and NO warning (even the local-frame
warning disappears). Same for a single eps layer. Fix: treat one-layer-only
as membrane-like (map into the reconciled layer count) or at minimum warn.

**P2. N3 title guard counts characters, lasso's limit is bytes.**
`builder.py:443` `len(title) > 36`; lasso pads to 40 chars THEN UTF-8-encodes
(d3plot.py:1135-1139), so ANY non-ASCII char overflows the header: title
'Größe' → RuntimeError "byte checksum wrong", stub d3plot left on disk.
For a German-speaking user this is the likely real-world path. Fix: budget
by `len(title.encode('utf-8'))` (truncate to ≤36 bytes on a char boundary),
warn; add a non-ASCII test.

**P3 (records/tests, confirmed by orchestrator).** (a) CLI --title test
missing (mandated). (b) N1 eps-path test missing (fix is real, verified to
fail pre-fix as an uncommitted repro — commit it). (c) plan.md M7 tick
untruthful (N6 pattern): un-tick or reword to list the open items.

**Downgraded to minor (adjudicated):** perf-bound test cannot detect any
realistic regression (would need a fixture-scaled bound or the 66 MB file);
NODE_ID==0 masking has no sanity check (only pathological inputs affected).

**Minors (spot-verified, unadjudicated):** plan.md:169 stale "intentionally
6%" tolerance line; spurious reconcile warning for 2-layer stress +
single eps (that path always worked); padded stress layer is zero-filled
while eps padding replicates (inconsistent); parts whose cells are all
masked survive as element-less parts; mask loop is pure-Python per-vertex;
dead `hasattr` guard builder.py:384; TENSILE fixture output changed 208→202
shells with no golden to validate against; zug masking test asserts count
only; malformed-value error context untested; vtk_reader docstring oversells
the perf change; C2 cross-block renumbering collision policy still
inconsistent (repro fires unchanged; contract item 4, not taken).

## Contract for the next fix round (M8)

1. P1: honour `n_shell_layers` at builder.py:218/:227/:292 + 3 crash tests.
2. P1b: handle/warn the single-layer-only series (stress and eps).
3. P2: byte-budgeted title truncation + non-ASCII test.
4. P3: CLI --title test, N1-eps test, truthful plan.md.
5. Opportunistically: the minors list above, C2 policy unification
   (seed per-block renumbering from the global max ELEMENT_ID).

Acceptance: full suite green including the new tests; each new test
demonstrated to fail on 0c1389c; no perf regression (66 MB benchmark or an
honest bound); no new warnings lost. Re-verify with the same workflow ritual.
