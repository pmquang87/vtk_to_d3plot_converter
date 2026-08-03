# M9 + Tier-1 verification (2026-08-03, commit 819a3df) and pyradioss tensor writer (1fd484b)

Verified by 20 agents across two workflows: 7 verifiers (M9 contract, P2
title, Tier-1 items, test-rigor with mutations, fresh-eyes diff, pyradioss
writer, end-to-end chain) + 13 adversarial adjudications (8 in-flow, 5
overflow). Suite at verification: converter 76 passed / 0 skipped; pyradioss
worktree: all VTK-writer-touching tests green (48+42+14, full 1130-suite not
awaited).

## Part A — pyradioss tensor writer (1fd484b): CLEAN PASS, merged as PR #45

- TENSORS layout, official names, VONM/EPSP + id arrays preserved: verified
  against the converter's parser and real goldens. Schema test not weakened.
- The frame question was answered properly: solid `sig` is global-frame
  Cauchy (Jaumann `srota3` update in a fixed global basis — verified in
  solid_hexa8.py:384-396/:429, documented in module docstring + tests).
- Tests pin values and FAIL on pre-fix code. End-to-end chain proven:
  pyradioss → VTK → converter → lasso: shell stress arrives as a real
  2-layer tensor (max abs err 3.2e-09 vs Model), solid tensor 9.6e-08
  (f32); the OLD writer's synthetic-uniaxial path even had the wrong sign
  on σxx. LS-PrePost renders v-M from the real tensors, values exact.
- Merge into the main repo already happened (PR #45, verbatim, no conflict
  damage).
- One DOWNGRADED-minor: tensor blocks are emitted unconditionally (not
  gated on /ANIM/ELEM selection) — anim size ~2.8×, write ~2.1× on
  VONM-only decks. Enhancement, not defect: add /ANIM/TENS-style gating.

## Part B — converter M9 contract: all four core items REAL

| Item | Status | Notes |
|---|---|---|
| R1 generic-eps regression | **FIXED** — generic survives (both layers), false warning gone; test fails pre-fix | inverse trade: _Upper now dropped silently when generic wins (minor); the literally-named all-three-arrays test doesn't fail pre-fix (naming/coverage nit) |
| R2 layer slotting | **FIXED** — all four *_only paths slot correctly (exact array_equal proof), physically consistent pairing, warnings name the slot | eps half + stress-lower-only have zero test coverage — ×3.7 mutations there survive (downgraded minor) |
| P2 title | **FIXED via transliteration** — ä→ae etc., other non-ASCII→'?', no crash/stub on any input incl. real surrogates and bytes; LS-PrePost PNG shows "Bauteil Groesse Zugversuch 2026" | minors: --f64 budget keeps 76 chars but lasso reads 72 (4 vanish silently); surrogate/bytes guards untested (the "surrogate" test uses U+FFFD, not a surrogate); the M8 vacuous test was left vacuous |
| Item 4 test hardening | **FIXED** — both named ×3.7 mutations now fail the suite | |
| Item 5 leftovers | **NOT taken** (4th time for C2 policy; membrane+upper still silent) | |

## Part C — Tier-1 items: 4 real, 1 critical defect, 1 regressed

**T1 erosion (CONFIRMED CRITICAL, C1).** Polarity is CORRECT (1=alive —
genani.F "0.0:deleted 1.0:activated"; the roadmap's "1=eroded" was wrong,
now corrected there). TENSILE acceptance exact: 3 shells die, fringe max
0.750→0.237. BUT the raw pass-through (builder.py:479/486/493) marks
never-integrated RIGID-part elements dead: on the repo's own zug fixture,
parts 1 'Einspannung' and 4 'Zugzeite' (rigid grips) carry
EROSION_STATUS=0 from state 0 → 288/864 shells (33%) hidden from t=0.
The Vortex golden marks them ALIVE; LS-PrePost renders the specimen
without its clamping ends (visually confirmed). No test catches it.
**Fix (measured against both fixtures + goldens): latch 1→0 transitions —
an element is dead only after having been observed alive; elements never
equal to 1 in any state are forced alive with a warning naming the count
and part ids. Latching also kills the latent 0→1 resurrection
pass-through. Port-dialect OFF branch stays as is.**

**T5 atomic writes (REGRESSED, 3 CONFIRMED majors).** The roadmap premise
was FALSE — pre-M9, lasso's write_d3plot already removed the entire stale
family (`[0-9]+` regex; measured: 1/2/3-digit members all removed). The
"fix": (a) narrowed cleanup to exactly-2-digit glob → 1- and 3-digit stale
members now survive; (b) the failure handler unlinks out/out01
unconditionally → a conversion failing before touching the output DELETES
the previous good family (pre-M9 preserved it); (c) the glob is built
without escaping → an output name containing '[1]' deletes files of a
DIFFERENT family (brackets are legal Windows names — LS-PrePost's own dir).
Both atomic-write tests are hollow (pass pre-fix; the failure path is
never triggered). The one true gain: no truncated stub after a mid-write
crash. **Fix: only ever delete files this run created; restore whole-family
cleanup with re.escape/glob.escape; real failure-injection tests.**

**T2 unmapped warnings (PARTIAL, 1 CONFIRMED major).** Works, but absmax is
frozen at the first state carrying the key → transient data reports
absmax=0, including the exact 2015.09 array that motivated the feature.
One-line fix (max-accumulate) + pin the number. Minor: hardcoded whitelist
instead of consumed-keys diff; case-sensitivity mismatch.

**T3 part titles: FIXED** (nmmat cliff handled on zug, German names
transliterated, partial mappings fall back correctly; display load-proven,
on-screen label not headlessly verifiable). **T4 SPH skip: FIXED** (minor:
pure-SPH file yields empty model; count/type not pinned by test).
**T6 CLI: HALF** — glob expansion works (untested); per-state progress and
lspp_check state-default NOT done; default title degenerates to 'vtk' for
the standard <model>/vtk layout.

Downgraded minors worth listing: cp1252 --part-names file crashes with a
raw traceback (fails loudly pre-write, nothing corrupted — but a German
user's Notepad default); UTF-8-BOM part-names file silently ignored.

## Contract for the next fix round (M10)

1. **C1 erosion latch** (the measured rule above) + tests: zug 864 alive
   matching the Vortex golden's is_alive, TENSILE exactly {2,3} dead at
   states 15/16+, resurrection clamped. Must fail on 819a3df.
2. **Atomic-write rework** per T5 fix above + failure-injection tests
   (monkeypatched writer raise; pre-existing family must survive).
3. **T2 absmax over all states** + pin 2015.09 on zug.
4. Minors: --f64 title budget 72; real surrogate/bytes tests; R2-eps +
   stress-lower-only value tests; de-vacuum the M8 P2 test; part-names
   encoding tolerance (BOM, cp1252 → clean error or auto-detect);
   erosion+OFF coexistence warning; finish T6 (progress, lspp_check
   --state default last).
5. Records: plan.md tick truthfulness (5th round watch); roadmap
   corrections are committed separately by the verifier.

Acceptance: suite green incl. new tests; new tests fail on 819a3df; zug
is_alive equals the Vortex golden; no deletion of pre-existing output on
injected failure.
