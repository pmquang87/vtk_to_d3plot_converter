# M8 fix verification (2026-08-02, commits b9c01b5 + 15d7b2a)

Verified by a 12-agent workflow: 4 verifiers (P1/P1b re-running the exact
6-combination crash matrix, P2/P3 byte-matrix + CLI, test-rigor audit against
a 0c1389c worktree with mutation spot-checks, fresh-eyes diff review) +
adversarial adjudication of 8 new major claims (5 CONFIRMED, 3 DOWNGRADED to
minor; the 1 unadjudicated claim is a duplicate of a confirmed one).
Suite state at verification: 54 passed, 0 skipped (incl. lspp + pyradioss).

## Scoreboard against the M7-verification contract

| ID | Status | Regression test | Notes |
|---|---|---|---|
| P1 (3 layer-crash sites) | **FIXED** — all 6 combinations convert; formerly-crashing C/D/E verified value-correct on read-back (synthetic-uniaxial VM 100/200 exact); LS-PrePost loads all three | YES, 3 tests, each fails pre-fix with the exact lasso ValueError | mutation check: restoring a hardcoded 1 fails the suite. No test pins padded-layer CONTENT |
| P1b (single-layer-only drop) | **FIXED core** — present layer mapped + truthful warning; pre-fix silent drop reproduced | YES, 2 tests, fail pre-fix | but introduced R1 (major regression, below) and R2 (mis-slotting, minor); tests assert presence only — a ×3.7 value mutation survives the suite |
| P2 (byte-budget title) | **PARTIAL** — crash + stub file genuinely closed (12-case matrix, CLI rc=0, no stubs, warning iff truncated) | YES, 2 tests, fail pre-fix | **the "byte budget" is an ASCII filter** (below); German titles lose everything from the first umlaut |
| P3a CLI --title test | **DONE** — spawns the real CLI | YES | passes on pre-fix code — expected (pins M7 behavior) |
| P3b N1-eps test | **DONE** | YES | passes on pre-fix code — expected (pins M7 behavior) |
| P3c truthful records | **PARTIAL** — M7 record honestly corrected, stale plan.md:169 tolerance line fixed; but the M8 tick claims "P2 (non-ASCII title)" done without qualification | | 4th consecutive over-tick, though this time everything EXCEPT P2 is real |
| M7 minors (opportunistic) | plan.md:169 FIXED; padding asymmetry NOT fixed; spurious reconcile warning NOT fixed; C2 cross-block policy NOT taken (3rd time) | | |
| Regression checks | M1 local-frame warning, M7 masking count, N1 aggregation, N5 one-filled wording all survived; dead hasattr guard removed; no earlier test weakened (diff = additions only) | | |

## NEW defects introduced/remaining (adjudicated)

**R1 (CONFIRMED major, regression). Generic eps discarded when Upper exists
without Lower.** `builder.py:335` places `has_shell_eps_upper_only` BEFORE
`has_shell_eps_single`, and the flag (`:132-134`) does not exclude the generic
`2DELEM_Plastic_Strain` array. Real anim_to_vtk exports carry all three eps
arrays; strip only `_Lower` and HEAD discards the generic array entirely,
hard-zeros layer 1, and emits a factually false warning ("only ..._Upper
found"). Pre-M8 populated both layers from the generic array — this is the
first data-loss regression since M6. Fix: exclude generic from the *_only
flags (or reorder branches) + a fixture carrying all three arrays.

**R2 (DOWNGRADED minor). Single-layer data always lands in layer 0.** The
four *_only branches write index 0 even when reconciled to 2 layers; the
2-layer convention elsewhere is layer0=lower/layer1=upper, so an upper-only
source pairs upper stress with lower eps at the same "integration point"
(proven by exact array equality on W3). Disclosed in the warning, contrived
input — hence minor. Fix: map the present layer to its own slot.

**P2-root (CONFIRMED major ×3, one shared cause). Not a byte budget.**
`builder.py:538` `_padded_bytes(t) = len(t.ljust(40).encode("utf-8"))` pads
to 40 CHARS then encodes, so the value is >40 iff ANY non-ASCII char remains;
the loop strips from the tail until the title is pure ASCII. Measured:
'Größe' → 'Gr'; 'Zugversuch Größe 3' → 'Zugversuch Gr'; 'ä'×20 → '' (title
destroyed); 'Bauteil Größe Zugversuch 2026' → 'Bauteil Gr' (trailing ASCII
discarded too). Underlying constraint: lasso pads the title to 40 chars and
THEN encodes, so any non-ASCII char structurally overflows the 40-byte header
— a true UTF-8 byte budget cannot work against this lasso version. Real
remedies: (a) transliterate umlauts (ä→ae, ö→oe, ü→ue, ß→ss) before the
36-char budget, warn; and/or (b) investigate lasso's `_str_codec` (latin-1 /
cp1252 would make 1 char = 1 byte) and verify LS-PrePost renders it.
Additionally CONFIRMED: `test_p2_non_ascii_long_title_truncation` is vacuous
(an empty title satisfies every assertion) and its docstring describes
char-boundary truncation the code does not perform.

**Minors:** `_padded_bytes` hardcodes 40 but the field is 10×wordsize (80
bytes with --f64, untested combination); unguarded `.encode()` raises on
unpaired surrogates from Windows argv; bytes-typed title now raises
AttributeError; membrane+upper coexistence drops upper with no warning;
dead `else 1` guards in the new eps branches; P1b/P1 presence-only asserts.

## Contract for the next fix round (M9)

1. R1: flag/branch-order fix + all-three-eps-arrays regression test
   (must fail on b9c01b5).
2. P2 properly: transliteration and/or lasso codec route (verify what
   LS-PrePost actually displays for latin-1 umlaut bytes); budget against
   the actual wordsize (--f64 = 80 bytes); make the vacuous test pin exact
   read-back; guard surrogates/bytes-input.
3. R2: slot the present layer correctly (upper→1, lower→0) + test.
4. Harden P1/P1b tests to pin values (the ×3.7 mutation must fail).
5. Leftovers, opportunistically: padding asymmetry + reconcile-warning
   wording, membrane+upper warning, C2 cross-block policy.

Acceptance: full suite green including new tests; new tests demonstrated to
fail on b9c01b5; W3/zug/TENSILE goldens byte-stable vs HEAD except where a
fix intentionally changes them (document each such change).
