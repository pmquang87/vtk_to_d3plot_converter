# Antigravity kickoff prompts

Copy-paste these into Antigravity IDE (project folder opened at
`C:\Users\pmqua\PycharmProjects\vtk_to_d3plot_converter`).
One milestone = one NEW conversation. Model advice: **Claude Opus 4.6
(thinking)** for M2–M4 (binary/format work), **Gemini 3.1 Pro (High)** fine
for M1 and M5; mixing models spreads quota.

## Conversation 0 (optional, recommended): assumption check

```
/grill-me Read AGENTS.md and @docs/plan.md. You will implement Milestone 1
soon. Before any code: ask me every clarifying question you have about the
VTK dialects (@docs/vtk_input_spec.md), the lasso-python constraints, and the
test fixtures. Do not write code in this conversation.
```

## Per-milestone prompt (N = 1..5)

```
Read AGENTS.md and @docs/plan.md. Implement Milestone N ONLY, exactly as
specified there (files, interfaces, test code, acceptance criteria).

Process, in order:
1. Produce an Implementation Plan artifact for Milestone N and STOP for my
   review. List: files you will create/modify, the failing tests you will
   write first (they are given in the plan - copy them faithfully), and any
   point where you intend to deviate from docs/plan.md (deviations need my
   approval).
2. After approval: write the failing tests, show them failing with
   .venv\Scripts\python.exe -m pytest -q -m milestoneN
3. Implement until that marker is green. Use tools/reference/make_d3plot.py
   as the template for all lasso-python array population.
4. Run the FULL suite: .venv\Scripts\python.exe -m pytest -q  - everything
   must be green (lspp/pyradioss tests may auto-skip if tools are missing,
   but on this machine LS-PrePost IS installed, so lspp tests must RUN).
5. Where the milestone requires an LS-PrePost check, run
   .venv\Scripts\python.exe tools\lspp_check.py <output.d3plot> and show me
   the JSON. For render checks, produce the PNG and show it.
6. Tick the Milestone N checkbox in docs/plan.md, fill in any "record here"
   items (e.g. the verified wedge mapping in M2), commit everything with
   git add -A and git commit -m "feat(mN): <summary>".
7. Produce a Walkthrough artifact and stop. Do NOT start Milestone N+1.

Hard rules (also in AGENTS.md): only .venv\Scripts\python.exe - never bare
python; no pip install; one flat terminal command per call; d3plot paths in
LS-PrePost cfiles need backslashes; golden comparisons join by ID, never by
index.
```

Alternative if the `/milestone` workflow command is picked up from
`.agents/workflows/milestone.md`, simply run:

```
/milestone 1
```

## Review checklist for YOU at each milestone gate

- Implementation Plan artifact matches docs/plan.md (no silent scope creep)?
- Tests were shown RED before implementation?
- Full suite green, lspp tests executed (not skipped)?
- For M2: did you look at the tetra/wedge/pyramid PNG yourself?
- For M3/M4: golden diffs quoted numerically in the walkthrough?
- docs/plan.md checkbox ticked + committed?
