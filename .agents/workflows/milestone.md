---
description: Implement one milestone of docs/plan.md, test-first, with review gate
---

1. Read AGENTS.md and docs/plan.md in full.
2. Identify Milestone $1 in docs/plan.md. Do not touch any other milestone.
3. Produce an Implementation Plan artifact for Milestone $1 only: files to
   create/modify, the failing tests to write first, expected shapes/interfaces
   (they are fully specified in docs/plan.md). STOP and wait for review.
4. After approval: write the failing tests, run
   `.venv\Scripts\python.exe -m pytest -q -m milestone$1` to show them red.
5. Implement minimally until that marker is green, then run the FULL suite:
   `.venv\Scripts\python.exe -m pytest -q`
6. If the milestone has an lspp step, run tools\lspp_check.py as specified and
   show the JSON summary (and view the PNG if one was produced).
7. Tick the milestone checkbox in docs/plan.md, update any "record here"
   notes the plan asks for (e.g. the verified wedge mapping).
8. `git add -A` then `git commit -m "feat(m$1): <summary>"`.
9. Produce a Walkthrough artifact and stop. The next milestone gets a fresh
   conversation.
