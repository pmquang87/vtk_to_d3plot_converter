# Antigravity setup for this project

One-time preparation so the coding agents (Gemini 3.1 Pro / Claude Opus 4.6) work smoothly.
Sources: official Antigravity docs + Google AI forum threads. Researched 2026-08-02.
Full research: an Opus agent report; key facts repeated here so this file stands alone.

## Which app

You have two Antigravity programs installed:

| App | Version | Use |
|---|---|---|
| Antigravity IDE (`...\Programs\Antigravity IDE\Antigravity IDE.exe`) | 2.1.1 | **Use this one.** VS Code fork, line-by-line review, debugging. |
| Antigravity 2.0 (`...\Programs\Antigravity\Antigravity.exe`) | 2.4.3 | Agent manager for parallel worktree tasks. Not needed here. |

Both share the same agent harness, rules, skills and MCP config (`~/.gemini/`).
Google's own guidance: use the IDE for "single focused coding tasks with code visibility" — exactly this project.

## What this repo already provides (do not recreate)

- `AGENTS.md` — agent rules: env, commands, terminal discipline, domain rules. Officially read by Antigravity (since IDE v1.20.3).
- `docs/plan.md` — the full milestone plan. `AGENTS.md` points to it with `@docs/plan.md`.
- `.agents/rules/env.md` — Always-On rule pinning the Python interpreter (workaround for the documented "agent ignores the venv" bug, confirmed by Google staff on the AI forum).
- `.venv` — Python 3.14.2 with all dependencies installed (vtk, numpy, lasso-python, meshio, pytest, ruff).
- `tests/data/golden/` — golden reference data (same physical states as VTK and as d3plot).
- Git repository with an initial commit. Commit after every milestone — it is your only undo.

## IDE settings to change once

Open user settings JSON (Ctrl+Shift+P → "Open User Settings (JSON)") and add:

```jsonc
{
  // Windows terminal-hang fix from the Google AI forum threads:
  "terminal.integrated.defaultProfile.windows": "PowerShell",   // pwsh 7, NOT "Windows PowerShell"
  "terminal.integrated.windowsEnableConpty": false
}
```

Known Windows bugs these avoid: commands stuck on "Running…" forever (EOF never signalled),
and the `c:\` vs `C:\` path-comparison hang (bug thread 129882).

## Agent permission settings

There is **no sandbox on Windows** — the allow/deny lists are the only guardrail.

- Terminal auto-execution: **Request Review** (IDE). Do not use Turbo/Always Proceed.
- Allow list (token-prefix matched):
  `.venv\Scripts\python.exe`, `.venv\Scripts\pip.exe`, `git status`, `git diff`, `git log`, `git add`, `git commit`
- Deny list:
  `Remove-Item`, `rm`, `del`, `rmdir`, `git push`, `git reset --hard`, `Invoke-WebRequest`, `curl`, `wget`
- Outside-of-workspace file access: **Ask**. Keeps agents out of `E:\foxcore_data\` and `E:\openradioss_run\`.
- Respect `.gitignore`: on (keeps `.venv` out of the context index).
- Browser URL allowlist: `www.dynasupport.com`, `github.com`, `raw.githubusercontent.com`, `pypi.org`,
  `open-lasso-python.github.io`, `docs.vtk.org`, `examples.vtk.org` — the agent will need the format docs.

## Model choice and quota

- Available on consumer tiers: Gemini 3.6/3.5 Flash, Gemini 3.1 Pro, Claude Sonnet 4.6 (thinking), Claude Opus 4.6 (thinking), GPT-OSS-120b. Model is sticky per conversation.
- Recommendation: **Claude Opus 4.6 (thinking)** for the binary-writer milestones (M2–M5) — byte-offset work rewards thinking models. Gemini 3.1 Pro (High) is fine for scaffolding/reader milestones and spreads quota load.
- Quota: Pro refreshes every 5 h up to a weekly cap, and limits are dynamic. Users report multi-hour lockouts of Gemini 3.1 Pro High. Mixing models across milestones reduces the risk of being stranded mid-milestone.
- Decide the "AI Credit Overages" setting deliberately (Never = hard stop at quota).

## How to run the work

1. Open the **project folder** in Antigravity IDE.
2. One milestone = one conversation. Long sessions degrade ("context rot" is the documented failure mode); the plan file + git history carry state between conversations.
3. Kickoff prompt for each milestone (also in `docs/plan.md`, header):

   > Read AGENTS.md and @docs/plan.md. Implement Milestone N only.
   > Produce an Implementation Plan artifact first and stop for my review.

4. Review the Implementation Plan artifact (inline comments, like Google Docs), approve, let it work.
5. It must finish by running `pytest` and showing the output, then produce a Walkthrough artifact.
6. Verify yourself (do not trust the walkthrough alone):
   ```powershell
   .venv\Scripts\python.exe -m pytest -q
   ```
7. `git commit` after every green milestone.
8. Tick the milestone checkbox in `docs/plan.md`.
9. Start a **new conversation** for the next milestone.

Optional: `/grill-me` on milestone 1 makes the agent ask its clarifying questions first — useful for surfacing wrong assumptions about the d3plot layout before any code exists.
Avoid `/goal` (runs to completion without review) until the loop has earned trust.

## MCP (optional)

Config is shared across IDE/2.0/CLI at `~/.gemini/config/mcp_config.json` (currently empty).
Not needed for this project. If you ever add servers: the key is `serverUrl`, not `url` — configs copied from Cursor/VS Code fail silently.
