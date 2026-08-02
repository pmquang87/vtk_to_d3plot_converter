---
trigger: always_on
description: Python interpreter and terminal discipline for this workspace
---

Always run Python through `.venv\Scripts\python.exe` in the workspace root
(e.g. `.venv\Scripts\python.exe -m pytest -q`). Never use a global interpreter
(C:\Python314 or anything on PATH), never `activate` the venv, never bare
`python` or `pip`. Dependencies are preinstalled; do not run pip install.
One flat command per terminal call; no `&&`, no pipes, no output redirection.
