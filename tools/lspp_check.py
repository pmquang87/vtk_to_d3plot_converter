"""Headless LS-PrePost verification of a d3plot family.

Verified invocation pattern (2026-08-02, LS-PrePost 2026R1 [4.13], Windows 11):
run ``lsprepost4.13.exe c=<cfile>``; the ``open d3plot`` path inside the cfile
must use BACKSLASHES (forward slashes crash the d3plot reader); success is
judged from ``lspost.msg`` in the working directory, not from stdout alone.

Usage:
    .venv\\Scripts\\python.exe tools\\lspp_check.py model.d3plot
    .venv\\Scripts\\python.exe tools\\lspp_check.py model.d3plot --fringe vonmises --png out.png

As a library:
    from tools.lspp_check import check_d3plot
    r = check_d3plot("C:/x/model.d3plot", fringe="vonmises", png="C:/x/out.png")
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_EXES = [
    r"C:\Program Files\Ansys\LS-PrePost-2026R1[4.13]\lsprepost4.13.exe",
    r"C:\Program Files\Ansys\LS-PrePost-2026R1[4.13]DP\lsprepost4.13DP.exe",
]

FRINGE_IDS = {"vonmises": 9, "epsp": 7}


def find_lsprepost() -> str | None:
    import os

    env = os.environ.get("LSPP_EXE")
    if env and Path(env).is_file():
        return env
    for exe in DEFAULT_EXES:
        if Path(exe).is_file():
            return exe
    return None


def check_d3plot(
    d3plot_path: str | Path,
    *,
    fringe: str | None = None,
    png: str | Path | None = None,
    state: int | None = None,
    timeout: float = 240.0,
) -> dict:
    """Load a d3plot in LS-PrePost batch mode and report what it saw.

    Returns dict with keys: loaded (bool), n_states (int|None), exit_code,
    msg (path of lspost.msg), detail (str). Raises RuntimeError if LS-PrePost
    is not installed.
    """
    exe = find_lsprepost()
    if exe is None:
        raise RuntimeError("LS-PrePost not found (set LSPP_EXE or install 2026R1)")

    d3 = Path(d3plot_path).resolve()
    if not d3.is_file():
        return {"loaded": False, "n_states": None, "exit_code": None,
                "msg": None, "detail": f"no such file: {d3}"}

    workdir = Path(tempfile.mkdtemp(prefix="lspp_check_"))
    lines = [f'open d3plot "{d3!s}"', "ac"]
    if fringe:
        if state is not None:
            lines.append(f"state {state};")
        else:
            # Default to last state — most interesting for post-processing
            lines.append("state last;")
        lines += [f"fringe {FRINGE_IDS.get(fringe, fringe)}", "pfringe",
                  "range actele;", "isometric x"]
    if png:
        p = Path(png).resolve().as_posix()  # print path MAY use forward slashes
        lines.append(f'print png "{p}" LANDSCAPE opaque VEC A4 dpi 150 enlisted "OGL1x1"')
    lines.append("exit")
    cfile = workdir / "check.cfile"
    cfile.write_text("\n".join(lines) + "\n", encoding="ascii")

    t0 = time.time()
    proc = subprocess.run(
        [exe, f"c={cfile}"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    msg_file = workdir / "lspost.msg"
    msg = msg_file.read_text(errors="replace") if msg_file.is_file() else ""
    blob = msg + "\n" + (proc.stdout or "") + "\n" + (proc.stderr or "")

    m = re.search(r"Total number of states\s*=\s*(\d+)", blob)
    n_states = int(m.group(1)) if m else None
    finished = "Finished reading model" in blob
    invalid = [ln for ln in blob.splitlines() if "Invalid command" in ln]
    loaded = proc.returncode == 0 and finished and not invalid

    return {
        "loaded": loaded,
        "n_states": n_states,
        "exit_code": proc.returncode,
        "wordsize": 8 if "64-bit ieee" in blob else (4 if "32-bit ieee" in blob else None),
        "elapsed_s": round(time.time() - t0, 1),
        "msg": str(msg_file),
        "detail": ("ok" if loaded else
                   f"finished={finished} invalid={invalid[:3]} rc={proc.returncode}"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("d3plot")
    ap.add_argument("--fringe", choices=sorted(FRINGE_IDS), default=None)
    ap.add_argument("--png", default=None)
    ap.add_argument("--state", type=int, default=None)
    args = ap.parse_args()
    r = check_d3plot(args.d3plot, fringe=args.fringe, png=args.png, state=args.state)
    print(json.dumps(r, indent=2))
    return 0 if r["loaded"] else 1


if __name__ == "__main__":
    sys.exit(main())
