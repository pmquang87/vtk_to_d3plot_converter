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


def cfile_encoding() -> str:
    """Encoding of the command file: the ANSI code page LS-PrePost reads paths in."""
    import locale

    return locale.getencoding()


def _short_path(p: Path) -> str | None:
    """Windows 8.3 short form of an EXISTING path, or None (non-Windows / no short name)."""
    import os

    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    fn = ctypes.windll.kernel32.GetShortPathNameW
    fn.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    fn.restype = wintypes.DWORD
    n = fn(str(p), None, 0)
    if n == 0:
        return None
    buf = ctypes.create_unicode_buffer(n)
    if fn(str(p), buf, n) == 0:
        return None
    return buf.value


def cfile_path(p: Path, enc: str, notes: list[str]) -> str:
    """Spell path ``p`` so that the command file can carry it.

    Order: ASCII path as is; else the Windows 8.3 short path of the parent
    folder plus the unchanged file name, if that is ASCII (the file name must
    stay long: LS-PrePost finds the family files ``<name>01``, ``<name>02`` by
    appending to it, and the PNG does not exist yet); else the path itself if
    the ANSI code page ``enc`` can encode it (with a note: LS-PrePost 4.13 was
    seen to crash on such a path); else the path with unencodable
    characters replaced by '?' - LS-PrePost will then not find it, and
    ``notes`` says so.
    """
    s = str(p)
    if s.isascii():
        return s
    short = _short_path(p.parent) if p.parent.exists() else None
    if short is not None:
        cand = short.rstrip("\\") + "\\" + p.name
        if cand.isascii():
            notes.append(f"non-ASCII path {s!r} written with 8.3 short folder {cand!r}")
            return cand
    try:
        s.encode(enc)
        # Seen 2026-10-11 with LS-PrePost 4.13: a cp1252 'ü' path loads the
        # states but the run then crashes (0xC0000409) - hence the note.
        notes.append(
            f"non-ASCII path {s!r} has no ASCII 8.3 short form; written in {enc} - "
            "LS-PrePost may fail on it"
        )
        return s
    except UnicodeEncodeError:
        pass
    replaced = s.encode(enc, errors="replace").decode(enc)
    notes.append(
        f"path {s!r} cannot be written to the command file (not representable in {enc}, "
        f"no 8.3 short name); written as {replaced!r} - LS-PrePost cannot open it"
    )
    return replaced


def check_d3plot(
    d3plot_path: str | Path,
    *,
    fringe: str | None = None,
    png: str | Path | None = None,
    state: int | None = None,
    timeout: float = 240.0,
) -> dict:
    """Load a d3plot in LS-PrePost batch mode and report what it saw.

    ``state`` is the 1-based LS-PrePost state number; when given, ``state N;``
    is written with or without ``fringe``, so the PNG shows that state. With
    ``fringe`` and no ``state`` the last state is shown.

    Returns dict with keys (the same keys on every return): loaded (bool),
    n_states (int|None), exit_code, wordsize (4|8|None), elapsed_s,
    msg (path of lspost.msg), cfile (path of the command file), path_notes
    (how non-ASCII paths were written), detail (str). Raises RuntimeError if
    LS-PrePost is not installed.
    """
    exe = find_lsprepost()
    if exe is None:
        raise RuntimeError("LS-PrePost not found (set LSPP_EXE or install 2026R1)")

    d3 = Path(d3plot_path).resolve()
    if not d3.is_file():
        return {
            "loaded": False,
            "n_states": None,
            "exit_code": None,
            "wordsize": None,
            "elapsed_s": 0.0,
            "msg": None,
            "cfile": None,
            "path_notes": [],
            "detail": f"no such file: {d3}",
        }

    workdir = Path(tempfile.mkdtemp(prefix="lspp_check_"))
    enc = cfile_encoding()
    path_notes: list[str] = []
    lines = [f'open d3plot "{cfile_path(d3, enc, path_notes)}"', "ac"]
    if state is not None:
        # 1-based LS-PrePost state number; written with or without a fringe.
        lines.append(f"state {state};")
    elif fringe:
        # No state given: fringe the last state. Resolve the state count from
        # the d3plot for a concrete state command ('state last;' is a silent
        # LS-PrePost no-op).
        try:
            from lasso.dyna import ArrayType
            from lasso.dyna import D3plot as _D3plot

            _dp = _D3plot(str(d3))
            _ts = _dp.arrays.get(ArrayType.global_timesteps)
            _n = len(_ts) if _ts is not None else 1
        except Exception:  # noqa: BLE001
            _n = 1
        lines.append(f"state {_n};")
    if fringe:
        lines += [
            f"fringe {FRINGE_IDS.get(fringe, fringe)}",
            "pfringe",
            "range actele;",
            "isometric x",
        ]
    if png:
        # print path MAY use forward slashes
        p = cfile_path(Path(png).resolve(), enc, path_notes).replace("\\", "/")
        lines.append(f'print png "{p}" LANDSCAPE opaque VEC A4 dpi 150 enlisted "OGL1x1"')
    lines.append("exit")
    cfile = workdir / "check.cfile"
    # Every path in `lines` went through cfile_path, so the text is encodable.
    cfile.write_text("\n".join(lines) + "\n", encoding=enc)

    t0 = time.time()
    try:
        proc = subprocess.run(
            [exe, f"c={cfile}"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout_text = (
            (exc.stdout or b"").decode("utf-8", "replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr_text = (
            (exc.stderr or b"").decode("utf-8", "replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        returncode = -1

    msg_file = workdir / "lspost.msg"
    msg = msg_file.read_text(errors="replace") if msg_file.is_file() else ""
    blob = msg + "\n" + stdout_text + "\n" + stderr_text

    m = re.search(r"Total number of states\s*=\s*(\d+)", blob)
    n_states = int(m.group(1)) if m else None
    finished = "Finished reading model" in blob
    invalid = [ln for ln in blob.splitlines() if "Invalid command" in ln]
    loaded = returncode == 0 and finished and not invalid

    detail = "ok" if loaded else f"finished={finished} invalid={invalid[:3]} rc={returncode}"
    if path_notes:
        detail += "; " + "; ".join(path_notes)
    return {
        "loaded": loaded,
        "n_states": n_states,
        "exit_code": returncode,
        "wordsize": 8 if "64-bit ieee" in blob else (4 if "32-bit ieee" in blob else None),
        "elapsed_s": round(time.time() - t0, 1),
        "msg": str(msg_file),
        "cfile": str(cfile),
        "path_notes": path_notes,
        "detail": detail,
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
