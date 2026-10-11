"""vtk2d3plot public API — convert VTK time series to LS-DYNA d3plot.

Milestone 2: working convert() pipeline with geometry + kinematics.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

# File-name comparison for the output family: case-insensitive exactly when the
# platform's file system is (ntpath.normcase lower-cases, posixpath.normcase is
# the identity). Module-level so tests can simulate the other platform.
_normcase = os.path.normcase


def _family_matcher(base_name: str) -> Callable[[str], bool]:
    """Return a predicate: is *name* ``base_name`` or ``base_name`` + ASCII digits?

    Mirrors lasso's own family rule ``<base>[0-9]+`` (``[0-9]``, not ``\\d``,
    which would also match non-ASCII digits such as full-width ones).
    """
    pattern = re.compile(re.escape(_normcase(base_name)) + "[0-9]*")
    return lambda name: pattern.fullmatch(_normcase(name)) is not None


def _family_sort_key(base_name: str, name: str) -> int:
    """Numeric state-file index of a family member (0 for the base file)."""
    suffix = name[len(base_name) :]
    return int(suffix) if suffix else 0


def _write_family(d3, out: Path) -> list[Path]:
    """Write the d3plot family ``out``, ``out01``, ... so that it replaces the old one.

    Every scratch file lives in ONE private folder ``_vtk2d3plot_<random>`` that
    is created in the output folder (same file system, so each move is a
    rename): ``new/`` takes the freshly written family, ``old/`` the backups of
    the existing family members. No other name in the output folder is used, so
    a user's file that is not a family member (``box.d3plot.bak``, ...) is never
    overwritten or deleted (bug report 2026-10-11, suspicion S14).

    On any failure every final file this run created is removed and the old
    family members are moved back, whether or not there were any (S11); output
    folders this run created are removed again when they are empty. A backup
    that cannot be moved back stays in the private folder, and a new file that
    cannot be removed stays where it is; the exception gets a note naming them.
    Returns the final files written.
    """
    import shutil
    import tempfile

    made_dirs = [p for p in (out.parent, *out.parent.parents) if not p.exists()]
    out.parent.mkdir(parents=True, exist_ok=True)
    in_family = _family_matcher(out.name)
    pre_existing = [f for f in out.parent.iterdir() if f.is_file() and in_family(f.name)]

    work = Path(tempfile.mkdtemp(prefix="_vtk2d3plot_", dir=out.parent))
    new_dir, old_dir = work / "new", work / "old"
    backups: list[tuple[Path, Path]] = []  # (original name, backup)
    created: list[Path] = []
    keep_work = False
    try:
        new_dir.mkdir()
        old_dir.mkdir()
        # single_file=False: one state file per state (LS-DYNA convention)
        d3.write_d3plot(str(new_dir / "tmp.d3plot"), single_file=False)
        moves = [
            (f, out.parent / f"{out.name}{f.name[len('tmp.d3plot') :]}")
            for f in sorted(new_dir.iterdir())
            if f.is_file() and f.name.startswith("tmp.d3plot")
        ]
        # A directory at a final family name would swallow the file that
        # shutil.move puts there: refuse before anything is moved or removed.
        for _, target in moves:
            if target.is_dir():
                raise IsADirectoryError(
                    f"output family file name is an existing directory: {target}"
                )
        for f in pre_existing:
            bak = old_dir / f.name
            shutil.move(str(f), str(bak))
            backups.append((f, bak))
        for f, target in moves:
            # Claimed before the move: all old members are in old/ now, so a
            # file at this name after a failed move can only be our partial copy.
            created.append(target)
            shutil.move(str(f), str(target))
    except BaseException as exc:
        stuck = []
        for target in created:
            try:
                if target.is_file():
                    target.unlink()
            except OSError:
                stuck.append(str(target))
        if stuck:
            exc.add_note(
                f"vtk2d3plot: {len(stuck)} new d3plot family file(s) could not be removed: "
                f"{', '.join(stuck)}"
            )
        lost = []
        for orig, bak in backups:
            try:
                shutil.move(str(bak), str(orig))
            except OSError:
                lost.append(bak.name)
        if lost:
            keep_work = True
            exc.add_note(
                f"vtk2d3plot: {len(lost)} file(s) of the previous d3plot family could not "
                f"be moved back and are kept in {old_dir}: {', '.join(lost)}"
            )
        else:
            _remove_tree(work)
            for d in made_dirs:  # deepest first; only while empty
                try:
                    d.rmdir()
                except OSError:
                    break
        raise
    finally:
        if not keep_work:
            _remove_tree(work)
    return created


def _remove_tree(path: Path) -> None:
    """Remove *path* recursively, also read-only files (a read-only old family
    member ends up in the backup folder); never raises."""
    import shutil
    import stat

    def clear_readonly(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    try:
        shutil.rmtree(path, onexc=clear_readonly)
    except OSError:
        pass


class WarningCode(str, Enum):
    """Machine-readable warning codes for pipeline integration.

    The code of a warning comes from its message TEMPLATE (see
    ``_WARNING_TEMPLATES``); the README lists what each code means.
    """

    NO_DISPLACEMENT = "NO_DISPLACEMENT"
    SYNTHETIC_STRESS = "SYNTHETIC_STRESS"
    SHELL_LOCAL_FRAME = "SHELL_LOCAL_FRAME"
    STRAIN_MEMBRANE_IGNORED = "STRAIN_MEMBRANE_IGNORED"
    LAYER_MISMATCH = "LAYER_MISMATCH"
    EROSION_FORCED_ALIVE = "EROSION_FORCED_ALIVE"
    SOLID_INTERNAL_ENERGY_HISTVAR = "SOLID_INTERNAL_ENERGY_HISTVAR"
    SKIPPED_CELLS = "SKIPPED_CELLS"
    KNOWN_UNSUPPORTED = "KNOWN_UNSUPPORTED"
    UNMAPPED_ARRAYS = "UNMAPPED_ARRAYS"
    TITLE_SANITIZED = "TITLE_SANITIZED"
    ZERO_FILL = "ZERO_FILL"
    INVERTED_TETRAS = "INVERTED_TETRAS"
    # Added in milestone 14 (bug report 2026-10-11, suspicion S13): templates
    # that had no fitting code before and fell back to OTHER or a wrong code.
    ALIVE_SOURCE = "ALIVE_SOURCE"
    ALIVE_FILL = "ALIVE_FILL"
    EROSION_NO_RESURRECTION = "EROSION_NO_RESURRECTION"
    SHELL_STRESS_ROTATED = "SHELL_STRESS_ROTATED"
    ROTATION_IGNORED = "ROTATION_IGNORED"
    DEGENERATE_SHELL_FRAMES = "DEGENERATE_SHELL_FRAMES"
    SINGLE_LAYER_SOURCE = "SINGLE_LAYER_SOURCE"
    PHANTOM_CELLS_MASKED = "PHANTOM_CELLS_MASKED"
    CASE_COLLISION = "CASE_COLLISION"
    PART_NAMES_UNUSED = "PART_NAMES_UNUSED"
    PVD_TIME_OVERRIDE = "PVD_TIME_OVERRIDE"
    T01_GLOBALS = "T01_GLOBALS"
    OTHER = "OTHER"


# Warning message templates -> code (bug report 2026-10-11, suspicion S13).
#
# Each pattern is matched with ``re.match``, i.e. anchored at the START of the
# message, and describes the fixed text the package writes there. User-controlled
# text (array, file and part names, titles) only ever follows that fixed start,
# so it cannot change the code. The patterns are mutually exclusive (a test
# checks that every known template matches exactly one row), so the row order
# does not matter. A message that matches no row gets ``WarningCode.OTHER``.
#
# To classify a new warning: add ONE row ``(_t(r"^fixed start of the message"),
# WarningCode.X)`` here, a sample message to ``TEMPLATE_SAMPLES`` in
# tests/test_suspicions_api_cli.py, and (for a new code) a README entry.
def _t(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


_WARNING_TEMPLATES: list[tuple[re.Pattern[str], WarningCode]] = [
    # --- kinematics ---------------------------------------------------------
    (_t(r"No DISPLACEMENT array found; "), WarningCode.NO_DISPLACEMENT),
    # --- stress / strain sources ----------------------------------------------
    (_t(r"(?:solid |shell )?stress tensor is synthetic-uniaxial from "), WarningCode.SYNTHETIC_STRESS),
    (_t(r"solid internal energy mapped to element_solid_history_variables"), WarningCode.SOLID_INTERNAL_ENERGY_HISTVAR),
    (_t(r"shell (?:stress|strain) in element-local coordinates"), WarningCode.SHELL_LOCAL_FRAME),
    (_t(r"shell stress from 2DELEM_Stress_\(membrane\) — single layer, element-local coordinates"), WarningCode.SHELL_LOCAL_FRAME),
    (_t(r"shell strain from 2DELEM_Strain_\(membrane\) — single layer"), WarningCode.SHELL_LOCAL_FRAME),
    (_t(r"only 2DELEM_Stress_\((?:upper|lower)\) found .*, element-local coordinates$"), WarningCode.SHELL_LOCAL_FRAME),
    (_t(r"only 2DELEM_Strain_\((?:upper|lower)\) found "), WarningCode.SHELL_LOCAL_FRAME),
    (_t(r"only 2DELEM_Plastic_Strain_(?:Upper|Lower) found "), WarningCode.SINGLE_LAYER_SOURCE),
    (_t(r"2DELEM_Strain_\(membrane\) ignored because "), WarningCode.STRAIN_MEMBRANE_IGNORED),
    (_t(r"shell element results have differing integration layer counts"), WarningCode.LAYER_MISMATCH),
    # --- shell stress rotation (--rotate-shell-stress) --------------------------
    (_t(r"shell stress rotated (?:from the element frame|with the OpenRadioss local frame)"), WarningCode.SHELL_STRESS_ROTATED),
    (_t(r"shell stress from 2DELEM_Stress_\(membrane\) — single layer, rotated from "), WarningCode.SHELL_STRESS_ROTATED),
    (_t(r"only 2DELEM_Stress_\((?:upper|lower)\) found .*, rotated from the element frame"), WarningCode.SHELL_STRESS_ROTATED),
    (_t(r"rotate_shell_stress ignored: "), WarningCode.ROTATION_IGNORED),
    (_t(r"\d+ shells have a degenerate local frame "), WarningCode.DEGENERATE_SHELL_FRAMES),
    # --- element deletion (is_alive) -------------------------------------------
    (_t(r"both OFF and EROSION_STATUS present; "), WarningCode.ALIVE_SOURCE),
    (_t(r"element is_alive from EROSION_STATUS "), WarningCode.ALIVE_SOURCE),
    (_t(r"EROSION_STATUS: \d+ \w+ elements never alive in any state "), WarningCode.EROSION_FORCED_ALIVE),
    (_t(r"EROSION_STATUS: \d+ \w+ elements return to 1 after being eroded "), WarningCode.EROSION_NO_RESURRECTION),
    (_t(r"element_\w+_is_alive: one-filled \(all-alive\) at states "), WarningCode.ALIVE_FILL),
    # --- arrays missing in some states / layout fills --------------------------
    (_t(r"(?:element|node)_\w+: (?:\S+ )?zero-filled at "), WarningCode.ZERO_FILL),
    # --- mesh -------------------------------------------------------------------
    (_t(r"Skipped \d+ unsupported cells "), WarningCode.SKIPPED_CELLS),
    (_t(r"Fixed \d+ tetras with negative volume "), WarningCode.INVERTED_TETRAS),
    (_t(r"Masked \d+ cells referencing zero-ID nodes "), WarningCode.PHANTOM_CELLS_MASKED),
    # --- arrays not written -----------------------------------------------------
    (_t(r"known but unsupported arrays \(no d3plot equivalent\): "), WarningCode.KNOWN_UNSUPPORTED),
    (_t(r"unmapped arrays ignored: "), WarningCode.UNMAPPED_ARRAYS),
    (_t(r"(?:cell|point) arrays '"), WarningCode.CASE_COLLISION),
    # --- titles and part names --------------------------------------------------
    (_t(r"(?:part -?\d+ )?title(?::| truncated from )"), WarningCode.TITLE_SANITIZED),
    (_t(r"part_names entries for part IDs not in the d3plot were ignored: "), WarningCode.PART_NAMES_UNUSED),
    # --- series inputs ----------------------------------------------------------
    (_t(r"PVD manifest timestep "), WarningCode.PVD_TIME_OVERRIDE),
    (_t(r"T01 (?:file |CSV )"), WarningCode.T01_GLOBALS),
    (_t(r"global_total_energy = "), WarningCode.T01_GLOBALS),
]  # fmt: skip


def classify_warning(message: str) -> WarningCode:
    """Machine-readable code of a warning message (``OTHER`` if no template matches)."""
    for pattern, code in _WARNING_TEMPLATES:
        if pattern.match(message):
            return code
    return WarningCode.OTHER


@dataclass
class StructuredWarning:
    """A single warning with machine-readable code and human-readable message."""

    code: WarningCode
    message: str

    def to_dict(self) -> dict:
        return {"code": self.code.value, "message": self.message}


@dataclass
class ConversionReport:
    """Summary returned by :func:`convert`."""

    output: str
    n_states: int
    n_nodes: int
    n_solids: int
    n_shells: int
    n_beams: int
    n_parts: int
    dialect: str
    n_inverted_tetras: int = 0
    warnings: list[str] = field(default_factory=list)
    structured_warnings: list[StructuredWarning] = field(default_factory=list)
    # Every file of the written family, in order: the geometry file
    # ``<output>`` first, then one state file per state ``<output>01`` ...
    output_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert StructuredWarning objects properly
        d["structured_warnings"] = [w.to_dict() for w in self.structured_warnings]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def convert(
    inputs: str | Path | Sequence[str | Path],
    output: str | Path,
    *,
    title: str = "",
    f64: bool = True,
    part_names: dict[int, str] | None = None,
    progress: Callable[[int, int], None] | None = None,
    rotate_shell_stress: bool = False,
) -> ConversionReport:
    """Convert a VTK time series to an LS-DYNA d3plot family.

    Parameters
    ----------
    inputs : str | Path | Sequence[str | Path]
        A directory containing VTK files, a glob-able stem, or an
        explicit list of ``.vtk`` file paths.
    output : str | Path
        Path of the output d3plot family. The geometry goes to ``<output>``
        and each state to its own file ``<output>01``, ``<output>02``, ...
        (one state file per state, LS-DYNA convention), so a series with
        N states writes N + 1 files; :attr:`ConversionReport.output_files`
        lists them. Keep all of them together. Existing members of the same
        family (``<output>`` + ASCII digits, compared case-insensitively
        only where the file system is) are replaced, including stale
        higher-numbered state files. Must not be an existing directory
        (``IsADirectoryError``). The replacement uses one private scratch
        folder ``_vtk2d3plot_<random>`` next to the output and no other name,
        so files that are not family members (e.g. ``<output>.bak``) are never
        touched; on failure the previous family is restored and no new family
        file is left behind.
    title : str
        Optional title string embedded in the d3plot header.
    f64 : bool
        If True, write 64-bit output (default: True).
    progress : callable, optional
        ``progress(i, n)`` is called once per state (``i`` = 0 .. n-1) while
        the states are loaded.

    Returns
    -------
    ConversionReport
    """
    from .builder import build_d3plot
    from .mapping import map_topology
    from .series import load_series

    out = Path(output)
    if out.is_dir():
        raise IsADirectoryError(
            f"output path is an existing directory: {out} "
            f"(give a file path, e.g. {out / 'model.d3plot'})"
        )

    series = load_series(inputs, progress_cb=progress)
    model = map_topology(series.states[0])
    d3, warn_list = build_d3plot(
        series,
        model,
        title=title,
        f64=f64,
        part_names=part_names,
        # Each state is reported once, while it is loaded (bug report
        # 2026-10-11, suspicion S25: forwarding the callback here as well made
        # the CLI print every "Processing state i/N" line twice).
        progress=None,
        rotate_shell_stress=rotate_shell_stress,
    )
    # Series-level warnings (PVD manifest time overrides, a T01 file without
    # data rows, ...) come first, like the loader runs first.
    warn_list[0:0] = series.warnings

    if model.n_skipped_cells > 0:
        types_str = ", ".join(str(t) for t in sorted(model.skipped_cell_types))
        warn_list.insert(
            0,
            f"Skipped {model.n_skipped_cells} unsupported cells "
            f"(VTK types: {types_str}); these are not included in the d3plot",
        )

    if model.n_inverted_tetras > 0:
        warn_list.append(
            f"Fixed {model.n_inverted_tetras} tetras with negative volume by swapping nodes to match LS-DYNA winding order."
        )

    written = _write_family(d3, out)

    # Classify warnings into structured codes for machine-readable pipeline use
    structured = [StructuredWarning(code=classify_warning(w), message=w) for w in warn_list]

    return ConversionReport(
        output=str(out),
        n_states=len(series.states),
        n_nodes=model.node_ids.shape[0],
        n_solids=model.solid_conn.shape[0],
        n_shells=model.shell_conn.shape[0],
        n_beams=model.beam_conn.shape[0],
        n_parts=model.part_ids.shape[0],
        dialect=series.dialect,
        n_inverted_tetras=model.n_inverted_tetras,
        warnings=warn_list,
        structured_warnings=structured,
        output_files=[
            str(p) for p in sorted(written, key=lambda p: _family_sort_key(out.name, p.name))
        ],
    )
