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


class WarningCode(str, Enum):
    """Machine-readable warning codes for pipeline integration."""

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
    OTHER = "OTHER"


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
        (``IsADirectoryError``).
    title : str
        Optional title string embedded in the d3plot header.
    f64 : bool
        If True, write 64-bit output (default: True).

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
        progress=progress,
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

    import shutil
    import tempfile

    out.parent.mkdir(parents=True, exist_ok=True)

    # Record which files existed before we start (for safe failure cleanup)
    pre_existing = set()
    in_family = _family_matcher(out.name)
    if out.parent.exists():
        for f in out.parent.iterdir():
            if f.is_file() and in_family(f.name):
                pre_existing.add(f)

    bak_files = {}
    new_tmp_files = []
    written: list[Path] = []

    # Atomic write: write to temp, then rename on success
    tmp_dir = out.parent
    try:
        with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
            tmp_base = Path(td) / "tmp.d3plot"
            # single_file=False: one state file per state (LS-DYNA convention)
            d3.write_d3plot(str(tmp_base), single_file=False)

            # A directory at a final family name would swallow the file that
            # shutil.move puts there: refuse before anything is moved or removed.
            for f in Path(td).iterdir():
                if f.is_file() and f.name.startswith("tmp.d3plot"):
                    target = out.parent / f"{out.name}{f.name[len('tmp.d3plot') :]}"
                    if target.is_dir():
                        raise IsADirectoryError(
                            f"output family file name is an existing directory: {target}"
                        )

            # Move new files to temp names in the target dir first
            for f in Path(td).iterdir():
                if f.is_file() and f.name.startswith("tmp.d3plot"):
                    suffix = f.name[len("tmp.d3plot") :]
                    tmp_target = out.parent / f"_vtk2d3plot_tmp_{out.name}{suffix}"
                    shutil.move(str(f), str(tmp_target))
                    new_tmp_files.append(tmp_target)

            # Backup pre-existing files
            for f in pre_existing:
                bak_path = f.with_name(f.name + ".bak")
                if bak_path.exists():
                    try:
                        bak_path.unlink()
                    except OSError:
                        pass
                if f.exists():
                    shutil.move(str(f), str(bak_path))
                    bak_files[f] = bak_path

            # Move the temps to final names
            for tmp_f in new_tmp_files:
                suffix = tmp_f.name[len(f"_vtk2d3plot_tmp_{out.name}") :]
                final_name = out.parent / f"{out.name}{suffix}"
                shutil.move(str(tmp_f), str(final_name))
                written.append(final_name)

            # Success! Delete backups
            for bak in bak_files.values():
                try:
                    if bak.exists():
                        bak.unlink()
                except OSError:
                    pass
    except Exception:
        # Failure! Restore backups
        for orig, bak in bak_files.items():
            if bak.exists():
                try:
                    # Windows requires destination to not exist for shutil.move
                    if orig.exists():
                        orig.unlink()
                    shutil.move(str(bak), str(orig))
                except OSError:
                    pass

        # Clean up any leftover tmp files
        for f in new_tmp_files:
            try:
                if f.exists():
                    f.unlink()
            except OSError:
                pass

        # Clean up final names if we were in the middle of replacing them,
        # but only if they weren't pre-existing (since backups restored those).
        if bak_files:
            for tmp_f in new_tmp_files:
                suffix = tmp_f.name[len(f"_vtk2d3plot_tmp_{out.name}") :]
                final_name = out.parent / f"{out.name}{suffix}"
                try:
                    if final_name.exists() and final_name not in pre_existing:
                        final_name.unlink()
                except OSError:
                    pass
        raise

    # Classify warnings into structured codes for machine-readable pipeline use
    structured = []
    for w in warn_list:
        wl = w.lower()
        if "no displacement" in wl:
            code = WarningCode.NO_DISPLACEMENT
        elif "synthetic" in wl or "synthetic-uniaxial" in wl:
            code = WarningCode.SYNTHETIC_STRESS
        elif "local coordinates" in wl or "local frame" in wl:
            code = WarningCode.SHELL_LOCAL_FRAME
        elif "membrane" in wl and "ignored" in wl:
            code = WarningCode.STRAIN_MEMBRANE_IGNORED
        elif "layer" in wl and "reconciled" in wl:
            code = WarningCode.LAYER_MISMATCH
        elif "forced alive" in wl or "erosion" in wl:
            code = WarningCode.EROSION_FORCED_ALIVE
        elif "history_variables" in wl:
            code = WarningCode.SOLID_INTERNAL_ENERGY_HISTVAR
        elif "skipped" in wl and "unsupported cells" in wl:
            code = WarningCode.SKIPPED_CELLS
        elif "known but unsupported" in wl:
            code = WarningCode.KNOWN_UNSUPPORTED
        elif "unmapped" in wl:
            code = WarningCode.UNMAPPED_ARRAYS
        elif "title" in wl and ("truncat" in wl or "transliterat" in wl or "replaced" in wl):
            code = WarningCode.TITLE_SANITIZED
        elif "zero-filled" in wl:
            code = WarningCode.ZERO_FILL
        elif "fixed" in wl and "negative volume" in wl:
            code = WarningCode.INVERTED_TETRAS
        else:
            code = WarningCode.OTHER
        structured.append(StructuredWarning(code=code, message=w))

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
