"""vtk2d3plot public API — convert VTK time series to LS-DYNA d3plot.

Milestone 2: working convert() pipeline with geometry + kinematics.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path


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
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def convert(
    inputs: str | Path | Sequence[str | Path],
    output: str | Path,
    *,
    title: str = "",
    f64: bool = False,
    part_names: dict[int, str] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ConversionReport:
    """Convert a VTK time series to an LS-DYNA d3plot family.

    Parameters
    ----------
    inputs : str | Path | Sequence[str | Path]
        A directory containing VTK files, a glob-able stem, or an
        explicit list of ``.vtk`` file paths.
    output : str | Path
        Path for the output d3plot (``<output>`` + ``<output>01`` will
        be written).
    title : str
        Optional title string embedded in the d3plot header.
    f64 : bool
        If True, write 64-bit output (default: 32-bit).

    Returns
    -------
    ConversionReport
    """
    from .builder import build_d3plot
    from .mapping import map_topology
    from .series import load_series

    series = load_series(inputs)
    model = map_topology(series.states[0])
    d3, warn_list = build_d3plot(
        series,
        model,
        title=title,
        f64=f64,
        part_names=part_names,
        progress=progress,
    )

    if model.n_skipped_cells > 0:
        types_str = ", ".join(str(t) for t in sorted(model.skipped_cell_types))
        warn_list.insert(
            0,
            f"Skipped {model.n_skipped_cells} unsupported cells "
            f"(VTK types: {types_str}); these are not included in the d3plot",
        )

    import re
    import shutil
    import tempfile

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Record which files existed before we start (for safe failure cleanup)
    pre_existing = set()
    escaped = re.escape(out.name)
    pattern = re.compile(f"^{escaped}(\\d*)$", re.IGNORECASE)
    if out.parent.exists():
        for f in out.parent.iterdir():
            if f.is_file() and pattern.match(f.name):
                pre_existing.add(f)

    bak_files = {}
    new_tmp_files = []

    # Atomic write: write to temp, then rename on success
    tmp_dir = out.parent
    try:
        with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
            tmp_base = Path(td) / "tmp.d3plot"
            d3.write_d3plot(str(tmp_base))

            # Move new files to temp names in the target dir first
            tmp_target_base = out.parent / f"_vtk2d3plot_tmp_{out.name}"
            shutil.move(str(tmp_base), str(tmp_target_base))
            new_tmp_files.append(tmp_target_base)
            
            tmp_states = Path(td) / "tmp.d3plot01"
            tmp_target_states = out.parent / f"_vtk2d3plot_tmp_{out.name}01"
            if tmp_states.exists():
                shutil.move(str(tmp_states), str(tmp_target_states))
                new_tmp_files.append(tmp_target_states)

            # Backup pre-existing files
            for f in pre_existing:
                bak_path = f.with_name(f.name + ".bak")
                if f.exists():
                    shutil.move(str(f), str(bak_path))
                    bak_files[f] = bak_path

            # Move the temps to final names
            shutil.move(str(tmp_target_base), str(out))
            if tmp_target_states.exists():
                shutil.move(str(tmp_target_states), str(out.parent / f"{out.name}01"))
            
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
            try:
                if out.exists() and out not in pre_existing:
                    out.unlink()
                out01 = out.parent / f"{out.name}01"
                if out01.exists() and out01 not in pre_existing:
                    out01.unlink()
            except OSError:
                pass
        raise

    return ConversionReport(
        output=str(out),
        n_states=len(series.states),
        n_nodes=model.node_ids.shape[0],
        n_solids=model.solid_conn.shape[0],
        n_shells=model.shell_conn.shape[0],
        n_beams=model.beam_conn.shape[0],
        n_parts=model.part_ids.shape[0],
        dialect=series.dialect,
        warnings=warn_list,
    )
