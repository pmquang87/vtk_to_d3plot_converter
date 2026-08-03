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
    if out.exists():
        pre_existing.add(out)
    out01 = out.parent / f"{out.name}01"
    if out01.exists():
        pre_existing.add(out01)

    # Track files created by this run
    files_created: list[Path] = []

    # Atomic write: write to temp, then rename on success
    tmp_dir = out.parent
    try:
        with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
            tmp_base = Path(td) / "tmp.d3plot"
            d3.write_d3plot(str(tmp_base))

            # Remove stale family members from a previous run.
            # Use re.escape to handle special chars (brackets, etc.) in the name.
            # Match the base name followed by one or more digits.
            escaped = re.escape(out.name)
            pattern = re.compile(f"^{escaped}(\\d+)$")
            for f in sorted(out.parent.iterdir()):
                if f.is_file() and pattern.match(f.name):
                    f.unlink()
            if out.exists():
                out.unlink()

            # Move the new files
            shutil.move(str(tmp_base), str(out))
            files_created.append(out)
            tmp_states = Path(td) / "tmp.d3plot01"
            if tmp_states.exists():
                shutil.move(str(tmp_states), str(out.parent / f"{out.name}01"))
                files_created.append(out.parent / f"{out.name}01")
    except Exception:
        # Clean up ONLY files this run created (not pre-existing ones)
        for f in files_created:
            if f.exists() and f not in pre_existing:
                f.unlink()
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
