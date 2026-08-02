"""vtk2d3plot public API — convert VTK time series to LS-DYNA d3plot.

Milestone 2: working convert() pipeline with geometry + kinematics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
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


def convert(
    inputs: str | Path | Sequence[str | Path],
    output: str | Path,
    *,
    title: str = "",
    f64: bool = False,
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
    d3, warn_list = build_d3plot(series, model, title=title, f64=f64)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    d3.write_d3plot(str(out))

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
