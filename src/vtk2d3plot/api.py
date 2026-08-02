"""vtk2d3plot public API — convert VTK time series to LS-DYNA d3plot.

Full implementation arrives in milestone 2; this module provides the
ConversionReport dataclass and the convert() signature so that downstream
tests and the CLI can import them from M1 onward.
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

    Raises
    ------
    NotImplementedError
        Until milestone 2 is implemented.
    """
    raise NotImplementedError("convert() is implemented in milestone 2")
