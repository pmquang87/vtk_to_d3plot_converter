"""vtk2d3plot - convert VTK unstructured-grid time series to LS-DYNA d3plot.

Package skeleton; the real modules are implemented milestone by milestone
per docs/plan.md (vtk_reader, series, mapping, builder, api, cli).
"""

__version__ = "0.1.0"

from .api import ConversionReport, convert  # noqa: F401
