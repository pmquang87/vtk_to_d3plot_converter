import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .vtk_reader import VtkState, read_vtk


@dataclass
class VtkSeries:
    states: list[VtkState]
    dialect: str
    times: np.ndarray


class TopologyError(Exception):
    pass


def load_series(inputs: str | Path | Sequence[str | Path]) -> VtkSeries:
    if isinstance(inputs, (str, Path)):
        path = Path(inputs)
        if path.is_dir():
            files = list(path.glob("*.vtk"))
            paths = _sort_and_filter_files(files)
        elif path.suffix.lower() == ".vtk":
            paths = [path]
        else:
            # Stem form
            parent = path.parent
            stem = path.name
            if not parent.exists() and not path.is_absolute():
                parent = Path(".")

            pattern = re.compile(re.escape(stem) + r"A\d+\.vtk$", re.IGNORECASE)
            files = [f for f in parent.glob("*.vtk") if pattern.match(f.name)]
            paths = _sort_and_filter_files(files)
    elif isinstance(inputs, Sequence):
        paths = [Path(p) for p in inputs]
    else:
        raise TypeError(f"Invalid inputs type: {type(inputs)}")

    if not paths:
        raise ValueError("No files found to load.")

    states = []
    for p in paths:
        states.append(read_vtk(p))

    filenames = [p.name for p in paths]

    times_list = []
    for i, state in enumerate(states):
        if getattr(state, "time", None) is None:
            raise ValueError(f"Time is None for file {filenames[i]}")
        times_list.append(state.time)

    times = np.array(times_list, dtype=np.float64)

    diffs = np.diff(times)
    if not np.all(diffs > 0):
        idx = np.where(diffs <= 0)[0][0]
        raise ValueError(
            f"Times are not strictly increasing (non-monotonic order): file {filenames[idx]} (time {times[idx]}) "
            f"is followed by file {filenames[idx + 1]} (time {times[idx + 1]}). "
            f"Decreasing or duplicate time."
        )

    state0 = states[0]
    title0 = getattr(state0, "title", "")
    if re.search(r"pyradioss", title0, re.IGNORECASE):
        dialect = "port"
    elif getattr(state0, "time", None) is not None:
        dialect = "official"
    else:
        dialect = "unknown"

    for i in range(1, len(states)):
        state = states[i]
        fname = filenames[i]

        if len(state.points) != len(state0.points):
            raise TopologyError(f"Points count mismatch in {fname}")

        if not np.array_equal(state.cell_types, state0.cell_types):
            raise TopologyError(f"cell_types mismatch in {fname}")

        if len(state.cells) != len(state0.cells) or not all(
            np.array_equal(a, b) for a, b in zip(state.cells, state0.cells)
        ):
            raise TopologyError(f"cells mismatch in {fname}")

        for field in ["NODE_ID", "ELEMENT_ID", "PART_ID"]:
            pd0 = getattr(state0, "point_data", {})
            pd_curr = getattr(state, "point_data", {})
            if field in pd0 and (
                field not in pd_curr or not np.array_equal(pd0[field], pd_curr[field])
            ):
                raise TopologyError(f"point_data {field} mismatch in {fname}")

            cd0 = getattr(state0, "cell_data", {})
            cd_curr = getattr(state, "cell_data", {})
            if field in cd0 and (
                field not in cd_curr or not np.array_equal(cd0[field], cd_curr[field])
            ):
                raise TopologyError(f"cell_data {field} mismatch in {fname}")

    return VtkSeries(states=states, dialect=dialect, times=times)


def _sort_and_filter_files(files: list[Path]) -> list[Path]:
    if not files:
        return []

    stem_re = re.compile(r"^(.*?)A(\d+)\.vtk$", re.IGNORECASE)

    parsed = []
    stems = set()
    for f in files:
        match = stem_re.match(f.name)
        if match:
            s, num = match.groups()
            stems.add(s)
            parsed.append((s, int(num), f))
        else:
            parsed.append((None, None, f))

    if len(stems) > 1:
        stems_list = ", ".join(sorted(stems))
        raise ValueError(f"Multiple stems found: {stems_list}")

    if len(stems) == 0:
        warnings.warn("No files matched the stem pattern, falling back to lexicographic sort.")
        return sorted(files, key=lambda f: f.name)

    valid_files = [p for p in parsed if p[0] is not None]
    valid_files.sort(key=lambda p: p[1])
    return [p[2] for p in valid_files]
