import re
import warnings
from collections.abc import Callable, Sequence
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


def load_series(
    inputs: str | Path | Sequence[str | Path],
    progress_cb: Callable[[int, int], None] | None = None
) -> VtkSeries:
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
    total = len(paths)
    for i, p in enumerate(paths):
        if progress_cb:
            progress_cb(i, total)
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

        if not np.array_equal(state.cell_connectivity, state0.cell_connectivity):
            raise TopologyError(f"cell_connectivity mismatch in {fname}")

        if not np.array_equal(state.cell_offsets, state0.cell_offsets):
            raise TopologyError(f"cell_offsets mismatch in {fname}")

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


@dataclass
class LazyVtkSeries:
    """Memory-efficient series: holds metadata + paths, yields states on demand.

    Only one state is in memory at a time during iteration. The reference
    state (state0) is kept for topology/ID lookups.
    """

    state0: VtkState
    paths: list[Path]
    dialect: str
    times: np.ndarray
    _array_keys_cache: dict[str, set[str]] | None = None

    @property
    def n_states(self) -> int:
        return len(self.paths)

    def iter_states(
        self,
        progress_cb: Callable[[int, int], None] | None = None,
    ):
        """Yield (index, VtkState) for each state, loading lazily."""
        total = len(self.paths)
        for i, p in enumerate(self.paths):
            if progress_cb:
                progress_cb(i, total)
            yield i, read_vtk(p)

    def detect_array_keys(self) -> dict[str, set[str]]:
        """Scan all states to build union of cell_data and point_data keys.

        This does a full pass reading all files but discards states immediately,
        keeping only the key sets. Used for the detection pass in streaming mode.
        """
        if self._array_keys_cache is not None:
            return self._array_keys_cache

        cell_keys: set[str] = set()
        point_keys: set[str] = set()
        for _, state in self.iter_states():
            cell_keys.update(state.cell_data.keys())
            point_keys.update(state.point_data.keys())
        self._array_keys_cache = {"cell": cell_keys, "point": point_keys}
        return self._array_keys_cache


def load_series_lazy(
    inputs: str | Path | Sequence[str | Path],
    progress_cb: Callable[[int, int], None] | None = None,
) -> LazyVtkSeries:
    """Load a VTK series lazily — only reads state0 + metadata eagerly.

    The returned ``LazyVtkSeries`` can iterate over states one at a time
    via :meth:`~LazyVtkSeries.iter_states`, keeping memory usage constant.
    """
    if isinstance(inputs, (str, Path)):
        path = Path(inputs)
        if path.is_dir():
            files = list(path.glob("*.vtk"))
            paths = _sort_and_filter_files(files)
        elif path.suffix.lower() == ".vtk":
            paths = [path]
        else:
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

    # Read all states just for metadata (times, topology check), then discard
    filenames = [p.name for p in paths]
    states_meta: list[tuple[float, int]] = []  # (time, n_points)

    state0 = read_vtk(paths[0])
    if progress_cb:
        progress_cb(0, len(paths))
    if state0.time is None:
        raise ValueError(f"Time is None for file {filenames[0]}")
    states_meta.append((state0.time, len(state0.points)))

    for i in range(1, len(paths)):
        if progress_cb:
            progress_cb(i, len(paths))
        state = read_vtk(paths[i])
        fname = filenames[i]
        if state.time is None:
            raise ValueError(f"Time is None for file {fname}")
        if len(state.points) != len(state0.points):
            raise TopologyError(f"Points count mismatch in {fname}")
        if not np.array_equal(state.cell_types, state0.cell_types):
            raise TopologyError(f"cell_types mismatch in {fname}")
        if not np.array_equal(state.cell_connectivity, state0.cell_connectivity):
            raise TopologyError(f"cell_connectivity mismatch in {fname}")
        if not np.array_equal(state.cell_offsets, state0.cell_offsets):
            raise TopologyError(f"cell_offsets mismatch in {fname}")
        states_meta.append((state.time, len(state.points)))
        # State is discarded here — only metadata kept

    times = np.array([t for t, _ in states_meta], dtype=np.float64)
    diffs = np.diff(times)
    if not np.all(diffs > 0):
        idx = int(np.where(diffs <= 0)[0][0])
        raise ValueError(
            f"Times are not strictly increasing: file {filenames[idx]} (time {times[idx]}) "
            f"is followed by file {filenames[idx + 1]} (time {times[idx + 1]})"
        )

    title0 = getattr(state0, "title", "")
    if re.search(r"pyradioss", title0, re.IGNORECASE):
        dialect = "port"
    elif state0.time is not None:
        dialect = "official"
    else:
        dialect = "unknown"

    return LazyVtkSeries(
        state0=state0,
        paths=paths,
        dialect=dialect,
        times=times,
    )
