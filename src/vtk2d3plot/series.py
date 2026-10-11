import math
import re
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .vtk_reader import VtkState, read_vtk

# File extensions picked up in directory and stem mode, compared case-insensitively
# (runA000.VTK and runA000.vtk are the same kind of file on every platform).
_VTK_SUFFIXES = (".vtk", ".vtu")

# Identity arrays: present in every state or in none, and equal in every state
# (plan.md design decision 8).
_IDENTITY_FIELDS = ("NODE_ID", "ELEMENT_ID", "PART_ID")


def parse_pvd(path: Path) -> list[tuple[float | None, Path]]:
    """Parse a ParaView ``.pvd`` collection into ``(timestep, file)`` entries.

    A DataSet without a ``timestep`` attribute gets ``None`` (the file's own
    time is then used). Entries are sorted by timestep when every entry has
    one; otherwise the document order is kept.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "VTKFile" or root.attrib.get("type") != "Collection":
        raise ValueError("Invalid PVD file")

    collection = root.find("Collection")
    if collection is None:
        raise ValueError("Missing Collection element")

    entries: list[tuple[float | None, Path]] = []
    parent_dir = path.parent
    for dataset in collection.findall("DataSet"):
        ts_attr = dataset.attrib.get("timestep")
        ts = float(ts_attr) if ts_attr is not None else None
        file_name = dataset.attrib.get("file", "")
        # Resolve relative path
        if not Path(file_name).is_absolute():
            vtu_path = parent_dir / file_name
        else:
            vtu_path = Path(file_name)
        entries.append((ts, vtu_path))
    if all(ts is not None for ts, _ in entries):
        entries.sort(key=lambda x: x[0])
    return entries


@dataclass
class T01Data:
    times: np.ndarray
    columns: dict[str, np.ndarray]


@dataclass
class VtkSeries:
    states: list[VtkState]
    dialect: str
    times: np.ndarray
    t01_data: T01Data | None = None
    # Series-level notices for the ConversionReport (manifest time overrides,
    # empty T01 file, ...), in the report's warning style.
    warnings: list[str] = field(default_factory=list)


class TopologyError(Exception):
    pass


def find_t01_csv(vtk_paths: list[Path]) -> Path | None:
    if not vtk_paths:
        return None
    stem_re = re.compile(r"^(.*?)A(\d+)\.(vtk|vtu)$", re.IGNORECASE)
    match = stem_re.match(vtk_paths[0].name)
    if not match:
        return None
    stem = match.group(1)
    t01_path = vtk_paths[0].parent / f"{stem}T01.csv"
    if t01_path.exists():
        return t01_path
    return None


def parse_t01_csv(path: Path) -> T01Data:
    """Parse a ``{run}T01.csv`` time-history file.

    A file without data rows (only its comment and header lines, e.g. from a run
    that stopped before its first time-history output) returns empty ``times``
    and ``columns == {}``, so no global variable is resampled from it.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 2:
        return T01Data(times=np.array([]), columns={})

    header_line = lines[1].strip()
    headers = [h.strip() for h in header_line.split(",")]

    data = []
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        data.append([float(x) for x in line.split(",")])

    if "TIME" not in headers:
        raise ValueError("TIME column missing from T01 CSV")

    if not data:
        return T01Data(times=np.array([]), columns={})
    data = np.array(data)

    time_idx = headers.index("TIME")
    times = data[:, time_idx]

    columns = {}
    for i, h in enumerate(headers):
        if h != "TIME":
            columns[h] = data[:, i]

    return T01Data(times=times, columns=columns)


# ── shared helpers for load_series / load_series_lazy ───────────────────────


def _list_vtk_files(folder: Path) -> list[Path]:
    """All .vtk/.vtu files in *folder*, extension compared case-insensitively."""
    return [f for f in folder.iterdir() if f.suffix.lower() in _VTK_SUFFIXES and f.is_file()]


def _resolve_inputs(
    inputs: str | Path | Sequence[str | Path],
) -> tuple[list[Path], list[float | None] | None]:
    """Turn the user's input into an ordered file list.

    Returns ``(paths, manifest_times)``; ``manifest_times`` is the per-file
    PVD timestep list for ``.pvd`` input and ``None`` otherwise.
    """
    manifest_times: list[float | None] | None = None
    if isinstance(inputs, (str, Path)):
        path = Path(inputs)
        if path.suffix.lower() == ".pvd":
            pvd_entries = parse_pvd(path)
            paths = [e[1] for e in pvd_entries]
            manifest_times = [e[0] for e in pvd_entries]
        elif path.is_dir():
            paths = _sort_and_filter_files(_list_vtk_files(path))
        elif path.suffix.lower() in _VTK_SUFFIXES:
            paths = [path]
        else:
            # Stem form: "<folder>/<stem>" means "<folder>/<stem>A<n>.vtk|vtu".
            # A bare stem ("run") has parent "." and so means ./runA*.vtk.
            parent = path.parent
            stem = path.name
            if not parent.is_dir():
                raise FileNotFoundError(
                    f"Folder '{parent}' of the stem input '{inputs}' does not exist"
                )
            pattern = re.compile(re.escape(stem) + r"A\d+\.(vtk|vtu)$", re.IGNORECASE)
            files = [f for f in _list_vtk_files(parent) if pattern.match(f.name)]
            paths = _sort_and_filter_files(files)
    elif isinstance(inputs, Sequence):
        # Explicit list: caller order is kept (plan.md design decision 9).
        paths = [Path(p) for p in inputs]
    else:
        raise TypeError(f"Invalid inputs type: {type(inputs)}")

    if not paths:
        raise ValueError("No files found to load.")
    return paths, manifest_times


def _times_agree(a: float, b: float) -> bool:
    # official-dialect TIME is printed with 6 significant digits
    return math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-12)


def _resolve_time(
    own_time: float | None,
    manifest_time: float | None,
    fname: str,
    warn_list: list[str],
) -> float:
    """State time: the PVD manifest timestep when there is one, else the file's own.

    The manifest wins on disagreement, as in VTK's/ParaView's PVD reader, which
    takes the DataSet ``timestep`` attribute and ignores times stored in the
    data files: the manifest is the series index its writer built for exactly
    this purpose. The override is reported, never silent.
    """
    if manifest_time is None:
        if own_time is None:
            raise ValueError(f"Time is None for file {fname}")
        return own_time
    if own_time is not None and not _times_agree(own_time, manifest_time):
        warn_list.append(
            f"PVD manifest timestep {manifest_time!r} used for {fname}; "
            f"it overrides the file's own time {own_time!r}"
        )
    return manifest_time


def _check_times(times: np.ndarray, filenames: list[str]) -> None:
    diffs = np.diff(times)
    # ~(d > 0) also catches NaN differences, which (d <= 0) misses
    bad = np.flatnonzero(~(diffs > 0))
    if bad.size:
        idx = int(bad[0])
        raise ValueError(
            f"Times are not strictly increasing (non-monotonic order): file {filenames[idx]} (time {times[idx]}) "
            f"is followed by file {filenames[idx + 1]} (time {times[idx + 1]}). "
            f"Decreasing or duplicate time."
        )


def _check_topology(state0: VtkState, state: VtkState, fname: str) -> None:
    if len(state.points) != len(state0.points):
        raise TopologyError(f"Points count mismatch in {fname}")
    if not np.array_equal(state.cell_types, state0.cell_types):
        raise TopologyError(f"cell_types mismatch in {fname}")
    if not np.array_equal(state.cell_connectivity, state0.cell_connectivity):
        raise TopologyError(f"cell_connectivity mismatch in {fname}")
    if not np.array_equal(state.cell_offsets, state0.cell_offsets):
        raise TopologyError(f"cell_offsets mismatch in {fname}")


def _check_identity(state0: VtkState, state: VtkState, fname0: str, fname: str) -> None:
    """NODE_ID / ELEMENT_ID / PART_ID: in all states or in none, and equal.

    Called for every state against state 0; presence must match state 0, so an
    array missing from state 0 must be missing everywhere, and equality with
    state 0 makes every pair of states equal.
    """
    for kind in ("point_data", "cell_data"):
        d0 = getattr(state0, kind, {})
        d = getattr(state, kind, {})
        for name in _IDENTITY_FIELDS:
            in0, inn = name in d0, name in d
            if in0 and not inn:
                raise TopologyError(f"{kind} {name} present in {fname0} but missing in {fname}")
            if inn and not in0:
                raise TopologyError(f"{kind} {name} missing in {fname0} but present in {fname}")
            if in0 and not np.array_equal(d0[name], d[name]):
                raise TopologyError(f"{kind} {name} mismatch in {fname} (differs from {fname0})")


def _detect_dialect(state0: VtkState, own_time0: float | None) -> str:
    # own_time0 is the file's own time, before any PVD manifest override:
    # dialect is provenance of the file content.
    title0 = getattr(state0, "title", "")
    if re.search(r"pyradioss", title0, re.IGNORECASE):
        return "port"
    if own_time0 is not None:
        return "official"
    return "unknown"


def _load_t01(paths: list[Path], warn_list: list[str]) -> T01Data | None:
    t01_path = find_t01_csv(paths)
    if t01_path is None:
        return None
    t01 = parse_t01_csv(t01_path)
    if t01.times.size == 0:
        warn_list.append(
            f"T01 file {t01_path.name} has no data rows; "
            f"global energies from it (IE, KE, EN) are not written"
        )
    return t01


def load_series(
    inputs: str | Path | Sequence[str | Path], progress_cb: Callable[[int, int], None] | None = None
) -> VtkSeries:
    paths, manifest_times = _resolve_inputs(inputs)

    states = []
    total = len(paths)
    for i, p in enumerate(paths):
        if progress_cb:
            progress_cb(i, total)
        states.append(read_vtk(p))

    filenames = [p.name for p in paths]
    warn_list: list[str] = []

    own_time0 = getattr(states[0], "time", None)
    times_list = []
    for i, state in enumerate(states):
        mt = manifest_times[i] if manifest_times is not None else None
        t = _resolve_time(getattr(state, "time", None), mt, filenames[i], warn_list)
        state.time = t
        times_list.append(t)

    times = np.array(times_list, dtype=np.float64)
    _check_times(times, filenames)

    state0 = states[0]
    dialect = _detect_dialect(state0, own_time0)

    for i in range(1, len(states)):
        _check_topology(state0, states[i], filenames[i])
        _check_identity(state0, states[i], filenames[0], filenames[i])

    t01_data = _load_t01(paths, warn_list)

    return VtkSeries(
        states=states, dialect=dialect, times=times, t01_data=t01_data, warnings=warn_list
    )


def _sort_and_filter_files(files: list[Path]) -> list[Path]:
    if not files:
        return []

    stem_re = re.compile(r"^(.*?)A(\d+)\.(vtk|vtu)$", re.IGNORECASE)

    parsed = []
    stems = set()
    for f in files:
        match = stem_re.match(f.name)
        if match:
            s, num, _ = match.groups()
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
    t01_data: T01Data | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def n_states(self) -> int:
        return len(self.paths)

    def iter_states(
        self,
        progress_cb: Callable[[int, int], None] | None = None,
    ):
        """Yield (index, VtkState) for each state, loading lazily.

        Each state carries the series time (``self.times[i]``), which for PVD
        input is the manifest timestep.
        """
        total = len(self.paths)
        for i, p in enumerate(self.paths):
            if progress_cb:
                progress_cb(i, total)
            state = read_vtk(p)
            state.time = float(self.times[i])
            yield i, state

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
    paths, manifest_times = _resolve_inputs(inputs)

    # Read all states just for metadata (times, topology check), then discard
    filenames = [p.name for p in paths]
    warn_list: list[str] = []
    times_list: list[float] = []

    def manifest_time(i: int) -> float | None:
        return manifest_times[i] if manifest_times is not None else None

    state0 = read_vtk(paths[0])
    if progress_cb:
        progress_cb(0, len(paths))
    own_time0 = state0.time
    state0.time = _resolve_time(own_time0, manifest_time(0), filenames[0], warn_list)
    times_list.append(state0.time)

    for i in range(1, len(paths)):
        if progress_cb:
            progress_cb(i, len(paths))
        state = read_vtk(paths[i], metadata_only=True)
        fname = filenames[i]
        times_list.append(_resolve_time(state.time, manifest_time(i), fname, warn_list))
        _check_topology(state0, state, fname)
        # Identity arrays (bug report 2026-10-11 #33, c34) are not compared here
        # yet: metadata_only reading drops NODE_ID/ELEMENT_ID/PART_ID. Once the
        # reader keeps them in metadata mode, add
        #     _check_identity(state0, state, filenames[0], fname)
        # State is discarded here — only metadata kept

    times = np.array(times_list, dtype=np.float64)
    _check_times(times, filenames)

    dialect = _detect_dialect(state0, own_time0)
    t01_data = _load_t01(paths, warn_list)

    return LazyVtkSeries(
        state0=state0,
        paths=paths,
        dialect=dialect,
        times=times,
        t01_data=t01_data,
        warnings=warn_list,
    )
