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

    datasets = collection.findall("DataSet")
    # One file per time is supported. A multi-part collection (several pieces per
    # timestep, told apart by the "part" attribute) would need the pieces merged
    # into one mesh; refuse it clearly instead of a "duplicate time" error later.
    parts = {d.attrib["part"] for d in datasets if "part" in d.attrib}
    if len(parts) > 1:
        parts_str = ", ".join(sorted(parts, key=lambda p: (len(p), p)))
        raise ValueError(
            f"{path.name} is a multi-part collection (parts {parts_str}); "
            f"multi-part PVD files are not supported, give one file per timestep"
        )

    entries: list[tuple[float | None, Path]] = []
    parent_dir = path.parent
    for dataset in datasets:
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
    # Malformed data rows that were skipped (1-based file line numbers), e.g. a
    # last row cut off by a run that crashed while writing.
    skipped_lines: list[int] = field(default_factory=list)
    # Rows dropped because a later row has an equal or earlier TIME (a solver
    # restart rewrites the history from its restart time; a duplicate row is
    # replaced by the later one), so ``times`` is strictly increasing.
    superseded_rows: int = 0


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
    and ``columns == {}``, so no global variable is resampled from it. A data
    row with the wrong number of fields, a non-numeric value or a non-finite
    TIME (e.g. a last row cut off by a crash) is skipped and its line number
    recorded in ``skipped_lines``; malformed rows never abort a conversion.

    The returned ``times`` are strictly increasing (they are interpolated at
    the state times). A row whose TIME is not after the previous kept row
    supersedes every kept row at or after its TIME: a solver restart rewrites
    the history from the restart time on, and of duplicate rows the last one
    wins. The number of dropped rows is ``superseded_rows``.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 2:
        return T01Data(times=np.array([]), columns={})

    header_line = lines[1].strip()
    headers = [h.strip() for h in header_line.split(",")]
    if "TIME" not in headers:
        raise ValueError("TIME column missing from T01 CSV")
    time_idx = headers.index("TIME")

    data: list[list[float]] = []
    skipped: list[int] = []
    superseded = 0
    for line_no, line in enumerate(lines[2:], start=3):
        line = line.strip()
        if not line:
            continue
        fields = line.split(",")
        try:
            if len(fields) != len(headers):
                raise ValueError
            row = [float(x) for x in fields]
            if not math.isfinite(row[time_idx]):
                raise ValueError
        except ValueError:
            skipped.append(line_no)
            continue
        # Restart semantics: the later rows supersede kept rows at or after this time.
        while data and data[-1][time_idx] >= row[time_idx]:
            data.pop()
            superseded += 1
        data.append(row)

    if not data:
        return T01Data(
            times=np.array([]), columns={}, skipped_lines=skipped, superseded_rows=superseded
        )
    arr = np.array(data)
    times = arr[:, time_idx]

    columns = {}
    for i, h in enumerate(headers):
        if h != "TIME":
            columns[h] = arr[:, i]

    return T01Data(times=times, columns=columns, skipped_lines=skipped, superseded_rows=superseded)


# ── shared helpers for load_series / load_series_lazy ───────────────────────


def _list_vtk_files(folder: Path) -> list[Path]:
    """All .vtk/.vtu files in *folder*, extension compared case-insensitively."""
    return [f for f in folder.iterdir() if f.suffix.lower() in _VTK_SUFFIXES and f.is_file()]


def _ignored_files_warning(files: list[Path], kept: list[Path]) -> str | None:
    """Report the .vtk/.vtu files of the input folder that are not series states.

    The folder path is left out on purpose: the user gave it, and api.convert
    classifies warnings by substrings that a folder name could contain.
    """
    kept_set = set(kept)
    ignored = sorted((f.name for f in files if f not in kept_set), key=_natural_key)
    if not ignored:
        return None
    shown = ", ".join(ignored[:10]) + (", ..." if len(ignored) > 10 else "")
    return (
        f"Ignored {len(ignored)} file(s) in the input folder without the "
        f"<stem>A<number>.vtk/.vtu state name: {shown}"
    )


def _resolve_inputs(
    inputs: str | Path | Sequence[str | Path],
    warn_list: list[str],
) -> tuple[list[Path], list[float | None] | None]:
    """Turn the user's input into an ordered file list.

    Returns ``(paths, manifest_times)``; ``manifest_times`` is the per-file
    PVD timestep list for ``.pvd`` input and ``None`` otherwise. Notices
    (files of a series folder that are ignored) go to *warn_list*.
    """
    manifest_times: list[float | None] | None = None
    if isinstance(inputs, (str, Path)):
        path = Path(inputs)
        if path.suffix.lower() == ".pvd":
            pvd_entries = parse_pvd(path)
            paths = [e[1] for e in pvd_entries]
            manifest_times = [e[0] for e in pvd_entries]
        elif path.is_dir():
            files = _list_vtk_files(path)
            paths = _sort_and_filter_files(files)
            notice = _ignored_files_warning(files, paths)
            if notice:
                warn_list.append(notice)
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


def _check_state(
    i: int,
    state: VtkState,
    state0: VtkState,
    manifest_time: float | None,
    filenames: list[str],
    times_list: list[float],
    warn_list: list[str],
) -> float:
    """All checks of file *i*, in the order both loaders share (open point (e)).

    1. time present, 2. time after the previous file's time, 3. time finite,
    4. topology vs state 0, 5. identity arrays vs state 0. The loaders call this
    per file in file order right after reading it, so both report the first
    problem in file order with the same exception. Appends the time to
    *times_list* and returns it.
    """
    fname = filenames[i]
    t = float(_resolve_time(getattr(state, "time", None), manifest_time, fname, warn_list))
    if i > 0:
        prev = times_list[-1]
        # not (t > prev) also catches NaN, which (t <= prev) misses
        if not t > prev:
            raise ValueError(
                f"Times are not strictly increasing (non-monotonic order): file {filenames[i - 1]} (time {prev}) "
                f"is followed by file {fname} (time {t}). "
                f"Decreasing or duplicate time."
            )
    # A lone NaN (state 0) or +inf passes the order test.
    if not math.isfinite(t):
        raise ValueError(f"Time is not finite: file {fname} (time {t})")
    if i > 0:
        _check_topology(state0, state, fname)
        _check_identity(state0, state, filenames[0], fname)
    times_list.append(t)
    return t


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
    # Dialect is provenance of the file content (plan.md design decision 7):
    # a pyradioss title -> "port"; a FIELD TIME (legacy FIELD TIME, VTU
    # FieldData TimeValue / TIME) -> "official"; anything else -> "unknown". A
    # time from a non-pyradioss title "t=...", a VTU PointData TIME or only a
    # PVD manifest timestep is not a FIELD TIME. own_time0 is the file's own
    # time, before any PVD manifest override.
    title0 = getattr(state0, "title", "")
    if re.search(r"pyradioss", title0, re.IGNORECASE):
        return "port"
    if own_time0 is not None and getattr(state0, "time_source", None) == "field":
        return "official"
    return "unknown"


def _load_t01(paths: list[Path], warn_list: list[str]) -> T01Data | None:
    t01_path = find_t01_csv(paths)
    if t01_path is None:
        return None
    t01 = parse_t01_csv(t01_path)
    if t01.skipped_lines:
        warn_list.append(
            f"T01 file {t01_path.name}: {len(t01.skipped_lines)} malformed data row(s) "
            f"skipped (wrong field count or non-numeric value; lines "
            f"{', '.join(map(str, t01.skipped_lines[:5]))}"
            f"{', ...' if len(t01.skipped_lines) > 5 else ''})"
        )
    if t01.superseded_rows:
        warn_list.append(
            f"T01 file {t01_path.name}: {t01.superseded_rows} data row(s) superseded by later "
            f"rows with an equal or earlier TIME (solver restart or duplicate output); "
            f"the later rows are used"
        )
    if t01.times.size == 0:
        warn_list.append(
            f"T01 file {t01_path.name} has no data rows; "
            f"global energies from it (internal, kinetic, total) are not written"
        )
    return t01


def load_series(
    inputs: str | Path | Sequence[str | Path], progress_cb: Callable[[int, int], None] | None = None
) -> VtkSeries:
    warn_list: list[str] = []
    paths, manifest_times = _resolve_inputs(inputs, warn_list)
    filenames = [p.name for p in paths]

    # Read and check file by file, so the first problem in file order is the one
    # reported, exactly as load_series_lazy does (open point (e)).
    states: list[VtkState] = []
    times_list: list[float] = []
    own_time0: float | None = None
    total = len(paths)
    for i, p in enumerate(paths):
        if progress_cb:
            progress_cb(i, total)
        state = read_vtk(p)
        if i == 0:
            own_time0 = getattr(state, "time", None)
        mt = manifest_times[i] if manifest_times is not None else None
        state0 = states[0] if states else state
        state.time = _check_state(i, state, state0, mt, filenames, times_list, warn_list)
        states.append(state)

    times = np.array(times_list, dtype=np.float64)
    dialect = _detect_dialect(states[0], own_time0)
    t01_data = _load_t01(paths, warn_list)

    return VtkSeries(
        states=states, dialect=dialect, times=times, t01_data=t01_data, warnings=warn_list
    )


def _natural_key(name: str) -> tuple:
    """Sort key comparing ASCII digit runs as numbers: out_2 before out_10.

    re.split with a capture group alternates text and digit runs, so equal
    positions always hold the same type; the full name breaks ties.
    """
    parts = re.split(r"([0-9]+)", name)
    return (
        [int(p) if i % 2 else p.casefold() for i, p in enumerate(parts)],
        name,
    )


def _same_state_error(a: Path, b: Path, what: str) -> ValueError:
    a, b = sorted((a, b), key=lambda p: (p.name, str(p)))
    # Names alone are ambiguous when a CLI glob spans folders (r1/runA000.vtk, r2/runA000.vtk).
    na, nb = (a.name, b.name) if a.parent == b.parent else (str(a), str(b))
    return ValueError(
        f"{na} and {nb} are the same state ({what}); remove one of them "
        f"or pass an explicit file list"
    )


def _sort_and_filter_files(files: list[Path]) -> list[Path]:
    """Order a folder's / glob's files as states.

    With ``<stem>A<number>.vtk|vtu`` names: only those files, by state number
    (one stem allowed). Without: all files, in natural order of their names.
    Two files holding the same state (``runA1.vtk`` and ``runA001.vtk``, or
    the same name as ``.vtk`` and ``.vtu``) raise ``ValueError``.
    """
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
        warnings.warn(
            "No files matched the <stem>A<number> state name pattern, "
            "falling back to natural sort of the file names."
        )
        # Same folder and name, different extension (a CLI glob can span folders:
        # step_1/out.vtk and step_2/out.vtk are different states).
        seen_names: dict[tuple[Path, str], Path] = {}
        for f in sorted(files, key=lambda f: f.name):
            key = (f.parent, f.stem)
            if key in seen_names:
                raise _same_state_error(seen_names[key], f, "same name, different extension")
            seen_names[key] = f
        return sorted(files, key=lambda f: (_natural_key(f.name), _natural_key(str(f))))

    valid_files = [p for p in parsed if p[0] is not None]
    seen_nums: dict[int, Path] = {}
    for s, num, f in sorted(valid_files, key=lambda p: p[2].name):
        if num in seen_nums:
            raise _same_state_error(seen_nums[num], f, f"state number {num} of stem '{s}'")
        seen_nums[num] = f
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
    warn_list: list[str] = []
    paths, manifest_times = _resolve_inputs(inputs, warn_list)

    # Read all states just for metadata (times, topology check), then discard
    filenames = [p.name for p in paths]
    times_list: list[float] = []

    def manifest_time(i: int) -> float | None:
        return manifest_times[i] if manifest_times is not None else None

    state0 = read_vtk(paths[0])
    if progress_cb:
        progress_cb(0, len(paths))
    own_time0 = state0.time
    state0.time = _check_state(
        0, state0, state0, manifest_time(0), filenames, times_list, warn_list
    )

    for i in range(1, len(paths)):
        if progress_cb:
            progress_cb(i, len(paths))
        # metadata_only keeps NODE_ID / ELEMENT_ID / PART_ID (vtk_reader
        # _METADATA_KEEP), so the lazy loader applies the same identity rule as
        # load_series (bug report 2026-10-11 #33, c34). It does not parse point
        # coordinates or result arrays, so their parse errors surface only later.
        state = read_vtk(paths[i], metadata_only=True)
        _check_state(i, state, state0, manifest_time(i), filenames, times_list, warn_list)
        # State is discarded here — only metadata kept

    times = np.array(times_list, dtype=np.float64)
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
