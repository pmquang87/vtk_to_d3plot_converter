import base64
import re
import struct
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class VtkState:
    path: str
    title: str  # raw line 2
    time: float | None  # from FIELD TIME or title regex r"t=([0-9.eE+\-]+)"
    points: np.ndarray  # (n_points, 3) float64
    cell_types: np.ndarray  # (n_cells,) int32, VTK type ids
    cell_connectivity: np.ndarray  # flat 1D int32 of all node indices
    cell_offsets: np.ndarray  # (n_cells+1,) int32, start offset per cell
    point_data: dict[str, np.ndarray] = field(default_factory=dict)
    cell_data: dict[str, np.ndarray] = field(default_factory=dict)

    def get_cell(self, i: int) -> np.ndarray:
        """Return the 0-based node indices for cell *i*."""
        return self.cell_connectivity[self.cell_offsets[i] : self.cell_offsets[i + 1]]

    @property
    def n_cells(self) -> int:
        return len(self.cell_types)


def _read_tokens(
    lines: Iterator[str],
    count: int,
    initial_tokens: list[str] | None = None,
    filepath: str | None = None,
) -> list[str]:
    """Read exactly `count` whitespace-separated tokens from the line iterator."""
    tokens = initial_tokens.copy() if initial_tokens else []
    while len(tokens) < count:
        try:
            line = next(lines)
            tokens.extend(line.split())
        except StopIteration:
            break
    if len(tokens) < count:
        file_ctx = f" in file '{filepath}'" if filepath else ""
        raise ValueError(f"Expected {count} tokens, got {len(tokens)}{file_ctx}")
    return tokens[:count]


def _skip_tokens(
    lines: Iterator[str],
    count: int,
    initial_tokens: list[str] | None = None,
) -> None:
    idx = 0
    if initial_tokens:
        idx += min(len(initial_tokens), count)
    
    while idx < count:
        try:
            line = next(lines)
        except StopIteration:
            break
        tokens = line.split()
        if not tokens:
            continue
        idx += min(len(tokens), count - idx)


def _read_numeric_block(
    lines: Iterator[str],
    count: int,
    dtype,
    filepath: str | None = None,
    initial_tokens: list[str] | None = None,
) -> np.ndarray:
    """Read exactly `count` tokens and convert to a numpy array.

    Uses str.split() + pre-allocated numpy array slice assignment for speed
    and lower memory usage (avoiding huge lists of strings).
    """
    arr = np.empty(count, dtype=dtype)
    idx = 0

    if initial_tokens:
        n = min(len(initial_tokens), count)
        try:
            arr[:n] = initial_tokens[:n]
        except (ValueError, OverflowError) as exc:
            for j, tok in enumerate(initial_tokens[:n]):
                try:
                    dtype(tok)
                except (ValueError, OverflowError):
                    file_ctx = f" in file '{filepath}'" if filepath else ""
                    raise ValueError(f"Malformed numeric value {tok!r} at position {idx + j}{file_ctx}") from exc
            raise
        idx += n

    while idx < count:
        try:
            line = next(lines)
        except StopIteration:
            break
        tokens = line.split()
        if not tokens:
            continue
        n = min(len(tokens), count - idx)
        try:
            arr[idx : idx + n] = tokens[:n]
        except (ValueError, OverflowError) as exc:
            for j, tok in enumerate(tokens[:n]):
                try:
                    dtype(tok)
                except (ValueError, OverflowError):
                    file_ctx = f" in file '{filepath}'" if filepath else ""
                    raise ValueError(f"Malformed numeric value {tok!r} at position {idx + j}{file_ctx}") from exc
            raise
        idx += n

    if idx < count:
        file_ctx = f" in file '{filepath}'" if filepath else ""
        raise ValueError(f"Expected {count} tokens, got {idx}{file_ctx}")
    return arr


def read_vtu(path: str | Path, metadata_only: bool = False) -> VtkState:
    path = str(path)
    
    root_attrib = {}
    piece = None
    appended_encoding = "base64"
    appended_data_pos = -1
    
    with open(path, "rb") as f:
        context = ET.iterparse(f, events=("start", "end"))
        try:
            for event, elem in context:
                if event == "start":
                    if elem.tag == "VTKFile":
                        root_attrib = elem.attrib
                    elif elem.tag == "AppendedData":
                        appended_encoding = elem.attrib.get("encoding", "base64")
                        break
                elif event == "end" and elem.tag == "Piece":
                    piece = elem
        except ET.ParseError:
            pass

    if piece is None:
        raise ValueError("Missing Piece element")

    n_points = int(piece.attrib.get("NumberOfPoints", 0))
    n_cells = int(piece.attrib.get("NumberOfCells", 0))

    import mmap
    with open(path, "rb") as f:
        f.seek(0, 2)
        fsize = f.tell()
        if fsize > 0:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                idx = mm.find(b"<AppendedData")
                if idx != -1:
                    tag_end = mm.find(b">", idx)
                    if tag_end != -1:
                        under_idx = mm.find(b"_", tag_end)
                        if under_idx != -1:
                            appended_data_pos = under_idx + 1

                if appended_encoding == "base64":
                    if appended_data_pos != -1:
                        end_idx = mm.find(b"</AppendedData>", appended_data_pos)
                        if end_idx != -1:
                            raw_b64 = mm[appended_data_pos:end_idx]
                        else:
                            raw_b64 = mm[appended_data_pos:]
                        appended_data_cache = base64.b64decode(raw_b64.strip())
                    else:
                        appended_data_cache = b""
                else:
                    appended_data_cache = b""
        else:
            appended_data_cache = b""

    def parse_data_array(elem) -> np.ndarray:
        dtype_str = elem.attrib.get("type", "Float64")
        dt_map = {
            "Float64": np.float64,
            "Float32": np.float32,
            "Int32": np.int32,
            "Int64": np.int64,
            "UInt32": np.uint32,
            "UInt64": np.uint64,
            "UInt8": np.uint8,
            "Int8": np.int8,
        }
        dt = dt_map.get(dtype_str, np.float64)
        ncomp = int(elem.attrib.get("NumberOfComponents", 1))
        fmt = elem.attrib.get("format", "ascii")

        if fmt == "ascii":
            text = elem.text or ""
            arr = np.fromstring(text, sep=" ", dtype=dt)
        elif fmt == "binary":
            text = elem.text or ""
            raw = base64.b64decode(text.strip())
            header_type = root_attrib.get("header_type", "UInt32")
            if header_type == "UInt64":
                header_size = 8
                data_size = struct.unpack("<Q", raw[:8])[0]
            else:
                header_size = 4
                data_size = struct.unpack("<I", raw[:4])[0]
            arr = np.frombuffer(
                raw, dtype=dt, count=data_size // np.dtype(dt).itemsize, offset=header_size
            ).copy()
        elif fmt == "appended":
            offset = int(elem.attrib.get("offset", 0))
            header_type = root_attrib.get("header_type", "UInt32")
            if appended_encoding == "base64":
                if header_type == "UInt64":
                    header_size = 8
                    data_size = struct.unpack("<Q", appended_data_cache[offset : offset + 8])[0]
                else:
                    header_size = 4
                    data_size = struct.unpack("<I", appended_data_cache[offset : offset + 4])[0]
                arr = np.frombuffer(
                    appended_data_cache,
                    dtype=dt,
                    count=data_size // np.dtype(dt).itemsize,
                    offset=offset + header_size,
                ).copy()
            elif appended_encoding == "raw":
                if appended_data_pos == -1:
                    raise ValueError("Could not find raw appended data offset")
                with open(path, "rb") as f2:
                    f2.seek(appended_data_pos + offset)
                    if header_type == "UInt64":
                        header_size = 8
                        data_size = struct.unpack("<Q", f2.read(8))[0]
                    else:
                        header_size = 4
                        data_size = struct.unpack("<I", f2.read(4))[0]
                    
                    raw_bytes = f2.read(data_size)
                    arr = np.frombuffer(raw_bytes, dtype=dt).copy()
            else:
                raise ValueError(f"Unknown appended encoding {appended_encoding}")
        else:
            raise ValueError(f"Unknown format {fmt}")

        if ncomp > 1:
            arr = arr.reshape((-1, ncomp))
        return arr

    points = np.zeros((n_points, 3), dtype=np.float64) if metadata_only else np.zeros((0, 3), dtype=np.float64)
    if n_points > 0 and not metadata_only:
        pts_elem = piece.find("Points")
        if pts_elem is not None:
            da = pts_elem.find("DataArray")
            if da is not None:
                points = parse_data_array(da)

    cell_types = np.zeros(0, dtype=np.int32)
    cell_connectivity = np.zeros(0, dtype=np.int32)
    cell_offsets = np.zeros(1, dtype=np.int32)

    if n_cells > 0:
        cells_elem = piece.find("Cells")
        if cells_elem is not None:
            for da in cells_elem.findall("DataArray"):
                name = da.attrib.get("Name", "")
                if name == "types":
                    cell_types = parse_data_array(da).astype(np.int32)
                elif name == "connectivity":
                    cell_connectivity = parse_data_array(da).astype(np.int32)
                elif name == "offsets":
                    offsets_arr = parse_data_array(da).astype(np.int32)
                    cell_offsets = np.concatenate(([0], offsets_arr))

    point_data = {}
    time_val = None
    pd_elem = piece.find("PointData")
    if pd_elem is not None:
        for da in pd_elem.findall("DataArray"):
            name = da.attrib.get("Name", "")
            if name:
                if metadata_only:
                    if name.upper() == "TIME":
                        arr = parse_data_array(da)
                        if arr.size > 0:
                            time_val = float(arr.flat[0])
                else:
                    arr = parse_data_array(da)
                    point_data[name] = arr
                    if name.upper() == "TIME" and arr.size > 0:
                        time_val = float(arr.flat[0])

    cell_data = {}
    if not metadata_only:
        cd_elem = piece.find("CellData")
        if cd_elem is not None:
            for da in cd_elem.findall("DataArray"):
                name = da.attrib.get("Name", "")
                if name:
                    cell_data[name] = parse_data_array(da)

    return VtkState(
        path=path,
        title="VTU File",
        time=time_val,
        points=points.astype(np.float64),
        cell_types=cell_types,
        cell_connectivity=cell_connectivity,
        cell_offsets=cell_offsets,
        point_data=point_data,
        cell_data=cell_data,
    )


def read_vtk(path: str | Path, metadata_only: bool = False) -> VtkState:
    path = str(path)

    if str(path).lower().endswith(".vtu"):
        return read_vtu(path, metadata_only=metadata_only)

    with open(path, "rb") as f:
        header = f.read(200).decode("utf-8-sig", errors="ignore").strip()
        if header.startswith(("<?xml", "<VTKFile")):
            return read_vtu(path, metadata_only=metadata_only)

    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        # global skip for blank lines
        def non_blank_lines():
            for line in f:
                line = line.strip()
                if line:
                    yield line

        lines = non_blank_lines()

        # Line 1: # vtk DataFile Version X.Y
        try:
            line1 = next(lines)
        except StopIteration:
            raise ValueError("Empty file")

        if not line1.lower().startswith("# vtk datafile version"):
            raise ValueError(f"Invalid VTK header: {line1}")

        # Line 2: Title
        title = next(lines)
        time_val = None
        m = re.search(r"t=([0-9.eE+\-]+)", title)
        if m:
            time_val = float(m.group(1))

        # Line 3: ASCII
        line3 = next(lines)
        if not line3.upper().startswith("ASCII"):
            raise ValueError(f"Expected ASCII format, got: {line3}")

        # Line 4: DATASET UNSTRUCTURED_GRID
        line4 = next(lines)
        if not line4.upper().startswith("DATASET UNSTRUCTURED_GRID"):
            raise ValueError(f"Expected DATASET UNSTRUCTURED_GRID, got: {line4}")

        points = np.zeros((0, 3), dtype=np.float64)
        cell_connectivity = np.zeros(0, dtype=np.int32)
        cell_offsets = np.zeros(1, dtype=np.int32)  # [0]
        cell_types = np.zeros(0, dtype=np.int32)
        point_data = {}
        cell_data = {}

        current_data_dict = None
        current_n = 0

        for line in lines:
            tokens = line.split()
            if not tokens:
                continue

            cmd = tokens[0].upper()

            if cmd == "FIELD":
                # FIELD FieldData N
                num_fields = int(tokens[2])
                for _ in range(num_fields):
                    header = next(lines).split()
                    name = header[0]
                    ncomp = int(header[1])
                    ntuples = int(header[2])
                    val_type = header[3]

                    num_vals = ncomp * ntuples
                    if metadata_only and name.upper() != "TIME":
                        _skip_tokens(lines, num_vals, initial_tokens=header[4:])
                    else:
                        vals = _read_tokens(lines, num_vals, filepath=path, initial_tokens=header[4:])
                        if name.upper() == "TIME" and num_vals > 0:
                            time_val = float(vals[0])
            elif cmd == "POINTS":
                n_points = int(tokens[1])
                if metadata_only:
                    _skip_tokens(lines, n_points * 3, initial_tokens=tokens[3:])
                    points = np.zeros((n_points, 3), dtype=np.float64)
                else:
                    arr = _read_numeric_block(lines, n_points * 3, dtype=np.float64, filepath=path, initial_tokens=tokens[3:])
                    points = arr.reshape((n_points, 3))
            elif cmd == "CELLS":
                n_cells = int(tokens[1])
                size = int(tokens[2])
                arr = _read_numeric_block(lines, size, dtype=np.int32, filepath=path, initial_tokens=tokens[3:])

                # Build flat connectivity + offsets from npts idx0 idx1 ...
                offsets_list = [0]
                conn_parts = []
                idx = 0
                for _ in range(n_cells):
                    npts = int(arr[idx])
                    conn_parts.append(arr[idx + 1 : idx + 1 + npts])
                    offsets_list.append(offsets_list[-1] + npts)
                    idx += 1 + npts
                cell_connectivity = (
                    np.concatenate(conn_parts) if conn_parts else np.zeros(0, dtype=np.int32)
                )
                cell_offsets = np.array(offsets_list, dtype=np.int32)
            elif cmd == "CELL_TYPES":
                n_cells = int(tokens[1])
                cell_types = _read_numeric_block(lines, n_cells, dtype=np.int32, filepath=path, initial_tokens=tokens[2:])
            elif cmd == "POINT_DATA":
                current_n = int(tokens[1])
                current_data_dict = point_data
            elif cmd == "CELL_DATA":
                current_n = int(tokens[1])
                current_data_dict = cell_data
            elif cmd == "SCALARS":
                if current_data_dict is None:
                    continue
                name = tokens[1]
                val_type = tokens[2].lower()
                dtype = np.int32 if "int" in val_type else np.float64
                tokens_upper = [t.upper() for t in tokens]
                if "LOOKUP_TABLE" in tokens_upper:
                    idx = tokens_upper.index("LOOKUP_TABLE")
                    ncomp = int(tokens[3]) if idx > 3 else 1
                    initial_tokens = tokens[idx + 2:]
                else:
                    ncomp = int(tokens[3]) if len(tokens) > 3 else 1
                    initial_tokens = tokens[4:] if len(tokens) > 3 else tokens[3:]
                    try:
                        peek_line = next(lines)
                    except StopIteration:
                        peek_line = ""

                    if peek_line.upper().startswith("LOOKUP_TABLE"):
                        initial_tokens.extend(peek_line.split()[2:])
                    else:
                        initial_tokens.extend(peek_line.split())

                num_vals = current_n * ncomp
                if metadata_only and name.upper() != "TIME":
                    _skip_tokens(lines, num_vals, initial_tokens=initial_tokens)
                else:
                    arr = _read_numeric_block(
                        lines, num_vals, dtype=dtype, filepath=path, initial_tokens=initial_tokens
                    )
                    if ncomp > 1:
                        arr = arr.reshape((current_n, ncomp))
                    current_data_dict[name] = arr
            elif cmd == "VECTORS":
                if current_data_dict is None:
                    continue
                name = tokens[1]
                val_type = tokens[2].lower()
                dtype = np.int32 if "int" in val_type else np.float64
                num_vals = current_n * 3
                if metadata_only and name.upper() != "TIME":
                    _skip_tokens(lines, num_vals, initial_tokens=tokens[3:])
                else:
                    arr = _read_numeric_block(lines, num_vals, dtype=dtype, filepath=path, initial_tokens=tokens[3:])
                    current_data_dict[name] = arr.reshape((current_n, 3))
            elif cmd == "TENSORS":
                if current_data_dict is None:
                    continue
                name = tokens[1]
                val_type = tokens[2].lower()
                dtype = np.int32 if "int" in val_type else np.float64
                num_vals = current_n * 9
                if metadata_only and name.upper() != "TIME":
                    _skip_tokens(lines, num_vals, initial_tokens=tokens[3:])
                else:
                    arr = _read_numeric_block(lines, num_vals, dtype=dtype, filepath=path, initial_tokens=tokens[3:])
                    current_data_dict[name] = arr.reshape((current_n, 3, 3))

        return VtkState(
            path=path,
            title=title,
            time=time_val,
            points=points,
            cell_types=cell_types,
            cell_connectivity=cell_connectivity,
            cell_offsets=cell_offsets,
            point_data=point_data,
            cell_data=cell_data,
        )
