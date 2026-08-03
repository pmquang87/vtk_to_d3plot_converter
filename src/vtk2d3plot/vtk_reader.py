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


def _read_numeric_block(
    lines: Iterator[str],
    count: int,
    dtype,
    filepath: str | None = None,
    initial_tokens: list[str] | None = None,
) -> np.ndarray:
    """Read exactly `count` tokens and convert to a numpy array.

    Uses str.split() + np.array() for speed (benchmarked faster than
    np.fromstring(sep=...) which is NumPy's deprecated slow text parser).
    """
    tokens = list(initial_tokens) if initial_tokens else []
    while len(tokens) < count:
        try:
            line = next(lines)
            tokens.extend(line.split())
        except StopIteration:
            break
    if len(tokens) < count:
        file_ctx = f" in file '{filepath}'" if filepath else ""
        raise ValueError(f"Expected {count} tokens, got {len(tokens)}{file_ctx}")
    try:
        arr = np.array(tokens[:count], dtype=dtype)
    except (ValueError, OverflowError) as exc:
        # Find the first offending token for diagnostic context
        for j, tok in enumerate(tokens[:count]):
            try:
                dtype(tok)
            except (ValueError, OverflowError):
                file_ctx = f" in file '{filepath}'" if filepath else ""
                raise ValueError(
                    f"Malformed numeric value {tok!r} at position {j}{file_ctx}"
                ) from exc
        raise  # shouldn't reach here but be safe
    return arr


def read_vtu(path: str | Path) -> VtkState:
    path = str(path)
    tree = ET.parse(path)
    root = tree.getroot()

    if root.tag != "VTKFile" or root.attrib.get("type") != "UnstructuredGrid":
        raise ValueError("Expected VTKFile type='UnstructuredGrid'")

    grid = root.find("UnstructuredGrid")
    if grid is None:
        raise ValueError("Missing UnstructuredGrid element")

    piece = grid.find("Piece")
    if piece is None:
        raise ValueError("Missing Piece element")

    n_points = int(piece.attrib.get("NumberOfPoints", 0))
    n_cells = int(piece.attrib.get("NumberOfCells", 0))

    appended_data_elem = root.find("AppendedData")
    appended_data = b""
    if appended_data_elem is not None:
        appended_encoding = appended_data_elem.attrib.get("encoding", "base64")
        text = appended_data_elem.text
        if text and text.startswith("_"):
            text = text[1:]
        if appended_encoding == "base64" and text:
            appended_data = base64.b64decode(text.strip())
        elif appended_encoding == "raw" and text:
            # Note: raw binary inside XML is brittle if read as string,
            # but usually it's read by binary parser. We do our best.
            appended_data = text.encode("latin1")

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
            header_type = root.attrib.get("header_type", "UInt32")
            if header_type == "UInt64":
                header_size = 8
                data_size = struct.unpack("<Q", raw[:8])[0]
            else:
                header_size = 4
                data_size = struct.unpack("<I", raw[:4])[0]
            arr = np.frombuffer(
                raw, dtype=dt, count=data_size // np.dtype(dt).itemsize, offset=header_size
            )
        elif fmt == "appended":
            offset = int(elem.attrib.get("offset", 0))
            header_type = root.attrib.get("header_type", "UInt32")
            if header_type == "UInt64":
                header_size = 8
                data_size = struct.unpack("<Q", appended_data[offset : offset + 8])[0]
            else:
                header_size = 4
                data_size = struct.unpack("<I", appended_data[offset : offset + 4])[0]
            arr = np.frombuffer(
                appended_data,
                dtype=dt,
                count=data_size // np.dtype(dt).itemsize,
                offset=offset + header_size,
            )
        else:
            raise ValueError(f"Unknown format {fmt}")

        if ncomp > 1:
            arr = arr.reshape((-1, ncomp))
        return arr

    points = np.zeros((0, 3), dtype=np.float64)
    if n_points > 0:
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
    pd_elem = piece.find("PointData")
    time_val = None
    if pd_elem is not None:
        for da in pd_elem.findall("DataArray"):
            name = da.attrib.get("Name", "")
            if name:
                arr = parse_data_array(da)
                point_data[name] = arr
                if name.upper() == "TIME" and arr.size > 0:
                    time_val = float(arr.flat[0])

    cell_data = {}
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


def read_vtk(path: str | Path) -> VtkState:
    path = str(path)

    if str(path).lower().endswith(".vtu"):
        return read_vtu(path)

    with open(path, "rb") as f:
        header = f.read(200).decode("utf-8-sig", errors="ignore").strip()
        if header.startswith(("<?xml", "<VTKFile")):
            return read_vtu(path)

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
                    vals = _read_tokens(lines, num_vals, filepath=path)
                    if name.upper() == "TIME" and num_vals > 0:
                        time_val = float(vals[0])
            elif cmd == "POINTS":
                n_points = int(tokens[1])
                arr = _read_numeric_block(lines, n_points * 3, dtype=np.float64, filepath=path)
                points = arr.reshape((n_points, 3))
            elif cmd == "CELLS":
                n_cells = int(tokens[1])
                size = int(tokens[2])
                arr = _read_numeric_block(lines, size, dtype=np.int32, filepath=path)

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
                cell_types = _read_numeric_block(lines, n_cells, dtype=np.int32, filepath=path)
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
                ncomp = int(tokens[3]) if len(tokens) > 3 else 1
                dtype = np.int32 if "int" in val_type else np.float64

                try:
                    peek_line = next(lines)
                except StopIteration:
                    peek_line = ""

                initial_tokens = []
                if peek_line.upper().startswith("LOOKUP_TABLE"):
                    pass  # consumed
                else:
                    initial_tokens = peek_line.split()

                num_vals = current_n * ncomp
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
                arr = _read_numeric_block(lines, current_n * 3, dtype=dtype, filepath=path)
                current_data_dict[name] = arr.reshape((current_n, 3))
            elif cmd == "TENSORS":
                if current_data_dict is None:
                    continue
                name = tokens[1]
                val_type = tokens[2].lower()
                dtype = np.int32 if "int" in val_type else np.float64
                arr = _read_numeric_block(lines, current_n * 9, dtype=dtype, filepath=path)
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
