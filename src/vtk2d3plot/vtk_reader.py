import base64
import lzma
import re
import xml.etree.ElementTree as ET
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class VtkState:
    path: str
    title: str  # raw line 2
    # legacy: FIELD TIME, else title "t=<float>"; VTU: FieldData TimeValue/TIME, else PointData TIME
    time: float | None
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
                    raise ValueError(
                        f"Malformed numeric value {tok!r} at position {idx + j}{file_ctx}"
                    ) from exc
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
                    raise ValueError(
                        f"Malformed numeric value {tok!r} at position {idx + j}{file_ctx}"
                    ) from exc
            raise
        idx += n

    if idx < count:
        file_ctx = f" in file '{filepath}'" if filepath else ""
        raise ValueError(f"Expected {count} tokens, got {idx}{file_ctx}")
    return arr


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Point/cell arrays that are still READ (not skipped) in metadata_only mode:
# the time, and the identity arrays the lazy loader compares between states
# (bug report 2026-10-11 #33, c34).
_METADATA_KEEP = frozenset({"TIME", "NODE_ID", "ELEMENT_ID", "PART_ID"})


def _keep_in_metadata(name: str) -> bool:
    """True if array *name* is read even when ``metadata_only=True``."""
    return name.upper() in _METADATA_KEEP


# Title time: "t=" must start a word (so "count=5" / "Result=x" do not match)
# and be followed by a well-formed float that is not glued to further word
# characters ("t=end", "t=1e" give no time). A trailing sentence period is
# allowed ("... at t=1.5.").
_TITLE_TIME_RE = re.compile(r"\bt=([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(?!\w|\.\w)")


def _title_time(title: str) -> float | None:
    m = _TITLE_TIME_RE.search(title)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# Legacy VTK data type names -> numpy dtype. Unknown names raise. The keys are
# every name vtkDataWriter writes (lower-cased); unsigned 64-bit types share
# int64 with unsigned_long (a value above 2**63-1 raises "Malformed numeric value").
_LEGACY_DTYPES = {
    "bit": np.int32,
    "char": np.int32,
    "signed_char": np.int32,
    "unsigned_char": np.int32,
    "short": np.int32,
    "unsigned_short": np.int32,
    "int": np.int32,
    "unsigned_int": np.int64,
    "long": np.int64,
    "unsigned_long": np.int64,
    "vtktypeint64": np.int64,
    "vtktypeuint64": np.int64,
    "vtkidtype": np.int64,
    "float": np.float64,
    "double": np.float64,
}
# Legacy string array types (FIELD arrays only); read as words, never as numbers.
_LEGACY_STRING_TYPES = frozenset({"string", "utf8_string"})


def _as_tensor_if_9(arr: np.ndarray) -> np.ndarray:
    """A 9-component (n, 9) array is a row-major 3x3 tensor: return it as (n, 3, 3).

    VTU and legacy FIELD arrays carry tensors as 9 flat components; the
    VtkState contract (and the legacy TENSORS path) stores them as (n, 3, 3).
    """
    if arr.ndim == 2 and arr.shape[1] == 9:
        return arr.reshape((arr.shape[0], 3, 3))
    return arr


def _legacy_dtype(val_type: str, path: str):
    try:
        return _LEGACY_DTYPES[val_type.lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported legacy VTK data type {val_type!r} in file '{path}'"
        ) from None


# VTK XML DataArray type names -> numpy dtype. Unknown names raise.
_VTU_DTYPES = {
    "Int8": np.int8,
    "UInt8": np.uint8,
    "Int16": np.int16,
    "UInt16": np.uint16,
    "Int32": np.int32,
    "UInt32": np.uint32,
    "Int64": np.int64,
    "UInt64": np.uint64,
    "Float32": np.float32,
    "Float64": np.float64,
}
_VTU_HEADER_TYPES = {"UInt32": np.uint32, "UInt64": np.uint64}
_VTU_TIME_FIELD_NAMES = ("TIMEVALUE", "TIME")


def _lz4_decompress(block: bytes, usize: int) -> bytes:
    try:
        import lz4.block
    except ImportError:
        raise ValueError(
            "VTU file uses vtkLZ4DataCompressor, which needs the 'lz4' Python package (not installed)"
        ) from None
    return lz4.block.decompress(block, uncompressed_size=usize)


_VTU_DECOMPRESSORS = {
    "vtkZLibDataCompressor": lambda block, usize: zlib.decompress(block),
    "vtkLZMADataCompressor": lambda block, usize: lzma.decompress(block),
    "vtkLZ4DataCompressor": _lz4_decompress,
}


def _b64decode_concat(text: bytes) -> bytes:
    """Decode base64 text that may be several independently padded blocks.

    VTK encodes a compressed array's header and its data as two base64
    blocks (each with its own '=' padding); a plain b64decode stops at the
    first padding. Splitting after every '=' run decodes both layouts.
    """
    text = bytes(text).translate(None, b" \t\r\n")
    body = text.rstrip(b"=")
    if b"=" not in body:
        return base64.b64decode(text)
    return b"".join(base64.b64decode(part) for part in re.split(rb"(?<==)(?=[^=])", text))


class _VtuDecoder:
    """Decode DataArrays of one VTU file (ascii, inline binary, appended)."""

    def __init__(
        self,
        path: str,
        root_attrib: dict,
        mm,
        data_start: int,
        data_end: int,
        encoding: str | None,
        offsets,
    ):
        self.path = path
        self.mm = mm
        self.data_start = data_start  # first byte after '_' (-1 if none)
        self.data_end = data_end
        self.encoding = encoding
        self.offsets = sorted(set(offsets))  # all appended offsets, to bound base64 blocks
        bo = root_attrib.get("byte_order", "LittleEndian")
        if bo not in ("LittleEndian", "BigEndian"):
            raise ValueError(f"Unsupported VTU byte_order {bo!r} in file '{path}'")
        self.bo = "<" if bo == "LittleEndian" else ">"
        ht = root_attrib.get("header_type", "UInt32")
        if ht not in _VTU_HEADER_TYPES:
            raise ValueError(f"Unsupported VTU header_type {ht!r} in file '{path}'")
        self.header_dt = np.dtype(_VTU_HEADER_TYPES[ht]).newbyteorder(self.bo)
        self.compressor = root_attrib.get("compressor", "") or None

    def _decompress_fn(self):
        try:
            return _VTU_DECOMPRESSORS[self.compressor]
        except KeyError:
            raise ValueError(
                f"Unsupported VTU compressor {self.compressor!r} in file '{self.path}' "
                f"(supported: {', '.join(_VTU_DECOMPRESSORS)})"
            ) from None

    def _payload(self, buf, pos: int, name: str) -> bytes:
        """Return the uncompressed bytes of the array whose header starts at buf[pos]."""
        hs = self.header_dt.itemsize
        if pos + hs > len(buf):
            raise ValueError(f"Truncated binary data for DataArray '{name}' in file '{self.path}'")
        if self.compressor is None:
            nbytes = int(np.frombuffer(buf, self.header_dt, count=1, offset=pos)[0])
            data = bytes(buf[pos + hs : pos + hs + nbytes])
            if len(data) != nbytes:
                raise ValueError(
                    f"Truncated binary data for DataArray '{name}' in file '{self.path}'"
                )
            return data
        decompress = self._decompress_fn()
        nblocks = int(np.frombuffer(buf, self.header_dt, count=1, offset=pos)[0])
        if pos + hs * (3 + nblocks) > len(buf):
            raise ValueError(
                f"Truncated compression header for DataArray '{name}' in file '{self.path}'"
            )
        hdr = np.frombuffer(buf, self.header_dt, count=3 + nblocks, offset=pos).astype(np.int64)
        block_size, last_size = int(hdr[1]), int(hdr[2])
        start = pos + hs * (3 + nblocks)
        parts = []
        for k, csize in enumerate(hdr[3:].tolist()):
            usize = last_size if (k == nblocks - 1 and last_size) else block_size
            chunk = bytes(buf[start : start + csize])
            if len(chunk) != csize:
                raise ValueError(
                    f"Truncated compressed data for DataArray '{name}' in file '{self.path}'"
                )
            block = decompress(chunk, usize)
            if len(block) != usize:
                raise ValueError(
                    f"Decompressed block {k} of DataArray '{name}' has {len(block)} bytes, "
                    f"header says {usize}, in file '{self.path}'"
                )
            parts.append(block)
            start += csize
        return b"".join(parts)

    def decode(self, elem) -> np.ndarray:
        name = elem.attrib.get("Name", "")
        type_str = elem.attrib.get("type", "Float64")
        if type_str not in _VTU_DTYPES:
            raise ValueError(
                f"Unsupported VTU DataArray type {type_str!r} (array '{name}') in file '{self.path}'"
            )
        dt = np.dtype(_VTU_DTYPES[type_str])
        ncomp = int(elem.attrib.get("NumberOfComponents", 1))
        fmt = elem.attrib.get("format", "ascii")

        if fmt == "ascii":
            arr = np.fromstring(elem.text or "", sep=" ", dtype=dt)
        else:
            if fmt == "binary":
                data = self._payload(_b64decode_concat((elem.text or "").encode("ascii")), 0, name)
            elif fmt == "appended":
                if self.data_start < 0:
                    raise ValueError(
                        f"DataArray '{name}' is appended but file '{self.path}' has no AppendedData"
                    )
                offset = int(elem.attrib.get("offset", 0))
                if self.encoding == "raw":
                    data = self._payload(self.mm, self.data_start + offset, name)
                elif self.encoding == "base64":
                    # Each array is its own base64 block; offsets count encoded
                    # characters. The block ends where the next array starts.
                    later = [o for o in self.offsets if o > offset]
                    end = self.data_start + later[0] if later else self.data_end
                    text = self.mm[self.data_start + offset : end]
                    data = self._payload(_b64decode_concat(text), 0, name)
                else:
                    raise ValueError(
                        f"Unknown appended encoding {self.encoding} in file '{self.path}'"
                    )
            else:
                raise ValueError(f"Unknown format {fmt}")
            if len(data) % dt.itemsize:
                raise ValueError(
                    f"DataArray '{name}' has {len(data)} bytes, not a multiple of {type_str} "
                    f"({dt.itemsize} bytes), in file '{self.path}'"
                )
            arr = np.frombuffer(data, dtype=dt.newbyteorder(self.bo)).astype(dt)

        if ncomp > 1:
            arr = arr.reshape((-1, ncomp))
        return arr


def _vtu_read_xml(path: str, mm):
    """Parse the XML part of a VTU. Returns (root, data_start, data_end, encoding)."""
    idx = mm.find(b"<AppendedData")
    try:
        if idx == -1:
            # Parse from the file (streamed) instead of copying the whole map.
            return ET.parse(path).getroot(), -1, -1, None
        tag_end = mm.find(b">", idx)
        if tag_end == -1:
            raise ValueError(f"Malformed AppendedData tag in file '{path}'")
        m = re.search(rb'encoding\s*=\s*"([^"]*)"', mm[idx:tag_end])
        encoding = m.group(1).decode("ascii", "replace") if m else "base64"
        under = mm.find(b"_", tag_end)
        if under == -1:
            raise ValueError(f"AppendedData without '_' marker in file '{path}'")
        data_start = under + 1
        if encoding == "base64":
            end = mm.find(b"</AppendedData>", data_start)
            data_end = end if end != -1 else len(mm)
        else:
            data_end = len(mm)
        root = ET.fromstring(mm[:idx] + b"</VTKFile>")
        return root, data_start, data_end, encoding
    except ET.ParseError as exc:
        raise ValueError(f"Malformed VTU XML in file '{path}': {exc}") from exc


def _concat_piece_data(
    per_piece: list[dict[str, np.ndarray]], what: str, path: str
) -> dict[str, np.ndarray]:
    if len(per_piece) == 1:
        return per_piece[0]
    names = list(per_piece[0])
    for k, d in enumerate(per_piece[1:], start=1):
        if set(d) != set(names):
            diff = sorted(set(d) ^ set(names))
            raise ValueError(
                f"Pieces of file '{path}' carry different {what} arrays (Piece 0 vs Piece {k}: {', '.join(diff)})"
            )
    return {n: np.concatenate([d[n] for d in per_piece]) for n in names}


def read_vtu(path: str | Path, metadata_only: bool = False) -> VtkState:
    import mmap

    path = str(path)
    with open(path, "rb") as f:
        f.seek(0, 2)
        if f.tell() == 0:
            raise ValueError(f"Empty file '{path}'")
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            root, data_start, data_end, encoding = _vtu_read_xml(path, mm)
            offsets = [
                int(da.attrib.get("offset", 0))
                for da in root.iter("DataArray")
                if da.attrib.get("format") == "appended"
            ]
            dec = _VtuDecoder(path, root.attrib, mm, data_start, data_end, encoding, offsets)
            return _read_vtu_tree(path, root, dec, metadata_only)


def _read_vtu_tree(path: str, root, dec: _VtuDecoder, metadata_only: bool) -> VtkState:
    pieces = list(root.iter("Piece"))
    if not pieces:
        raise ValueError("Missing Piece element")

    # Time: dataset FieldData TimeValue / TIME (VTK / ParaView convention) first.
    time_val = None
    for fd in root.iter("FieldData"):
        for da in fd.findall("DataArray"):
            if time_val is None and da.attrib.get("Name", "").upper() in _VTU_TIME_FIELD_NAMES:
                arr = dec.decode(da)
                if arr.size > 0:
                    time_val = float(arr.flat[0])

    pts_list, types_list, conn_list, offs_list = [], [], [], []
    pd_list: list[dict[str, np.ndarray]] = []
    cd_list: list[dict[str, np.ndarray]] = []
    n_points_total = 0
    n_conn_total = 0
    for piece in pieces:
        n_points = int(piece.attrib.get("NumberOfPoints", 0))
        n_cells = int(piece.attrib.get("NumberOfCells", 0))

        if n_points > 0 and not metadata_only:
            da = piece.find("Points/DataArray")
            if da is None:
                raise ValueError(
                    f"Piece with {n_points} points has no Points DataArray in file '{path}'"
                )
            pts_list.append(dec.decode(da).reshape((-1, 3)))

        if n_cells > 0:
            cells = {da.attrib.get("Name", ""): da for da in piece.findall("Cells/DataArray")}
            if "types" in cells:
                types_list.append(dec.decode(cells["types"]).astype(np.int32))
            if "connectivity" in cells:
                conn = dec.decode(cells["connectivity"]).astype(np.int32)
                conn_list.append(conn + n_points_total if n_points_total else conn)
            if "offsets" in cells:
                offs = dec.decode(cells["offsets"]).astype(np.int32)
                offs_list.append(offs + n_conn_total if n_conn_total else offs)
            if "connectivity" in cells:
                n_conn_total += len(conn_list[-1])

        point_data: dict[str, np.ndarray] = {}
        for da in piece.findall("PointData/DataArray"):
            name = da.attrib.get("Name", "")
            if name and (not metadata_only or _keep_in_metadata(name)):
                point_data[name] = _as_tensor_if_9(dec.decode(da))
        pd_list.append(point_data)

        cell_data: dict[str, np.ndarray] = {}
        for da in piece.findall("CellData/DataArray"):
            name = da.attrib.get("Name", "")
            if name and (not metadata_only or _keep_in_metadata(name)):
                cell_data[name] = _as_tensor_if_9(dec.decode(da))
        cd_list.append(cell_data)

        n_points_total += n_points

    if metadata_only:
        points = np.zeros((n_points_total, 3), dtype=np.float64)
    elif pts_list:
        points = np.concatenate(pts_list) if len(pts_list) > 1 else pts_list[0]
    else:
        points = np.zeros((0, 3), dtype=np.float64)

    cell_types = np.concatenate(types_list) if types_list else np.zeros(0, dtype=np.int32)
    cell_connectivity = np.concatenate(conn_list) if conn_list else np.zeros(0, dtype=np.int32)
    cell_offsets = np.concatenate(([0], *offs_list)) if offs_list else np.zeros(1, dtype=np.int32)

    point_data = _concat_piece_data(pd_list, "PointData", path)
    cell_data = _concat_piece_data(cd_list, "CellData", path)

    if time_val is None:
        for name, arr in point_data.items():
            if name.upper() == "TIME" and arr.size > 0:
                time_val = float(arr.flat[0])

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

        # Line 2: Title. Read straight from the file: it may be empty, and the
        # blank-skipping iterator would then take line 3 ('ASCII') as title.
        raw_title = next(f, None)
        if raw_title is None:
            raise ValueError(f"Truncated VTK file '{path}': missing title line")
        title = raw_title.strip()
        # Title time is a fallback only; a FIELD TIME array takes precedence.
        title_time = _title_time(title)
        field_time = None

        # Line 3: ASCII
        line3 = next(lines, None)
        if line3 is None:
            raise ValueError(f"Truncated VTK file '{path}': missing format line after the title")
        if not line3.upper().startswith("ASCII"):
            raise ValueError(f"Expected ASCII format, got: {line3}")

        # Line 4: DATASET UNSTRUCTURED_GRID
        line4 = next(lines, "")
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
                # FIELD FieldData N. At dataset level (before POINT_DATA /
                # CELL_DATA) only TIME is used; inside POINT_DATA / CELL_DATA
                # every array is a point / cell array (meshio writes all its
                # data this way).
                num_fields = int(tokens[2])
                for _ in range(num_fields):
                    header = next(lines).split()
                    name = header[0]
                    ncomp = int(header[1])
                    ntuples = int(header[2])
                    val_type = header[3]

                    num_vals = ncomp * ntuples
                    is_time = name.upper() == "TIME"
                    if current_data_dict is not None and ntuples != current_n and not is_time:
                        raise ValueError(
                            f"FIELD array '{name}' has {ntuples} tuples but the enclosing "
                            f"POINT_DATA / CELL_DATA has {current_n}, in file '{path}'"
                        )
                    if metadata_only and not _keep_in_metadata(name):
                        _skip_tokens(lines, num_vals, initial_tokens=header[4:])
                    elif current_data_dict is None or ntuples != current_n:
                        # dataset-level FIELD (official dialect TIME / CYCLE),
                        # or a 1-value TIME inside POINT_DATA: time only
                        vals = _read_tokens(
                            lines, num_vals, filepath=path, initial_tokens=header[4:]
                        )
                        if name.upper() == "TIME" and num_vals > 0:
                            field_time = float(vals[0])
                    elif val_type.lower() in _LEGACY_STRING_TYPES:
                        # vtkDataWriter writes one encoded word per string value.
                        # A string array has no d3plot field; keep it so the
                        # conversion report lists it as unmapped instead of the
                        # whole file failing (bug report 2026-10-11 #17 follow-up).
                        vals = _read_tokens(
                            lines, num_vals, filepath=path, initial_tokens=header[4:]
                        )
                        sarr = np.array(vals, dtype=str)
                        current_data_dict[name] = (
                            sarr.reshape((ntuples, ncomp)) if ncomp > 1 else sarr
                        )
                    else:
                        arr = _read_numeric_block(
                            lines,
                            num_vals,
                            dtype=_legacy_dtype(val_type, path),
                            filepath=path,
                            initial_tokens=header[4:],
                        )
                        if name.upper() == "TIME" and num_vals > 0:
                            field_time = float(arr[0])
                        current_data_dict[name] = _as_tensor_if_9(
                            arr.reshape((ntuples, ncomp)) if ncomp > 1 else arr
                        )
            elif cmd == "POINTS":
                n_points = int(tokens[1])
                if metadata_only:
                    _skip_tokens(lines, n_points * 3, initial_tokens=tokens[3:])
                    points = np.zeros((n_points, 3), dtype=np.float64)
                else:
                    arr = _read_numeric_block(
                        lines,
                        n_points * 3,
                        dtype=np.float64,
                        filepath=path,
                        initial_tokens=tokens[3:],
                    )
                    points = arr.reshape((n_points, 3))
            elif cmd == "CELLS":
                n_cells = int(tokens[1])
                size = int(tokens[2])
                arr = _read_numeric_block(
                    lines, size, dtype=np.int32, filepath=path, initial_tokens=tokens[3:]
                )

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
                cell_types = _read_numeric_block(
                    lines, n_cells, dtype=np.int32, filepath=path, initial_tokens=tokens[2:]
                )
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
                dtype = _legacy_dtype(tokens[2], path)
                tokens_upper = [t.upper() for t in tokens]
                if "LOOKUP_TABLE" in tokens_upper:
                    idx = tokens_upper.index("LOOKUP_TABLE")
                    ncomp = int(tokens[3]) if idx > 3 else 1
                    initial_tokens = tokens[idx + 2 :]
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
                if metadata_only and not _keep_in_metadata(name):
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
                dtype = _legacy_dtype(tokens[2], path)
                num_vals = current_n * 3
                if metadata_only and not _keep_in_metadata(name):
                    _skip_tokens(lines, num_vals, initial_tokens=tokens[3:])
                else:
                    arr = _read_numeric_block(
                        lines, num_vals, dtype=dtype, filepath=path, initial_tokens=tokens[3:]
                    )
                    current_data_dict[name] = arr.reshape((current_n, 3))
            elif cmd == "TENSORS":
                if current_data_dict is None:
                    continue
                name = tokens[1]
                dtype = _legacy_dtype(tokens[2], path)
                num_vals = current_n * 9
                if metadata_only and not _keep_in_metadata(name):
                    _skip_tokens(lines, num_vals, initial_tokens=tokens[3:])
                else:
                    arr = _read_numeric_block(
                        lines, num_vals, dtype=dtype, filepath=path, initial_tokens=tokens[3:]
                    )
                    current_data_dict[name] = arr.reshape((current_n, 3, 3))

        return VtkState(
            path=path,
            title=title,
            time=field_time if field_time is not None else title_time,
            points=points,
            cell_types=cell_types,
            cell_connectivity=cell_connectivity,
            cell_offsets=cell_offsets,
            point_data=point_data,
            cell_data=cell_data,
        )
