import re
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
    cells: list[np.ndarray]  # per cell, 0-based point indices (ragged)
    point_data: dict[str, np.ndarray] = field(default_factory=dict)
    cell_data: dict[str, np.ndarray] = field(default_factory=dict)


def _read_tokens(
    lines: Iterator[str], count: int, initial_tokens: list[str] | None = None, filepath: str | None = None
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
    lines: Iterator[str], count: int, dtype, filepath: str | None = None, initial_tokens: list[str] | None = None
) -> np.ndarray:
    """Read exactly `count` tokens directly into a numpy array."""
    collected_lines = []
    num_tokens = 0
    if initial_tokens:
        collected_lines.append(" ".join(initial_tokens))
        num_tokens += len(initial_tokens)
        
    while num_tokens < count:
        try:
            line = next(lines)
            collected_lines.append(line)
            num_tokens += len(line.split())
        except StopIteration:
            break
            
    if num_tokens < count:
        file_ctx = f" in file '{filepath}'" if filepath else ""
        raise ValueError(f"Expected {count} tokens, got {num_tokens}{file_ctx}")
        
    joined = " ".join(collected_lines)
    arr = np.fromstring(joined, dtype=dtype, sep=' ')
    return arr[:count]


def read_vtk(path: str | Path) -> VtkState:
    path = str(path)
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
        cells = []
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

                # Each cell: npts idx0 idx1 ...
                idx = 0
                for _ in range(n_cells):
                    npts = arr[idx]
                    cell_pts = arr[idx + 1 : idx + 1 + npts]
                    cells.append(cell_pts)
                    idx += 1 + npts
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
                arr = _read_numeric_block(lines, num_vals, dtype=dtype, filepath=path, initial_tokens=initial_tokens)

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
            cells=cells,
            point_data=point_data,
            cell_data=cell_data,
        )
