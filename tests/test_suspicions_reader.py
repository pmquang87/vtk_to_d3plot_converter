"""Suspicion round 2026-10-11, cluster A: VTK reader leads.

The leads come from the "Suspicions not checked" list of
docs/reviews/2026-10-11-external-bug-report.md:
- S03: VTU ``byte_order`` (already fixed in milestone 13; pins for the
  layouts the milestone-13 tests do not cover);
- S04: METADATA blocks (VTK >= 8 legacy writers) after arrays;
- S05: truncated / malformed legacy files must raise ValueError naming the file;
- S10: dialect label ("official" needs a FIELD TIME);
- S23: legacy VTK 5.1 (OFFSETS / CONNECTIVITY cell layout).

METADATA grammar, from VTK IO/Legacy vtkDataWriter::WriteArray /
WriteInformation and vtkDataReader::ReadArray / ReadInformation: after the
values of an array the writer may emit ``METADATA``, then optionally
``COMPONENT_NAMES`` and one encoded name per component on its own line
(a missing name is an empty line), then optionally ``INFORMATION <nkeys>``
and per key ``NAME <key> LOCATION <loc>`` plus a ``DATA ...`` line (a string
vector key: ``DATA <n>`` and one line per item), and always a closing blank
line. The reader skips blank lines, peeks for ``METADATA`` and reads the
block up to the blank line.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.suspicions

meshio = pytest.importorskip("meshio")

# ---------------------------------------------------------------------------
# synthetic legacy files
# ---------------------------------------------------------------------------

# Two states of one mixed mesh: hex, quad, triangle, line, tetra.
_PTS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
        [2, 0, 1],
        [2, 1, 1],
    ],
    dtype=np.float64,
)
_CELLS = [
    (12, [0, 1, 2, 3, 4, 5, 6, 7]),
    (9, [5, 8, 9, 6]),
    (5, [4, 5, 7]),
    (3, [8, 9]),
    (10, [0, 1, 3, 4]),
]
_NODE_ID = np.arange(10) + 1001
_ELEMENT_ID = np.array([101, 102, 103, 104, 105])
_PART_ID = np.array([1, 2, 2, 3, 1])

# METADATA blocks exactly as vtkDataWriter writes them (closing blank line included).
_MD_INFO = (
    "METADATA\n"
    "INFORMATION 2\n"
    "NAME L2_NORM_RANGE LOCATION vtkDataArray\n"
    "DATA 2 0 1.73205 \n"
    "NAME L2_NORM_FINITE_RANGE LOCATION vtkDataArray\n"
    "DATA 2 0 1.73205 \n"
    "\n"
)


def _md_names(ncomp: int) -> str:
    """COMPONENT_NAMES (2nd name missing -> empty line, as for a null name) + a string-vector key."""
    names = [f"C{i}%20name" if i != 1 else "" for i in range(ncomp)]
    return (
        "METADATA\nCOMPONENT_NAMES\n"
        + "".join(n + "\n" for n in names)
        + "INFORMATION 1\nNAME SOME_KEYS LOCATION vtkFoo\nDATA 2\nfirst%20item\nsecond\n\n"
    )


def _vals(a) -> str:
    return " ".join(
        repr(float(v)) if isinstance(v, float) else str(v) for v in np.ravel(a).tolist()
    )


def _legacy(
    version: str = "4.2",
    k: int = 0,
    metadata: bool = False,
    title: str = "vtk output",
    field_time: bool = True,
) -> str:
    """One state of the mixed mesh in VTK's legacy layout (4.2 or 5.1 cells)."""
    md3 = _md_names(3) if metadata else ""
    mdi = _MD_INFO if metadata else ""
    t = 0.25 * (k + 1)
    disp = _PTS * 0.01 * (k + 1)
    pts = _PTS + disp
    stress = np.array(
        [[[1.0 + i + k, 4.0, 5.0], [4.0, 2.0 + i, 6.0], [5.0, 6.0, 3.0 + i]] for i in range(5)]
    )
    eps = np.arange(5) * 0.125 + k
    out = [f"# vtk DataFile Version {version}\n{title}\nASCII\nDATASET UNSTRUCTURED_GRID\n"]
    if field_time:
        out.append(f"FIELD FieldData 2\nTIME 1 1 double\n{t!r}\n{mdi}CYCLE 1 1 int\n0\n{mdi}")
    out.append(f"POINTS 10 double\n{_vals(pts)}\n{mdi}")
    conn = [c for _, cell in _CELLS for c in cell]
    if version == "5.1":
        offs = np.cumsum([0] + [len(c) for _, c in _CELLS])
        out.append(f"CELLS {len(offs)} {len(conn)}\nOFFSETS vtktypeint64\n{_vals(offs)}\n{mdi}")
        out.append(f"CONNECTIVITY vtktypeint64\n{_vals(conn)}\n{mdi}")
    else:
        size = len(conn) + len(_CELLS)
        rows = "".join(f"{len(c)} {_vals(c)}\n" for _, c in _CELLS)
        out.append(f"CELLS {len(_CELLS)} {size}\n{rows}")
    out.append(f"CELL_TYPES {len(_CELLS)}\n{_vals([t for t, _ in _CELLS])}\n")
    out.append(
        f"POINT_DATA 10\nSCALARS NODE_ID int 1\nLOOKUP_TABLE default\n{_vals(_NODE_ID)}\n{mdi}"
    )
    out.append(f"VECTORS Displacement double\n{_vals(disp)}\n{md3}")
    out.append(f"FIELD FieldData 1\nVelocity 3 10 double\n{_vals(disp * 2.0)}\n{md3}")
    out.append(
        f"CELL_DATA 5\nSCALARS ELEMENT_ID int 1\nLOOKUP_TABLE default\n{_vals(_ELEMENT_ID)}\n{mdi}"
    )
    out.append(f"SCALARS PART_ID int 1\nLOOKUP_TABLE default\n{_vals(_PART_ID)}\n{mdi}")
    out.append(f"TENSORS 3DELEM_Stress double\n{_vals(stress)}\n{_md_names(9) if metadata else ''}")
    out.append(
        f"FIELD FieldData 2\n3DELEM_Plastic_Strain 1 5 double\n{_vals(eps)}\n{mdi}"
        f"EXTRA 2 5 double\n{_vals(np.arange(10) * 1.5)}\n{_md_names(2) if metadata else ''}"
    )
    return "".join(out)


def _write_series(folder, version="4.2", metadata=False, n=2, **kw):
    folder.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        (folder / f"runA{k:03d}.vtk").write_text(_legacy(version, k, metadata, **kw))
    return folder


def _assert_mixed_state(s, k=0, meta=False):
    assert s.time == pytest.approx(0.25 * (k + 1))
    assert s.cell_types.tolist() == [t for t, _ in _CELLS]
    assert s.cell_connectivity.tolist() == [c for _, cell in _CELLS for c in cell]
    assert s.cell_offsets.tolist() == np.cumsum([0] + [len(c) for _, c in _CELLS]).tolist()
    np.testing.assert_array_equal(s.point_data["NODE_ID"], _NODE_ID)
    np.testing.assert_array_equal(s.cell_data["ELEMENT_ID"], _ELEMENT_ID)
    np.testing.assert_array_equal(s.cell_data["PART_ID"], _PART_ID)
    assert s.points.shape == (10, 3)
    if meta:
        assert set(s.point_data) == {"NODE_ID"}
        assert set(s.cell_data) == {"ELEMENT_ID", "PART_ID"}
        return
    disp = _PTS * 0.01 * (k + 1)
    np.testing.assert_array_equal(s.points, _PTS + disp)
    np.testing.assert_array_equal(s.point_data["Displacement"], disp)
    np.testing.assert_array_equal(s.point_data["Velocity"], disp * 2.0)
    assert s.cell_data["3DELEM_Stress"].shape == (5, 3, 3)
    assert s.cell_data["3DELEM_Stress"][4, 0, 0] == 5.0 + k
    np.testing.assert_array_equal(s.cell_data["3DELEM_Plastic_Strain"], np.arange(5) * 0.125 + k)
    np.testing.assert_array_equal(s.cell_data["EXTRA"], (np.arange(10) * 1.5).reshape(5, 2))
    assert set(s.point_data) == {"NODE_ID", "Displacement", "Velocity"}
    assert set(s.cell_data) == {
        "ELEMENT_ID",
        "PART_ID",
        "3DELEM_Stress",
        "3DELEM_Plastic_Strain",
        "EXTRA",
    }


def _d3_arrays(path):
    from lasso.dyna import D3plot

    return D3plot(str(path)).arrays


def _assert_same_d3plot(a, b):
    """Every lasso array of the two d3plots is identical (same mesh, same order)."""
    da, db = _d3_arrays(a), _d3_arrays(b)
    assert set(da) == set(db)
    for key in da:
        np.testing.assert_array_equal(da[key], db[key], err_msg=str(key))
    return da


# ---------------------------------------------------------------------------
# S03 VTU byte_order (already fixed in milestone 13) - pins for layouts the
# milestone-13 test test_c03_vtu_big_endian_matches_little_endian does not run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fmt", "compressor", "header_type"),
    [
        ("appended-base64", None, "UInt32"),
        ("appended-base64", None, "UInt64"),
        ("binary", None, "UInt64"),
        ("binary", "lzma", "UInt64"),
        ("appended-base64", "zlib", "UInt64"),
    ],
)
@pytest.mark.parametrize("meta", [False, True])
def test_s03_vtu_big_endian_uncovered_layouts(tmp_path, fmt, compressor, header_type, meta):
    """S03: BigEndian headers and data read exactly like the LittleEndian file."""
    from test_bugreport_reader import _mixed_type_piece, _write_vtu

    from vtk2d3plot.vtk_reader import read_vtk

    piece = _mixed_type_piece()
    kw = {
        "fmt": fmt,
        "compressor": compressor,
        "header_type": header_type,
        "block_size": 40,
        "field_data": {"TimeValue": np.array([0.375])},
    }
    le = _write_vtu(tmp_path / "le.vtu", [piece], byte_order="LittleEndian", **kw)
    be = _write_vtu(tmp_path / "be.vtu", [piece], byte_order="BigEndian", **kw)
    assert le.read_bytes() != be.read_bytes().replace(b"BigEndian", b"LittleEndian")
    a, b = read_vtk(le, metadata_only=meta), read_vtk(be, metadata_only=meta)
    assert a.time == b.time == 0.375
    assert b.cell_connectivity.tolist() == list(range(8))
    np.testing.assert_array_equal(b.point_data["NODE_ID"], piece["point_data"]["NODE_ID"])
    np.testing.assert_array_equal(b.cell_data["ELEMENT_ID"], [4242])
    if not meta:
        np.testing.assert_array_equal(b.points, piece["points"])
        for name, ref in piece["point_data"].items():
            assert b.point_data[name].dtype == ref.dtype
            np.testing.assert_array_equal(b.point_data[name], ref)
            np.testing.assert_array_equal(a.point_data[name], ref)


# ---------------------------------------------------------------------------
# S04 METADATA blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("meta", [False, True])
def test_s04_metadata_in_field_blocks(tmp_path, meta):
    """S04: METADATA after FIELD arrays (dataset level and inside POINT_DATA / CELL_DATA)."""
    from vtk2d3plot.vtk_reader import read_vtk

    p = tmp_path / "a.vtk"
    p.write_text(_legacy("4.2", 0, metadata=True))
    _assert_mixed_state(read_vtk(p, metadata_only=meta), 0, meta)


@pytest.mark.parametrize("meta", [False, True])
def test_s04_metadata_after_every_array_kind(tmp_path, meta):
    """S04: METADATA (component names and information keys) after POINTS, SCALARS,
    VECTORS and TENSORS; component names that look like keywords are not parsed."""
    from vtk2d3plot.vtk_reader import read_vtk

    text = _legacy("4.2", 1, metadata=False, field_time=True)
    # component names that are legacy keywords / array headers: count-based skip only
    md_kw = "METADATA\nCOMPONENT_NAMES\nCELL_DATA\nFIELD\nVelocity 3 10 double\n\n"
    for before, md in (
        ("CELLS 5 ", _MD_INFO),  # after POINTS
        ("VECTORS Displacement", _MD_INFO),  # after SCALARS NODE_ID
        ("FIELD FieldData 1\nVelocity", md_kw),  # after VECTORS
        ("SCALARS PART_ID", _md_names(1)),  # after SCALARS ELEMENT_ID
        ("FIELD FieldData 2\n3DELEM", _md_names(9)),  # after TENSORS
    ):
        assert text.count(before) == 1
        text = text.replace(before, md + before)
    p = tmp_path / "a.vtk"
    p.write_text(text)
    _assert_mixed_state(read_vtk(p, metadata_only=meta), 1, meta)


def test_s04_metadata_without_blank_line_does_not_swallow_next_block(tmp_path):
    """S04 guard: a METADATA block missing its closing blank line (outside a FIELD block)
    ends at the next section keyword instead of swallowing that section."""
    from vtk2d3plot.vtk_reader import read_vtk

    text = _legacy("4.2", 0, metadata=False)
    md = _MD_INFO.rstrip("\n") + "\n"
    for before in ("CELLS 5 ", "VECTORS Displacement", "SCALARS PART_ID", "CELL_DATA 5"):
        assert text.count(before) == 1
        text = text.replace(before, md + before)
    p = tmp_path / "a.vtk"
    p.write_text(text)
    _assert_mixed_state(read_vtk(p), 0)


def test_s04_metadata_converts_like_plain_file(tmp_path):
    """S04: a series with METADATA blocks converts to the same d3plot as without them."""
    from vtk2d3plot.api import convert

    plain = _write_series(tmp_path / "plain" / "run")
    withmd = _write_series(tmp_path / "md" / "run", metadata=True)
    ra = convert(plain, tmp_path / "a.d3plot", title="t")
    rb = convert(withmd, tmp_path / "b.d3plot", title="t")
    assert ra.n_states == rb.n_states == 2
    assert ra.warnings == rb.warnings
    _assert_same_d3plot(tmp_path / "a.d3plot", tmp_path / "b.d3plot")


# ---------------------------------------------------------------------------
# S05 truncated / malformed legacy files
# ---------------------------------------------------------------------------

_OFFICIAL = """# vtk DataFile Version 3.0
vtk output
ASCII
DATASET UNSTRUCTURED_GRID
FIELD FieldData 2
TIME 1 1 double
0.25
CYCLE 1 1 int
0

POINTS 8 float
0 0 0 1 0 0 1 1 0
0 1 0 0 0 1 1 0 1
1 1 1 0 1 1

CELLS 2 14
8 0 1 2 3 4 5 6 7
4 0 1 2 3

CELL_TYPES 2
12
9

POINT_DATA 8
SCALARS NODE_ID int 1
LOOKUP_TABLE default
11 12 13 14 15 16 17 18

VECTORS Displacement float
0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0

CELL_DATA 2
SCALARS ELEMENT_ID int 1
LOOKUP_TABLE default
1 2

SCALARS PART_ID int 1
LOOKUP_TABLE default
1 2

TENSORS 3DELEM_Stress float
1 2 3 2 4 5 3 5 6
0 0 0 0 0 0 0 0 0

FIELD FieldData 1
EXTRA 2 2 double
1 2 3 4
"""

_PORT = """# vtk DataFile Version 3.0
pyradioss state t=1.000000000E-03
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 4 double
0 0 0 1 0 0 1 1 0 0 1 0
CELLS 1 5
4 0 1 2 3
CELL_TYPES 1
9
POINT_DATA 4
VECTORS DISPLACEMENT double
0 0 0 0 0 0 0 0 0 0 0 0
VECTORS VELOCITY double
0 0 0 0 0 0 0 0 0 0 0 0
CELL_DATA 1
SCALARS VONM double 1
LOOKUP_TABLE default
5.5
SCALARS EPSP double 1
LOOKUP_TABLE default
0.1
SCALARS OFF double 1
LOOKUP_TABLE default
1.0
"""


def _read_or_valueerror(fn, path):
    """Run fn(): success must give a self-consistent state, failure a ValueError naming path."""
    try:
        s = fn()
    except ValueError as exc:
        assert str(path) in str(exc), f"ValueError does not name the file: {exc}"
        return None
    if s is not None:
        assert len(s.cell_offsets) == len(s.cell_types) + 1
        assert s.cell_offsets[-1] == len(s.cell_connectivity)
        for d, n in ((s.point_data, len(s.points)), (s.cell_data, len(s.cell_types))):
            for name, arr in d.items():
                assert arr.shape[0] == n, (name, arr.shape, n)
    return s


def _line_cuts(data: bytes, keep_line=lambda line: True) -> list[int]:
    """Cut positions: the end of, inside, and just after the first word of each kept line."""
    cuts, start = {0, len(data)}, 0
    for line in data.split(b"\n"):
        end = start + len(line) + 1
        if keep_line(line):
            cuts.update({start + 1, start + len(line) // 2, min(end, len(data))})
            if b" " in line:
                cuts.add(start + line.index(b" ") + 1)
        start = end
    return sorted(c for c in cuts if c <= len(data))


@pytest.mark.parametrize("which", ["official", "port", "official51"])
@pytest.mark.parametrize("meta", [False, True])
def test_s05_truncated_anywhere(tmp_path, which, meta):
    """S05: a file cut at any byte (official / port) or inside / after any line (5.1 with
    METADATA) reads (a legitimately shorter file) or raises ValueError naming the file -
    never StopIteration / IndexError / KeyError."""
    from vtk2d3plot.vtk_reader import read_vtk

    text = {"official": _OFFICIAL, "port": _PORT, "official51": _legacy("5.1", 0, True)}[which]
    data = text.encode()
    cuts = _line_cuts(data) if which == "official51" else range(len(data) + 1)
    p = tmp_path / "runA001.vtk"
    n_ok = 0
    for cut in cuts:
        p.write_bytes(data[:cut])
        if _read_or_valueerror(lambda meta=meta: read_vtk(p, metadata_only=meta), p) is not None:
            n_ok += 1
    assert n_ok > 0  # the complete file (and a few shorter ones) read


def test_s05_truncated_real_port_file(tmp_path):
    """S05: the real port fixture cut inside / after every section line and every 25th line."""
    from conftest import PORT_DIR

    from vtk2d3plot.vtk_reader import read_vtk

    data = (PORT_DIR / "BOXIMPA000.vtk").read_bytes()
    count = iter(range(10**9))
    cuts = _line_cuts(data, lambda line: line[:1].isalpha() or next(count) % 25 == 0)
    assert len(cuts) > 50
    p = tmp_path / "BOXIMPA000.vtk"
    for cut in cuts:
        p.write_bytes(data[:cut])
        for meta in (False, True):
            _read_or_valueerror(lambda meta=meta: read_vtk(p, metadata_only=meta), p)


@pytest.mark.parametrize(
    ("fmt", "compressor"),
    [
        ("appended-base64", None),
        ("appended-base64", "zlib"),
        ("binary", "zlib"),
        ("appended-raw", None),
    ],
)
def test_s05_truncated_vtu_anywhere(tmp_path, fmt, compressor):
    """S05 (VTU side): a VTU cut at any byte raises ValueError naming the file (a bare
    binascii.Error from a cut base64 block named no file) or reads consistently."""
    from test_bugreport_reader import _hex_piece, _write_vtu

    from vtk2d3plot.vtk_reader import read_vtk

    src = _write_vtu(
        tmp_path / "src.vtu",
        [_hex_piece()],
        fmt=fmt,
        compressor=compressor,
        field_data={"TimeValue": np.array([0.5])},
    )
    data = src.read_bytes()
    p = tmp_path / "cut.vtu"
    # every 7th byte (all residues mod 4 of a base64 block) and every byte of the tail
    for cut in sorted(set(range(0, len(data), 7)) | set(range(len(data) - 120, len(data)))):
        p.write_bytes(data[:cut])
        for meta in (False, True):
            _read_or_valueerror(lambda meta=meta: read_vtk(p, metadata_only=meta), p)


def test_s05_lazy_iter_states_truncated_file(tmp_path):
    """S05: a file truncated after load_series_lazy gives ValueError from iter_states,
    not RuntimeError('generator raised StopIteration') (PEP 479)."""
    from vtk2d3plot.series import load_series_lazy

    (tmp_path / "runA000.vtk").write_text(_OFFICIAL)
    full = _OFFICIAL.replace("0.25", "0.5").encode()
    p = tmp_path / "runA001.vtk"
    for cut in _line_cuts(full)[:-1]:
        p.write_bytes(full)
        ser = load_series_lazy(tmp_path)
        p.write_bytes(full[:cut])
        try:
            list(ser.iter_states())
        except ValueError as exc:
            assert str(p) in str(exc)


def test_s05_truncated_series_through_convert(tmp_path):
    """S05: convert() over a series whose last file stops after a FIELD header."""
    from vtk2d3plot.api import convert

    (tmp_path / "runA000.vtk").write_text(_OFFICIAL)
    cut = _OFFICIAL.index("TIME 1 1 double")
    p = tmp_path / "runA001.vtk"
    p.write_text(_OFFICIAL[:cut])
    with pytest.raises(ValueError, match="runA001.vtk"):
        convert(tmp_path, tmp_path / "o.d3plot")


@pytest.mark.parametrize(
    "bad",
    [
        "POINTS\n",
        "POINTS x float\n",
        "POINTS -1 float\n",
        "CELLS 2\n",
        "CELLS 1 3\n8 0 1\n",
        "CELLS 2 3\n2 0 1\nCELL_TYPES 2\n3 3\n",
        "CELL_TYPES\n",
        "FIELD FieldData\n",
        "FIELD FieldData 1\n",
        "FIELD FieldData 1\nTIME 1\n",
        "POINT_DATA\n",
        "CELL_DATA abc\n",
        "POINT_DATA 2\nSCALARS X\n",
        "POINT_DATA 2\nVECTORS V\n",
        "POINT_DATA 2\nTENSORS T\n",
        "POINT_DATA 2\nSCALARS X double x\n1 2\n",
        "CELLS 1 4\n3 0 1 2\nCELL_TYPES 2\n5 5\n",
        "CELLS 1 4\n3 0 1 2\n",
    ],
)
@pytest.mark.parametrize("meta", [False, True])
def test_s05_malformed_header_lines(tmp_path, bad, meta):
    """S05: a malformed section header (missing / non-integer / negative count, cell
    rows that overrun the CELLS size, CELLS and CELL_TYPES counts that disagree)
    raises ValueError naming the file."""
    from vtk2d3plot.vtk_reader import read_vtk

    p = tmp_path / "a.vtk"
    head = "# vtk DataFile Version 3.0\nt\nASCII\nDATASET UNSTRUCTURED_GRID\n"
    pts = "POINTS 3 double\n0 0 0 1 0 0 0 1 0\n" if not bad.startswith("POINTS") else ""
    p.write_text(head + pts + bad)
    with pytest.raises(ValueError, match="a.vtk"):
        read_vtk(p, metadata_only=meta)


# ---------------------------------------------------------------------------
# S10 dialect label
# ---------------------------------------------------------------------------


def test_s10_title_time_without_pyradioss_is_unknown(tmp_path):
    """S10: a time taken from a non-pyradioss title 't=...' is not the official dialect
    (plan.md design decision 7: official needs a FIELD TIME)."""
    from vtk2d3plot.api import convert
    from vtk2d3plot.series import load_series, load_series_lazy

    for k in range(2):
        (tmp_path / f"runA{k:03d}.vtk").write_text(
            _legacy("4.2", k, title=f"my solver t={0.25 * (k + 1)}", field_time=False)
        )
    assert load_series(tmp_path).dialect == "unknown"
    assert load_series_lazy(tmp_path).dialect == "unknown"
    assert convert(tmp_path, tmp_path / "o.d3plot").dialect == "unknown"


@pytest.mark.parametrize(
    ("title", "field_time", "want"),
    [
        ("vtk output", True, "official"),
        ("pyradioss state t=2.5E-01", True, "port"),
        ("pyradioss state t=2.5E-01", False, "port"),
    ],
)
def test_s10_legacy_dialects(tmp_path, title, field_time, want):
    """S10: FIELD TIME -> official; a pyradioss title wins (patched port files stay port)."""
    from vtk2d3plot.series import load_series, load_series_lazy

    (tmp_path / "runA000.vtk").write_text(_legacy("4.2", 0, title=title, field_time=field_time))
    assert load_series(tmp_path).dialect == want
    assert load_series_lazy(tmp_path).dialect == want


def test_s10_vtu_dialects(tmp_path):
    """S10: a VTU FieldData TimeValue / TIME counts as a field time (official); a TIME
    PointData array or a PVD-only timestep does not (unknown)."""
    from test_bugreport_reader import _quad_piece, _write_vtu

    from vtk2d3plot.series import load_series, load_series_lazy

    fd = _write_vtu(
        tmp_path / "fdA000.vtu", [_quad_piece()], field_data={"TimeValue": np.array([0.5])}
    )
    piece = _quad_piece()
    piece["point_data"]["TIME"] = np.full(4, 0.5)
    pd = _write_vtu(tmp_path / "pdA000.vtu", [piece])
    none = _write_vtu(tmp_path / "noA000.vtu", [_quad_piece()])
    pvd = tmp_path / "s.pvd"
    pvd.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n'
        f'<DataSet timestep="0.5" part="0" file="{none.name}"/>\n</Collection>\n</VTKFile>\n'
    )
    for loader in (load_series, load_series_lazy):
        assert loader([fd]).dialect == "official"
        assert loader([pd]).dialect == "unknown"
        assert loader(pvd).dialect == "unknown"


# ---------------------------------------------------------------------------
# S23 legacy VTK 5.1 (OFFSETS / CONNECTIVITY)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metadata", [False, True])
def test_s23_vtk51_converts_like_vtk42(tmp_path, metadata):
    """S23: a hand-written VTK 9 style 5.1 series (optionally with METADATA after
    POINTS / OFFSETS / CONNECTIVITY / arrays) gives the same d3plot as the 4.2 series."""
    from lasso.dyna import ArrayType

    from vtk2d3plot.api import convert
    from vtk2d3plot.vtk_reader import read_vtk

    s42 = _write_series(tmp_path / "v42" / "run", "4.2", metadata)
    s51 = _write_series(tmp_path / "v51" / "run", "5.1", metadata)
    for meta in (False, True):
        _assert_mixed_state(read_vtk(s51 / "runA001.vtk", metadata_only=meta), 1, meta)
    ra = convert(s42, tmp_path / "a.d3plot", title="t")
    rb = convert(s51, tmp_path / "b.d3plot", title="t")
    assert (
        (ra.n_solids, ra.n_shells, ra.n_beams)
        == (rb.n_solids, rb.n_shells, rb.n_beams)
        == (2, 2, 1)
    )
    assert ra.warnings == rb.warnings
    d = _assert_same_d3plot(tmp_path / "a.d3plot", tmp_path / "b.d3plot")
    assert d[ArrayType.element_solid_ids].tolist() == [101, 105]
    assert d[ArrayType.element_shell_ids].tolist() == [102, 103]
    assert d[ArrayType.element_beam_ids].tolist() == [104]
    np.testing.assert_array_equal(d[ArrayType.node_ids], _NODE_ID)
    hexc = _NODE_ID[d[ArrayType.element_solid_node_indexes][0]]
    np.testing.assert_array_equal(hexc, _NODE_ID[:8])
    np.testing.assert_allclose(d[ArrayType.node_displacement][1], _PTS * 1.02, atol=1e-6)


def test_s23_meshio_vtk51_converts_like_vtk42(tmp_path):
    """S23: meshio 5.x writes legacy 5.1 by default (with vtktypeint32 / vtktypeint64 types);
    it converts to the same d3plot as meshio's 4.2 output of the same mesh."""
    from lasso.dyna import ArrayType

    from vtk2d3plot.api import convert

    cells = {}
    for t, c in _CELLS:
        cells.setdefault(t, []).append(c)
    names = {12: "hexahedron", 9: "quad", 5: "triangle", 3: "line", 10: "tetra"}
    order = [12, 9, 5, 3, 10]
    eids = dict(zip(order, ([101], [102], [103], [104], [105]), strict=True))
    for fmt, sub in (("vtk", "v51"), ("vtk42", "v42")):
        folder = tmp_path / sub / "run"
        folder.mkdir(parents=True)
        for k in range(2):
            mesh = meshio.Mesh(
                _PTS * (1.0 + 0.01 * (k + 1)),
                [(names[t], np.array(cells[t])) for t in order],
                point_data={
                    "NODE_ID": _NODE_ID.astype(np.int32),
                    "Displacement": _PTS * 0.01 * (k + 1),
                },
                cell_data={"ELEMENT_ID": [np.array(eids[t], np.int32) for t in order]},
            )
            p = folder / f"runA{k:03d}.vtk"
            meshio.write(p, mesh, file_format=fmt, binary=False)
            text = p.read_text()
            want = "# vtk DataFile Version 5.1" if fmt == "vtk" else "# vtk DataFile Version 4.2"
            assert text.startswith(want)
            assert ("OFFSETS" in text) == (fmt == "vtk")
            # meshio writes no time: give each state a title time
            lines = text.split("\n")
            lines[1] = f"meshio state t={0.25 * (k + 1)}"
            p.write_text("\n".join(lines))
    ra = convert(tmp_path / "v42" / "run", tmp_path / "a.d3plot", title="t")
    rb = convert(tmp_path / "v51" / "run", tmp_path / "b.d3plot", title="t")
    assert ra.n_states == rb.n_states == 2
    d = _assert_same_d3plot(tmp_path / "a.d3plot", tmp_path / "b.d3plot")
    assert sorted(d[ArrayType.element_solid_ids].tolist()) == [101, 105]
    np.testing.assert_array_equal(d[ArrayType.node_ids], _NODE_ID)


@pytest.mark.parametrize(
    "cells",
    [
        "CELLS 2 3\nOFFSETS vtktypeint64\n1 3\nCONNECTIVITY vtktypeint64\n0 1 2\n",  # offsets[0] != 0
        "CELLS 2 3\nOFFSETS vtktypeint64\n0 2\nCONNECTIVITY vtktypeint64\n0 1 2\n",  # last != size
        "CELLS 3 3\nOFFSETS vtktypeint64\n0 2 1\nCONNECTIVITY vtktypeint64\n0 1 2\n",  # decreasing
        "CELLS 0 0\nOFFSETS vtktypeint64\n\nCONNECTIVITY vtktypeint64\n\n",  # no offsets at all
        "CELLS 2 3\nOFFSETS vtktypeint64\n0 3\nCONNECTIVITY\n0 1 2\n",  # no type
        "CELLS 2 3\nOFFSETS vtktypeint64\n0 3\nPOINT_DATA 3\n",  # no CONNECTIVITY
        "CELLS 2 3\nOFFSETS vtktypeint64\n0 3\n",  # truncated
        "CELLS 2 3\nOFFSETS vtktypeint64\n0 3\nCONNECTIVITY vtktypeint64\n0 1\n",  # truncated
    ],
)
def test_s23_vtk51_bad_offsets_raise(tmp_path, cells):
    """S23: an inconsistent 5.1 OFFSETS / CONNECTIVITY block raises ValueError naming the file."""
    from vtk2d3plot.vtk_reader import read_vtk

    p = tmp_path / "bad.vtk"
    p.write_text(
        "# vtk DataFile Version 5.1\nt=1\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        "POINTS 3 double\n0 0 0 1 0 0 0 1 0\n" + cells + "CELL_TYPES 1\n5\n"
    )
    with pytest.raises(ValueError, match="bad.vtk"):
        read_vtk(p)
