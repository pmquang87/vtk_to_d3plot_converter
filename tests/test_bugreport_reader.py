"""Bug report 2026-10-11, cluster D: VTK / VTU reader.

Covers #13 (c03), #14 (c16), #15 reader side (c17), #16 (c29), #17 (c32),
#18 (c30), #29 (c18), #30 (c31).

The XML VTU files that meshio cannot write (appended data, several Pieces,
FieldData time) are written by ``_write_vtu`` below in the layout VTK uses:
- inline ``binary``: base64 text; uncompressed = one base64 stream
  ``header + data``; compressed = base64(header) + base64(compressed blocks);
- appended ``raw``: ``_`` then, per array, header + data bytes; offsets are
  byte positions after the ``_``;
- appended ``base64``: ``_`` then, per array, its OWN base64 block(s);
  offsets count encoded characters after the ``_``.
The compressed header is ``[n_blocks, block_size, last_block_size,
compressed_size_1 .. compressed_size_n]`` in the header type.
Every hand-written layout is cross-checked against meshio (independent
reader) inside the tests that use it.
"""

from __future__ import annotations

import base64
import binascii
import lzma
import zlib

import numpy as np
import pytest

pytestmark = pytest.mark.bugreport

meshio = pytest.importorskip("meshio")

_NP_OF_VTU = {
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
_VTU_OF_NP = {np.dtype(v): k for k, v in _NP_OF_VTU.items()}
_COMPRESSORS = {"zlib": ("vtkZLibDataCompressor", zlib), "lzma": ("vtkLZMADataCompressor", lzma)}


def _payload(
    arr: np.ndarray, compressor: str | None, header_type: str, block_size: int, bo: str = "<"
):
    """Return (header_bytes, data_bytes) for one array, VTK layout, byte order ``bo``."""
    hdt = np.dtype(np.uint64 if header_type == "UInt64" else np.uint32).newbyteorder(bo)
    raw = np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder(bo)).tobytes()
    if compressor is None:
        return np.array([len(raw)], dtype=hdt).tobytes(), raw
    mod = _COMPRESSORS[compressor][1]
    blocks = [raw[i : i + block_size] for i in range(0, len(raw), block_size)] or [b""]
    comp = [mod.compress(b) for b in blocks]
    last = len(blocks[-1]) if len(blocks[-1]) != block_size else 0
    header = np.array([len(blocks), block_size, last] + [len(c) for c in comp], dtype=hdt)
    return header.tobytes(), b"".join(comp)


def _write_vtu(
    path,
    pieces,
    *,
    fmt="ascii",
    compressor=None,
    header_type="UInt32",
    block_size=32768,
    field_data=None,
    byte_order="LittleEndian",
):
    """Write a VTU file in VTK's own layout.

    pieces: list of dicts with keys points (n,3), conn, offsets (end offsets),
    types, point_data {name: arr}, cell_data {name: arr}.
    fmt: "ascii" | "binary" | "appended-raw" | "appended-base64".
    field_data: {name: arr} written as dataset-level <FieldData>.
    byte_order: "LittleEndian" | "BigEndian" for headers and binary data.
    """
    bo = "<" if byte_order == "LittleEndian" else ">"
    appended = []  # list of bytes (raw) or str (base64) chunks
    pos = 0

    def data_array(name, arr, extra=""):
        nonlocal pos
        arr = np.asarray(arr)
        ncomp = arr.shape[1] if arr.ndim == 2 else 1
        typ = _VTU_OF_NP[arr.dtype]
        head = f'<DataArray type="{typ}" Name="{name}" NumberOfComponents="{ncomp}"{extra}'
        if fmt == "ascii":
            body = " ".join(repr(v.item()) for v in arr.reshape(-1))
            return f'{head} format="ascii">{body}</DataArray>\n'
        hdr, data = _payload(arr, compressor, header_type, block_size, bo)
        if compressor is None:
            enc = base64.b64encode(hdr + data).decode()
        else:
            enc = base64.b64encode(hdr).decode() + base64.b64encode(data).decode()
        if fmt == "binary":
            return f'{head} format="binary">\n{enc}\n</DataArray>\n'
        off = pos
        if fmt == "appended-raw":
            appended.append(hdr + data)
            pos += len(hdr) + len(data)
        else:
            appended.append(enc)
            pos += len(enc)
        return f'{head} format="appended" offset="{off}"/>\n'

    attrs = (
        f'type="UnstructuredGrid" version="1.0" byte_order="{byte_order}" '
        f'header_type="{header_type}"'
    )
    if compressor is not None:
        attrs += f' compressor="{_COMPRESSORS[compressor][0]}"'
    xml = [f'<?xml version="1.0"?>\n<VTKFile {attrs}>\n<UnstructuredGrid>\n']
    if field_data:
        xml.append("<FieldData>\n")
        for name, arr in field_data.items():
            arr = np.asarray(arr)
            xml.append(data_array(name, arr, f' NumberOfTuples="{arr.shape[0]}"'))
        xml.append("</FieldData>\n")
    for p in pieces:
        pts = np.asarray(p["points"])
        xml.append(f'<Piece NumberOfPoints="{len(pts)}" NumberOfCells="{len(p["types"])}">\n')
        xml.append("<Points>\n" + data_array("Points", pts) + "</Points>\n")
        xml.append("<Cells>\n")
        xml.append(data_array("connectivity", p["conn"]))
        xml.append(data_array("offsets", p["offsets"]))
        xml.append(data_array("types", p["types"]))
        xml.append("</Cells>\n")
        for tag, key in (("PointData", "point_data"), ("CellData", "cell_data")):
            if p.get(key):
                xml.append(f"<{tag}>\n")
                for name, arr in p[key].items():
                    xml.append(data_array(name, arr))
                xml.append(f"</{tag}>\n")
        xml.append("</Piece>\n")
    xml.append("</UnstructuredGrid>\n")
    out = "".join(xml).encode()
    if fmt.startswith("appended"):
        enc = "raw" if fmt == "appended-raw" else "base64"
        out += f'<AppendedData encoding="{enc}">\n   _'.encode()
        out += b"".join(c if isinstance(c, bytes) else c.encode() for c in appended)
        out += b"\n</AppendedData>\n"
    out += b"</VTKFile>\n"
    path.write_bytes(out)
    return path


def _b64decode_stops_at_padding() -> bool:
    """True if this CPython's non-strict b64decode stops at the first padded quad.

    meshio 5.3.5 decodes an appended base64 array with
    ``base64.b64decode(appended[offset:])``, i.e. with every later array's padded
    block still attached, and only works while the decoder stops at the first
    padding (CPython 3.12.15, 3.14.2). CPython 3.13.16 follows RFC 4648 sec. 3.3
    instead (Modules/binascii.c ``binascii_a2b_base64_impl``: a pad inside the
    data is skipped and decoding goes on), so meshio fails on these files with
    "Incorrect padding". The converter is not affected: its reader decodes each
    padded block on its own (``vtk_reader._b64decode_concat``).
    """
    try:
        return base64.b64decode(base64.b64encode(b"A") + base64.b64encode(b"BC")) == b"A"
    except binascii.Error:
        return False


_MESHIO_READS_APPENDED_BASE64 = _b64decode_stops_at_padding()


def _oracle(p, fmt):
    """meshio read of a hand-written file, or None where meshio cannot read the layout.

    meshio 5.3.5 converts raw appended data to base64 by rewriting each
    DataArray offset in place and then looking up the next raw offset with
    find(); a rewritten offset can equal a later raw offset, so it misreads
    valid raw files. Raw layouts are therefore not cross-checked with meshio.
    Appended base64 is cross-checked only on interpreters whose base64 decoder
    meshio can use (see ``_b64decode_stops_at_padding``); the reader itself is
    still checked against the source arrays on every interpreter.
    """
    if fmt == "appended-raw":
        return None
    if fmt == "appended-base64" and not _MESHIO_READS_APPENDED_BASE64:
        return None
    return meshio.read(p)


def _hex_piece(shift=0.0, pd=None, cd=None):
    pts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=np.float64,
    )
    pts[:, 0] += shift
    return {
        "points": pts,
        "conn": np.arange(8, dtype=np.int64),
        "offsets": np.array([8], dtype=np.int64),
        "types": np.array([12], dtype=np.uint8),
        "point_data": pd
        if pd is not None
        else {"Temperature": np.arange(8, dtype=np.float64) * 10.0 + 1.5},
        "cell_data": cd if cd is not None else {"VONM": np.array([123.25])},
    }


def _quad_piece(x0=0.0, temp=(1.0, 2.0, 3.0, 4.0), vonm=7.5):
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    pts[:, 0] += x0
    return {
        "points": pts,
        "conn": np.array([0, 1, 2, 3], dtype=np.int32),
        "offsets": np.array([4], dtype=np.int32),
        "types": np.array([9], dtype=np.uint8),
        "point_data": {"Temperature": np.array(temp, dtype=np.float64)},
        "cell_data": {"VONM": np.array([vonm], dtype=np.float64)},
    }


def _assert_hex(state, piece):
    assert state.points.shape == (8, 3)
    np.testing.assert_array_equal(state.points, piece["points"])
    assert state.cell_types.tolist() == [12]
    assert state.cell_connectivity.tolist() == list(range(8))
    assert state.cell_offsets.tolist() == [0, 8]
    np.testing.assert_array_equal(
        state.point_data["Temperature"], piece["point_data"]["Temperature"]
    )
    np.testing.assert_array_equal(state.cell_data["VONM"], [123.25])


def _grid_mesh(n=2500):
    """A strip of n-1 quads over 2*n points: big enough for several 32 KiB blocks."""
    x = np.arange(n, dtype=np.float64)
    pts = np.zeros((2 * n, 3))
    pts[:n, 0] = x
    pts[n:, 0] = x
    pts[n:, 1] = 1.0
    i = np.arange(n - 1)
    quads = np.stack([i, i + 1, n + i + 1, n + i], axis=1)
    temp = np.sin(np.arange(2 * n, dtype=np.float64))
    vel = np.arange(6 * n, dtype=np.float64).reshape(2 * n, 3) * 0.5
    vonm = np.cos(np.arange(n - 1, dtype=np.float64)) * 100.0
    eid = np.arange(n - 1, dtype=np.int32) + 1000
    return meshio.Mesh(
        pts,
        [("quad", quads)],
        point_data={"Temperature": temp, "Velocity": vel},
        cell_data={"VONM": [vonm], "ELEMENT_ID": [eid]},
    )


# ---------------------------------------------------------------------------
# #13 (c03) compressed VTU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compression", ["zlib", "lzma"])
@pytest.mark.parametrize("header_type", ["UInt32", "UInt64"])
def test_c03_meshio_compressed_vtu(tmp_path, compression, header_type):
    """Bug report 2026-10-11 #13 (c03): compressed VTU (meshio default) read as empty arrays."""
    from vtk2d3plot.vtk_reader import read_vtk

    mesh = _grid_mesh()
    p = tmp_path / "x.vtu"
    meshio.write(p, mesh, compression=compression, header_type=header_type)
    s = read_vtk(p)
    assert s.points.shape == (5000, 3)
    np.testing.assert_array_equal(s.points, mesh.points)
    assert s.cell_types.tolist() == [9] * 2499
    np.testing.assert_array_equal(s.cell_connectivity, mesh.cells[0].data.reshape(-1))
    np.testing.assert_array_equal(s.cell_offsets, np.arange(2500) * 4)
    np.testing.assert_array_equal(s.point_data["Temperature"], mesh.point_data["Temperature"])
    np.testing.assert_array_equal(s.point_data["Velocity"], mesh.point_data["Velocity"])
    np.testing.assert_array_equal(s.cell_data["VONM"], mesh.cell_data["VONM"][0])
    assert s.cell_data["ELEMENT_ID"].dtype == np.int32
    np.testing.assert_array_equal(s.cell_data["ELEMENT_ID"], np.arange(2499) + 1000)


@pytest.mark.parametrize("fmt", ["binary", "appended-raw", "appended-base64"])
@pytest.mark.parametrize("compression", ["zlib", "lzma"])
@pytest.mark.parametrize("header_type", ["UInt32", "UInt64"])
def test_c03_vtk_layout_compressed_multiblock(tmp_path, fmt, compression, header_type):
    """Bug report 2026-10-11 #13 (c03): compressed inline AND appended data, several blocks."""
    from vtk2d3plot.vtk_reader import read_vtk

    piece = _hex_piece()
    # block size 40 bytes: the 192-byte Points array spans 5 blocks, the last one partial
    p = _write_vtu(
        tmp_path / "h.vtu",
        [piece],
        fmt=fmt,
        compressor=compression,
        header_type=header_type,
        block_size=40,
    )
    ref = _oracle(p, fmt)  # the hand-written layout is what meshio expects
    if ref is not None:
        np.testing.assert_array_equal(ref.points, piece["points"])
    for meta in (False, True):
        s = read_vtk(p, metadata_only=meta)
        if meta:
            assert s.points.shape == (8, 3)
            assert s.cell_connectivity.tolist() == list(range(8))
            assert s.cell_offsets.tolist() == [0, 8]
        else:
            _assert_hex(s, piece)


def test_c03_unknown_compressor_raises(tmp_path):
    """Bug report 2026-10-11 #13 (c03): an unknown compressor must raise, never return empty arrays."""
    from vtk2d3plot.vtk_reader import read_vtk

    p = _write_vtu(tmp_path / "h.vtu", [_hex_piece()], fmt="binary", compressor="zlib")
    p.write_bytes(p.read_bytes().replace(b"vtkZLibDataCompressor", b"vtkFancyDataCompressor"))
    with pytest.raises(ValueError, match="vtkFancyDataCompressor"):
        read_vtk(p)
    with pytest.raises(ValueError, match="vtkFancyDataCompressor"):
        read_vtk(p, metadata_only=True)


def test_c03_lz4_without_module_raises(tmp_path):
    """Bug report 2026-10-11 #13 (c03): LZ4 data without the lz4 module gives a clear error."""
    try:
        import lz4.block  # noqa: F401

        pytest.skip("lz4 installed; the missing-module error path cannot be exercised")
    except ImportError:
        pass
    from vtk2d3plot.vtk_reader import read_vtk

    p = _write_vtu(tmp_path / "h.vtu", [_hex_piece()], fmt="binary", compressor="zlib")
    p.write_bytes(p.read_bytes().replace(b"vtkZLibDataCompressor", b"vtkLZ4DataCompressor"))
    with pytest.raises(ValueError, match="lz4"):
        read_vtk(p)


def _mixed_type_piece():
    """Hex piece with 1-, 4- and 8-byte arrays (byte order matters for all but UInt8)."""
    return _hex_piece(
        shift=0.125,
        pd={
            "Temperature": np.arange(8, dtype=np.float64) * 10.0 + 1.5,
            "Weight": (np.arange(8, dtype=np.float32) - 3.25) * 0.5,
            "NODE_ID": np.arange(8, dtype=np.int32) * 1000 + 70001,
        },
        cd={"VONM": np.array([123.25]), "ELEMENT_ID": np.array([4242], dtype=np.int32)},
    )


@pytest.mark.parametrize(
    ("fmt", "compressor", "header_type"),
    [
        ("binary", None, "UInt32"),
        ("appended-raw", None, "UInt32"),
        ("appended-raw", None, "UInt64"),
        ("binary", "zlib", "UInt32"),
        ("appended-raw", "zlib", "UInt64"),
        ("appended-base64", "zlib", "UInt32"),
    ],
)
def test_c03_vtu_big_endian_matches_little_endian(tmp_path, fmt, compressor, header_type):
    """Bug report 2026-10-11 #13 (c03): byte_order="BigEndian" VTU (headers AND data)
    reads exactly like the same mesh written LittleEndian."""
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
    # the two files really differ beyond the byte_order attribute
    assert le.read_bytes() != be.read_bytes().replace(b"BigEndian", b"LittleEndian")
    for meta in (False, True):
        a, b = read_vtk(le, metadata_only=meta), read_vtk(be, metadata_only=meta)
        assert b.time == a.time == 0.375
        assert b.points.shape == a.points.shape == (8, 3)
        assert b.cell_types.tolist() == a.cell_types.tolist() == [12]
        assert b.cell_connectivity.tolist() == a.cell_connectivity.tolist() == list(range(8))
        assert b.cell_offsets.tolist() == a.cell_offsets.tolist() == [0, 8]
        # metadata_only decodes only the identity arrays (and the time)
        keep = {"NODE_ID", "ELEMENT_ID"} if meta else None
        pdata = {k: v for k, v in piece["point_data"].items() if keep is None or k in keep}
        cdata = {k: v for k, v in piece["cell_data"].items() if keep is None or k in keep}
        assert set(b.point_data) == set(a.point_data) == set(pdata)
        assert set(b.cell_data) == set(a.cell_data) == set(cdata)
        for got_a, got_b, ref in (
            *((a.point_data[k], b.point_data[k], v) for k, v in pdata.items()),
            *((a.cell_data[k], b.cell_data[k], v) for k, v in cdata.items()),
        ):
            assert got_b.dtype == got_a.dtype == ref.dtype
            np.testing.assert_array_equal(got_b, ref)
            np.testing.assert_array_equal(got_a, ref)
        if not meta:
            np.testing.assert_array_equal(b.points, piece["points"])
            np.testing.assert_array_equal(a.points, piece["points"])


@pytest.mark.parametrize("fmt", ["binary", "appended-raw", "appended-base64"])
def test_c03_compressed_wrong_last_block_size_raises(tmp_path, monkeypatch, fmt):
    """Bug report 2026-10-11 #13 (c03): a compression header whose last-block size
    disagrees with the decompressed block is a corrupt file -> ValueError."""
    from vtk2d3plot.vtk_reader import read_vtk

    good_payload = _payload

    def wrong_last_size(arr, compressor, header_type, block_size, bo="<"):
        hdr, data = good_payload(arr, compressor, header_type, block_size, bo)
        hdt = np.dtype(np.uint64 if header_type == "UInt64" else np.uint32).newbyteorder(bo)
        h = np.frombuffer(hdr, dtype=hdt).copy()
        h[2] = (h[2] or block_size) - 1  # last block one byte shorter than it is
        return h.tobytes(), data

    monkeypatch.setitem(globals(), "_payload", wrong_last_size)
    p = _write_vtu(tmp_path / "h.vtu", [_hex_piece()], fmt=fmt, compressor="zlib", block_size=40)
    with pytest.raises(ValueError, match=r"Decompressed block \d+ of DataArray '\w+' has"):
        read_vtk(p)


def test_c03_compressed_vtu_series_converts(tmp_path):
    """Bug report 2026-10-11 #13 (c03): convert() of zlib VTUs gives the real mesh and values."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    files = []
    for k, t in enumerate((0.0, 0.5)):
        pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
        disp = np.zeros((4, 3))
        disp[:, 2] = 0.1 * k
        m = meshio.Mesh(
            pts + disp,
            [("quad", np.array([[0, 1, 2, 3]]))],
            point_data={"DISPLACEMENT": disp},
            cell_data={"VONM": [np.array([10.0 + k])]},
        )
        f = tmp_path / f"runA{k:03d}.vtu"
        meshio.write(f, m, compression="zlib")
        # meshio 5.3.5 does not write field_data to VTU; add the ParaView-style
        # dataset FieldData TimeValue by hand.
        fd = (
            '<UnstructuredGrid>\n<FieldData>\n<DataArray type="Float64" Name="TimeValue" '
            f'NumberOfTuples="1" format="ascii">{t!r}</DataArray>\n</FieldData>'
        )
        f.write_text(f.read_text().replace("<UnstructuredGrid>", fd, 1))
        files.append(f)
    rep = convert(files, tmp_path / "o.d3plot")
    assert rep.n_nodes == 4 and rep.n_shells == 1 and rep.n_states == 2
    d = D3plot(str(tmp_path / "o.d3plot"))
    np.testing.assert_allclose(d.arrays[ArrayType.global_timesteps], [0.0, 0.5])
    np.testing.assert_allclose(d.arrays[ArrayType.node_displacement][1, :, 2], [0.1] * 4)
    np.testing.assert_allclose(d.arrays[ArrayType.element_shell_stress][:, 0, 0, 0], [10.0, 11.0])


# ---------------------------------------------------------------------------
# #14 (c16) appended base64, one block per array
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compression", [None, "zlib"])
def test_c16_appended_base64_per_array_blocks(tmp_path, compression):
    """Bug report 2026-10-11 #14 (c16): appended base64 decoded as one stream (crash at 2nd array)."""
    from vtk2d3plot.vtk_reader import read_vtk

    piece = _hex_piece()
    p = _write_vtu(tmp_path / "h.vtu", [piece], fmt="appended-base64", compressor=compression)
    ref = _oracle(p, "appended-base64")
    if ref is not None:
        np.testing.assert_array_equal(
            ref.point_data["Temperature"].reshape(-1), piece["point_data"]["Temperature"]
        )
    _assert_hex(read_vtk(p), piece)
    s = read_vtk(p, metadata_only=True)
    assert s.cell_connectivity.tolist() == list(range(8)) and s.cell_types.tolist() == [12]


@pytest.mark.parametrize("compression", [None, "zlib"])
def test_c16_appended_raw_still_works(tmp_path, compression):
    """Bug report 2026-10-11 #14 (c16): the raw appended (mmap) path keeps working."""
    from vtk2d3plot.vtk_reader import read_vtk

    piece = _hex_piece()
    p = _write_vtu(tmp_path / "h.vtu", [piece], fmt="appended-raw", compressor=compression)
    _assert_hex(read_vtk(p), piece)


# ---------------------------------------------------------------------------
# #15 (c17) reader side: FieldData TimeValue / TIME
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["ascii", "binary", "appended-raw", "appended-base64"])
@pytest.mark.parametrize("name", ["TimeValue", "TIME"])
def test_c17_vtu_fielddata_time(tmp_path, fmt, name):
    """Bug report 2026-10-11 #15 (c17): read_vtu ignores FieldData TimeValue / TIME."""
    from vtk2d3plot.vtk_reader import read_vtk

    p = _write_vtu(
        tmp_path / "h.vtu", [_hex_piece()], fmt=fmt, field_data={name: np.array([0.375])}
    )
    assert read_vtk(p).time == 0.375
    assert read_vtk(p, metadata_only=True).time == 0.375


def test_c17_vtu_fielddata_time_series_converts(tmp_path):
    """Bug report 2026-10-11 #15 (c17): VTUs timed only by FieldData TimeValue convert."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    files = [
        _write_vtu(
            tmp_path / f"runA{k:03d}.vtu",
            [_hex_piece()],
            fmt="binary",
            field_data={"TimeValue": np.array([t])},
        )
        for k, t in enumerate((0.0, 0.5, 1.0))
    ]
    convert(files, tmp_path / "o.d3plot")
    d = D3plot(str(tmp_path / "o.d3plot"))
    np.testing.assert_allclose(d.arrays[ArrayType.global_timesteps], [0.0, 0.5, 1.0])


# ---------------------------------------------------------------------------
# #16 (c29) several Pieces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["ascii", "binary", "appended-raw", "appended-base64"])
def test_c29_multipiece_concatenated(tmp_path, fmt):
    """Bug report 2026-10-11 #16 (c29): a VTU with several Pieces keeps only the last one."""
    from vtk2d3plot.vtk_reader import read_vtk

    pieces = [_quad_piece(0.0, (1, 2, 3, 4), 7.5), _quad_piece(5.0, (5, 6, 7, 8), 9.5)]
    p = _write_vtu(tmp_path / "m.vtu", pieces, fmt=fmt, field_data={"TimeValue": np.array([0.25])})
    ref = _oracle(p, fmt)
    s = read_vtk(p)
    np.testing.assert_array_equal(
        s.points, np.concatenate([pieces[0]["points"], pieces[1]["points"]])
    )
    assert s.cell_connectivity.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert s.cell_offsets.tolist() == [0, 4, 8]
    assert s.cell_types.tolist() == [9, 9]
    np.testing.assert_array_equal(s.point_data["Temperature"], [1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(s.cell_data["VONM"], [7.5, 9.5])
    if (
        ref is not None
    ):  # meshio 5.3.5 keeps all Piece points (its multi-Piece cells are unreliable)
        np.testing.assert_array_equal(ref.points, s.points)
    m = read_vtk(p, metadata_only=True)
    assert m.points.shape == (8, 3) and m.time == 0.25
    assert m.cell_connectivity.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert m.cell_offsets.tolist() == [0, 4, 8]


def test_c29_multipiece_convert_n_nodes(tmp_path):
    """Bug report 2026-10-11 #16 (c29): convert() reports and writes all nodes of all Pieces."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    files = [
        _write_vtu(
            tmp_path / f"runA{k:03d}.vtu",
            [_quad_piece(0.0, vonm=1.0 + k), _quad_piece(5.0, vonm=2.0 + k)],
            fmt="binary",
            field_data={"TimeValue": np.array([float(k)])},
        )
        for k in range(2)
    ]
    rep = convert(files, tmp_path / "o.d3plot")
    assert rep.n_nodes == 8 and rep.n_shells == 2
    d = D3plot(str(tmp_path / "o.d3plot"))
    assert d.arrays[ArrayType.node_coordinates][:, 0].tolist() == [0, 1, 1, 0, 5, 6, 6, 5]
    np.testing.assert_allclose(d.arrays[ArrayType.element_shell_stress][1, :, 0, 0], [2.0, 3.0])


def test_c29_multipiece_inconsistent_arrays_raise(tmp_path):
    """Bug report 2026-10-11 #16 (c29): Pieces with different arrays raise instead of dropping data."""
    from vtk2d3plot.vtk_reader import read_vtk

    a = _quad_piece(0.0)
    b = _quad_piece(5.0)
    b["point_data"] = {"Pressure": np.zeros(4)}
    p = _write_vtu(tmp_path / "m.vtu", [a, b], fmt="ascii")
    with pytest.raises(ValueError, match="Temperature"):
        read_vtk(p)


# ---------------------------------------------------------------------------
# #17 (c32) FIELD arrays inside POINT_DATA / CELL_DATA
# ---------------------------------------------------------------------------


def test_c32_meshio_vtk42_field_arrays(tmp_path):
    """Bug report 2026-10-11 #17 (c32): FIELD arrays in POINT_DATA / CELL_DATA are thrown away."""
    from vtk2d3plot.vtk_reader import read_vtk

    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    m = meshio.Mesh(
        pts,
        [("quad", np.array([[0, 1, 2, 3]]))],
        point_data={
            "Temperature": np.array([1.0, 2.0, 3.0, 4.0]),
            "V": np.arange(12.0).reshape(4, 3),
        },
        cell_data={"VONM": [np.array([7.5])], "ID": [np.array([42], dtype=np.int32)]},
    )
    p = tmp_path / "m.vtk"
    meshio.write(p, m, file_format="vtk42", binary=False)
    assert "FIELD FieldData 2" in p.read_text()
    s = read_vtk(p)
    np.testing.assert_array_equal(s.point_data["Temperature"], [1.0, 2.0, 3.0, 4.0])
    assert s.point_data["V"].shape == (4, 3)
    np.testing.assert_array_equal(s.point_data["V"], np.arange(12.0).reshape(4, 3))
    np.testing.assert_array_equal(s.cell_data["VONM"], [7.5])
    assert s.cell_data["ID"].tolist() == [42] and s.cell_data["ID"].dtype.kind == "i"
    # metadata-only (lazy) path: arrays skipped, parsing stays in sync
    s2 = read_vtk(p, metadata_only=True)
    assert s2.cell_connectivity.tolist() == [0, 1, 2, 3] and s2.cell_types.tolist() == [9]


_OFFICIAL_FIELD = """# vtk DataFile Version 3.0
{title}
ASCII
DATASET UNSTRUCTURED_GRID
FIELD FieldData 2
TIME 1 1 double
{time}
CYCLE 1 1 int
0
POINTS 4 float
0 0 {dz} 1 0 {dz} 1 1 {dz} 0 1 {dz}
CELLS 1 5
4 0 1 2 3
CELL_TYPES 1
9
POINT_DATA 4
FIELD FieldData 2
DISPLACEMENT 3 4 double
0 0 {dz} 0 0 {dz} 0 0 {dz} 0 0 {dz}
Temperature 1 4 float
1 2 3 4
CELL_DATA 1
FIELD FieldData 1
VONM 1 1 double
{vonm}
"""


def test_c32_field_in_data_and_dataset_time(tmp_path):
    """Bug report 2026-10-11 #17 (c32): FIELD in POINT_DATA/CELL_DATA kept; dataset FIELD TIME still the time."""
    from vtk2d3plot.vtk_reader import read_vtk

    p = tmp_path / "a.vtk"
    p.write_text(_OFFICIAL_FIELD.format(title="x", time="0.75", dz="0.5", vonm="12.5"))
    s = read_vtk(p)
    assert s.time == 0.75
    assert set(s.point_data) == {"DISPLACEMENT", "Temperature"}
    assert set(s.cell_data) == {"VONM"}
    np.testing.assert_array_equal(s.point_data["DISPLACEMENT"][:, 2], [0.5] * 4)
    np.testing.assert_array_equal(s.point_data["Temperature"], [1, 2, 3, 4])
    np.testing.assert_array_equal(s.cell_data["VONM"], [12.5])
    m = read_vtk(p, metadata_only=True)
    assert m.time == 0.75 and m.point_data == {} and m.cell_data == {}


def test_c32_field_arrays_reach_d3plot(tmp_path):
    """Bug report 2026-10-11 #17 (c32): FIELD-stored DISPLACEMENT / VONM reach the d3plot."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    # POINTS are deformed (z = dz) and DISPLACEMENT = dz, so the undeformed
    # z = 0 is recovered only if the FIELD-stored DISPLACEMENT is read.
    for k in range(2):
        (tmp_path / f"runA{k:03d}.vtk").write_text(
            _OFFICIAL_FIELD.format(
                title="x", time=str(0.5 * k), dz=str(0.125 + 0.25 * k), vonm=str(10.0 + k)
            )
        )
    rep = convert(tmp_path, tmp_path / "o.d3plot")
    assert not any("No DISPLACEMENT" in w for w in rep.warnings)
    d = D3plot(str(tmp_path / "o.d3plot"))
    np.testing.assert_allclose(d.arrays[ArrayType.node_coordinates][:, 2], [0.0] * 4)
    np.testing.assert_allclose(d.arrays[ArrayType.node_displacement][1, :, 2], [0.375] * 4)
    np.testing.assert_allclose(d.arrays[ArrayType.element_shell_stress][:, 0, 0, 0], [10.0, 11.0])


def test_c32_string_field_array_in_cell_data(tmp_path):
    """Bug report 2026-10-11 #17 (c32) follow-up: a string FIELD array inside CELL_DATA
    must not make the whole file fail; the numeric array beside it is kept, and the
    conversion report lists the string array as unmapped."""
    from vtk2d3plot.api import convert
    from vtk2d3plot.vtk_reader import read_vtk

    for k in range(2):
        txt = _OFFICIAL_FIELD.format(title="x", time=str(0.5 * k), dz="0.0", vonm="42.0")
        txt = txt.replace(
            "CELL_DATA 1\nFIELD FieldData 1\n",
            "CELL_DATA 1\nFIELD FieldData 2\nLabel 1 1 string\nabc\n",
        )
        (tmp_path / f"runA{k:03d}.vtk").write_text(txt)
    s = read_vtk(tmp_path / "runA000.vtk")
    np.testing.assert_array_equal(s.cell_data["VONM"], [42.0])
    assert s.cell_data["Label"].tolist() == ["abc"]
    m = read_vtk(tmp_path / "runA000.vtk", metadata_only=True)
    assert m.cell_data == {} and m.time == 0.0

    rep = convert(tmp_path, tmp_path / "o.d3plot")
    unmapped = [w for w in rep.warnings if w.startswith("unmapped arrays")]
    assert len(unmapped) == 1 and "Label" in unmapped[0]


def test_c32_field_tuple_count_mismatch_raises(tmp_path):
    """Bug report 2026-10-11 #17 (c32): a FIELD array whose tuple count is not the POINT_DATA count raises."""
    from vtk2d3plot.vtk_reader import read_vtk

    p = tmp_path / "a.vtk"
    txt = _OFFICIAL_FIELD.format(title="x", time="0.75", dz="0.5", vonm="12.5")
    p.write_text(txt.replace("Temperature 1 4 float\n1 2 3 4", "Temperature 1 3 float\n1 2 3"))
    with pytest.raises(ValueError, match="Temperature"):
        read_vtk(p)


# ---------------------------------------------------------------------------
# #18 (c30) Int16 / UInt16 and the full VTK type list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["ascii", "binary", "appended-raw", "appended-base64"])
@pytest.mark.parametrize("vtype", list(_NP_OF_VTU))
def test_c30_all_vtk_types(tmp_path, fmt, vtype):
    """Bug report 2026-10-11 #18 (c30): Int16 / UInt16 decode as float64 garbage; audit all types."""
    from vtk2d3plot.vtk_reader import read_vtk

    dt = _NP_OF_VTU[vtype]
    vals = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=dt)
    if np.dtype(dt).kind == "u":
        vals[-1] = np.iinfo(dt).max
    elif np.dtype(dt).kind == "i":
        vals[-1] = np.iinfo(dt).min
    piece = _hex_piece(pd={"V": vals})
    p = _write_vtu(tmp_path / "t.vtu", [piece], fmt=fmt)
    s = read_vtk(p)
    assert s.point_data["V"].dtype == np.dtype(dt)
    np.testing.assert_array_equal(s.point_data["V"], vals)
    ref = _oracle(p, fmt)
    if ref is not None:
        np.testing.assert_array_equal(ref.point_data["V"].reshape(-1), vals)


def test_c30_int16_binary(tmp_path):
    """Bug report 2026-10-11 #18 (c30): Int16 [1, 2, 3, 4] reads as [1, 2, 3, 4], not [5.56e-309]."""
    from vtk2d3plot.vtk_reader import read_vtk

    piece = _quad_piece()
    piece["point_data"] = {"I16": np.array([1, 2, 3, 4], dtype=np.int16)}
    p = _write_vtu(tmp_path / "t.vtu", [piece], fmt="binary")
    assert read_vtk(p).point_data["I16"].tolist() == [1, 2, 3, 4]


def test_c30_unknown_type_raises(tmp_path):
    """Bug report 2026-10-11 #18 (c30): an unknown DataArray type raises, never falls back to float64."""
    from vtk2d3plot.vtk_reader import read_vtk

    piece = _quad_piece()
    piece["point_data"] = {"I16": np.array([1, 2, 3, 4], dtype=np.int16)}
    p = _write_vtu(tmp_path / "t.vtu", [piece], fmt="binary")
    p.write_bytes(p.read_bytes().replace(b'type="Int16"', b'type="Float16"'))
    with pytest.raises(ValueError, match="Float16"):
        read_vtk(p)


# Every type name vtkDataWriter emits for legacy files (VTK 9: vtkDataWriter::
# WriteArray), with the numpy kind the reader returns and a value that only the
# right width can hold (2**40 needs 64 bits).
_LEGACY_TYPE_CASES = {
    "bit": ("i", [1, 0, 1, 1]),
    "unsigned_char": ("i", [1, 2, 3, 255]),
    "char": ("i", [1, 2, 3, -4]),
    "signed_char": ("i", [1, 2, 3, -4]),
    "unsigned_short": ("i", [1, 2, 3, 65535]),
    "short": ("i", [1, 2, 3, -4]),
    "unsigned_int": ("i", [1, 2, 3, 4294967295]),
    "int": ("i", [1, 2, 3, -4]),
    "unsigned_long": ("i", [1, 2, 3, 2**40]),
    "long": ("i", [1, 2, 3, -(2**40)]),
    "vtktypeint64": ("i", [1, 2, 3, -(2**40)]),
    "vtktypeuint64": ("i", [1, 2, 3, 2**40]),
    "vtkIdType": ("i", [1, 2, 3, 2**40]),
    "float": ("f", [1.5, 2.0, 3.0, -4.25]),
    "double": ("f", [1.5, 2.0, 3.0, -4.25]),
}


@pytest.mark.parametrize("section", ["SCALARS", "FIELD"])
@pytest.mark.parametrize("vtype", list(_LEGACY_TYPE_CASES))
def test_c30_legacy_vtkdatawriter_type_names(tmp_path, vtype, section):
    """Bug report 2026-10-11 #18 (c30): the legacy type audit accepts every vtkDataWriter type name.

    signed_char, vtktypeint64 and vtktypeuint64 (VTK_LONG_LONG / VTK_UNSIGNED_LONG_LONG,
    what numpy int64 becomes on Windows) were rejected by the first audit table.
    """
    from vtk2d3plot.vtk_reader import read_vtk

    kind, vals = _LEGACY_TYPE_CASES[vtype]
    body = " ".join(str(v) for v in vals)
    if section == "SCALARS":
        block = f"SCALARS A {vtype} 1\nLOOKUP_TABLE default\n{body}\n"
    else:
        block = f"FIELD FieldData 1\nA 1 4 {vtype}\n{body}\n"
    p = tmp_path / "a.vtk"
    p.write_text(_LEGACY.format(title="x", field="") + "POINT_DATA 4\n" + block)
    s = read_vtk(p)
    assert s.point_data["A"].dtype.kind == kind
    assert s.point_data["A"].tolist() == vals
    # lazy path skips the array but must stay in sync and accept the type name
    m = read_vtk(p, metadata_only=True)
    assert m.point_data == {} and m.cell_connectivity.tolist() == [0, 1, 2, 3]


_TENSOR = np.array([[1.0, 4.0, 6.0], [4.0, 2.0, 5.0], [6.0, 5.0, 3.0]])


@pytest.mark.parametrize(
    ("fmt", "compressor"),
    [("ascii", None)]
    + [(f, c) for f in ("binary", "appended-raw", "appended-base64") for c in (None, "zlib")],
)
def test_c03_vtu_9component_tensor_is_3x3(tmp_path, fmt, compressor):
    """Bug report 2026-10-11 #13 (c03) follow-up: a 9-component VTU array (3DELEM_Stress) is (n, 3, 3).

    The VtkState contract and the legacy TENSORS path use (n, 3, 3); (n, 9) made
    convert() fail with 'too many indices for array'.
    """
    from vtk2d3plot.vtk_reader import read_vtk

    t = np.stack([_TENSOR, _TENSOR * 10.0])  # 2 cells
    piece = _hex_piece(cd={"3DELEM_Stress": t.reshape(2, 9)})
    piece["conn"] = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    piece["offsets"] = np.array([8, 16], dtype=np.int64)
    piece["types"] = np.array([12, 12], dtype=np.uint8)
    piece["point_data"] = {"VEC9": np.arange(72.0).reshape(8, 9)}
    p = _write_vtu(tmp_path / "t.vtu", [piece], fmt=fmt, compressor=compressor)
    s = read_vtk(p)
    assert s.cell_data["3DELEM_Stress"].shape == (2, 3, 3)
    np.testing.assert_array_equal(s.cell_data["3DELEM_Stress"], t)
    np.testing.assert_array_equal(s.point_data["VEC9"], np.arange(72.0).reshape(8, 3, 3))
    ref = _oracle(p, fmt)
    if ref is not None:
        np.testing.assert_array_equal(ref.cell_data["3DELEM_Stress"][0].reshape(2, 3, 3), t)


def test_c03_vtu_9component_tensor_converts(tmp_path):
    """Bug report 2026-10-11 #13 (c03) follow-up: VTU 3DELEM_Stress reaches element_solid_stress."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    files = [
        _write_vtu(
            tmp_path / f"runA{k:03d}.vtu",
            [_hex_piece(cd={"3DELEM_Stress": (_TENSOR * (k + 1)).reshape(1, 9)})],
            fmt="binary",
            compressor="zlib",
            field_data={"TimeValue": np.array([0.5 * k])},
        )
        for k in range(2)
    ]
    convert(files, tmp_path / "o.d3plot")
    d = D3plot(str(tmp_path / "o.d3plot"))
    s = d.arrays[ArrayType.element_solid_stress]  # (n_states, n_solids, 1, 6)
    # Voigt [xx, yy, zz, xy, xz, yz] = [t00, t11, t22, t01, t02, t12]
    np.testing.assert_allclose(s[0, 0, 0], [1, 2, 3, 4, 6, 5])
    np.testing.assert_allclose(s[1, 0, 0], [2, 4, 6, 8, 12, 10])


def test_c32_field_9component_is_3x3(tmp_path):
    """Bug report 2026-10-11 #17 (c32) follow-up: a 9-component FIELD array in CELL_DATA is (n, 3, 3)."""
    from vtk2d3plot.vtk_reader import read_vtk

    body = " ".join(str(v) for v in _TENSOR.reshape(-1))
    block = f"CELL_DATA 1\nFIELD FieldData 1\n3DELEM_Stress 9 1 double\n{body}\n"
    p = tmp_path / "a.vtk"
    p.write_text(_LEGACY.format(title="x", field="") + block)
    s = read_vtk(p)
    assert s.cell_data["3DELEM_Stress"].shape == (1, 3, 3)
    np.testing.assert_array_equal(s.cell_data["3DELEM_Stress"][0], _TENSOR)


# ---------------------------------------------------------------------------
# #29 (c18) title time regex
# ---------------------------------------------------------------------------

_LEGACY = """# vtk DataFile Version 3.0
{title}
ASCII
DATASET UNSTRUCTURED_GRID
{field}POINTS 4 double
0 0 0 1 0 0 1 1 0 0 1 0
CELLS 1 5
4 0 1 2 3
CELL_TYPES 1
9
"""
_FIELD_05 = "FIELD FieldData 1\nTIME 1 1 double\n0.5\n"


@pytest.mark.parametrize(
    "title", ["Result=extracted", "state at t=end", "crash test at t=1.5.", "t=1e", "t=."]
)
@pytest.mark.parametrize("meta", [False, True])
def test_c18_field_time_wins_over_garbage_title(tmp_path, title, meta):
    """Bug report 2026-10-11 #29 (c18): title regex crashes on free text; FIELD TIME takes precedence."""
    from vtk2d3plot.vtk_reader import read_vtk

    p = tmp_path / "a.vtk"
    p.write_text(_LEGACY.format(title=title, field=_FIELD_05))
    s = read_vtk(p, metadata_only=meta)
    assert s.time == 0.5 and s.title == title


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("pyradioss count=5 state", None),
        ("Result=extracted", None),
        ("state at t=end", None),
        ("abct=3.0", None),
        ("pyradioss state t=2.902412789E-04", 2.902412789e-04),
        ("t=1.5", 1.5),
        ("state t=-1.25e+02 x", -125.0),
        ("crash test at t=1.5.", 1.5),
    ],
)
def test_c18_title_time_word_boundary(tmp_path, title, expected):
    """Bug report 2026-10-11 #29 (c18): t= must start a word and carry a well-formed float."""
    from vtk2d3plot.vtk_reader import read_vtk

    p = tmp_path / "a.vtk"
    p.write_text(_LEGACY.format(title=title, field=""))
    assert read_vtk(p).time == expected
    assert read_vtk(p, metadata_only=True).time == expected


def test_c18_convert_with_free_text_title(tmp_path):
    """Bug report 2026-10-11 #29 (c18): convert() no longer crashes on 'state at t=end' titles."""
    from lasso.dyna import ArrayType, D3plot

    from vtk2d3plot.api import convert

    for k in range(2):
        field = f"FIELD FieldData 1\nTIME 1 1 double\n{0.5 * k}\n"
        (tmp_path / f"runA{k:03d}.vtk").write_text(
            _LEGACY.format(title="state at t=end", field=field)
        )
    convert(tmp_path, tmp_path / "o.d3plot")
    d = D3plot(str(tmp_path / "o.d3plot"))
    np.testing.assert_allclose(d.arrays[ArrayType.global_timesteps], [0.0, 0.5])


# ---------------------------------------------------------------------------
# #30 (c31) empty title line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("meta", [False, True])
def test_c31_empty_title(tmp_path, meta):
    """Bug report 2026-10-11 #30 (c31): an empty line 2 must give title '' and parse the rest."""
    from vtk2d3plot.vtk_reader import read_vtk

    p = tmp_path / "a.vtk"
    p.write_text(_LEGACY.format(title="", field=_FIELD_05))
    s = read_vtk(p, metadata_only=meta)
    assert s.title == ""
    assert s.time == 0.5
    assert s.cell_connectivity.tolist() == [0, 1, 2, 3]
    assert s.points.shape == (4, 3)
    if not meta:
        np.testing.assert_array_equal(s.points[2], [1, 1, 0])


@pytest.mark.parametrize("meta", [False, True])
def test_c31_truncated_after_title(tmp_path, meta):
    """Bug report 2026-10-11 #30 (c31) follow-up: a file that ends after the title says it is truncated."""
    from vtk2d3plot.vtk_reader import read_vtk

    p = tmp_path / "a.vtk"
    p.write_text("# vtk DataFile Version 3.0\nsome title\n")
    with pytest.raises(ValueError, match="Truncated VTK file"):
        read_vtk(p, metadata_only=meta)


# ---------------------------------------------------------------------------
# lazy loader (metadata_only path) over the new VTU layouts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["binary", "appended-raw", "appended-base64"])
def test_c03_c16_c17_c29_lazy_series(tmp_path, fmt):
    """Bug report 2026-10-11 #13 (c03), #14 (c16), #15 (c17), #16 (c29): lazy loader over compressed multi-Piece VTUs."""
    from vtk2d3plot.series import load_series_lazy

    files = [
        _write_vtu(
            tmp_path / f"runA{k:03d}.vtu",
            [_quad_piece(0.0, vonm=1.0 + k), _quad_piece(5.0, vonm=2.0 + k)],
            fmt=fmt,
            compressor="zlib",
            block_size=16,
            field_data={"TimeValue": np.array([0.25 * k])},
        )
        for k in range(3)
    ]
    ser = load_series_lazy(files)
    np.testing.assert_array_equal(ser.times, [0.0, 0.25, 0.5])
    assert ser.state0.points.shape == (8, 3)
    states = [s for _, s in ser.iter_states()]
    np.testing.assert_array_equal(states[2].cell_data["VONM"], [3.0, 4.0])
    assert states[2].cell_connectivity.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
