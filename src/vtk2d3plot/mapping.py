from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .vtk_reader import VtkState


@dataclass
class MappedModel:
    node_ids: np.ndarray  # (n_nodes,) int32; zero IDs renumbered max+1
    solid_conn: np.ndarray  # (n_solids, 8) int32, 0-based node indexes
    solid_ids: np.ndarray  # (n_solids,) int32
    solid_part_idx: np.ndarray  # (n_solids,) int32, 0-based part index
    shell_conn: np.ndarray  # (n_shells, 4) int32
    shell_ids: np.ndarray  # (n_shells,) int32
    shell_part_idx: np.ndarray  # (n_shells,) int32
    beam_conn: np.ndarray  # (n_beams, 5) int32; orientation cols zero
    beam_ids: np.ndarray  # (n_beams,) int32
    beam_part_idx: np.ndarray  # (n_beams,) int32
    part_ids: np.ndarray  # (n_parts,) int32
    solid_src: np.ndarray  # (n_solids,) int32 — VTK cell index
    shell_src: np.ndarray  # (n_shells,) int32
    beam_src: np.ndarray  # (n_beams,) int32
    n_phantom_masked: int = 0
    n_skipped_cells: int = 0
    skipped_cell_types: set = field(default_factory=set)  # set of unsupported VTK type ints


def map_topology(state0: VtkState) -> MappedModel:
    SOLID_TYPES = {
        12: [0, 1, 2, 3, 4, 5, 6, 7],  # hex
        10: [0, 1, 2, 3, 3, 3, 3, 3],  # tetra
        13: [0, 1, 4, 3, 2, 2, 5, 5],  # wedge
        14: [0, 1, 2, 3, 4, 4, 4, 4],  # pyramid
    }
    SHELL_TYPES = {
        9: [0, 1, 2, 3],  # quad
        5: [0, 1, 2, 2],  # triangle
    }
    BEAM_TYPES = {
        3: [0, 1, 0, 0, 0],  # line
    }

    n_nodes = state0.points.shape[0]

    # 1. Node IDs — identify zero-ID points for phantom cell masking
    zero_id_point_indices: set[int] = set()
    if "NODE_ID" in state0.point_data:
        node_ids = state0.point_data["NODE_ID"].astype(np.int32).copy()
        zero_mask = node_ids == 0
        if np.any(zero_mask):
            zero_id_point_indices = {int(idx) for idx in np.where(zero_mask)[0]}
            max_id = np.max(node_ids)
            num_zeros = np.sum(zero_mask)
            node_ids[zero_mask] = np.arange(max_id + 1, max_id + 1 + num_zeros, dtype=np.int32)
    else:
        node_ids = np.arange(1, n_nodes + 1, dtype=np.int32)

    solids, shells, beams = [], [], []
    solid_src_list, shell_src_list, beam_src_list = [], [], []
    n_phantom_masked = 0
    n_skipped = 0
    skipped_cell_types: set[int] = set()

    for i, (ct, cell) in enumerate(zip(state0.cell_types, state0.cells)):
        ct = int(ct)
        # M7: mask cells referencing zero-ID nodes (phantom geometry)
        if zero_id_point_indices and any(int(v) in zero_id_point_indices for v in cell):
            n_phantom_masked += 1
            continue
        if ct in SOLID_TYPES:
            mapping = SOLID_TYPES[ct]
            conn = cell[mapping]  # 0-based node indexes
            solids.append(conn)
            solid_src_list.append(i)
        elif ct in SHELL_TYPES:
            mapping = SHELL_TYPES[ct]
            conn = cell[mapping]
            shells.append(conn)
            shell_src_list.append(i)
        elif ct in BEAM_TYPES:
            mapping = BEAM_TYPES[ct]
            conn = cell[mapping]
            beams.append(conn)
            beam_src_list.append(i)
        else:
            n_skipped += 1
            skipped_cell_types.add(ct)

    solid_conn = (
        np.array(solids, dtype=np.int32).reshape(-1, 8)
        if solids
        else np.empty((0, 8), dtype=np.int32)
    )
    shell_conn = (
        np.array(shells, dtype=np.int32).reshape(-1, 4)
        if shells
        else np.empty((0, 4), dtype=np.int32)
    )
    beam_conn = (
        np.array(beams, dtype=np.int32).reshape(-1, 5)
        if beams
        else np.empty((0, 5), dtype=np.int32)
    )
    solid_src = np.array(solid_src_list, dtype=np.int32)
    shell_src = np.array(shell_src_list, dtype=np.int32)
    beam_src = np.array(beam_src_list, dtype=np.int32)

    # 3. Part IDs
    if "PART_ID" in state0.cell_data:
        raw_part_ids = state0.cell_data["PART_ID"].astype(np.int32)
        # Unique preserving order of appearance
        _, idx = np.unique(raw_part_ids, return_index=True)
        part_ids = raw_part_ids[np.sort(idx)]

        part_id_to_idx = {pid: i for i, pid in enumerate(part_ids)}

        solid_part_idx = (
            np.array([part_id_to_idx[raw_part_ids[i]] for i in solid_src], dtype=np.int32)
            if len(solid_src)
            else np.empty((0,), dtype=np.int32)
        )
        shell_part_idx = (
            np.array([part_id_to_idx[raw_part_ids[i]] for i in shell_src], dtype=np.int32)
            if len(shell_src)
            else np.empty((0,), dtype=np.int32)
        )
        beam_part_idx = (
            np.array([part_id_to_idx[raw_part_ids[i]] for i in beam_src], dtype=np.int32)
            if len(beam_src)
            else np.empty((0,), dtype=np.int32)
        )
    else:
        part_ids = np.array([1], dtype=np.int32)
        solid_part_idx = np.zeros_like(solid_src, dtype=np.int32)
        shell_part_idx = np.zeros_like(shell_src, dtype=np.int32)
        beam_part_idx = np.zeros_like(beam_src, dtype=np.int32)

    # 2. Element IDs
    n_solids = len(solid_src)
    n_shells = len(shell_src)
    n_beams = len(beam_src)

    if "ELEMENT_ID" in state0.cell_data:
        raw_elem_ids = state0.cell_data["ELEMENT_ID"].astype(np.int32)
        solid_ids = raw_elem_ids[solid_src].copy() if n_solids else np.empty((0,), dtype=np.int32)
        if n_solids:
            zero_mask = solid_ids == 0
            if np.any(zero_mask):
                max_id = np.max(solid_ids)
                solid_ids[zero_mask] = np.arange(
                    max_id + 1, max_id + 1 + np.sum(zero_mask), dtype=np.int32
                )

        shell_ids = raw_elem_ids[shell_src].copy() if n_shells else np.empty((0,), dtype=np.int32)
        if n_shells:
            zero_mask = shell_ids == 0
            if np.any(zero_mask):
                max_id = np.max(shell_ids)
                shell_ids[zero_mask] = np.arange(
                    max_id + 1, max_id + 1 + np.sum(zero_mask), dtype=np.int32
                )

        beam_ids = raw_elem_ids[beam_src].copy() if n_beams else np.empty((0,), dtype=np.int32)
        if n_beams:
            zero_mask = beam_ids == 0
            if np.any(zero_mask):
                max_id = np.max(beam_ids)
                beam_ids[zero_mask] = np.arange(
                    max_id + 1, max_id + 1 + np.sum(zero_mask), dtype=np.int32
                )
    else:
        solid_ids = np.arange(1, n_solids + 1, dtype=np.int32)
        shell_ids = np.arange(n_solids + 1, n_solids + n_shells + 1, dtype=np.int32)
        beam_ids = np.arange(
            n_solids + n_shells + 1, n_solids + n_shells + n_beams + 1, dtype=np.int32
        )

    return MappedModel(
        node_ids=node_ids,
        solid_conn=solid_conn,
        solid_ids=solid_ids,
        solid_part_idx=solid_part_idx,
        shell_conn=shell_conn,
        shell_ids=shell_ids,
        shell_part_idx=shell_part_idx,
        beam_conn=beam_conn,
        beam_ids=beam_ids,
        beam_part_idx=beam_part_idx,
        part_ids=part_ids,
        solid_src=solid_src,
        shell_src=shell_src,
        beam_src=beam_src,
        n_phantom_masked=n_phantom_masked,
        n_skipped_cells=n_skipped,
        skipped_cell_types=skipped_cell_types,
    )
