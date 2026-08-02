"""d3plot builder — assembles a lasso D3plot from VtkSeries + MappedModel."""

from __future__ import annotations

import numpy as np
from lasso.dyna import ArrayType, D3plot

from .mapping import MappedModel
from .series import VtkSeries


def build_d3plot(
    series: VtkSeries,
    model: MappedModel,
    *,
    title: str = "",
    f64: bool = False,
) -> tuple[D3plot, list[str]]:
    ftype = np.float64 if f64 else np.float32
    itype = np.int64 if f64 else np.int32

    x0 = None
    warn_list = []
    for state in series.states:
        if "DISPLACEMENT" in state.point_data:
            x0 = state.points - state.point_data["DISPLACEMENT"]
            break
    if x0 is None:
        x0 = series.states[0].points
        warn_list.append(
            "No DISPLACEMENT array found; using state 0 coordinates as undeformed geometry"
        )

    n_states = len(series.states)
    n_nodes = x0.shape[0]
    node_disp = np.zeros((n_states, n_nodes, 3), dtype=ftype)
    for i, state in enumerate(series.states):
        node_disp[i] = state.points.astype(ftype)

    node_vel = None
    has_vel = any("VELOCITY" in s.point_data for s in series.states)
    if has_vel:
        node_vel = np.zeros((n_states, n_nodes, 3), dtype=ftype)
        for i, state in enumerate(series.states):
            if "VELOCITY" in state.point_data:
                node_vel[i] = state.point_data["VELOCITY"].astype(ftype)

    d3 = D3plot()
    a = d3.arrays

    a[ArrayType.node_coordinates] = x0.astype(ftype)
    a[ArrayType.node_ids] = model.node_ids.astype(itype)

    if model.solid_conn.shape[0] > 0:
        a[ArrayType.element_solid_node_indexes] = model.solid_conn.astype(itype)
        a[ArrayType.element_solid_part_indexes] = model.solid_part_idx.astype(itype)
        a[ArrayType.element_solid_ids] = model.solid_ids.astype(itype)

    if model.shell_conn.shape[0] > 0:
        a[ArrayType.element_shell_node_indexes] = model.shell_conn.astype(itype)
        a[ArrayType.element_shell_part_indexes] = model.shell_part_idx.astype(itype)
        a[ArrayType.element_shell_ids] = model.shell_ids.astype(itype)

    if model.beam_conn.shape[0] > 0:
        a[ArrayType.element_beam_node_indexes] = model.beam_conn.astype(itype)
        a[ArrayType.element_beam_part_indexes] = model.beam_part_idx.astype(itype)
        a[ArrayType.element_beam_ids] = model.beam_ids.astype(itype)

    a[ArrayType.part_ids] = model.part_ids.astype(itype)
    a[ArrayType.part_titles_ids] = model.part_ids.astype(itype)
    a[ArrayType.part_titles] = np.array([b"Part %d" % int(pid) for pid in model.part_ids])

    a[ArrayType.global_timesteps] = series.times.astype(ftype)
    a[ArrayType.node_displacement] = node_disp

    if node_vel is not None:
        a[ArrayType.node_velocity] = node_vel

    if f64:
        d3.header.wordsize = 8
        d3.header.itype = np.int64
        d3.header.ftype = np.float64

    return d3, warn_list
