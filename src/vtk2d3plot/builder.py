"""d3plot builder — assembles a lasso D3plot from VtkSeries + MappedModel."""

from __future__ import annotations

import numpy as np
from lasso.dyna import ArrayType, D3plot

from .mapping import MappedModel
from .series import VtkSeries


def _find_key(d: dict, name: str) -> str | None:
    """Case-insensitive key lookup in a dict."""
    low = name.lower()
    for k in d:
        if k.lower() == low:
            return k
    return None


def _tensor_to_voigt(t: np.ndarray) -> np.ndarray:
    """Convert (n, 3, 3) symmetric tensor to (n, 6) Voigt: xx yy zz xy xz yz.

    The binary d3plot layout for stress slots 3-5 is [σxy, σxz, σyz].
    This matches the extraction order [t[0,1], t[0,2], t[1,2]].
    Root cause: anim_to_vtk.cpp (OpenRadioss) writes its 3×3 with yz at
    position (0,2) and zx at (1,2), so the Voigt extraction here produces
    the correct slot assignment.  Verified against Vortex golden d3plot.
    """
    return np.stack(
        [t[:, 0, 0], t[:, 1, 1], t[:, 2, 2], t[:, 0, 1], t[:, 0, 2], t[:, 1, 2]],
        axis=-1,
    )


def _any_state_has(series: VtkSeries, key: str, data_kind: str = "cell") -> bool:
    """Check if any state in the series contains the given key (M3 fix)."""
    for state in series.states:
        d = state.cell_data if data_kind == "cell" else state.point_data
        if key in d:
            return True
    return False


def build_d3plot(
    series: VtkSeries,
    model: MappedModel,
    *,
    title: str = "",
    f64: bool = False,
) -> tuple[D3plot, list[str]]:
    ftype = np.float64 if f64 else np.float32
    itype = np.int64 if f64 else np.int32

    warn_list: list[str] = []
    n_states = len(series.states)
    n_nodes = series.states[0].points.shape[0]
    n_solids = model.solid_conn.shape[0]
    n_shells = model.shell_conn.shape[0]
    n_beams = model.beam_conn.shape[0]

    # ----------------------------------------------------------------
    # Geometry recovery: x0 = POINTS - DISPLACEMENT (case-insensitive)
    # ----------------------------------------------------------------
    x0 = None
    for state in series.states:
        dk = _find_key(state.point_data, "DISPLACEMENT")
        if dk is not None:
            x0 = state.points - state.point_data[dk]
            break
    if x0 is None:
        x0 = series.states[0].points
        warn_list.append(
            "No DISPLACEMENT array found; using state 0 coordinates as undeformed geometry"
        )

    # ----------------------------------------------------------------
    # Kinematics
    # ----------------------------------------------------------------
    node_disp = np.zeros((n_states, n_nodes, 3), dtype=ftype)
    for i, state in enumerate(series.states):
        node_disp[i] = state.points.astype(ftype)

    node_vel = None
    if any(_find_key(s.point_data, "VELOCITY") for s in series.states):
        node_vel = np.zeros((n_states, n_nodes, 3), dtype=ftype)
        for i, state in enumerate(series.states):
            vk = _find_key(state.point_data, "VELOCITY")
            if vk is not None:
                node_vel[i] = state.point_data[vk].astype(ftype)

    # ----------------------------------------------------------------
    # Element results: detect which arrays are present (M3: union over ALL states)
    # ----------------------------------------------------------------

    # --- Official dialect: 3D solid stress tensor ---
    has_solid_stress = n_solids > 0 and _any_state_has(series, "3DELEM_Stress")
    # --- Official dialect: shell stress (upper/lower layers) ---
    has_shell_stress_2layer = n_shells > 0 and (
        _any_state_has(series, "2DELEM_Stress_(lower)")
        and _any_state_has(series, "2DELEM_Stress_(upper)")
    )
    # --- Official dialect: shell stress (membrane — single layer) (M2 fix) ---
    has_shell_stress_membrane = n_shells > 0 and (
        not has_shell_stress_2layer
        and _any_state_has(series, "2DELEM_Stress_(membrane)")
    )
    # --- Official dialect: 2DELEM_Von_Mises fallback (M2 fix) ---
    has_shell_vonmises_official = n_shells > 0 and (
        not has_shell_stress_2layer
        and not has_shell_stress_membrane
        and _any_state_has(series, "2DELEM_Von_Mises")
    )
    # --- Port dialect: VONM synthetic stress (C1 fix: removed n_shells>0 gate) ---
    has_vonm = _any_state_has(series, "VONM")
    # --- Official: solid plastic strain ---
    has_solid_eps = n_solids > 0 and _any_state_has(series, "3DELEM_Plastic_Strain")
    # --- Official: shell plastic strain (upper/lower) ---
    has_shell_eps_2layer = n_shells > 0 and (
        _any_state_has(series, "2DELEM_Plastic_Strain_Upper")
        and _any_state_has(series, "2DELEM_Plastic_Strain_Lower")
    )
    has_shell_eps_single = n_shells > 0 and _any_state_has(
        series, "2DELEM_Plastic_Strain"
    )
    # --- Port: EPSP (C1 fix: removed n_shells>0 gate) ---
    has_epsp = _any_state_has(series, "EPSP")
    # --- Port: OFF (element alive) ---
    has_off = _any_state_has(series, "OFF")

    # Shell stress layers count
    if has_shell_stress_2layer:
        n_shell_layers = 2
    elif has_shell_stress_membrane or has_shell_vonmises_official or has_vonm and n_shells > 0:
        n_shell_layers = 1
    else:
        n_shell_layers = 0

    # ================================================================
    # === Solid stress (n_states, n_solids, 1, 6) ===
    # ================================================================
    solid_stress = None
    solid_stress_fill: list[int] = []
    if has_solid_stress:
        solid_stress = np.zeros((n_states, n_solids, 1, 6), dtype=ftype)
        for i, state in enumerate(series.states):
            if "3DELEM_Stress" in state.cell_data:
                t = state.cell_data["3DELEM_Stress"][model.solid_src]  # (n_solids, 3, 3)
                solid_stress[i, :, 0, :] = _tensor_to_voigt(t).astype(ftype)
            else:
                solid_stress_fill.append(i)
    elif has_vonm and n_solids > 0:
        # C1 fix: port-dialect solids get synthetic stress from VONM
        solid_stress = np.zeros((n_states, n_solids, 1, 6), dtype=ftype)
        warn_list.append(
            "solid stress tensor is synthetic-uniaxial from VONM"
        )
        for i, state in enumerate(series.states):
            if "VONM" in state.cell_data:
                vonm = state.cell_data["VONM"][model.solid_src]
                solid_stress[i, :, 0, 0] = vonm.astype(ftype)
            else:
                solid_stress_fill.append(i)

    # ================================================================
    # === Shell stress (n_states, n_shells, n_layers, 6) ===
    # ================================================================
    shell_stress = None
    shell_stress_fill: list[int] = []
    if has_shell_stress_2layer:
        shell_stress = np.zeros((n_states, n_shells, 2, 6), dtype=ftype)
        for i, state in enumerate(series.states):
            lo_present = "2DELEM_Stress_(lower)" in state.cell_data
            hi_present = "2DELEM_Stress_(upper)" in state.cell_data
            if lo_present:
                t_lo = state.cell_data["2DELEM_Stress_(lower)"][model.shell_src]
                shell_stress[i, :, 0, :] = _tensor_to_voigt(t_lo).astype(ftype)
            if hi_present:
                t_hi = state.cell_data["2DELEM_Stress_(upper)"][model.shell_src]
                shell_stress[i, :, 1, :] = _tensor_to_voigt(t_hi).astype(ftype)
            if not lo_present and not hi_present:
                shell_stress_fill.append(i)
    elif has_shell_stress_membrane:
        # M2 fix: accept membrane as single-layer source
        shell_stress = np.zeros((n_states, n_shells, 1, 6), dtype=ftype)
        warn_list.append(
            "shell stress from 2DELEM_Stress_(membrane) — single layer (element-local coords)"
        )
        for i, state in enumerate(series.states):
            if "2DELEM_Stress_(membrane)" in state.cell_data:
                t_mem = state.cell_data["2DELEM_Stress_(membrane)"][model.shell_src]
                shell_stress[i, :, 0, :] = _tensor_to_voigt(t_mem).astype(ftype)
            else:
                shell_stress_fill.append(i)
    elif has_shell_vonmises_official:
        # M2 fix: fallback to 2DELEM_Von_Mises as synthetic-uniaxial
        shell_stress = np.zeros((n_states, n_shells, 1, 6), dtype=ftype)
        warn_list.append(
            "shell stress tensor is synthetic-uniaxial from 2DELEM_Von_Mises"
        )
        for i, state in enumerate(series.states):
            if "2DELEM_Von_Mises" in state.cell_data:
                vm = state.cell_data["2DELEM_Von_Mises"][model.shell_src]
                shell_stress[i, :, 0, 0] = vm.astype(ftype)
            else:
                shell_stress_fill.append(i)
    elif has_vonm and n_shells > 0:
        shell_stress = np.zeros((n_states, n_shells, 1, 6), dtype=ftype)
        warn_list.append(
            "stress tensor is synthetic-uniaxial from VONM"
        )
        for i, state in enumerate(series.states):
            if "VONM" in state.cell_data:
                vonm = state.cell_data["VONM"][model.shell_src]  # (n_shells,)
                shell_stress[i, :, 0, 0] = vonm.astype(ftype)
            else:
                shell_stress_fill.append(i)

    # ================================================================
    # === Solid plastic strain (n_states, n_solids, 1) ===
    # ================================================================
    solid_eps = None
    solid_eps_fill: list[int] = []
    if has_solid_eps:
        solid_eps = np.zeros((n_states, n_solids, 1), dtype=ftype)
        for i, state in enumerate(series.states):
            if "3DELEM_Plastic_Strain" in state.cell_data:
                eps = state.cell_data["3DELEM_Plastic_Strain"][model.solid_src]
                solid_eps[i, :, 0] = eps.astype(ftype)
            else:
                solid_eps_fill.append(i)
    elif has_epsp and n_solids > 0:
        # C1 fix: port-dialect solids get EPSP
        solid_eps = np.zeros((n_states, n_solids, 1), dtype=ftype)
        for i, state in enumerate(series.states):
            if "EPSP" in state.cell_data:
                solid_eps[i, :, 0] = state.cell_data["EPSP"][model.solid_src].astype(
                    ftype
                )
            else:
                solid_eps_fill.append(i)

    # ================================================================
    # === Shell plastic strain ===
    # ================================================================
    shell_eps = None
    shell_eps_fill: list[int] = []
    if has_shell_eps_2layer:
        shell_eps = np.zeros((n_states, n_shells, 2), dtype=ftype)
        for i, state in enumerate(series.states):
            lo = "2DELEM_Plastic_Strain_Lower" in state.cell_data
            hi = "2DELEM_Plastic_Strain_Upper" in state.cell_data
            if lo:
                shell_eps[i, :, 0] = state.cell_data["2DELEM_Plastic_Strain_Lower"][
                    model.shell_src
                ].astype(ftype)
            if hi:
                shell_eps[i, :, 1] = state.cell_data["2DELEM_Plastic_Strain_Upper"][
                    model.shell_src
                ].astype(ftype)
            if not lo and not hi:
                shell_eps_fill.append(i)
    elif has_shell_eps_single:
        # Replicate single scalar into both layers (official with no Upper/Lower split)
        n_layers = n_shell_layers if n_shell_layers > 0 else 1
        shell_eps = np.zeros((n_states, n_shells, n_layers), dtype=ftype)
        for i, state in enumerate(series.states):
            if "2DELEM_Plastic_Strain" in state.cell_data:
                eps = state.cell_data["2DELEM_Plastic_Strain"][model.shell_src].astype(
                    ftype
                )
                for layer in range(n_layers):
                    shell_eps[i, :, layer] = eps
            else:
                shell_eps_fill.append(i)
    elif has_epsp and n_shells > 0:
        shell_eps = np.zeros((n_states, n_shells, 1), dtype=ftype)
        for i, state in enumerate(series.states):
            if "EPSP" in state.cell_data:
                shell_eps[i, :, 0] = state.cell_data["EPSP"][model.shell_src].astype(
                    ftype
                )
            else:
                shell_eps_fill.append(i)

    # ================================================================
    # === Element alive (from OFF) — M6 fix: added beam alive ===
    # ================================================================
    shell_alive = None
    solid_alive = None
    beam_alive = None
    shell_alive_fill: list[int] = []
    solid_alive_fill: list[int] = []
    beam_alive_fill: list[int] = []
    if has_off:
        if n_shells > 0:
            shell_alive = np.ones((n_states, n_shells), dtype=ftype)
            for i, state in enumerate(series.states):
                if "OFF" in state.cell_data:
                    shell_alive[i] = state.cell_data["OFF"][model.shell_src].astype(
                        ftype
                    )
                else:
                    shell_alive_fill.append(i)
        if n_solids > 0:
            solid_alive = np.ones((n_states, n_solids), dtype=ftype)
            for i, state in enumerate(series.states):
                if "OFF" in state.cell_data:
                    solid_alive[i] = state.cell_data["OFF"][model.solid_src].astype(
                        ftype
                    )
                else:
                    solid_alive_fill.append(i)
        if n_beams > 0:
            beam_alive = np.ones((n_states, n_beams), dtype=ftype)
            for i, state in enumerate(series.states):
                if "OFF" in state.cell_data:
                    beam_alive[i] = state.cell_data["OFF"][model.beam_src].astype(
                        ftype
                    )
                else:
                    beam_alive_fill.append(i)

    # ================================================================
    # M4 fix: emit aggregated zero-fill warnings
    # ================================================================
    _fill_warnings = [
        ("element_solid_stress", solid_stress_fill),
        ("element_shell_stress", shell_stress_fill),
        ("element_solid_effective_plastic_strain", solid_eps_fill),
        ("element_shell_effective_plastic_strain", shell_eps_fill),
        ("element_shell_is_alive", shell_alive_fill),
        ("element_solid_is_alive", solid_alive_fill),
        ("element_beam_is_alive", beam_alive_fill),
    ]
    for array_name, fill_states in _fill_warnings:
        if fill_states:
            warn_list.append(
                f"{array_name}: zero-filled at states {fill_states} "
                f"(source array missing in those states)"
            )

    # ----------------------------------------------------------------
    # Assemble D3plot
    # ----------------------------------------------------------------
    d3 = D3plot()
    a = d3.arrays

    a[ArrayType.node_coordinates] = x0.astype(ftype)
    a[ArrayType.node_ids] = model.node_ids.astype(itype)

    if n_solids > 0:
        a[ArrayType.element_solid_node_indexes] = model.solid_conn.astype(itype)
        a[ArrayType.element_solid_part_indexes] = model.solid_part_idx.astype(itype)
        a[ArrayType.element_solid_ids] = model.solid_ids.astype(itype)

    if n_shells > 0:
        a[ArrayType.element_shell_node_indexes] = model.shell_conn.astype(itype)
        a[ArrayType.element_shell_part_indexes] = model.shell_part_idx.astype(itype)
        a[ArrayType.element_shell_ids] = model.shell_ids.astype(itype)

    if n_beams > 0:
        a[ArrayType.element_beam_node_indexes] = model.beam_conn.astype(itype)
        a[ArrayType.element_beam_part_indexes] = model.beam_part_idx.astype(itype)
        a[ArrayType.element_beam_ids] = model.beam_ids.astype(itype)

    a[ArrayType.part_ids] = model.part_ids.astype(itype)
    a[ArrayType.part_titles_ids] = model.part_ids.astype(itype)
    a[ArrayType.part_titles] = np.array(
        [b"Part %d" % int(pid) for pid in model.part_ids]
    )

    a[ArrayType.global_timesteps] = series.times.astype(ftype)
    a[ArrayType.node_displacement] = node_disp

    if node_vel is not None:
        a[ArrayType.node_velocity] = node_vel

    # Element results
    if solid_stress is not None:
        a[ArrayType.element_solid_stress] = solid_stress
    if shell_stress is not None:
        a[ArrayType.element_shell_stress] = shell_stress
    if solid_eps is not None:
        a[ArrayType.element_solid_effective_plastic_strain] = solid_eps
    if shell_eps is not None:
        a[ArrayType.element_shell_effective_plastic_strain] = shell_eps
    if shell_alive is not None:
        a[ArrayType.element_shell_is_alive] = shell_alive
    if solid_alive is not None:
        a[ArrayType.element_solid_is_alive] = solid_alive
    if beam_alive is not None:
        a[ArrayType.element_beam_is_alive] = beam_alive

    # M5 fix: set the d3plot header title
    if title:
        d3.header.title = title

    if f64:
        d3.header.wordsize = 8
        d3.header.itype = np.int64
        d3.header.ftype = np.float64

    return d3, warn_list
