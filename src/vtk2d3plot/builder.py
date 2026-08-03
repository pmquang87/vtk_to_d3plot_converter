"""d3plot builder — assembles a lasso D3plot from VtkSeries + MappedModel."""

from __future__ import annotations

from collections.abc import Callable

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


# German umlaut transliteration table for d3plot ASCII-only title constraint.
_TRANSLIT = {
    "\u00e4": "ae",
    "\u00f6": "oe",
    "\u00fc": "ue",
    "\u00df": "ss",
    "\u00c4": "Ae",
    "\u00d6": "Oe",
    "\u00dc": "Ue",
}


def _sanitize_title(title: str, wordsize: int = 4) -> tuple[str, list[str]]:
    """Transliterate and truncate a title for the d3plot header.

    lasso 2.0.4 pads the title to 10*wordsize chars then UTF-8-encodes.
    Non-ASCII chars produce more bytes than chars, crashing the writer.
    We transliterate German umlauts (ae/oe/ue/ss), replace remaining
    non-ASCII with '?', and truncate to 10*wordsize - 4 readable chars.

    Returns ``(sanitized_title, list_of_warnings)``.
    """
    warns: list[str] = []

    # Guard: bytes input
    if isinstance(title, bytes):
        title = title.decode("utf-8", errors="replace")

    # Guard: unpaired surrogates
    title = title.encode("utf-8", errors="replace").decode("utf-8")

    # Transliterate German umlauts
    original = title
    for src, dst in _TRANSLIT.items():
        title = title.replace(src, dst)
    if title != original:
        warns.append(f"title: transliterated non-ASCII chars ({original!r} \u2192 {title!r})")

    # Replace remaining non-ASCII with '?'
    ascii_title = title.encode("ascii", errors="replace").decode("ascii")
    if ascii_title != title:
        warns.append("title: replaced remaining non-ASCII chars with '?'")
        title = ascii_title

    # Truncate to readable char budget (10*wordsize - 4 per lasso read-back)
    max_chars = 10 * wordsize - 8 if wordsize == 8 else 10 * wordsize - 4  # 72 for f64, 36 for f32
    if len(title) > max_chars:
        warns.append(
            f"title truncated from {len(title)} to {max_chars} chars "
            f"(d3plot header: {max_chars} readable chars)"
        )
        title = title[:max_chars]

    return title, warns


def build_d3plot(
    series: VtkSeries,
    model: MappedModel,
    *,
    title: str = "",
    f64: bool = False,
    part_names: dict[int, str] | None = None,
    progress: Callable[[int, int], None] | None = None,
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
        if progress is not None:
            progress(i, n_states)
        node_disp[i] = state.points.astype(ftype)

    node_vel = None
    node_vel_fill: list[int] = []
    if any(_find_key(s.point_data, "VELOCITY") for s in series.states):
        node_vel = np.zeros((n_states, n_nodes, 3), dtype=ftype)
        for i, state in enumerate(series.states):
            vk = _find_key(state.point_data, "VELOCITY")
            if vk is not None:
                node_vel[i] = state.point_data[vk].astype(ftype)
            else:
                node_vel_fill.append(i)

    node_accel = None
    node_accel_fill: list[int] = []
    if any(_find_key(s.point_data, "ACCELERATION") for s in series.states):
        node_accel = np.zeros((n_states, n_nodes, 3), dtype=ftype)
        for i, state in enumerate(series.states):
            ak = _find_key(state.point_data, "ACCELERATION")
            if ak is not None:
                node_accel[i] = state.point_data[ak].astype(ftype)
            else:
                node_accel_fill.append(i)

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
    # --- P1b: single-layer-only stress (only upper OR only lower in entire series) ---
    has_shell_stress_upper_only = n_shells > 0 and (
        not has_shell_stress_2layer and _any_state_has(series, "2DELEM_Stress_(upper)")
    )
    has_shell_stress_lower_only = n_shells > 0 and (
        not has_shell_stress_2layer
        and not has_shell_stress_upper_only
        and _any_state_has(series, "2DELEM_Stress_(lower)")
    )
    # --- Official dialect: shell stress (membrane — single layer) (M2 fix) ---
    has_shell_stress_membrane = n_shells > 0 and (
        not has_shell_stress_2layer and _any_state_has(series, "2DELEM_Stress_(membrane)")
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
    # --- P1b: single-layer-only eps ---
    has_shell_eps_upper_only = n_shells > 0 and (
        not has_shell_eps_2layer
        and not _any_state_has(series, "2DELEM_Plastic_Strain")
        and _any_state_has(series, "2DELEM_Plastic_Strain_Upper")
    )
    has_shell_eps_lower_only = n_shells > 0 and (
        not has_shell_eps_2layer
        and not has_shell_eps_upper_only
        and not _any_state_has(series, "2DELEM_Plastic_Strain")
        and _any_state_has(series, "2DELEM_Plastic_Strain_Lower")
    )
    has_shell_eps_single = n_shells > 0 and _any_state_has(series, "2DELEM_Plastic_Strain")
    # --- Port: EPSP (C1 fix: removed n_shells>0 gate) ---
    has_epsp = _any_state_has(series, "EPSP")
    # --- Port: OFF (element alive) ---
    has_off = _any_state_has(series, "OFF")
    # --- Official: EROSION_STATUS (1=alive, 0=eroded) ---
    has_erosion_status = _any_state_has(series, "EROSION_STATUS")
    # --- Official: shell strain tensor (upper/lower layers) — M12 S1: 2-layer wins ---
    has_shell_strain_2layer = n_shells > 0 and (
        _any_state_has(series, "2DELEM_Strain_(lower)")
        and _any_state_has(series, "2DELEM_Strain_(upper)")
    )
    # --- M12 S1: single-layer-only strain (mirror stress _only flags) ---
    has_shell_strain_upper_only = n_shells > 0 and (
        not has_shell_strain_2layer and _any_state_has(series, "2DELEM_Strain_(upper)")
    )
    has_shell_strain_lower_only = n_shells > 0 and (
        not has_shell_strain_2layer
        and not has_shell_strain_upper_only
        and _any_state_has(series, "2DELEM_Strain_(lower)")
    )
    # --- Official: shell strain tensor (membrane — single layer) ---
    # M12 S1: membrane now falls AFTER 2-layer + _only flags
    has_shell_strain_membrane = n_shells > 0 and (
        not has_shell_strain_2layer
        and not has_shell_strain_upper_only
        and not has_shell_strain_lower_only
        and _any_state_has(series, "2DELEM_Strain_(membrane)")
    )
    # --- Official: solid strain tensor ---
    has_solid_strain = n_solids > 0 and _any_state_has(series, "3DELEM_Strain")
    # --- Official: shell thickness ---
    has_shell_thickness = n_shells > 0 and _any_state_has(series, "2DELEM_Thickness")

    # Shell stress layers count (from stress source)
    if has_shell_stress_2layer:
        stress_layers = 2
    elif (
        has_shell_stress_membrane
        or has_shell_stress_upper_only
        or has_shell_stress_lower_only
        or has_shell_vonmises_official
        or (has_vonm and n_shells > 0)
    ):
        stress_layers = 1
    else:
        stress_layers = 0

    # Shell eps layers count (from eps source)
    if has_shell_eps_2layer:
        eps_layers = 2
    elif (
        has_shell_eps_upper_only
        or has_shell_eps_lower_only
        or has_shell_eps_single
        or (has_epsp and n_shells > 0)
    ):
        eps_layers = 1
    else:
        eps_layers = 0

    # Shell strain layers count
    if has_shell_strain_2layer:
        strain_layers = 2
    elif (
        has_shell_strain_upper_only
        or has_shell_strain_lower_only
        or has_shell_strain_membrane
    ):
        strain_layers = 2
    else:
        strain_layers = 0

    # N4 fix: reconcile to one MAXINT — pick max, pad the smaller
    n_shell_layers = max(stress_layers, eps_layers, strain_layers)
    if (stress_layers > 0 or eps_layers > 0 or strain_layers > 0) and (
        (stress_layers > 0 and stress_layers != n_shell_layers)
        or (eps_layers > 0 and eps_layers != n_shell_layers)
        or (strain_layers > 0 and strain_layers != n_shell_layers)
    ):
        warn_list.append(
            f"shell element results have differing integration layer counts "
            f"(stress={stress_layers}, eps={eps_layers}, strain={strain_layers}) "
            f"— reconciled to {n_shell_layers} layers (d3plot MAXINT constraint)"
        )

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
        warn_list.append("solid stress tensor is synthetic-uniaxial from VONM")
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
        shell_stress = np.zeros((n_states, n_shells, n_shell_layers, 6), dtype=ftype)
        for i, state in enumerate(series.states):
            lo_present = "2DELEM_Stress_(lower)" in state.cell_data
            hi_present = "2DELEM_Stress_(upper)" in state.cell_data
            if lo_present:
                t_lo = state.cell_data["2DELEM_Stress_(lower)"][model.shell_src]
                shell_stress[i, :, 0, :] = _tensor_to_voigt(t_lo).astype(ftype)
            else:
                # N1 fix: track partial-layer fill
                shell_stress_fill.append((i, "2DELEM_Stress_(lower)"))
            if hi_present:
                t_hi = state.cell_data["2DELEM_Stress_(upper)"][model.shell_src]
                shell_stress[i, :, 1, :] = _tensor_to_voigt(t_hi).astype(ftype)
            else:
                shell_stress_fill.append((i, "2DELEM_Stress_(upper)"))
        # M1 fix: warn about element-local coordinates
        warn_list.append(
            "shell stress in element-local coordinates (from anim_to_vtk); "
            "von Mises / principal invariants are exact, "
            "but raw component values differ from global frame"
        )
    elif has_shell_stress_membrane:
        # M2 fix: accept membrane as single-layer source
        shell_stress = np.zeros((n_states, n_shells, n_shell_layers, 6), dtype=ftype)
        warn_list.append(
            "shell stress from 2DELEM_Stress_(membrane) — single layer, element-local coordinates; "
            "von Mises / principal invariants are exact"
        )
        for i, state in enumerate(series.states):
            if "2DELEM_Stress_(membrane)" in state.cell_data:
                t_mem = state.cell_data["2DELEM_Stress_(membrane)"][model.shell_src]
                shell_stress[i, :, 0, :] = _tensor_to_voigt(t_mem).astype(ftype)
            else:
                shell_stress_fill.append(i)
    elif has_shell_stress_upper_only:
        # P1b fix: only upper layer exists — map into correct slot
        shell_stress = np.zeros((n_states, n_shells, n_shell_layers, 6), dtype=ftype)
        upper_layer = min(1, n_shell_layers - 1)
        warn_list.append(
            f"only 2DELEM_Stress_(upper) found (lower missing from all states); "
            f"mapped to layer {upper_layer}, element-local coordinates"
        )
        for i, state in enumerate(series.states):
            if "2DELEM_Stress_(upper)" in state.cell_data:
                t = state.cell_data["2DELEM_Stress_(upper)"][model.shell_src]
                shell_stress[i, :, upper_layer, :] = _tensor_to_voigt(t).astype(ftype)
            else:
                shell_stress_fill.append(i)
    elif has_shell_stress_lower_only:
        # P1b fix: only lower layer exists — map into layer 0
        shell_stress = np.zeros((n_states, n_shells, n_shell_layers, 6), dtype=ftype)
        warn_list.append(
            "only 2DELEM_Stress_(lower) found (upper missing from all states); "
            "mapped to layer 0, element-local coordinates"
        )
        for i, state in enumerate(series.states):
            if "2DELEM_Stress_(lower)" in state.cell_data:
                t = state.cell_data["2DELEM_Stress_(lower)"][model.shell_src]
                shell_stress[i, :, 0, :] = _tensor_to_voigt(t).astype(ftype)
            else:
                shell_stress_fill.append(i)
    elif has_shell_vonmises_official:
        # M2 fix: fallback to 2DELEM_Von_Mises as synthetic-uniaxial
        shell_stress = np.zeros((n_states, n_shells, n_shell_layers, 6), dtype=ftype)
        warn_list.append("shell stress tensor is synthetic-uniaxial from 2DELEM_Von_Mises")
        for i, state in enumerate(series.states):
            if "2DELEM_Von_Mises" in state.cell_data:
                vm = state.cell_data["2DELEM_Von_Mises"][model.shell_src]
                shell_stress[i, :, 0, 0] = vm.astype(ftype)
            else:
                shell_stress_fill.append(i)
    elif has_vonm and n_shells > 0:
        shell_stress = np.zeros((n_states, n_shells, n_shell_layers, 6), dtype=ftype)
        warn_list.append("stress tensor is synthetic-uniaxial from VONM")
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
                solid_eps[i, :, 0] = state.cell_data["EPSP"][model.solid_src].astype(ftype)
            else:
                solid_eps_fill.append(i)

    # ================================================================
    # === Shell plastic strain ===
    # ================================================================
    shell_eps = None
    shell_eps_fill: list[int] = []
    if has_shell_eps_2layer:
        shell_eps = np.zeros((n_states, n_shells, n_shell_layers), dtype=ftype)
        for i, state in enumerate(series.states):
            lo = "2DELEM_Plastic_Strain_Lower" in state.cell_data
            hi = "2DELEM_Plastic_Strain_Upper" in state.cell_data
            if lo:
                shell_eps[i, :, 0] = state.cell_data["2DELEM_Plastic_Strain_Lower"][
                    model.shell_src
                ].astype(ftype)
            else:
                shell_eps_fill.append((i, "2DELEM_Plastic_Strain_Lower"))
            if hi:
                shell_eps[i, :, 1] = state.cell_data["2DELEM_Plastic_Strain_Upper"][
                    model.shell_src
                ].astype(ftype)
            else:
                shell_eps_fill.append((i, "2DELEM_Plastic_Strain_Upper"))
    elif has_shell_eps_upper_only:
        # P1b fix: only upper eps exists — map into correct slot
        shell_eps = np.zeros(
            (n_states, n_shells, n_shell_layers if n_shell_layers > 0 else 1), dtype=ftype
        )
        upper_eps_layer = min(1, (n_shell_layers if n_shell_layers > 0 else 1) - 1)
        warn_list.append(
            f"only 2DELEM_Plastic_Strain_Upper found (Lower missing from all states); "
            f"mapped to layer {upper_eps_layer}"
        )
        for i, state in enumerate(series.states):
            if "2DELEM_Plastic_Strain_Upper" in state.cell_data:
                shell_eps[i, :, upper_eps_layer] = state.cell_data["2DELEM_Plastic_Strain_Upper"][
                    model.shell_src
                ].astype(ftype)
            else:
                shell_eps_fill.append(i)
    elif has_shell_eps_lower_only:
        # P1b fix: only lower eps exists — map into layer 0
        shell_eps = np.zeros(
            (n_states, n_shells, n_shell_layers if n_shell_layers > 0 else 1), dtype=ftype
        )
        warn_list.append(
            "only 2DELEM_Plastic_Strain_Lower found (Upper missing from all states); "
            "mapped to layer 0"
        )
        for i, state in enumerate(series.states):
            if "2DELEM_Plastic_Strain_Lower" in state.cell_data:
                shell_eps[i, :, 0] = state.cell_data["2DELEM_Plastic_Strain_Lower"][
                    model.shell_src
                ].astype(ftype)
            else:
                shell_eps_fill.append(i)
    elif has_shell_eps_single:
        # Replicate single scalar into both layers (official with no Upper/Lower split)
        n_layers = n_shell_layers if n_shell_layers > 0 else 1
        shell_eps = np.zeros((n_states, n_shells, n_layers), dtype=ftype)
        for i, state in enumerate(series.states):
            if "2DELEM_Plastic_Strain" in state.cell_data:
                eps = state.cell_data["2DELEM_Plastic_Strain"][model.shell_src].astype(ftype)
                for layer in range(n_layers):
                    shell_eps[i, :, layer] = eps
            else:
                shell_eps_fill.append(i)
    elif has_epsp and n_shells > 0:
        n_layers = n_shell_layers if n_shell_layers > 0 else 1
        shell_eps = np.zeros((n_states, n_shells, n_layers), dtype=ftype)
        for i, state in enumerate(series.states):
            if "EPSP" in state.cell_data:
                eps_val = state.cell_data["EPSP"][model.shell_src].astype(ftype)
                for layer in range(n_layers):
                    shell_eps[i, :, layer] = eps_val
            else:
                shell_eps_fill.append(i)

    # ================================================================
    # === Shell strain tensor (n_states, n_shells, 2, 6) ===
    # ================================================================
    # M12 S1: 2-layer wins over membrane (mirror stress precedence)
    shell_strain = None
    shell_strain_fill: list = []
    if has_shell_strain_2layer:
        n_layers = max(1, n_shell_layers)
        upper_layer = min(1, n_layers - 1)
        shell_strain = np.zeros((n_states, n_shells, n_layers, 6), dtype=ftype)
        for i, state in enumerate(series.states):
            lo_present = "2DELEM_Strain_(lower)" in state.cell_data
            hi_present = "2DELEM_Strain_(upper)" in state.cell_data
            if lo_present:
                t = state.cell_data["2DELEM_Strain_(lower)"][model.shell_src]
                shell_strain[i, :, 0, :] = _tensor_to_voigt(t).astype(ftype)
            else:
                shell_strain_fill.append((i, "2DELEM_Strain_(lower)"))
            if hi_present:
                t = state.cell_data["2DELEM_Strain_(upper)"][model.shell_src]
                shell_strain[i, :, upper_layer, :] = _tensor_to_voigt(t).astype(ftype)
            else:
                shell_strain_fill.append((i, "2DELEM_Strain_(upper)"))
        # M12: local-frame warning for strain (M1 class)
        warn_list.append(
            "shell strain in element-local coordinates (from anim_to_vtk); "
            "principal invariants are exact, "
            "but raw component values differ from global frame"
        )
    elif has_shell_strain_upper_only:
        # M12 S1: only upper strain exists — duplicate into both layers
        n_layers = max(1, n_shell_layers)
        upper_layer = min(1, n_layers - 1)
        shell_strain = np.zeros((n_states, n_shells, n_layers, 6), dtype=ftype)
        warn_list.append(
            "only 2DELEM_Strain_(upper) found (lower missing from all states); "
            "duplicated to both layers. "
            "shell strain in element-local coordinates (from anim_to_vtk); "
            "principal invariants are exact, "
            "but raw component values differ from global frame"
        )
        for i, state in enumerate(series.states):
            if "2DELEM_Strain_(upper)" in state.cell_data:
                t = state.cell_data["2DELEM_Strain_(upper)"][model.shell_src]
                voigt = _tensor_to_voigt(t).astype(ftype)
                shell_strain[i, :, 0, :] = voigt
                shell_strain[i, :, upper_layer, :] = voigt
            else:
                shell_strain_fill.append(i)
    elif has_shell_strain_lower_only:
        # M12 S1: only lower strain exists — duplicate into both layers
        n_layers = max(1, n_shell_layers)
        upper_layer = min(1, n_layers - 1)
        shell_strain = np.zeros((n_states, n_shells, n_layers, 6), dtype=ftype)
        warn_list.append(
            "only 2DELEM_Strain_(lower) found (upper missing from all states); "
            "duplicated to both layers. "
            "shell strain in element-local coordinates (from anim_to_vtk); "
            "principal invariants are exact, "
            "but raw component values differ from global frame"
        )
        for i, state in enumerate(series.states):
            if "2DELEM_Strain_(lower)" in state.cell_data:
                t = state.cell_data["2DELEM_Strain_(lower)"][model.shell_src]
                voigt = _tensor_to_voigt(t).astype(ftype)
                shell_strain[i, :, 0, :] = voigt
                shell_strain[i, :, upper_layer, :] = voigt
            else:
                shell_strain_fill.append(i)
    elif has_shell_strain_membrane:
        n_layers = max(1, n_shell_layers)
        upper_layer = min(1, n_layers - 1)
        shell_strain = np.zeros((n_states, n_shells, n_layers, 6), dtype=ftype)
        warn_list.append(
            "shell strain from 2DELEM_Strain_(membrane) — single layer duplicated to both layers. "
            "shell strain in element-local coordinates (from anim_to_vtk); "
            "principal invariants are exact, "
            "but raw component values differ from global frame"
        )
        for i, state in enumerate(series.states):
            if "2DELEM_Strain_(membrane)" in state.cell_data:
                t = state.cell_data["2DELEM_Strain_(membrane)"][model.shell_src]
                voigt = _tensor_to_voigt(t).astype(ftype)
                shell_strain[i, :, 0, :] = voigt
                shell_strain[i, :, upper_layer, :] = voigt
            else:
                shell_strain_fill.append(i)

    # ================================================================
    # === Solid strain tensor (n_states, n_solids, 1, 6) ===
    # ================================================================
    solid_strain = None
    solid_strain_fill: list[int] = []
    if has_solid_strain:
        solid_strain = np.zeros((n_states, n_solids, 1, 6), dtype=ftype)
        for i, state in enumerate(series.states):
            if "3DELEM_Strain" in state.cell_data:
                t = state.cell_data["3DELEM_Strain"][model.solid_src]  # (n_solids, 3, 3)
                solid_strain[i, :, 0, :] = _tensor_to_voigt(t).astype(ftype)
            else:
                solid_strain_fill.append(i)

    # ================================================================
    # === Shell thickness (n_states, n_shells) ===
    # ================================================================
    shell_thickness = None
    shell_thickness_fill: list[int] = []
    if has_shell_thickness:
        shell_thickness = np.zeros((n_states, n_shells), dtype=ftype)
        for i, state in enumerate(series.states):
            if "2DELEM_Thickness" in state.cell_data:
                shell_thickness[i, :] = state.cell_data["2DELEM_Thickness"][model.shell_src].astype(
                    ftype
                )
            else:
                shell_thickness_fill.append(i)

    # ================================================================
    # === Element alive (from OFF or EROSION_STATUS) — M6 fix: added beam alive ===
    # ================================================================
    shell_alive = None
    solid_alive = None
    beam_alive = None
    shell_alive_fill: list[int] = []
    solid_alive_fill: list[int] = []
    beam_alive_fill: list[int] = []

    # Determine alive source: OFF (port dialect) or EROSION_STATUS (official dialect)
    alive_key = None
    if has_off and has_erosion_status:
        warn_list.append(
            "both OFF and EROSION_STATUS present; using OFF (port dialect takes priority)"
        )
    if has_off:
        alive_key = "OFF"
    elif has_erosion_status:
        alive_key = "EROSION_STATUS"
        warn_list.append(
            "element is_alive from EROSION_STATUS (1=alive, 0=eroded); "
            "deleted elements will be hidden in LS-PrePost"
        )

    if alive_key is not None:
        if n_shells > 0:
            shell_alive = np.ones((n_states, n_shells), dtype=ftype)
            for i, state in enumerate(series.states):
                if alive_key in state.cell_data:
                    shell_alive[i] = state.cell_data[alive_key][model.shell_src].astype(ftype)
                else:
                    shell_alive_fill.append(i)
        if n_solids > 0:
            solid_alive = np.ones((n_states, n_solids), dtype=ftype)
            for i, state in enumerate(series.states):
                if alive_key in state.cell_data:
                    solid_alive[i] = state.cell_data[alive_key][model.solid_src].astype(ftype)
                else:
                    solid_alive_fill.append(i)
        if n_beams > 0:
            beam_alive = np.ones((n_states, n_beams), dtype=ftype)
            for i, state in enumerate(series.states):
                if alive_key in state.cell_data:
                    beam_alive[i] = state.cell_data[alive_key][model.beam_src].astype(ftype)
                else:
                    beam_alive_fill.append(i)

        if alive_key == "EROSION_STATUS":
            for alive_arr, src_arr, fill_list, type_name in [
                (shell_alive, model.shell_src, shell_alive_fill, "shell"),
                (solid_alive, model.solid_src, solid_alive_fill, "solid"),
                (beam_alive, model.beam_src, beam_alive_fill, "beam"),
            ]:
                if alive_arr is None:
                    continue
                # M12 fix: compute ever_alive only over states with real data
                # (not filled states which default to 1.0)
                fill_set = set(fill_list)
                present_mask = np.array(
                    [si not in fill_set for si in range(n_states)], dtype=bool
                )
                if present_mask.any():
                    ever_alive = np.any(alive_arr[present_mask] == 1.0, axis=0)
                else:
                    ever_alive = np.ones(alive_arr.shape[1], dtype=bool)
                never_alive_mask = ~ever_alive  # elements with EROSION_STATUS=0 in every present state
                n_never = int(never_alive_mask.sum())
                if n_never > 0:
                    # Force never-alive elements to alive (they are rigid/non-integrated)
                    alive_arr[:, never_alive_mask] = 1.0
                    # Find which part IDs these elements belong to
                    pid_data = series.states[0].cell_data.get("PART_ID")
                    if pid_data is not None:
                        affected_pids = sorted(
                            {
                                int(pid_data[src_arr[i]])
                                for i in range(len(src_arr))
                                if never_alive_mask[i]
                            }
                        )
                        warn_list.append(
                            f"EROSION_STATUS: {n_never} {type_name} elements never alive in any state "
                            f"(parts {affected_pids}); forced alive (rigid/non-integrated)"
                        )
                    else:
                        warn_list.append(
                            f"EROSION_STATUS: {n_never} {type_name} elements never alive in any state; "
                            f"forced alive (rigid/non-integrated)"
                        )

                # Step 2: Latch 1→0 transitions (element dies permanently, no resurrection)
                for si in range(1, n_states):
                    # Elements that were dead at previous state stay dead
                    was_dead = alive_arr[si - 1] == 0.0
                    alive_arr[si, was_dead] = 0.0

    # ================================================================
    # M4/N1 fix: emit aggregated fill warnings (partial-layer aware)
    # ================================================================
    # Simple fills (whole state missing) — list[int]
    _simple_fills = [
        ("element_solid_stress", solid_stress_fill),
        ("element_solid_effective_plastic_strain", solid_eps_fill),
        ("element_solid_strain", solid_strain_fill),
        ("element_shell_thickness", shell_thickness_fill),
        ("node_velocity", node_vel_fill),
        ("node_acceleration", node_accel_fill),
    ]
    for array_name, fill_states in _simple_fills:
        if fill_states:
            warn_list.append(
                f"{array_name}: zero-filled at states {fill_states} "
                f"(source array missing in those states)"
            )

    # N1: partial-layer fills — list[int] or list[tuple[int, str]]
    _layer_fills = [
        ("element_shell_stress", shell_stress_fill),
        ("element_shell_effective_plastic_strain", shell_eps_fill),
        ("element_shell_strain", shell_strain_fill),
    ]
    for array_name, fill_entries in _layer_fills:
        if not fill_entries:
            continue
        # Entries may be int (whole state) or (int, layer_name) tuples
        if isinstance(fill_entries[0], tuple):
            by_layer: dict[str, list[int]] = {}
            for state_idx, layer_name in fill_entries:
                by_layer.setdefault(layer_name, []).append(state_idx)
            for layer_name, states in sorted(by_layer.items()):
                warn_list.append(
                    f"{array_name}: {layer_name} zero-filled at states {states} "
                    f"(source array missing in those states)"
                )
        else:
            warn_list.append(
                f"{array_name}: zero-filled at states {fill_entries} "
                f"(source array missing in those states)"
            )

    # N5 fix: is_alive arrays are ONE-filled (all-alive), not zero-filled
    _alive_fills = [
        ("element_shell_is_alive", shell_alive_fill),
        ("element_solid_is_alive", solid_alive_fill),
        ("element_beam_is_alive", beam_alive_fill),
    ]
    for array_name, fill_states in _alive_fills:
        if fill_states:
            warn_list.append(
                f"{array_name}: one-filled (all-alive) at states {fill_states} "
                f"(source {alive_key} array missing in those states)"
            )

    # M7: phantom cell masking warning
    if model.n_phantom_masked > 0:
        warn_list.append(
            f"Masked {model.n_phantom_masked} cells referencing zero-ID nodes "
            f"(phantom geometry, matching Vortex behavior)"
        )

    # ================================================================
    # M9: unmapped-array warnings
    # ================================================================
    _RECOGNIZED_POINT = {
        "DISPLACEMENT",
        "Displacement",
        "VELOCITY",
        "Velocity",
        "ACCELERATION",
        "Acceleration",
        "NODE_ID",
    }
    _RECOGNIZED_CELL = {
        "ELEMENT_ID",
        "PART_ID",
        "VONM",
        "EPSP",
        "OFF",
        "EROSION_STATUS",
        "3DELEM_Stress",
        "3DELEM_Strain",
        "3DELEM_Plastic_Strain",
        "2DELEM_Stress_(lower)",
        "2DELEM_Stress_(upper)",
        "2DELEM_Stress_(membrane)",
        "2DELEM_Plastic_Strain",
        "2DELEM_Plastic_Strain_Upper",
        "2DELEM_Plastic_Strain_Lower",
        "2DELEM_Von_Mises",
        "2DELEM_Strain_(membrane)",
        "2DELEM_Strain_(lower)",
        "2DELEM_Strain_(upper)",
        "2DELEM_Thickness",
    }
    unmapped_cell: dict[str, float] = {}
    unmapped_point: dict[str, float] = {}
    for state in series.states:
        for k, v in state.cell_data.items():
            if k not in _RECOGNIZED_CELL:
                try:
                    val = float(np.abs(v).max())
                except (TypeError, ValueError):
                    val = 0.0
                unmapped_cell[k] = max(unmapped_cell.get(k, 0.0), val)
        for k, v in state.point_data.items():
            if k not in _RECOGNIZED_POINT:
                try:
                    val = float(np.abs(v).max())
                except (TypeError, ValueError):
                    val = 0.0
                unmapped_point[k] = max(unmapped_point.get(k, 0.0), val)
    all_unmapped = {**unmapped_point, **unmapped_cell}
    if all_unmapped:
        items = ", ".join(f"{k} (absmax={v:.4g})" for k, v in sorted(all_unmapped.items()))
        warn_list.append(f"unmapped arrays ignored: {items}")

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
    if part_names:
        titles_list = []
        for pid in model.part_ids:
            name = part_names.get(int(pid), f"Part {int(pid)}")
            # Transliterate part names for ASCII safety (lasso encodes as UTF-8 bytes)
            name_clean = name
            for src, dst in _TRANSLIT.items():
                name_clean = name_clean.replace(src, dst)
            name_clean = name_clean.encode("ascii", errors="replace").decode("ascii")
            titles_list.append(name_clean.encode("utf-8"))
        a[ArrayType.part_titles] = np.array(titles_list)
    else:
        a[ArrayType.part_titles] = np.array([b"Part %d" % int(pid) for pid in model.part_ids])

    a[ArrayType.global_timesteps] = series.times.astype(ftype)
    a[ArrayType.node_displacement] = node_disp

    if node_vel is not None:
        a[ArrayType.node_velocity] = node_vel
    if node_accel is not None:
        a[ArrayType.node_acceleration] = node_accel

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
    if shell_strain is not None:
        a[ArrayType.element_shell_strain] = shell_strain
    if solid_strain is not None:
        a[ArrayType.element_solid_strain] = solid_strain
    if shell_thickness is not None:
        a[ArrayType.element_shell_thickness] = shell_thickness

    if shell_strain is not None or solid_strain is not None:
        d3.header.has_element_strain = True

    # M5 fix: set the d3plot header title
    # M9 fix: transliterate non-ASCII, then truncate to readable char budget.
    # lasso's pack() does "{0:N}".format(title).encode("utf-8") where N=10*wordsize.
    # Any non-ASCII char makes encoded bytes > N, crashing the writer.
    # Solution: transliterate German umlauts, replace remaining non-ASCII with '?',
    # then truncate to readable chars (10*wordsize - 4).
    if title:
        wordsize_val = 8 if f64 else 4
        title, title_warns = _sanitize_title(title, wordsize_val)
        warn_list.extend(title_warns)
        if title:
            d3.header.title = title

    if f64:
        d3.header.wordsize = 8
        d3.header.itype = np.int64
        d3.header.ftype = np.float64

    return d3, warn_list
