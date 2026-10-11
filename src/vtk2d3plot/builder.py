"""d3plot builder — assembles a lasso D3plot from VtkSeries + MappedModel."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from lasso.dyna import ArrayType, D3plot

from .mapping import MappedModel
from .series import VtkSeries
from .shell_rotation import compute_shell_local_frames, rotate_stress_to_global


def _find_key(d: dict, name: str) -> str | None:
    """Case-insensitive key lookup in a dict — the ONE lookup rule for all gathers.

    Presence flags (``_any_state_has``) ignore case, so every gather must find
    the array the same way (bug report 2026-10-11 #5).  An exact-case match
    wins; otherwise the case-insensitive matches are taken in sorted order, so
    the choice is deterministic.  When several names differ only in case,
    ``_case_collision_warnings`` reports which one was used.
    """
    if name in d:
        return name
    low = name.lower()
    matches = sorted(k for k in d if k.lower() == low)
    return matches[0] if matches else None


def _cell_array(state, name: str) -> np.ndarray | None:
    """Cell array ``name`` of one state (case-insensitive, see ``_find_key``), or None."""
    k = _find_key(state.cell_data, name)
    return None if k is None else state.cell_data[k]


# Canonical names the builder gathers (cell / point data).  Used only to report
# names that differ only in case within one state (no silent mixing).
_GATHERED_CELL_NAMES = (
    "VONM", "EPSP", "OFF", "EROSION_STATUS",
    "3DELEM_Stress", "3DELEM_Strain", "3DELEM_Plastic_Strain", "3DELEM_Von_Mises",
    "3DELEM_Specific_Energy",
    "2DELEM_Stress_(lower)", "2DELEM_Stress_(upper)", "2DELEM_Stress_(membrane)",
    "2DELEM_Plastic_Strain", "2DELEM_Plastic_Strain_Upper", "2DELEM_Plastic_Strain_Lower",
    "2DELEM_Von_Mises", "2DELEM_Strain_(membrane)", "2DELEM_Strain_(lower)",
    "2DELEM_Strain_(upper)", "2DELEM_Thickness", "2DELEM_Specific_Energy",
)  # fmt: skip
_GATHERED_POINT_NAMES = ("DISPLACEMENT", "VELOCITY", "ACCELERATION")


def _case_collision_warnings(series: VtkSeries) -> list[str]:
    """Warn when one state holds several gathered arrays whose names differ only in case."""
    found: dict[tuple[str, tuple[str, ...], str, str], list[int]] = {}
    for i, state in enumerate(series.states):
        for kind, d, names in (
            ("cell", state.cell_data, _GATHERED_CELL_NAMES),
            ("point", state.point_data, _GATHERED_POINT_NAMES),
        ):
            for name in names:
                low = name.lower()
                keys = tuple(sorted(k for k in d if k.lower() == low))
                if len(keys) > 1:
                    used = _find_key(d, name)
                    how = "exact name" if used == name else "first in sorted order"
                    found.setdefault((kind, keys, used, how), []).append(i)
    warns = []
    for (kind, keys, used, how), states in found.items():
        names = ", ".join(f"'{k}'" for k in keys)
        warns.append(
            f"{kind} arrays {names} differ only in case at states {states}; "
            f"using '{used}' ({how}), the others are ignored"
        )
    return warns


def _tensor_to_voigt(t: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Convert (n, 3, 3) symmetric tensor to (n, 6) Voigt: xx yy zz xy xz yz.

    The binary d3plot layout for stress slots 3-5 is [σxy, σxz, σyz].
    This matches the extraction order [t[0,1], t[0,2], t[1,2]].
    Root cause: anim_to_vtk.cpp (OpenRadioss) writes its 3×3 with yz at
    position (0,2) and zx at (1,2), so the Voigt extraction here produces
    the correct slot assignment.  Verified against Vortex golden d3plot.
    """
    if out is None:
        out = np.empty((t.shape[0], 6), dtype=t.dtype)
    out[:, 0] = t[:, 0, 0]
    out[:, 1] = t[:, 1, 1]
    out[:, 2] = t[:, 2, 2]
    out[:, 3] = t[:, 0, 1]
    out[:, 4] = t[:, 0, 2]
    out[:, 5] = t[:, 1, 2]
    return out


def _any_state_has(series: VtkSeries, key: str, data_kind: str = "cell") -> bool:
    """Check if any state in the series contains the given key (M3 fix)."""
    low_key = key.lower()
    for state in series.states:
        d = state.cell_data if data_kind == "cell" else state.point_data
        for k in d:
            if k.lower() == low_key:
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


def _build_kinematics(series, n_nodes, ftype, progress, n_states, get_point_key, warn_list):
    # Geometry recovery: x0 = POINTS - DISPLACEMENT (case-insensitive)
    x0 = None
    for i, state in enumerate(series.states):
        dk = get_point_key(i, "DISPLACEMENT")
        if dk is not None:
            x0 = state.points - state.point_data[dk]
            break
    if x0 is None:
        x0 = series.states[0].points
        warn_list.append(
            "No DISPLACEMENT array found; using state 0 coordinates as undeformed geometry"
        )

    # Kinematics
    node_disp = np.empty((n_states, n_nodes, 3), dtype=ftype)
    for i, state in enumerate(series.states):
        if progress is not None:
            progress(i, n_states)
        node_disp[i] = state.points.astype(ftype)

    node_vel = None
    node_vel_fill: list[int] = []
    if any(get_point_key(i, "VELOCITY") for i, s in enumerate(series.states)):
        node_vel = np.zeros((n_states, n_nodes, 3), dtype=ftype)
        for i, state in enumerate(series.states):
            vk = get_point_key(i, "VELOCITY")
            if vk is not None:
                node_vel[i] = state.point_data[vk].astype(ftype)
            else:
                node_vel_fill.append(i)

    node_accel = None
    node_accel_fill: list[int] = []
    if any(get_point_key(i, "ACCELERATION") for i, s in enumerate(series.states)):
        node_accel = np.zeros((n_states, n_nodes, 3), dtype=ftype)
        for i, state in enumerate(series.states):
            ak = get_point_key(i, "ACCELERATION")
            if ak is not None:
                node_accel[i] = state.point_data[ak].astype(ftype)
            else:
                node_accel_fill.append(i)

    return x0, node_disp, node_vel, node_accel, node_vel_fill, node_accel_fill

def _build_solid_results(
    series, model, ftype, n_states, n_solids, warn_list,
    has_solid_stress, has_vonm, has_solid_vonmises,
    has_solid_eps, has_epsp,
    has_solid_internal_energy, has_solid_strain, _tensor_to_voigt
):
    # ================================================================
    # === Solid stress (n_states, n_solids, 1, 6) ===
    # ================================================================
    solid_stress = None
    solid_stress_fill: list[int] = []
    if has_solid_stress:
        solid_stress = np.zeros((n_states, n_solids, 1, 6), dtype=ftype)
        for i, state in enumerate(series.states):
            arr = _cell_array(state, "3DELEM_Stress")
            if arr is not None:
                t = arr[model.solid_src]  # (n_solids, 3, 3)
                _tensor_to_voigt(t, out=solid_stress[i, :, 0, :])
            else:
                solid_stress_fill.append(i)
    elif has_vonm and n_solids > 0:
        # C1 fix: port-dialect solids get synthetic stress from VONM
        solid_stress = np.zeros((n_states, n_solids, 1, 6), dtype=ftype)
        warn_list.append("solid stress tensor is synthetic-uniaxial from VONM")
        for i, state in enumerate(series.states):
            arr = _cell_array(state, "VONM")
            if arr is not None:
                vonm = arr[model.solid_src]
                solid_stress[i, :, 0, 0] = vonm.astype(ftype)
            else:
                solid_stress_fill.append(i)
    elif has_solid_vonmises:
        # Official dialect: 3DELEM_Von_Mises scalar → synthetic-uniaxial solid stress
        solid_stress = np.zeros((n_states, n_solids, 1, 6), dtype=ftype)
        warn_list.append("solid stress tensor is synthetic-uniaxial from 3DELEM_Von_Mises")
        for i, state in enumerate(series.states):
            arr = _cell_array(state, "3DELEM_Von_Mises")
            if arr is not None:
                vonm = arr[model.solid_src]
                solid_stress[i, :, 0, 0] = vonm.astype(ftype)
            else:
                solid_stress_fill.append(i)

    # ================================================================
    # === Solid plastic strain (n_states, n_solids, 1) ===
    # ================================================================
    solid_eps = None
    solid_eps_fill: list[int] = []
    if has_solid_eps:
        solid_eps = np.zeros((n_states, n_solids, 1), dtype=ftype)
        for i, state in enumerate(series.states):
            arr = _cell_array(state, "3DELEM_Plastic_Strain")
            if arr is not None:
                eps = arr[model.solid_src]
                solid_eps[i, :, 0] = eps.astype(ftype)
            else:
                solid_eps_fill.append(i)
    elif has_epsp and n_solids > 0:
        # C1 fix: port-dialect solids get EPSP
        solid_eps = np.zeros((n_states, n_solids, 1), dtype=ftype)
        for i, state in enumerate(series.states):
            arr = _cell_array(state, "EPSP")
            if arr is not None:
                solid_eps[i, :, 0] = arr[model.solid_src].astype(ftype)
            else:
                solid_eps_fill.append(i)

    # ================================================================
    # === Solid internal energy (mapped to history variables) ===
    # ================================================================
    solid_history_vars = None
    solid_history_vars_fill: list[int] = []
    if has_solid_internal_energy:
        warn_list.append(
            "solid internal energy mapped to element_solid_history_variables (index 0)"
        )
        solid_history_vars = np.zeros((n_states, n_solids, 1, 1), dtype=ftype)
        for i, state in enumerate(series.states):
            arr = _cell_array(state, "3DELEM_Specific_Energy")
            if arr is not None:
                ie = arr[model.solid_src]
                solid_history_vars[i, :, 0, 0] = ie.astype(ftype)
            else:
                solid_history_vars_fill.append(i)

    # ================================================================
    # === Solid strain tensor (n_states, n_solids, 1, 6) ===
    # ================================================================
    solid_strain = None
    solid_strain_fill: list[int] = []
    if has_solid_strain:
        solid_strain = np.zeros((n_states, n_solids, 1, 6), dtype=ftype)
        for i, state in enumerate(series.states):
            arr = _cell_array(state, "3DELEM_Strain")
            if arr is not None:
                t = arr[model.solid_src]  # (n_solids, 3, 3)
                _tensor_to_voigt(t, out=solid_strain[i, :, 0, :])
            else:
                solid_strain_fill.append(i)


    return solid_stress, solid_stress_fill, solid_eps, solid_eps_fill, solid_history_vars, solid_history_vars_fill, solid_strain, solid_strain_fill

def _build_shell_results(
    series, model, ftype, n_states, n_shells, n_shell_layers, warn_list, rotate_shell_stress,
    has_shell_stress_2layer, has_shell_stress_membrane, has_shell_stress_upper_only,
    has_shell_stress_lower_only, has_shell_vonmises_official, has_vonm,
    has_shell_eps_2layer, has_shell_eps_upper_only, has_shell_eps_lower_only,
    has_shell_eps_single, has_epsp,
    has_shell_internal_energy,
    has_shell_strain_2layer, has_shell_strain_upper_only, has_shell_strain_lower_only,
    has_shell_strain_membrane,
    has_shell_thickness, _tensor_to_voigt, rotate_stress_to_global, compute_shell_local_frames
):
    # ================================================================
    # === Shell stress (n_states, n_shells, n_layers, 6) ===
    # ================================================================
    shell_stress = None
    shell_stress_fill: list[int] = []
    if has_shell_stress_2layer:
        shell_stress = np.zeros((n_states, n_shells, n_shell_layers, 6), dtype=ftype)
        for i, state in enumerate(series.states):
            arr_lo = _cell_array(state, "2DELEM_Stress_(lower)")
            arr_hi = _cell_array(state, "2DELEM_Stress_(upper)")
            if arr_lo is not None:
                t_lo = arr_lo[model.shell_src]
                _tensor_to_voigt(t_lo, out=shell_stress[i, :, 0, :])
            else:
                # N1 fix: track partial-layer fill
                shell_stress_fill.append((i, "2DELEM_Stress_(lower)"))
            if arr_hi is not None:
                t_hi = arr_hi[model.shell_src]
                _tensor_to_voigt(t_hi, out=shell_stress[i, :, 1, :])
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
            arr = _cell_array(state, "2DELEM_Stress_(membrane)")
            if arr is not None:
                t_mem = arr[model.shell_src]
                _tensor_to_voigt(t_mem, out=shell_stress[i, :, 0, :])
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
            arr = _cell_array(state, "2DELEM_Stress_(upper)")
            if arr is not None:
                t = arr[model.shell_src]
                _tensor_to_voigt(t, out=shell_stress[i, :, upper_layer, :])
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
            arr = _cell_array(state, "2DELEM_Stress_(lower)")
            if arr is not None:
                t = arr[model.shell_src]
                _tensor_to_voigt(t, out=shell_stress[i, :, 0, :])
            else:
                shell_stress_fill.append(i)
    elif has_shell_vonmises_official:
        # M2 fix: fallback to 2DELEM_Von_Mises as synthetic-uniaxial
        shell_stress = np.zeros((n_states, n_shells, n_shell_layers, 6), dtype=ftype)
        warn_list.append("shell stress tensor is synthetic-uniaxial from 2DELEM_Von_Mises")
        for i, state in enumerate(series.states):
            arr = _cell_array(state, "2DELEM_Von_Mises")
            if arr is not None:
                vm = arr[model.shell_src]
                shell_stress[i, :, 0, 0] = vm.astype(ftype)
            else:
                shell_stress_fill.append(i)
    elif has_vonm and n_shells > 0:
        shell_stress = np.zeros((n_states, n_shells, n_shell_layers, 6), dtype=ftype)
        warn_list.append("stress tensor is synthetic-uniaxial from VONM")
        for i, state in enumerate(series.states):
            arr = _cell_array(state, "VONM")
            if arr is not None:
                vonm = arr[model.shell_src]  # (n_shells,)
                shell_stress[i, :, 0, 0] = vonm.astype(ftype)
            else:
                shell_stress_fill.append(i)

    if shell_stress is not None and rotate_shell_stress:
        for i, state in enumerate(series.states):
            R = compute_shell_local_frames(state.points, model.shell_conn)
            shell_stress[i] = rotate_stress_to_global(shell_stress[i], R)

        for w_idx in range(len(warn_list)):
            if "shell stress" in warn_list[w_idx] or "2DELEM_Stress" in warn_list[w_idx]:
                warn_list[w_idx] = (
                    warn_list[w_idx]
                    .replace(
                        "element-local coordinates (from anim_to_vtk); von Mises / principal invariants are exact, but raw component values differ from global frame",
                        "rotated to global frame (using standard diagonal-based local frame)",
                    )
                    .replace(
                        "element-local coordinates; von Mises / principal invariants are exact",
                        "rotated to global frame",
                    )
                    .replace(", element-local coordinates", ", rotated to global frame")
                )

    # ================================================================
    # === Shell internal energy (n_states, n_shells) ===
    # ================================================================
    shell_internal_energy = None
    shell_internal_energy_fill: list[int] = []
    if has_shell_internal_energy:
        shell_internal_energy = np.zeros((n_states, n_shells), dtype=ftype)
        for i, state in enumerate(series.states):
            arr = _cell_array(state, "2DELEM_Specific_Energy")
            if arr is not None:
                ie = arr[model.shell_src]
                shell_internal_energy[i, :] = ie.astype(ftype)
            else:
                shell_internal_energy_fill.append(i)

    # ================================================================
    # === Shell plastic strain ===
    # ================================================================
    shell_eps = None
    shell_eps_fill: list[int] = []
    if has_shell_eps_2layer:
        shell_eps = np.zeros((n_states, n_shells, n_shell_layers), dtype=ftype)
        for i, state in enumerate(series.states):
            arr_lo = _cell_array(state, "2DELEM_Plastic_Strain_Lower")
            arr_hi = _cell_array(state, "2DELEM_Plastic_Strain_Upper")
            if arr_lo is not None:
                shell_eps[i, :, 0] = arr_lo[model.shell_src].astype(ftype)
            else:
                shell_eps_fill.append((i, "2DELEM_Plastic_Strain_Lower"))
            if arr_hi is not None:
                shell_eps[i, :, 1] = arr_hi[model.shell_src].astype(ftype)
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
            arr = _cell_array(state, "2DELEM_Plastic_Strain_Upper")
            if arr is not None:
                shell_eps[i, :, upper_eps_layer] = arr[model.shell_src].astype(ftype)
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
            arr = _cell_array(state, "2DELEM_Plastic_Strain_Lower")
            if arr is not None:
                shell_eps[i, :, 0] = arr[model.shell_src].astype(ftype)
            else:
                shell_eps_fill.append(i)
    elif has_shell_eps_single:
        # Replicate single scalar into both layers (official with no Upper/Lower split)
        n_layers = n_shell_layers if n_shell_layers > 0 else 1
        shell_eps = np.zeros((n_states, n_shells, n_layers), dtype=ftype)
        for i, state in enumerate(series.states):
            arr = _cell_array(state, "2DELEM_Plastic_Strain")
            if arr is not None:
                eps = arr[model.shell_src].astype(ftype)
                for layer in range(n_layers):
                    shell_eps[i, :, layer] = eps
            else:
                shell_eps_fill.append(i)
    elif has_epsp and n_shells > 0:
        n_layers = n_shell_layers if n_shell_layers > 0 else 1
        shell_eps = np.zeros((n_states, n_shells, n_layers), dtype=ftype)
        for i, state in enumerate(series.states):
            arr = _cell_array(state, "EPSP")
            if arr is not None:
                eps_val = arr[model.shell_src].astype(ftype)
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
            arr_lo = _cell_array(state, "2DELEM_Strain_(lower)")
            arr_hi = _cell_array(state, "2DELEM_Strain_(upper)")
            if arr_lo is not None:
                t = arr_lo[model.shell_src]
                _tensor_to_voigt(t, out=shell_strain[i, :, 0, :])
            else:
                shell_strain_fill.append((i, "2DELEM_Strain_(lower)"))
            if arr_hi is not None:
                t = arr_hi[model.shell_src]
                _tensor_to_voigt(t, out=shell_strain[i, :, upper_layer, :])
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
            arr = _cell_array(state, "2DELEM_Strain_(upper)")
            if arr is not None:
                t = arr[model.shell_src]
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
            arr = _cell_array(state, "2DELEM_Strain_(lower)")
            if arr is not None:
                t = arr[model.shell_src]
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
            arr = _cell_array(state, "2DELEM_Strain_(membrane)")
            if arr is not None:
                t = arr[model.shell_src]
                voigt = _tensor_to_voigt(t).astype(ftype)
                shell_strain[i, :, 0, :] = voigt
                shell_strain[i, :, upper_layer, :] = voigt
            else:
                shell_strain_fill.append(i)

    # ================================================================
    # === Shell thickness (n_states, n_shells) ===
    # ================================================================
    shell_thickness = None
    shell_thickness_fill: list[int] = []
    if has_shell_thickness:
        shell_thickness = np.zeros((n_states, n_shells), dtype=ftype)
        for i, state in enumerate(series.states):
            arr = _cell_array(state, "2DELEM_Thickness")
            if arr is not None:
                shell_thickness[i, :] = arr[model.shell_src].astype(ftype)
            else:
                shell_thickness_fill.append(i)


    return shell_stress, shell_stress_fill, shell_eps, shell_eps_fill, shell_internal_energy, shell_internal_energy_fill, shell_strain, shell_strain_fill, shell_thickness, shell_thickness_fill

def build_d3plot(
    series: VtkSeries,
    model: MappedModel,
    *,
    title: str = "",
    f64: bool = False,
    part_names: dict[int, str] | None = None,
    progress: Callable[[int, int], None] | None = None,
    rotate_shell_stress: bool = False,
) -> tuple[D3plot, list[str]]:
    ftype = np.float64 if f64 else np.float32
    itype = np.int64 if f64 else np.int32

    # ----------------------------------------------------------------
    # Key lookups: one rule for every gather (_find_key — exact case first,
    # then case-insensitive), matching the case-insensitive presence flags.
    # Cell gathers use the module-level _cell_array().
    # ----------------------------------------------------------------
    def get_point_key(i: int, k: str) -> str | None:
        return _find_key(series.states[i].point_data, k)

    warn_list: list[str] = _case_collision_warnings(series)
    n_states = len(series.states)
    n_nodes = series.states[0].points.shape[0]
    n_solids = model.solid_conn.shape[0]
    n_shells = model.shell_conn.shape[0]
    n_beams = model.beam_conn.shape[0]

    x0, node_disp, node_vel, node_accel, node_vel_fill, node_accel_fill = _build_kinematics(series, n_nodes, ftype, progress, n_states, get_point_key, warn_list)

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
    # --- Official: EROSION_STATUS (1=alive, 0=eroded) — RESOLVED from anim_to_vtk.cpp:
    #     Written for ALL families (solids, shells, beams, trusses, springs).
    #     Same polarity as OFF in pyradioss. Elements in rigid/unintegrated parts
    #     may emit 0 from state 0 (handled by the ever_alive logic below).
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
    if (
        n_shells > 0
        and _any_state_has(series, "2DELEM_Strain_(membrane)")
        and not has_shell_strain_membrane
    ):
        warn_list.append(
            "2DELEM_Strain_(membrane) ignored because upper/lower layers were also found"
        )

    # --- Official: solid strain tensor ---
    has_solid_strain = n_solids > 0 and _any_state_has(series, "3DELEM_Strain")
    # --- Official: shell thickness ---
    has_shell_thickness = n_shells > 0 and _any_state_has(series, "2DELEM_Thickness")
    # --- Official: shell internal energy ---
    has_shell_internal_energy = n_shells > 0 and _any_state_has(series, "2DELEM_Specific_Energy")
    # --- Official: solid internal energy ---
    has_solid_internal_energy = n_solids > 0 and _any_state_has(series, "3DELEM_Specific_Energy")
    # --- Official: 3DELEM_Von_Mises scalar (fallback for solid stress) ---
    has_solid_vonmises = n_solids > 0 and (
        not has_solid_stress and not has_vonm and _any_state_has(series, "3DELEM_Von_Mises")
    )

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
    if has_shell_strain_2layer or (
        has_shell_strain_upper_only or has_shell_strain_lower_only or has_shell_strain_membrane
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


    solid_stress, solid_stress_fill, solid_eps, solid_eps_fill, solid_history_vars, solid_history_vars_fill, solid_strain, solid_strain_fill = _build_solid_results(
        series, model, ftype, n_states, n_solids, warn_list,
        has_solid_stress, has_vonm, has_solid_vonmises,
        has_solid_eps, has_epsp,
        has_solid_internal_energy, has_solid_strain, _tensor_to_voigt
    )

    shell_stress, shell_stress_fill, shell_eps, shell_eps_fill, shell_internal_energy, shell_internal_energy_fill, shell_strain, shell_strain_fill, shell_thickness, shell_thickness_fill = _build_shell_results(
        series, model, ftype, n_states, n_shells, n_shell_layers, warn_list, rotate_shell_stress,
        has_shell_stress_2layer, has_shell_stress_membrane, has_shell_stress_upper_only,
        has_shell_stress_lower_only, has_shell_vonmises_official, has_vonm,
        has_shell_eps_2layer, has_shell_eps_upper_only, has_shell_eps_lower_only,
        has_shell_eps_single, has_epsp,
        has_shell_internal_energy,
        has_shell_strain_2layer, has_shell_strain_upper_only, has_shell_strain_lower_only,
        has_shell_strain_membrane,
        has_shell_thickness, _tensor_to_voigt, rotate_stress_to_global, compute_shell_local_frames
    )

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
                arr = _cell_array(state, alive_key)
                if arr is not None:
                    shell_alive[i] = arr[model.shell_src].astype(ftype)
                else:
                    shell_alive_fill.append(i)
        if n_solids > 0:
            solid_alive = np.ones((n_states, n_solids), dtype=ftype)
            for i, state in enumerate(series.states):
                arr = _cell_array(state, alive_key)
                if arr is not None:
                    solid_alive[i] = arr[model.solid_src].astype(ftype)
                else:
                    solid_alive_fill.append(i)
        if n_beams > 0:
            beam_alive = np.ones((n_states, n_beams), dtype=ftype)
            for i, state in enumerate(series.states):
                arr = _cell_array(state, alive_key)
                if arr is not None:
                    beam_alive[i] = arr[model.beam_src].astype(ftype)
                else:
                    beam_alive_fill.append(i)

        if alive_key == "EROSION_STATUS":
            for alive_arr, part_idx, fill_list, type_name in [
                (shell_alive, model.shell_part_idx, shell_alive_fill, "shell"),
                (solid_alive, model.solid_part_idx, solid_alive_fill, "solid"),
                (beam_alive, model.beam_part_idx, beam_alive_fill, "beam"),
            ]:
                if alive_arr is None:
                    continue
                # M12 fix: compute ever_alive only over states with real data
                # (not filled states which default to 1.0)
                fill_set = set(fill_list)
                present_mask = np.array([si not in fill_set for si in range(n_states)], dtype=bool)
                if present_mask.any():
                    ever_alive = np.any(alive_arr[present_mask] == 1.0, axis=0)
                else:
                    ever_alive = np.ones(alive_arr.shape[1], dtype=bool)
                never_alive_mask = (
                    ~ever_alive
                )  # elements with EROSION_STATUS=0 in every present state
                n_never = int(never_alive_mask.sum())
                if n_never > 0:
                    # Force never-alive elements to alive (they are rigid/non-integrated)
                    alive_arr[:, never_alive_mask] = 1.0
                    # Part IDs of these elements as written to the d3plot (the model's
                    # part table), not a second lookup of the raw PART_ID array.
                    affected_pids = sorted(
                        {int(p) for p in model.part_ids[part_idx[never_alive_mask]]}
                    )
                    warn_list.append(
                        f"EROSION_STATUS: {n_never} {type_name} elements never alive in any state "
                        f"(parts {affected_pids}); forced alive (rigid/non-integrated)"
                    )

                # Step 2: Latch 1→0 transitions (element dies permanently, no resurrection).
                # An element is dead only after it has been observed alive (M9 rule,
                # bug report 2026-10-11 #6): leading zeros of an element activated after
                # state 0 are kept as they are and do not latch.  Only states with real
                # data count as observations (filled states default to 1.0).
                seen_alive = present_mask[0] & (alive_arr[0] == 1.0)
                revived_any = np.zeros(alive_arr.shape[1], dtype=bool)
                revived_states: list[int] = []
                for si in range(1, n_states):
                    # Elements that were dead at previous state stay dead
                    was_dead = seen_alive & (alive_arr[si - 1] == 0.0)
                    if present_mask[si]:
                        # Report only the state of each element's first revival; a
                        # revival that continues into later states is not a new one.
                        revived = was_dead & (alive_arr[si] != 0.0) & ~revived_any
                        if revived.any():
                            revived_any |= revived
                            revived_states.append(si)
                    alive_arr[si, was_dead] = 0.0
                    if present_mask[si]:
                        seen_alive |= alive_arr[si] == 1.0
                if revived_states:
                    warn_list.append(
                        f"EROSION_STATUS: {int(revived_any.sum())} {type_name} elements return "
                        f"to 1 after being eroded (states {revived_states}); "
                        f"kept eroded (no resurrection)"
                    )

    # ================================================================
    # M4/N1 fix: emit aggregated fill warnings (partial-layer aware)
    # ================================================================
    # Simple fills (whole state missing) — list[int]
    _simple_fills = [
        ("element_solid_stress", solid_stress_fill),
        ("element_solid_effective_plastic_strain", solid_eps_fill),
        ("element_solid_strain", solid_strain_fill),
        ("element_solid_history_variables", solid_history_vars_fill),
        ("element_shell_thickness", shell_thickness_fill),
        ("element_shell_internal_energy", shell_internal_energy_fill),
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
    # NODE_ID is read with exact case (mapping.py); the kinematic arrays are
    # resolved per state below (mapped_point), case-insensitively.
    _RECOGNIZED_POINT = {"NODE_ID"}
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
        "3DELEM_Von_Mises",
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
        "2DELEM_Specific_Energy",
        "3DELEM_Specific_Energy",
    }
    # Arrays we recognize but cannot map to a d3plot field — suppress noise
    _KNOWN_BUT_UNSUPPORTED = {
        "Contact_Forces",
        "Contact_Pressure_/_Normal",
        "Contact_Pressure_/_Tangent",
        "Tied_Contact_Forces",
        "Mass_Change",
        "Time_Step",
        "1DELEM_Force_&_Moment_F1",
        "1DELEM_Force_&_Moment_F2",
        "1DELEM_Force_&_Moment_F3",
        "1DELEM_Force_&_Moment_M1",
        "1DELEM_Force_&_Moment_M2",
        "1DELEM_Force_&_Moment_M3",
        "1DELEM_Force_&_Moment_M4",
        "1DELEM_Force_&_Moment_M5",
        "1DELEM_Force_&_Moment_M6",
        "1DELEM_Plastic_Strain",
        "1DELEM_Specific_Energy",
        "1DELEM_Von_Mises",
        "2DELEM_Hourglass_Energy_per_unit_mass",
        "3DELEM_Hourglass_Energy_per_unit_mass",
    }
    unmapped_cell: dict[str, float] = {}
    unsupported: dict[str, float] = {}
    unmapped_point: dict[str, float] = {}
    for i_state, state in enumerate(series.states):
        # Bug report 2026-10-11 #35 (c28): the kinematic point arrays are mapped
        # case-insensitively through get_point_key, so they count as recognised
        # under exactly the key get_point_key resolves to (a shadowed duplicate
        # such as "Velocity" next to "velocity" stays reported as unmapped).
        mapped_point = {
            get_point_key(i_state, name) for name in ("DISPLACEMENT", "VELOCITY", "ACCELERATION")
        }
        for k, v in state.cell_data.items():
            if k not in _RECOGNIZED_CELL:
                try:
                    val = float(np.abs(v).max())
                except (TypeError, ValueError):
                    val = 0.0
                if k in _KNOWN_BUT_UNSUPPORTED:
                    unsupported[k] = max(unsupported.get(k, 0.0), val)
                else:
                    unmapped_cell[k] = max(unmapped_cell.get(k, 0.0), val)
        for k, v in state.point_data.items():
            if k not in _RECOGNIZED_POINT and k not in mapped_point:
                try:
                    val = float(np.abs(v).max())
                except (TypeError, ValueError):
                    val = 0.0
                if k in _KNOWN_BUT_UNSUPPORTED:
                    unsupported[k] = max(unsupported.get(k, 0.0), val)
                else:
                    unmapped_point[k] = max(unmapped_point.get(k, 0.0), val)
    if unsupported:
        items = ", ".join(f"{k} (absmax={v:.4g})" for k, v in sorted(unsupported.items()))
        warn_list.append(f"known but unsupported arrays (no d3plot equivalent): {items}")
    all_unmapped = {**unmapped_point, **unmapped_cell}
    if all_unmapped:
        items = ", ".join(f"{k} (absmax={v:.4g})" for k, v in sorted(all_unmapped.items()))
        warn_list.append(f"unmapped arrays ignored: {items}")

    # T01 global variables (bug report 2026-10-11 #3 / #27: see _t01_globals)
    ie_resampled, ke_resampled, te_resampled = _t01_globals(series, ftype, warn_list)

    # d3plot record-layout consistency (bug report 2026-10-11 #7, #8, #26)
    solid_strain, shell_strain, solid_eps = _layout_zero_fills(
        n_states,
        n_solids,
        n_shells,
        ftype,
        warn_list,
        solid_stress=solid_stress,
        solid_eps=solid_eps,
        solid_history_vars=solid_history_vars,
        solid_strain=solid_strain,
        shell_strain=shell_strain,
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
    a[ArrayType.part_titles] = _part_titles(model.part_ids, part_names, warn_list)

    a[ArrayType.global_timesteps] = series.times.astype(ftype)
    if ie_resampled is not None:
        a[ArrayType.global_internal_energy] = ie_resampled
    if ke_resampled is not None:
        a[ArrayType.global_kinetic_energy] = ke_resampled
    if te_resampled is not None:
        a[ArrayType.global_total_energy] = te_resampled

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
    if shell_internal_energy is not None:
        a[ArrayType.element_shell_internal_energy] = shell_internal_energy
    if solid_history_vars is not None:
        a[ArrayType.element_solid_history_variables] = solid_history_vars
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


# ----------------------------------------------------------------------
# Bug report 2026-10-11 helpers (T01 globals, layout zero-fills, part titles)
# ----------------------------------------------------------------------

# Columns of the pyradioss T01 CSV (pyfemsolver/output/time_history.py, the only
# T01 writer the converter reads) that make up the d3plot global "total energy".
# LS-DYNA convention (glstat): total = kinetic + internal + hourglass + sliding
# interface (contact) + damping energy.  pyfemsolver maps these to KE, IE, HE,
# CE and DE (engine/engine.py energy dict: "DE": state.e_damp = /DAMP dissipation).
# EN is pyfemsolver's numerical-dissipation ledger (engine.py: e_num, "numerical
# dissipation of the elements") and EW the external work; neither is part of the
# LS-DYNA total, which must show such a loss as a drop instead of hiding it.
_T01_TOTAL_ENERGY_COLUMNS = ("IE", "KE", "HE", "CE", "DE")


def _t01_globals(series, ftype, warn_list: list[str]):
    """Resample the T01 energies at the VTK state times.

    Returns ``(internal, kinetic, total)``, each ``(n_states,)`` or ``None``.
    A T01 file without data rows is skipped with a warning (bug report
    2026-10-11 #27); the total energy is the sum of the energy columns, never
    the EN column (bug report 2026-10-11 #3).
    """
    t01 = getattr(series, "t01_data", None)
    if t01 is None:
        return None, None, None
    if len(t01.times) == 0:
        warn_list.append(
            "T01 CSV has no data rows; global energies (internal, kinetic, total) not written"
        )
        return None, None, None

    def _resample(values: np.ndarray) -> np.ndarray:
        return np.interp(series.times, t01.times, values).astype(ftype)

    ie = _resample(t01.columns["IE"]) if "IE" in t01.columns else None
    ke = _resample(t01.columns["KE"]) if "KE" in t01.columns else None
    te = None
    if ie is not None and ke is not None:
        te_cols = [c for c in _T01_TOTAL_ENERGY_COLUMNS if c in t01.columns]
        te = _resample(np.sum([t01.columns[c] for c in te_cols], axis=0))
        unused = [c for c in t01.columns if c not in _T01_TOTAL_ENERGY_COLUMNS]
        warn_list.append(
            f"global_total_energy = {'+'.join(te_cols)} from T01 (LS-DYNA convention: "
            f"kinetic + internal + hourglass + contact + damping; EN numerical "
            f"dissipation and EW external work excluded); T01 columns not written: "
            f"{', '.join(unused) if unused else 'none'}"
        )
    else:
        warn_list.append("T01 CSV lacks the IE and/or KE column; global_total_energy not written")
    return ie, ke, te


def _layout_zero_fills(
    n_states: int,
    n_solids: int,
    n_shells: int,
    ftype,
    warn_list: list[str],
    *,
    solid_stress,
    solid_eps,
    solid_history_vars,
    solid_strain,
    shell_strain,
):
    """Zero-fill the state arrays the lasso 2.0.4 record layout needs.

    lasso derives the d3plot layout flags from array PRESENCE:

    * ISTRN is one global flag, set when any element family carries strain
      (``D3plotWriterSettings.build_header``).  Shells then need 12 strain words
      each and solids 6 strain words at the end of NEIPH.  A family without its
      strain array gets a broken record: solids compute NEIPH - 6 < 0 history
      words and drop them (bug report #7), shells without any state array make
      the writer raise (bug report #26).  So: zero-fill the missing strain.
    * The writer always writes the 7 basic solid words (stress + eff. plastic
      strain), but IOSHL1/IOSHL2 are only non-zero when a stress / plastic-strain
      array exists.  Readers then skip fewer words and shift history variables
      and strain by one word (bug report #8).  So: whenever solids carry state
      data, zero-fill the solid plastic strain (it sets IOSHL2=999 and, via
      IOSHL1's solid branch, IOSHL1>=999).
    """
    reason_strain = "required because element strain sets the global d3plot ISTRN flag"
    if solid_strain is not None or shell_strain is not None:
        if n_solids > 0 and solid_strain is None:
            solid_strain = np.zeros((n_states, n_solids, 1, 6), dtype=ftype)
            warn_list.append(
                f"element_solid_strain: zero-filled at all states (no solid strain "
                f"source array; {reason_strain})"
            )
        if n_shells > 0 and shell_strain is None:
            shell_strain = np.zeros((n_states, n_shells, 2, 6), dtype=ftype)
            warn_list.append(
                f"element_shell_strain: zero-filled at all states (no shell strain "
                f"source array; {reason_strain})"
            )
    solids_have_data = any(
        arr is not None for arr in (solid_stress, solid_history_vars, solid_strain)
    )
    if n_solids > 0 and solid_eps is None and solids_have_data:
        solid_eps = np.zeros((n_states, n_solids, 1), dtype=ftype)
        warn_list.append(
            "element_solid_effective_plastic_strain: zero-filled at all states (no solid "
            "plastic-strain source array; required so that readers skip the always-written "
            "plastic-strain word of the solid record)"
        )
    return solid_strain, shell_strain, solid_eps


def _part_titles(part_ids: np.ndarray, part_names: dict[int, str] | None, warn_list):
    """Part title bytes for lasso, sanitised like the header title.

    lasso 2.0.4 pads each part title to 18*4 = 72 characters and only then
    UTF-8-encodes it (``d3plot.py`` part-title writer, ``title_wordsize = 4``
    for every file wordsize), so a multi-byte character lengthens the record
    and shifts every later one (bug report 2026-10-11 #9).  The 72-character
    budget is independent of f32/f64 output, which is ``_sanitize_title``'s
    ``wordsize=8`` budget (10*8 - 8 = 72).
    """
    titles = []
    for pid in part_ids:
        pid = int(pid)
        name = part_names.get(pid) if part_names else None
        if name is None:
            titles.append(b"Part %d" % pid)
            continue
        clean, warns = _sanitize_title(name, wordsize=8)
        warn_list.extend(f"part {pid} {w}" for w in warns)
        titles.append(clean.encode("ascii"))
    return np.array(titles)
