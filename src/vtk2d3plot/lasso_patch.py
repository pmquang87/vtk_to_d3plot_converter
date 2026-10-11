"""Monkey-patches for lasso-python 2.0.4 SPH writing bugs (lasso issue #101).

The converter itself never writes SPH data. These patches only matter for
other lasso use in the same Python process (``import vtk2d3plot`` applies them
globally). Non-SPH writes are unaffected: every patched method returns 0
without writing when the model has no SPH nodes (``nmsph <= 0``), exactly like
the originals.

Bugs fixed (lasso 2.0.4, ``lasso/dyna/d3plot.py``):

1. ``_write_geom_sph_element_data_flags`` writes the SPH variable count as
   ``isphfg(1)``. The reader (``_read_sph_element_data_flags``) and the LS-DYNA
   manual treat ``isphfg(1)`` as the length of the flag block (11 words) and
   advance the geometry position by it, so every later geometry section is
   misread. Patched: ``isphfg(1) = 11``.
2. ``_write_states_sph`` reads ``numsph`` / ``isphfg2..10`` from the writer
   header, which ``D3plotWriterSettings.build_header`` never sets (KeyError),
   fills its buffer with column slices of width ``n_particles`` instead of the
   variable width (broadcast errors), and checks the byte count against shell
   header fields (``nv2d``/``nel4``/``numrbe``). Patched: the state block is
   written directly, with the variable layout derived from the arrays present
   - the same rule the (patched) flag writer uses, so flags and data agree.
3. ``_write_geom_sph_node_and_materials`` concatenates 1-D arrays along
   axis 1. Patched: column stack; a missing ``sph_node_material_index`` raises
   a clear ValueError.

Per-particle state layout (matches ``_read_states_sph``): material/deletion
word, then radius, pressure, stress(6), eff. plastic strain, density, internal
energy, n neighbours, strain(6), mass, history variables - each only when its
array is present.
"""

from __future__ import annotations

import logging

import numpy as np
from lasso.dyna.array_type import ArrayType
from lasso.dyna.d3plot import D3plot

logger = logging.getLogger(__name__)

FORTRAN_OFFSET = 1

# (array type, number of words per particle), in d3plot order after the
# material/deletion word. The order is the order of isphfg2..isphfg10.
_SPH_STATE_VARS: tuple[tuple[str, int], ...] = (
    (ArrayType.sph_radius, 1),  # isphfg2
    (ArrayType.sph_pressure, 1),  # isphfg3
    (ArrayType.sph_stress, 6),  # isphfg4
    (ArrayType.sph_effective_plastic_strain, 1),  # isphfg5
    (ArrayType.sph_density, 1),  # isphfg6
    (ArrayType.sph_internal_energy, 1),  # isphfg7
    (ArrayType.sph_n_neighbors, 1),  # isphfg8
    (ArrayType.sph_strain, 6),  # isphfg9
    (ArrayType.sph_mass, 1),  # isphfg10
)

_ISPHFG1_BLOCK_LENGTH = 11


def _sph_flags(d3: D3plot) -> tuple[list[int], int]:
    """Return ([isphfg2..isphfg10], n_history_vars) from the arrays present."""
    flags = [width if at in d3.arrays else 0 for at, width in _SPH_STATE_VARS]
    n_hist = 0
    if ArrayType.sph_history_vars in d3.arrays:
        hist = d3.arrays[ArrayType.sph_history_vars]
        if hist.ndim != 3:
            raise ValueError(
                f"Array '{ArrayType.sph_history_vars}' was expected to have 3 dimensions "
                "(n_timesteps, n_sph_particles, n_sph_history_vars)."
            )
        n_hist = int(hist.shape[2])
    return flags, n_hist


def _patched_write_geom_sph_element_data_flags(self, fp, settings) -> int:
    """Write the 11-word SPH flag block with isphfg(1) = 11 (block length)."""
    if settings.header["nmsph"] <= 0:
        return 0

    flags, n_hist = _sph_flags(self)
    words = [_ISPHFG1_BLOCK_LENGTH, *flags, n_hist]

    n_bytes_written = 0
    for word in words:
        n_bytes_written += fp.write(settings.pack(int(word)))

    n_bytes_expected = _ISPHFG1_BLOCK_LENGTH * settings.wordsize
    if n_bytes_written != n_bytes_expected:
        raise RuntimeError(f"byte checksum wrong: {n_bytes_expected} != {n_bytes_written}")
    return n_bytes_written


def _patched_write_states_sph(self, fp, i_timestep, settings) -> int:
    """Write one state's SPH block: n_particles x (1 + sum(flags) + n_hist) words."""
    if settings.header["nmsph"] <= 0:
        return 0

    n_particles = int(settings.header["nmsph"])
    flags, n_hist = _sph_flags(self)
    n_vars = 1 + sum(flags) + n_hist

    sph_data = np.zeros((n_particles, n_vars), dtype=np.float64)

    # material number (1-based), negative when the particle is deleted.
    # lasso reads sph_deletion as (word < 0); a bool array means True=deleted,
    # a numeric array keeps that raw sign convention.
    # (the geometry writer already required sph_node_material_index)
    mat = self.arrays[ArrayType.sph_node_material_index].reshape(-1).astype(np.float64)
    mat = mat + FORTRAN_OFFSET
    if ArrayType.sph_deletion in self.arrays:
        dele = np.asarray(self.arrays[ArrayType.sph_deletion][i_timestep]).reshape(-1)
        deleted = dele.astype(bool) if dele.dtype == np.bool_ else dele < 0
        mat = np.where(deleted, -mat, mat)
    sph_data[:, 0] = mat

    col = 1
    for (at, width), flag in zip(_SPH_STATE_VARS, flags, strict=True):
        if not flag:
            continue
        arr = np.asarray(self.arrays[at][i_timestep], dtype=np.float64)
        sph_data[:, col : col + width] = arr.reshape(n_particles, width)
        col += width
    if n_hist:
        hist = np.asarray(self.arrays[ArrayType.sph_history_vars][i_timestep], dtype=np.float64)
        sph_data[:, col : col + n_hist] = hist.reshape(n_particles, n_hist)
        col += n_hist

    n_bytes_written = fp.write(settings.pack(sph_data, dtype_hint=np.floating))

    n_bytes_expected = n_particles * n_vars * settings.wordsize
    if n_bytes_written != n_bytes_expected:
        raise RuntimeError(f"byte checksum wrong: {n_bytes_expected} != {n_bytes_written}")
    return n_bytes_written


def _patched_write_geom_sph_node_and_materials(self, fp, settings) -> int:
    """Write the SPH node / material list as (nmsph, 2) Fortran-indexed ints."""
    nmsph = settings.header["nmsph"]

    if nmsph <= 0:
        return 0

    if ArrayType.sph_node_material_index not in self.arrays:
        raise ValueError(
            "Writing SPH nodes requires ArrayType.sph_node_material_index "
            "(0-based SPH material index per particle) besides sph_node_indexes."
        )

    array_dims = {ArrayType.sph_node_indexes: 0, ArrayType.sph_node_material_index: 0}
    self.check_array_dims(array_dims, "n_sph_nodes", nmsph)

    sph_node_indexes = self.arrays[ArrayType.sph_node_indexes].reshape(-1) + FORTRAN_OFFSET
    sph_node_material_index = (
        self.arrays[ArrayType.sph_node_material_index].reshape(-1) + FORTRAN_OFFSET
    )
    sph_data = np.column_stack((sph_node_indexes, sph_node_material_index))

    n_bytes_written = fp.write(settings.pack(sph_data, dtype_hint=np.integer))

    n_bytes_expected = nmsph * settings.wordsize * 2
    if n_bytes_written != n_bytes_expected:
        raise RuntimeError(f"byte checksum wrong: {n_bytes_expected} != {n_bytes_written}")

    return n_bytes_written


_PATCHES = {
    "_write_geom_sph_element_data_flags": _patched_write_geom_sph_element_data_flags,
    "_write_states_sph": _patched_write_states_sph,
    "_write_geom_sph_node_and_materials": _patched_write_geom_sph_node_and_materials,
}


def apply_patches():
    """Apply all local monkey-patches to third-party libraries (idempotent)."""
    for name, func in _PATCHES.items():
        if getattr(D3plot, name) is not func:
            setattr(D3plot, name, func)
            logger.debug("Applied lasso-python SPH monkey-patch: D3plot.%s", name)
