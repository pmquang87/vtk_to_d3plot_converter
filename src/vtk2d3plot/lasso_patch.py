"""Monkey-patches for lasso-python bugs."""

import logging

from lasso.dyna.array_type import ArrayType
from lasso.dyna.d3plot import D3plot

logger = logging.getLogger(__name__)

_orig_write_states_sph = D3plot._write_states_sph

def _patched_write_states_sph(self, fp, i_timestep, settings):
    """
    Patched version of lasso.dyna.D3plot._write_states_sph to fix lasso-python bug #101.
    The original method incorrectly calculates expected SPH bytes using 2D shell variables
    (nv2d, nel4, numrbe) instead of SPH variables, raising a RuntimeError.
    It also misses numsph and isphfg* flags in settings.header.
    """
    # Inject missing SPH flags into settings.header that are not copied by D3plotWriterSettings
    for field in ["numsph"] + [f"isphfg{i}" for i in range(2, 11)]:
        if field not in settings.header:
            settings.header[field] = getattr(self.header, field, 0) if hasattr(self, "header") else 0

    try:
        return _orig_write_states_sph(self, fp, i_timestep, settings)
    except (RuntimeError, KeyError) as e:
        is_bug_101 = False
        if isinstance(e, RuntimeError) and "byte checksum wrong" in str(e) or isinstance(e, KeyError) and str(e) in ["'nv2d'", "'nel4'", "'numrbe'"]:
            is_bug_101 = True
            
        if is_bug_101:
            # The exception is raised AFTER the data is successfully written.
            # We intercept it and return the correct byte count.
            
            # Recompute n_particles using the same logic as the original method
            array_dims = {
                ArrayType.sph_node_indexes: 0,
                ArrayType.sph_deletion: 1,
                ArrayType.sph_radius: 1,
                ArrayType.sph_pressure: 1,
                ArrayType.sph_stress: 1,
                ArrayType.sph_effective_plastic_strain: 1,
                ArrayType.sph_density: 1,
                ArrayType.sph_internal_energy: 1,
                ArrayType.sph_n_neighbors: 1,
                ArrayType.sph_strain: 1,
                ArrayType.sph_mass: 1,
            }
            try:
                n_particles = self.check_array_dims(array_dims, "n_particles")
            except Exception:  # noqa: BLE001
                n_particles = settings.header.get("nmsph", 0)
                
            n_sph_variables = settings.header.get("numsph", 0)
            n_bytes_written = n_particles * n_sph_variables * settings.wordsize
            
            logger.debug(
                "lasso-python bug #101 caught and bypassed in _write_states_sph. "
                "Calculated %d bytes written.", n_bytes_written
            )
            return n_bytes_written
        raise

def _patched_write_geom_sph_node_and_materials(self, fp, settings):
    """
    Patched version of lasso.dyna.D3plot._write_geom_sph_node_and_materials.
    The original method tries to concatenate 1D arrays along axis 1, which crashes.
    """
    nmsph = settings.header["nmsph"]

    if nmsph <= 0:
        return 0

    # Avoid check_ndim failure by manually doing what the original intended,
    # but with column_stack instead of concatenate(axis=1) on 1D arrays.
    
    # Original checks
    array_dims = {ArrayType.sph_node_indexes: 0, ArrayType.sph_node_material_index: 0}
    self.check_array_dims(array_dims, "n_sph_nodes", nmsph)

    # Convert to 1D, add offset, and reshape to (N, 1) so concatenate(axis=1) works,
    # OR just use np.column_stack.
    import numpy as np
    sph_node_indexes = self.arrays[ArrayType.sph_node_indexes].reshape(-1, 1) + 1 # FORTRAN_OFFSET is 1
    sph_node_material_index = self.arrays[ArrayType.sph_node_material_index].reshape(-1, 1) + 1
    sph_data = np.concatenate((sph_node_indexes, sph_node_material_index), axis=1)

    # write
    n_bytes_written = fp.write(settings.pack(sph_data, dtype_hint=np.integer))

    # check
    n_bytes_expected = nmsph * settings.wordsize * 2
    if n_bytes_written != n_bytes_expected:
        raise RuntimeError(f"byte checksum wrong: {n_bytes_expected} != {n_bytes_written}")

    return n_bytes_written


_orig_write_geom_sph = D3plot._write_geom_sph_node_and_materials

def apply_patches():
    """Apply all local monkey-patches to third-party libraries."""
    if D3plot._write_states_sph is not _patched_write_states_sph:
        D3plot._write_states_sph = _patched_write_states_sph
        logger.debug("Applied monkey-patch for lasso-python bug #101 (SPH write checksum).")
        
    if D3plot._write_geom_sph_node_and_materials is not _patched_write_geom_sph_node_and_materials:
        D3plot._write_geom_sph_node_and_materials = _patched_write_geom_sph_node_and_materials
        logger.debug("Applied monkey-patch for lasso-python SPH geometry write crash.")
