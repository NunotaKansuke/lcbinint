"""Experimental end-to-end JAX inverse-ray entry points."""

from functools import partial

import jax
import jax.numpy as jnp

from ._config import as_float64, require_x64
from .cpp_backend import (
    binary_inverse_ray_polar_ffi,
    binary_inverse_ray_cartesian_ffi,
    binary_inverse_ray_fixed_support_ffi,
    cpp_cartesian_epoch_ffi_available,
    cpp_polar_epoch_ffi_available,
    cpp_fixed_support_ffi_available,
    cpp_macro_tile_discovery_ffi_available,
    discover_binary_macro_tiles_ffi,
)
from .discovery import discover_binary_macro_tiles
from .integrate import binary_inverse_ray_fixed_support
from .multipole import (
    binary_hexadecapole,
    binary_point_source_magnification,
)
from .polar import binary_inverse_ray_polar
from .source_plane import binary_source_plane_quadrature
from .types import (
    AutoInverseRayResult,
    HybridMagnificationResult,
    InverseRayResult,
)


def _use_ffi_cartesian_backend(cartesian_backend, kernel):
    if cartesian_backend not in ("auto", "jax", "ffi"):
        raise ValueError("cartesian_backend must be 'auto', 'jax', or 'ffi'")
    if cartesian_backend == "ffi":
        if kernel != "real":
            raise ValueError("the FFI Cartesian backend requires kernel='real'")
        return True
    if cartesian_backend == "jax":
        # The public JAX result remains transformable through the registered
        # JVP.  On CPU its support must nevertheless come from the certified
        # component discovery, rather than the old fixed-limb JAX raster.
        return kernel == "real" and cpp_fixed_support_ffi_available()
    return (
        cartesian_backend == "auto"
        and kernel == "real"
        and cpp_fixed_support_ffi_available()
    )


def _binary_inverse_ray(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c,
    limb_d,
    *,
    resolution,
    tile_size,
    tile_capacity,
    limb_samples,
    kernel,
    moment_mode,
    cartesian_backend,
    root_backend,
):
    source_x, source_y, separation, mass_ratio, source_radius, limb_c, limb_d = (
        as_float64(value)
        for value in (
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
        )
    )
    cell_size = jax.lax.stop_gradient(source_radius / resolution)
    use_ffi = _use_ffi_cartesian_backend(cartesian_backend, kernel)
    use_fused_ffi = (
        use_ffi and root_backend != "jax" and cpp_cartesian_epoch_ffi_available()
    )
    subdivision = 4
    if use_fused_ffi:
        return binary_inverse_ray_cartesian_ffi(
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
            cell_size=cell_size,
            tile_size=tile_size,
            tile_capacity=tile_capacity,
            limb_samples=limb_samples,
            moment_mode=moment_mode,
            boundary_subdivision=subdivision,
        )
    # Support discovery is topology, not a differentiable part of the cell
    # quadrature.  On CPU use the certified C++ discovery whenever the root
    # backend is not explicitly forced to JAX, even if the caller selected
    # the JAX integration kernel.  This keeps the latter's AD path while
    # avoiding a fixed limb raster as its component oracle.
    certified_discovery = (
        root_backend != "jax" and cpp_macro_tile_discovery_ffi_available()
    )
    discovery_function = (
        discover_binary_macro_tiles_ffi
        if certified_discovery
        else discover_binary_macro_tiles
    )
    discovery = discovery_function(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        cell_size,
        tile_size=tile_size,
        tile_capacity=tile_capacity,
        limb_samples=limb_samples,
        root_backend=root_backend,
    )
    if use_ffi:
        integrated = binary_inverse_ray_fixed_support_ffi(
            discovery.tile_origins,
            discovery.tile_mask,
            cell_size,
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
            tile_size=tile_size,
            moment_mode=moment_mode,
        )
    else:
        integrated = binary_inverse_ray_fixed_support(
            discovery.tile_origins,
            discovery.tile_mask,
            cell_size,
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
            tile_size=tile_size,
            kernel=kernel,
            moment_mode=moment_mode,
        )
    return _cartesian_result(discovery, integrated)


def _cartesian_result(discovery, integrated):
    support_valid = ~(discovery.overflow | discovery.root_failure)
    return InverseRayResult(
        magnification=integrated.magnification,
        moments=integrated.moments,
        boundary_cells=integrated.boundary_cells,
        active_cells=integrated.active_cells,
        tile_count=discovery.visited_count,
        discovery_overflow=discovery.overflow,
        root_failure=discovery.root_failure,
        support_valid=support_valid,
    )


def _auto_result(result, used_polar):
    return AutoInverseRayResult(
        magnification=result.magnification,
        moments=result.moments,
        boundary_cells=result.boundary_cells,
        active_cells=result.active_cells,
        support_count=result.tile_count,
        discovery_overflow=result.discovery_overflow,
        root_failure=result.root_failure,
        support_valid=result.support_valid,
        used_polar=jnp.asarray(used_polar),
    )


def _point_source_magnification(
    source_x,
    source_y,
    separation,
    mass_ratio,
    root_backend,
):
    return jax.lax.stop_gradient(
        binary_point_source_magnification(
            source_x,
            source_y,
            separation,
            mass_ratio,
            root_backend=root_backend,
        ).magnification
    )


@partial(
    jax.jit,
    static_argnames=(
        "tile_size",
        "tile_capacity",
        "limb_samples",
        "kernel",
        "cartesian_backend",
        "root_backend",
    ),
)
def binary_inverse_ray(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c=0.0,
    limb_d=0.0,
    *,
    resolution=64,
    tile_size=16,
    tile_capacity=1024,
    limb_samples=16,
    kernel="real",
    cartesian_backend="auto",
    root_backend="auto",
):
    """Discover image support and integrate a finite binary-lens source."""

    require_x64()
    return _binary_inverse_ray(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        limb_c,
        limb_d,
        resolution=resolution,
        tile_size=tile_size,
        tile_capacity=tile_capacity,
        limb_samples=limb_samples,
        kernel=kernel,
        moment_mode="two_coefficient",
        cartesian_backend=cartesian_backend,
        root_backend=root_backend,
    )


@partial(
    jax.jit,
    static_argnames=(
        "tile_size",
        "tile_capacity",
        "limb_samples",
        "kernel",
        "cartesian_backend",
        "root_backend",
    ),
)
def binary_inverse_ray_uniform(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    *,
    resolution=64,
    tile_size=16,
    tile_capacity=1024,
    limb_samples=16,
    kernel="real",
    cartesian_backend="auto",
    root_backend="auto",
):
    """Specialized inverse-ray path for a uniform source."""

    require_x64()
    return _binary_inverse_ray(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        0.0,
        0.0,
        resolution=resolution,
        tile_size=tile_size,
        tile_capacity=tile_capacity,
        limb_samples=limb_samples,
        kernel=kernel,
        moment_mode="uniform",
        cartesian_backend=cartesian_backend,
        root_backend=root_backend,
    )


@partial(
    jax.jit,
    static_argnames=(
        "tile_size",
        "tile_capacity",
        "limb_samples",
        "kernel",
        "cartesian_backend",
        "root_backend",
    ),
)
def binary_inverse_ray_linear(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c,
    *,
    resolution=64,
    tile_size=16,
    tile_capacity=1024,
    limb_samples=16,
    kernel="real",
    cartesian_backend="auto",
    root_backend="auto",
):
    """Specialized inverse-ray path for linear limb darkening."""

    require_x64()
    return _binary_inverse_ray(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        limb_c,
        0.0,
        resolution=resolution,
        tile_size=tile_size,
        tile_capacity=tile_capacity,
        limb_samples=limb_samples,
        kernel=kernel,
        moment_mode="linear",
        cartesian_backend=cartesian_backend,
        root_backend=root_backend,
    )


@partial(
    jax.jit,
    static_argnames=(
        "tile_size",
        "tile_capacity",
        "limb_samples",
        "kernel",
        "polar_resolution",
        "polar_angular_bins",
        "polar_radial_capacity",
        "polar_band_capacity",
        "polar_limb_samples",
        "polar_angular_chunk_size",
        "polar_boundary_subdivision",
        "moment_mode",
        "cartesian_backend",
        "root_backend",
    ),
)
def binary_inverse_ray_auto(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c=0.0,
    limb_d=0.0,
    *,
    resolution=64,
    tile_size=16,
    tile_capacity=1024,
    limb_samples=16,
    kernel="real",
    polar_resolution=128,
    polar_angular_bins=4096,
    polar_radial_capacity=256,
    polar_band_capacity=4,
    polar_limb_samples=32,
    polar_angular_chunk_size=1024,
    polar_boundary_subdivision=2,
    polar_magnification_threshold=80.0,
    polar_max_source_radius=0.01,
    polar_force=False,
    polar_allowed=True,
    polar_fallback_on_overflow=True,
    moment_mode="two_coefficient",
    cartesian_backend="auto",
    root_backend="auto",
):
    """Automatically dispatch between Cartesian and polar inverse rays.

    Tiny high-magnification sources take the polar path before Cartesian tile
    discovery. Other sources use Cartesian integration, with polar as a
    fixed-capacity overflow fallback.
    """

    require_x64()
    source_x, source_y, separation, mass_ratio, source_radius, limb_c, limb_d = (
        as_float64(value)
        for value in (
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
        )
    )
    if moment_mode not in ("uniform", "linear", "two_coefficient"):
        raise ValueError(
            "moment_mode must be 'uniform', 'linear', or 'two_coefficient'"
        )

    def polar_path(_):
        polar_options = {
            "resolution": polar_resolution,
            "angular_bins": polar_angular_bins,
            "radial_capacity": polar_radial_capacity,
            "band_capacity": polar_band_capacity,
            "limb_samples": polar_limb_samples,
            "angular_chunk_size": polar_angular_chunk_size,
            "boundary_subdivision": polar_boundary_subdivision,
            "moment_mode": moment_mode,
        }
        use_polar_ffi = (
            kernel == "real"
            and root_backend != "jax"
            and cpp_polar_epoch_ffi_available()
        )
        if use_polar_ffi:
            result = binary_inverse_ray_polar_ffi(
                source_x,
                source_y,
                separation,
                mass_ratio,
                source_radius,
                limb_c,
                limb_d,
                **polar_options,
            )
        else:
            result = binary_inverse_ray_polar(
                source_x,
                source_y,
                separation,
                mass_ratio,
                source_radius,
                limb_c,
                limb_d,
                kernel=kernel,
                root_backend=root_backend,
                **polar_options,
            )
        return _auto_result(result, True)

    def cartesian_or_fallback(_):
        cell_size = jax.lax.stop_gradient(source_radius / resolution)
        use_ffi = _use_ffi_cartesian_backend(cartesian_backend, kernel)
        use_fused_ffi = (
            use_ffi and root_backend != "jax" and cpp_cartesian_epoch_ffi_available()
        )

        def fallback(_):
            return polar_path(None)

        if use_fused_ffi:
            subdivision = 4 if moment_mode == "two_coefficient" else 3
            cartesian = binary_inverse_ray_cartesian_ffi(
                source_x,
                source_y,
                separation,
                mass_ratio,
                source_radius,
                limb_c,
                limb_d,
                cell_size=cell_size,
                tile_size=tile_size,
                tile_capacity=tile_capacity,
                limb_samples=limb_samples,
                moment_mode=moment_mode,
                boundary_subdivision=subdivision,
            )
            use_fallback = (
                cartesian.discovery_overflow
                & jnp.asarray(polar_allowed)
                & jnp.asarray(polar_fallback_on_overflow)
            )
            return jax.lax.cond(
                use_fallback,
                fallback,
                lambda _: _auto_result(cartesian, False),
                None,
            )

        discovery_function = (
            discover_binary_macro_tiles_ffi
            if use_ffi and cpp_macro_tile_discovery_ffi_available()
            else discover_binary_macro_tiles
        )
        discovery = discovery_function(
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            cell_size,
            tile_size=tile_size,
            tile_capacity=tile_capacity,
            limb_samples=limb_samples,
            root_backend=root_backend,
        )

        def integrate_cartesian(_):
            subdivision = 4 if moment_mode == "two_coefficient" else 3
            if use_ffi:
                integrated = binary_inverse_ray_fixed_support_ffi(
                    discovery.tile_origins,
                    discovery.tile_mask,
                    cell_size,
                    source_x,
                    source_y,
                    separation,
                    mass_ratio,
                    source_radius,
                    limb_c,
                    limb_d,
                    tile_size=tile_size,
                    moment_mode=moment_mode,
                    boundary_subdivision=subdivision,
                )
            else:
                integrated = binary_inverse_ray_fixed_support(
                    discovery.tile_origins,
                    discovery.tile_mask,
                    cell_size,
                    source_x,
                    source_y,
                    separation,
                    mass_ratio,
                    source_radius,
                    limb_c,
                    limb_d,
                    tile_size=tile_size,
                    kernel=kernel,
                    moment_mode=moment_mode,
                    boundary_subdivision=subdivision,
                )
            return _auto_result(_cartesian_result(discovery, integrated), False)

        use_fallback = (
            discovery.overflow
            & jnp.asarray(polar_allowed)
            & jnp.asarray(polar_fallback_on_overflow)
        )
        return jax.lax.cond(use_fallback, fallback, integrate_cartesian, None)

    point_magnification = _point_source_magnification(
        source_x,
        source_y,
        separation,
        mass_ratio,
        root_backend,
    )
    preselect_polar = (
        (
            jnp.asarray(polar_force)
            | (point_magnification >= polar_magnification_threshold)
        )
        & (source_radius <= polar_max_source_radius)
        & (source_radius > 0.0)
        & jnp.asarray(polar_allowed)
    )
    def polar_or_cartesian(_):
        polar = polar_path(None)
        # The preselect is a prediction, and it is wrong in one place: where the
        # caustic enters the source disk the polar grid finds no valid support,
        # returning roughly half the magnification with `support_valid` clear,
        # while the Cartesian grid at the same resolution integrates the epoch
        # normally.  A preselected epoch had no second route -- the whole
        # `cartesian_or_fallback` branch was unreachable once `preselect_polar`
        # was set -- so it fell closed with a usable answer one branch away.
        # Mirror the Cartesian -> polar overflow fallback in the opposite
        # direction: this is the same recourse, for the same reason.
        return jax.lax.cond(
            polar.support_valid,
            lambda _: polar,
            cartesian_or_fallback,
            None,
        )

    return jax.lax.cond(
        preselect_polar,
        polar_or_cartesian,
        cartesian_or_fallback,
        None,
    )


def _multipole_dispatch_masks(
    hexadecapole,
    mass_ratio,
    source_radius,
    absolute_tolerance,
    relative_tolerance,
    multipole_safety_factor,
    polar_min_mass_ratio,
    polar_require_topology_stable,
):
    budget = absolute_tolerance + relative_tolerance * jnp.maximum(
        jnp.abs(hexadecapole.magnification),
        1.0,
    )
    correction_scale = jnp.maximum(
        jnp.abs(hexadecapole.quadrupole_correction),
        budget,
    )
    expansion_well_ordered = (
        hexadecapole.estimated_error <= 0.25 * correction_scale
    ) & (
        jnp.abs(hexadecapole.quadrupole_correction)
        <= 0.1 * jnp.maximum(jnp.abs(hexadecapole.magnification), 1.0)
    )
    accept_multipole = jax.lax.stop_gradient(
        (source_radius >= 0.0)
        & hexadecapole.topology_stable
        & ~hexadecapole.root_failure
        & jnp.isfinite(hexadecapole.magnification)
        & expansion_well_ordered
        & (multipole_safety_factor * hexadecapole.estimated_error <= budget)
    )
    absolute_mass_ratio = jnp.abs(mass_ratio)
    symmetric_mass_ratio = jnp.minimum(
        absolute_mass_ratio,
        jnp.where(
            absolute_mass_ratio > 0.0,
            1.0 / absolute_mass_ratio,
            0.0,
        ),
    )
    polar_allowed = jax.lax.stop_gradient(
        (
            ~jnp.asarray(polar_require_topology_stable)
            | hexadecapole.topology_stable
        )
        & (symmetric_mass_ratio >= polar_min_mass_ratio)
    )
    return accept_multipole, polar_allowed


@partial(
    jax.jit,
    static_argnames=(
        "tile_size",
        "tile_capacity",
        "limb_samples",
        "kernel",
        "polar_resolution",
        "polar_angular_bins",
        "polar_radial_capacity",
        "polar_band_capacity",
        "polar_limb_samples",
        "polar_angular_chunk_size",
        "polar_boundary_subdivision",
        "moment_mode",
        "source_plane_fallback",
        "source_plane_rule",
        "source_plane_coarse_order",
        "source_plane_fine_order",
        "source_plane_angular_multiplier",
        "expanded_cartesian_fallback",
        "expanded_coarse_resolution",
        "expanded_fine_resolution",
        "expanded_coarse_tile_capacity",
        "expanded_fine_tile_capacity",
        "expanded_limb_samples",
        "cartesian_backend",
        "root_backend",
    ),
)
def binary_magnification_auto(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c=0.0,
    limb_d=0.0,
    *,
    absolute_tolerance=1.0e-4,
    relative_tolerance=1.0e-4,
    multipole_safety_factor=4.0,
    resolution=64,
    tile_size=16,
    tile_capacity=1024,
    limb_samples=16,
    kernel="real",
    polar_resolution=128,
    polar_angular_bins=4096,
    polar_radial_capacity=256,
    polar_band_capacity=4,
    polar_limb_samples=32,
    polar_angular_chunk_size=1024,
    polar_boundary_subdivision=2,
    polar_magnification_threshold=80.0,
    polar_max_source_radius=0.01,
    polar_min_mass_ratio=5.0e-3,
    polar_require_topology_stable=True,
    polar_force=False,
    polar_fallback_on_overflow=False,
    source_plane_fallback=True,
    source_plane_rule="chord",
    source_plane_coarse_order=16,
    source_plane_fine_order=32,
    source_plane_angular_multiplier=4,
    expanded_cartesian_fallback=False,
    expanded_coarse_resolution=64,
    expanded_fine_resolution=128,
    expanded_coarse_tile_capacity=4096,
    expanded_fine_tile_capacity=16384,
    expanded_limb_samples=32,
    moment_mode="two_coefficient",
    cartesian_backend="auto",
    root_backend="auto",
):
    """Dispatch safely-far epochs to a differentiable hexadecapole expansion.

    Method codes are 0 for hexadecapole, 1 for Cartesian inverse rays, 2 for
    polar inverse rays, and 3 for source-plane quadrature. Source-plane
    quadrature is attempted only when fixed-capacity image support is invalid,
    and is accepted only when its independent coarse/fine check converges.
    ``expanded_cartesian_fallback=True`` adds a costly coarse/fine Cartesian
    retry after source-plane rejection; it is disabled by default so trajectory
    callers can bucket only failed epochs into that executable.
    The multipole error check remains an empirical guard rather than a proof
    that no unresolved caustic lies inside the source.
    """

    require_x64()
    source_x, source_y, separation, mass_ratio, source_radius, limb_c, limb_d = (
        as_float64(value)
        for value in (
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
        )
    )
    hexadecapole = binary_hexadecapole(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        limb_c,
        limb_d,
        root_backend=root_backend,
    )
    accept_multipole, polar_allowed = _multipole_dispatch_masks(
        hexadecapole,
        mass_ratio,
        source_radius,
        absolute_tolerance,
        relative_tolerance,
        multipole_safety_factor,
        polar_min_mass_ratio,
        polar_require_topology_stable,
    )

    def multipole_path(_):
        return HybridMagnificationResult(
            magnification=hexadecapole.magnification,
            method=jnp.asarray(0, dtype=jnp.int32),
            estimated_error=hexadecapole.estimated_error,
            support_valid=jnp.asarray(True),
            used_multipole=jnp.asarray(True),
            used_polar=jnp.asarray(False),
            used_source_plane=jnp.asarray(False),
            used_expanded_cartesian=jnp.asarray(False),
        )

    def inverse_ray_path(_):
        result = binary_inverse_ray_auto(
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
            resolution=resolution,
            tile_size=tile_size,
            tile_capacity=tile_capacity,
            limb_samples=limb_samples,
            kernel=kernel,
            polar_resolution=polar_resolution,
            polar_angular_bins=polar_angular_bins,
            polar_radial_capacity=polar_radial_capacity,
            polar_band_capacity=polar_band_capacity,
            polar_limb_samples=polar_limb_samples,
            polar_angular_chunk_size=polar_angular_chunk_size,
            polar_boundary_subdivision=polar_boundary_subdivision,
            polar_magnification_threshold=polar_magnification_threshold,
            polar_max_source_radius=polar_max_source_radius,
            polar_force=polar_force,
            polar_allowed=polar_allowed,
            polar_fallback_on_overflow=polar_fallback_on_overflow,
            moment_mode=moment_mode,
            cartesian_backend=cartesian_backend,
            root_backend=root_backend,
        )

        def image_plane_result(_):
            return HybridMagnificationResult(
                magnification=result.magnification,
                method=jnp.where(result.used_polar, 2, 1).astype(jnp.int32),
                estimated_error=jnp.asarray(jnp.nan),
                support_valid=jnp.asarray(True),
                used_multipole=jnp.asarray(False),
                used_polar=result.used_polar,
                used_source_plane=jnp.asarray(False),
                used_expanded_cartesian=jnp.asarray(False),
            )

        def source_plane_result(_):
            source_plane = binary_source_plane_quadrature(
                source_x,
                source_y,
                separation,
                mass_ratio,
                source_radius,
                limb_c,
                limb_d,
                rule=source_plane_rule,
                coarse_order=source_plane_coarse_order,
                fine_order=source_plane_fine_order,
                angular_multiplier=source_plane_angular_multiplier,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                root_backend=root_backend,
            )
            valid = source_plane.converged

            def accepted_source_plane(_):
                return HybridMagnificationResult(
                    magnification=source_plane.magnification,
                    method=jnp.asarray(3, dtype=jnp.int32),
                    estimated_error=source_plane.estimated_error,
                    support_valid=jnp.asarray(True),
                    used_multipole=jnp.asarray(False),
                    used_polar=jnp.asarray(False),
                    used_source_plane=jnp.asarray(True),
                    used_expanded_cartesian=jnp.asarray(False),
                )

            def rejected_source_plane(_):
                return HybridMagnificationResult(
                    magnification=jnp.asarray(jnp.nan),
                    method=jnp.asarray(3, dtype=jnp.int32),
                    estimated_error=source_plane.estimated_error,
                    support_valid=jnp.asarray(False),
                    used_multipole=jnp.asarray(False),
                    used_polar=jnp.asarray(False),
                    used_source_plane=jnp.asarray(True),
                    used_expanded_cartesian=jnp.asarray(False),
                )

            def expanded_cartesian_result(_):
                coarse = _binary_inverse_ray(
                    source_x,
                    source_y,
                    separation,
                    mass_ratio,
                    source_radius,
                    limb_c,
                    limb_d,
                    resolution=expanded_coarse_resolution,
                    tile_size=tile_size,
                    tile_capacity=expanded_coarse_tile_capacity,
                    limb_samples=expanded_limb_samples,
                    kernel=kernel,
                    moment_mode=moment_mode,
                    cartesian_backend=cartesian_backend,
                    root_backend=root_backend,
                )
                fine = _binary_inverse_ray(
                    source_x,
                    source_y,
                    separation,
                    mass_ratio,
                    source_radius,
                    limb_c,
                    limb_d,
                    resolution=expanded_fine_resolution,
                    tile_size=tile_size,
                    tile_capacity=expanded_fine_tile_capacity,
                    limb_samples=expanded_limb_samples,
                    kernel=kernel,
                    moment_mode=moment_mode,
                    cartesian_backend=cartesian_backend,
                    root_backend=root_backend,
                )
                error = jnp.abs(fine.magnification - coarse.magnification)
                retry_budget = absolute_tolerance + relative_tolerance * jnp.maximum(
                    jnp.abs(fine.magnification), 1.0
                )
                retry_valid = (
                    coarse.support_valid
                    & fine.support_valid
                    & jnp.isfinite(error)
                    & (error <= retry_budget)
                )
                return HybridMagnificationResult(
                    magnification=jnp.where(retry_valid, fine.magnification, jnp.nan),
                    method=jnp.asarray(1, dtype=jnp.int32),
                    estimated_error=error,
                    support_valid=retry_valid,
                    used_multipole=jnp.asarray(False),
                    used_polar=jnp.asarray(False),
                    used_source_plane=jnp.asarray(False),
                    used_expanded_cartesian=jnp.asarray(True),
                )

            if expanded_cartesian_fallback:
                return jax.lax.cond(
                    valid,
                    accepted_source_plane,
                    expanded_cartesian_result,
                    None,
                )
            return jax.lax.cond(
                valid,
                accepted_source_plane,
                rejected_source_plane,
                None,
            )

        def invalid_image_plane_result(_):
            return HybridMagnificationResult(
                magnification=jnp.asarray(jnp.nan),
                method=jnp.where(result.used_polar, 2, 1).astype(jnp.int32),
                estimated_error=jnp.asarray(jnp.nan),
                support_valid=jnp.asarray(False),
                used_multipole=jnp.asarray(False),
                used_polar=result.used_polar,
                used_source_plane=jnp.asarray(False),
                used_expanded_cartesian=jnp.asarray(False),
            )

        if source_plane_fallback:
            return jax.lax.cond(
                result.support_valid,
                image_plane_result,
                source_plane_result,
                None,
            )
        return jax.lax.cond(
            result.support_valid,
            image_plane_result,
            invalid_image_plane_result,
            None,
        )

    return jax.lax.cond(accept_multipole, multipole_path, inverse_ray_path, None)
