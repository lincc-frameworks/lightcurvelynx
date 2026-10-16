"""A resolved strong-lens model that wraps a single physical source. Each sampled source has
its own lens system, and every lensed image becomes its own row (and light curve) in the
simulation, with its own sky position, arrival time, and magnification.
"""

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame

from lightcurvelynx.base_models import FunctionNode
from lightcurvelynx.math_nodes.basic_math_node import BasicMathNode
from lightcurvelynx.models.multi_object_model import MultiObjectModel


class _LensImageExpansionNode(FunctionNode):
    """Expand each lens system in the GraphState into one row per active image.

    Each system's active images (the leading ``num_images`` entries of the fixed-width
    arrays) are sorted by arrival time and their delays are normalized so the earliest
    image has a delay of zero. The GraphState is then repeated so every value sampled so
    far is shared by all of a system's image rows.

    Parameters
    ----------
    image_x, image_y : parameter
        Fixed-width image offsets (east, north) from the lens origin in arcseconds.
    macro_magnifications : parameter
        Fixed-width absolute (dimensionless) image magnifications.
    time_delays : parameter
        Fixed-width image arrival delays in observer-frame days.
    num_images : parameter, optional
        The number of active leading entries per system. If None, all entries are active.
    **kwargs : dict, optional
        Additional keyword arguments, such as ``node_label``.
    """

    def __init__(self, *, image_x, image_y, macro_magnifications, time_delays, num_images=None, **kwargs):
        # Inputs and outputs share one parameter namespace, so the inputs are prefixed.
        inputs = {
            "all_image_x": image_x,
            "all_image_y": image_y,
            "all_macro_magnifications": macro_magnifications,
            "all_time_delays": time_delays,
        }
        if num_images is not None:
            inputs["num_images"] = num_images

        super().__init__(
            self._non_func,
            outputs=["system_id", "image_id", "image_x", "image_y", "macro_magnification", "time_delay"],
            **inputs,
            **kwargs,
        )

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Expand the GraphState and save the per-image values.

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values. This object is modified
            in place (expanded) as it is sampled.
        rng_info : numpy.random._generator.Generator, optional
            Unused.
        **kwargs : dict, optional
            Additional function arguments.

        Returns
        -------
        list
            The per-image system_id, image_id, image_x, image_y, macro_magnification,
            and time_delay values.
        """
        args = self._build_inputs(graph_state, **kwargs)
        delays = np.atleast_2d(np.asarray(args["all_time_delays"], dtype=float))
        num_systems, width = delays.shape
        if "num_images" in args:
            counts = np.atleast_1d(np.asarray(args["num_images"])).astype(int)
        else:
            counts = np.full(num_systems, width)

        # Sort each system's active images by delay, pushing the inactive padding to the end.
        active = np.arange(width) < counts[:, np.newaxis]
        order = np.argsort(np.where(active, delays, np.inf), axis=1, kind="stable")

        def take(values):
            # After sorting, each row's active entries lead the row, so the active mask
            # selects them in system-major order.
            values = np.atleast_2d(np.asarray(values, dtype=float))
            return np.take_along_axis(values, order, axis=1)[active]

        starts = np.cumsum(counts) - counts  # The index of each system's first image.
        time_delay = take(delays)
        time_delay -= np.repeat(time_delay[starts], counts)
        sample_offset = graph_state.sample_offset or 0
        system_id = np.repeat(np.arange(num_systems) + sample_offset, counts)
        image_id = np.arange(np.sum(counts)) - np.repeat(starts, counts)

        graph_state.repeat(counts)
        results = [
            system_id,
            image_id,
            take(args["all_image_x"]),
            take(args["all_image_y"]),
            take(args["all_macro_magnifications"]),
            time_delay,
        ]
        if graph_state.num_samples == 1:
            results = [value[0] for value in results]
        self._save_results(results, graph_state)
        return results


class _LensImagePositionNode(FunctionNode):
    """Compute the lens origin and an image's sky position from the unlensed source position.

    The source and image offsets are measured in the lens-centred frame
    (``SkyOffsetFrame(origin=lens)``). The lens origin is the point from which the source
    appears at (``source_x``, ``source_y``). It is found with a closed-form first guess
    followed by two fixed-point corrections, which are accurate to better than 1e-6 arcsec
    for arcsecond offsets, even near the poles.

    Parameters
    ----------
    source_ra, source_dec : parameter
        The unlensed source position in degrees.
    source_x, source_y : parameter
        The source offset (east, north) from the lens origin in arcseconds.
    image_x, image_y : parameter
        The image offset (east, north) from the lens origin in arcseconds.
    **kwargs : dict, optional
        Additional keyword arguments, such as ``node_label``.
    """

    def __init__(self, *, source_ra, source_dec, source_x, source_y, image_x, image_y, **kwargs):
        super().__init__(
            self._non_func,
            outputs=["lens_ra", "lens_dec", "ra", "dec"],
            source_ra=source_ra,
            source_dec=source_dec,
            source_x=source_x,
            source_y=source_y,
            image_x=image_x,
            image_y=image_y,
            **kwargs,
        )

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Compute and save the lens and image positions.

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        rng_info : numpy.random._generator.Generator, optional
            Unused.
        **kwargs : dict, optional
            Additional function arguments.

        Returns
        -------
        list
            The lens RA, lens Dec, image RA, and image Dec in degrees.
        """
        args = self._build_inputs(graph_state, **kwargs)
        source_x = np.asarray(args["source_x"]) * u.arcsec
        source_y = np.asarray(args["source_y"]) * u.arcsec
        source = SkyCoord(np.asarray(args["source_ra"]) * u.deg, np.asarray(args["source_dec"]) * u.deg)

        # Start from the source shifted back by its offset (exact on the equator), then
        # move the lens until the source appears at the requested lens-frame offset.
        lens = source.spherical_offsets_by(-source_x, -source_y)
        for _ in range(2):
            seen = source.transform_to(SkyOffsetFrame(origin=lens))
            lens = lens.spherical_offsets_by(seen.lon.wrap_at(180 * u.deg) - source_x, seen.lat - source_y)

        image = lens.spherical_offsets_by(
            np.asarray(args["image_x"]) * u.arcsec,
            np.asarray(args["image_y"]) * u.arcsec,
        )
        results = [lens.ra.deg, lens.dec.deg, image.ra.deg, image.dec.deg]
        self._save_results(results, graph_state)
        return results


class ResolvedStrongLensModel(MultiObjectModel):
    """A strongly lensed source whose images are resolved into separate objects.

    Each sampled source has its own lens system. Sampling expands every system into one
    row per active image, so ``num_samples`` counts lens systems while evaluation and
    simulation results contain one row (and light curve) per image. Each image row has
    its own ``ra``/``dec`` (the image position), ``t0`` (the source ``t0`` plus the image's
    delay), and magnification. The source model itself is never modified.

    The image inputs follow a fixed-width convention (``I`` entries per system, of which
    the leading ``num_images`` are active and the rest are padding that is ignored):

    ==================== ========== ========= ==========================================
    Parameter            One system S systems Meaning
    ==================== ========== ========= ==========================================
    source_x, source_y   scalar     (S,)      Source offset (east, north) from the lens
                                              origin in arcseconds.
    image_x, image_y     (I,)       (S, I)    Image offsets (east, north) from the lens
                                              origin in arcseconds.
    macro_magnifications (I,)       (S, I)    Absolute (dimensionless) magnifications.
    time_delays          (I,)       (S, I)    Arrival delays in observer-frame days.
    num_images           int        (S,)      Number of active leading entries.
    ==================== ========== ========= ==========================================

    Offsets are measured in the lens-centred frame. The source model's own ``ra``/``dec`` is
    the unlensed source position, from which the lens origin and image positions are
    derived. Delays are normalized so the earliest image has a delay of zero, and
    ``image_id`` numbers the images in arrival order. Use
    ``param_cols=["<label>.system_id", "<label>.image_id"]`` in ``simulate_lightcurves`` to
    regroup images by system.

    Sampling semantics: the source's parameters (including those of effects attached to the
    source, and of rest-frame effects added through this wrapper) are sampled once per
    system and shared by all of its images. Parameters reachable only through this wrapper,
    such as those of observer-frame effects added to it or nodes linked to its ``ra``/``dec``,
    are sampled per image.

    Notes
    -----
    * This model must be the top-level model passed to ``simulate_lightcurves``; it is not
      supported as a child of another multi-object model. Composite sources (any
      ``MultiObjectModel``, e.g. a host plus a transient) are not supported.
    * Each image is evaluated separately at ``t - time_delay``. A source that draws new
      random numbers at evaluation time gets independent randomness per image, and models
      that re-zero their time axis (e.g. ``EzTaoXWrapperModel``) will not show delays.
      Deterministic transient models are unaffected.
    * No seeing, PSF, or blending is modeled; every image is treated as resolved.

    Parameters
    ----------
    source_model : BasePhysicalModel
        The (single) physical source to lens. It is evaluated once per image.
    source_x, source_y : parameter
        The source offset (east, north) from the lens origin in arcseconds.
    image_x, image_y : parameter
        The fixed-width image offsets (east, north) from the lens origin in arcseconds.
    macro_magnifications : parameter
        The fixed-width absolute (dimensionless) image magnifications.
    time_delays : parameter
        The fixed-width image arrival delays in observer-frame days.
    num_images : parameter, optional
        The number of active leading image entries. If None, all entries are active.
    node_label : str, optional
        The label for this node.

    Attributes
    ----------
    source_model : BasePhysicalModel
        The lensed source.
    """

    def __init__(
        self,
        source_model,
        *,
        source_x,
        source_y,
        image_x,
        image_y,
        macro_magnifications,
        time_delays,
        num_images=None,
        node_label=None,
    ):
        images = _LensImageExpansionNode(
            image_x=image_x,
            image_y=image_y,
            macro_magnifications=macro_magnifications,
            time_delays=time_delays,
            num_images=num_images,
            node_label=None if node_label is None else f"{node_label}_images",
        )
        positions = _LensImagePositionNode(
            source_ra=source_model.ra,
            source_dec=source_model.dec,
            source_x=source_x,
            source_y=source_y,
            image_x=images.image_x,
            image_y=images.image_y,
            node_label=None if node_label is None else f"{node_label}_positions",
        )
        image_t0 = BasicMathNode(
            "t0 + time_delay",
            t0=source_model.t0,
            time_delay=images.time_delay,
            node_label=None if node_label is None else f"{node_label}_t0",
        )
        super().__init__(
            [source_model],
            ra=positions.ra,
            dec=positions.dec,
            redshift=source_model.redshift,
            t0=image_t0,
            distance=source_model.distance,
            node_label=node_label,
        )
        self.source_model = source_model

        extra_params = {
            "system_id": (images.system_id, "Index of the lens system."),
            "image_id": (images.image_id, "Index of the image within its system, in arrival order."),
            "image_x": (images.image_x, "Image offset east of the lens origin (arcsec)."),
            "image_y": (images.image_y, "Image offset north of the lens origin (arcsec)."),
            "macro_magnification": (images.macro_magnification, "Absolute image magnification."),
            "time_delay": (images.time_delay, "Image delay after the earliest image (days)."),
            "lens_ra": (positions.lens_ra, "Right ascension of the lens origin (degrees)."),
            "lens_dec": (positions.lens_dec, "Declination of the lens origin (degrees)."),
            "source_x": (source_x, "Source offset east of the lens origin (arcsec)."),
            "source_y": (source_y, "Source offset north of the lens origin (arcsec)."),
        }
        for name, (setter, description) in extra_params.items():
            self.add_parameter(name, setter, description=description, allow_gradient=False)

        # The source applies its own redshift; the wrapper must not apply it again.
        self.apply_redshift = False
