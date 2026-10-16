"""A resolved strong-lens model that wraps a single physical source. Each sampled source has
its own lens system, and every lensed image becomes its own row (and light curve) in the
simulation, with its own sky position, arrival time, and magnification.
"""

import numpy as np

from lightcurvelynx.base_models import FunctionNode


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
