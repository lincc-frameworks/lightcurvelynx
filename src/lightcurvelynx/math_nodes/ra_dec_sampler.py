"""Samplers used for generating (RA, dec) coordinates."""

import logging
from pathlib import Path

import numpy as np
from astropy.coordinates import Angle, SkyCoord
from cdshealpix.nested import healpix_to_skycoord
from citation_compass import CiteClass
from mocpy import MOC

from lightcurvelynx.math_nodes.given_sampler import TableSampler
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.obstable.obs_table import ObsTable


class UniformRADEC(NumpyRandomFunc):
    """A FunctionNode that uniformly samples (RA, dec) over a sphere,

    Attributes
    ----------
    use_degrees : bool
        The default return unit. If True returns samples in degrees.
        Otherwise, if False, returns samples in radians.
    """

    def __init__(self, outputs=None, seed=None, use_degrees=True, **kwargs):
        self.use_degrees = use_degrees

        # Override key arguments. We create a uniform sampler function, but
        # won't need it because the subclass overloads compute().
        func_name = "uniform"
        outputs = ["ra", "dec"]
        super().__init__(func_name, outputs=outputs, seed=seed, **kwargs)

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Return the given values.

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : dict, optional
            Additional function arguments.

        Returns
        -------
        results : any
            The result of the computation. This return value is provided so that testing
            functions can easily access the results.
        """
        rng = rng_info if rng_info is not None else self._rng

        # Generate the random (RA, dec) lists.
        ra = rng.uniform(0.0, 2.0 * np.pi, size=graph_state.num_samples)
        dec = np.arcsin(rng.uniform(-1.0, 1.0, size=graph_state.num_samples))
        if self.use_degrees:
            ra = np.degrees(ra)
            dec = np.degrees(dec)

        # If we are generating a single sample, return floats.
        if graph_state.num_samples == 1:
            ra = ra[0]
            dec = dec[0]

        # Set the outputs and return the results. This takes the place of
        # function node's _save_results() function because we know the outputs.
        graph_state.set(self.node_string, "ra", ra)
        graph_state.set(self.node_string, "dec", dec)
        return [ra, dec]


class ObsTableRADECSampler(TableSampler):
    """A FunctionNode that randomly samples RA and dec (and any extra columns) from
    around given data. This node is used for both sampling from a list of pointings
    (where the radius is the field of view) or sampling from a list of true objects.

    Parameters
    ----------
    data : ObsTable, Pandas DataFrame, NestedFrame, or dict
        The data to use for sampling. Must contain 'ra' and 'dec' columns.
    extra_cols : list of str, optional
        A list of extra column names to include in the sampling.
        Default: None
    radius : float, optional
        The sampling radius around the given points. If the points represent the
        center of a pointing, this is the radius of the field of view in degrees.
        If the points represent exact true objects, this can be a noise factor.
        Use 0.0 to return the exact given points.
        If None and data is an ObsTable, uses the value from the ObsTable.
        Default: None
    **kwargs : dict, optional
        Additional keyword arguments to pass to the parent class constructor.
    """

    def __init__(self, data, *, extra_cols=None, radius=None, **kwargs):
        if isinstance(data, ObsTable):
            if radius is None:
                radius = data.radius
            else:
                logging.getLogger(__name__).info(
                    f"Using provided radius {radius} instead of ObsTable radius {data.radius}."
                )
        if radius is None or radius < 0.0:
            raise ValueError(f"Invalid radius: {radius}")
        self.radius = radius

        # Start with RA, dec, and (optionally) time.
        data_dict = {
            "ra": data["ra"],
            "dec": data["dec"],
        }
        if "time" in data:
            data_dict["time"] = data["time"]

        # Add any extra columns (without duplicates).
        if extra_cols is not None:
            for col in extra_cols:
                if col not in data_dict:
                    data_dict[col] = data[col]

        super().__init__(data_dict, in_order=False, **kwargs)

    @classmethod
    def from_hats(cls, path, *, radius=None, extra_cols=None, **kwargs):
        """Create a GivenRADECSampler from the observations in a HATS Catalog.

        Note
        ----
        If you have an existing Dask client, it may be used.
        See LSDB documentation for details: https://docs.lsdb.io/en/latest/

        Parameters
        ----------
        path : str or Path
            The base path of the HATS data directory.
        radius : float, optional
            The sampling radius (noise factor) in degrees. Use 0.0 to return
            the exact pointings.
            Default: 0.0
        extra_cols : list of str, optional
            A list of extra column names to include in the sampling.
            Default: None
        **kwargs : dict, optional
            Additional keyword arguments to pass to the constructor.

        Returns
        -------
        GivenRADECSampler
            The created GivenRADECSampler object.
        """
        # See if the (optional) LSDB package is installed.
        try:
            from lsdb import read_hats
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "The lsdb package is required to read HATS catalogs. "
                "Please install it via 'pip install lsdb' or 'conda install conda-forge::lsdb'."
            ) from err

        # Compute the full list of columns to load from HATS.
        cols_to_load = ["ra", "dec"]
        if extra_cols is not None:
            cols_to_load.extend(extra_cols)
        columns = list(set(cols_to_load))  # Remove any duplicates.

        data = read_hats(path, columns=columns).compute()
        return cls(data, extra_cols=extra_cols, radius=radius, **kwargs)

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Return the given values.

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : dict, optional
            Additional function arguments.

        Returns
        -------
        results : any
            The result of the computation. This return value is provided so that testing
            functions can easily access the results.
        """
        # Sample the center RA, dec, and times without the radius. Results is a vector
        # of arrays, one per output column. The first two are guaranteed to be RA and dec.
        results = super().compute(graph_state, rng_info=rng_info, **kwargs)

        if self.radius > 0.0:
            rng = rng_info if rng_info is not None else np.random.default_rng()
            center = SkyCoord(ra=results[0], dec=results[1], unit="deg")

            # Add an offset from the center of the pointing defined by an offset angle (phi)
            # and offset radius (theta).  Both are defined in radians.
            offset_dir = rng.uniform(0.0, 2.0 * np.pi, size=graph_state.num_samples)
            cos_radius = np.cos(np.radians(self.radius))
            offset_amt = np.arccos(rng.uniform(cos_radius, 1.0, size=graph_state.num_samples))
            new_coords = center.directional_offset_by(
                position_angle=Angle(offset_dir, "radian"),
                separation=Angle(offset_amt, "radian"),
            )

            # Replace the results' RA and dec with the new pointing (in degrees).
            results[0] = new_coords.ra.deg
            results[1] = new_coords.dec.deg

            # Resave the results to transfer them to the graph state.
            self._save_results(results, graph_state)

        return results


class ObsTableUniformRADECSampler(NumpyRandomFunc):
    """A FunctionNode that samples RA and dec uniformly from the area covered
    by an ObsTable.  RA and dec are returned in degrees.

    Note
    ----
    This uses rejection sampling where it randomly guesses an (RA, dec) then checks if that
    point falls within the survey. If not, it repeats the process until it finds a valid point
    or reaches `max_iterations` iterations (then returns the last sample). This sampling method
    can be quite slow or even generate out-of-survey samples if the coverage is small.

    Attributes
    ----------
    data : ObsTable
        The ObsTable object to use for sampling.
    radius : float
        The radius of the field of view of the observations in degrees.
    max_iterations : int
        The maximum number of iterations to perform. Default: 1000

    Parameters
    ----------
    data : ObsTable
        The ObsTable object to use for sampling.
    radius : float, optional
        The search radius around the center of the pointing. If None, uses the
        value from the ObsTable.
    outputs : list of str, optional
        The list of output names. Default: ["ra", "dec"]
    seed : int, optional
        The random seed to use for the internal random number generator. Default: None
    max_iterations : int, optional
        The maximum number of iterations to perform. Default: 1000
    **kwargs : dict, optional
        Additional keyword arguments to pass to the parent class constructor.
    """

    def __init__(self, data, *, radius=None, outputs=None, seed=None, max_iterations=1000, **kwargs):
        if isinstance(data, ObsTable):
            if radius is None:
                radius = data.radius
            else:
                logging.getLogger(__name__).info(
                    f"Using provided radius {radius} instead of ObsTable radius {data.radius}."
                )
        if radius is None or radius < 0.0:
            raise ValueError(f"Invalid radius: {radius}")
        self.radius = radius

        if len(data) == 0:  # pragma: no cover
            raise ValueError("ObsTable data cannot be empty.")
        self.data = data

        if max_iterations <= 0:
            raise ValueError("Invalid max_iterations: {max_iterations}")
        self.max_iterations = max_iterations

        # Override key arguments. We create a uniform sampler function, but
        # won't need it because the subclass overloads compute().
        func_name = "uniform"
        outputs = ["ra", "dec"]
        super().__init__(func_name, outputs=outputs, seed=seed, **kwargs)

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Return the given values.

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : dict, optional
            Additional function arguments.

        Returns
        -------
        (ra, dec) : tuple of floats or np.ndarray
            If a single sample is generated, returns a tuple of floats. Otherwise,
            returns a tuple of np.ndarrays.
        """
        rng = rng_info if rng_info is not None else self._rng

        ra = np.zeros(graph_state.num_samples)
        dec = np.zeros(graph_state.num_samples)
        mask = np.full(graph_state.num_samples, False)
        num_missing = graph_state.num_samples

        # Rejection sampling to ensure the samples are within the ObsTable coverage.
        # This can take many iterations if the coverage is small.
        iter_num = 1
        while num_missing > 0 and iter_num <= self.max_iterations:
            # Generate new samples for the missing ones.
            new_ra = np.degrees(rng.uniform(0.0, 2.0 * np.pi, size=num_missing))
            new_dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=num_missing)))
            ra[~mask] = new_ra
            dec[~mask] = new_dec

            # Check if the new samples are within the ObsTable coverage. We don't
            # need to recheck the ones that were already valid.
            mask[~mask] = self.data.is_observed(new_ra, new_dec, radius=self.radius)
            num_missing = np.sum(~mask)
            iter_num += 1

        # If we are generating a single sample, return floats.
        if graph_state.num_samples == 1:
            ra = ra[0]
            dec = dec[0]

        # Set the outputs and return the results. This takes the place of
        # function node's _save_results() function because we know the outputs.
        graph_state.set(self.node_string, "ra", ra)
        graph_state.set(self.node_string, "dec", dec)
        return (ra, dec)


class ApproximateMOCSampler(NumpyRandomFunc, CiteClass):
    """A FunctionNode that samples RA and dec (approximately) from the coverage of
    a MOCPy Multi-Order Coverage Map object.

    The depth parameter controls the approximation level. Higher depths provide
    better accuracy but require more memory and computation time. We recommend at
    least depth=12 for reasonable accuracy.

    References
    ----------
    * MOCPY: https://github.com/cds-astro/mocpy/
    * CDS Healpix: https://github.com/cds-astro/cds-healpix-python
    * MOC: Pierre Fernique, Thomas Boch, Tom Donaldson, Daniel Durand , Wil O'Mullane, Martin Reinecke,
    and Mark Taylor. MOC - HEALPix Multi-Order Coverage map Version 1.0. IVOA Recommendation 02 June 2014,
    pages 602, Jun 2014. doi:10.5479/ADS/bib/2014ivoa.spec.0602F.

    Attributes
    ----------
    healpix_list : list of int
        The list of healpix pixel IDs that cover the MOC at the given depth.
    depth : int
        The healpix depth to use as an approximation. Must be [2, 29].
    """

    def __init__(self, moc, *, outputs=None, seed=None, depth=12, **kwargs):
        if depth < 2 or depth > 29:
            raise ValueError("Depth must be [2, 29]. Received {depth}")
        self.depth = depth
        self.healpix_list = moc.to_order(depth).flatten()

        # Override key arguments. We create a uniform sampler function, but
        # won't need it because the subclass overloads compute().
        func_name = "uniform"
        outputs = ["ra", "dec"]
        super().__init__(func_name, outputs=outputs, seed=seed, **kwargs)

    @classmethod
    def from_file(cls, filename, format="fits", **kwargs):
        """Create an ApproximateMOCSampler from a MOC file.

        This file can be created from a mocpy.MOC object using its save() function.

        Parameters
        ----------
        filename : str or Path
            The path to the MOC file. Supported formats include FITS, JSON, and ASCII.
        format : str, optional
            The format of the MOC file. Supported formats include 'fits', 'json', and
            'ascii'. Default is 'fits'.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the constructor.

        Returns
        -------
        ApproximateMOCSampler
            The created ApproximateMOCSampler object.
        """
        filename = Path(filename)
        if not filename.is_file():
            raise FileNotFoundError(f"MOC file not found: {filename}")

        moc = MOC.load(filename, format=format)
        return cls(moc, **kwargs)

    @classmethod
    def from_obstable(
        cls,
        obstable,
        *,
        depth=12,
        use_footprint=False,
        radius=None,
        **kwargs,
    ):
        """Create an ApproximateMOCSampler from an ObsTable object.

        The depth parameter controls the approximation level. Higher depths provide
        better accuracy but require more memory and computation time. We recommend at
        least depth=12 for reasonable accuracy.

        Parameters
        ----------
        obstable : ObsTable
            The ObsTable object to use for creating the MOC.
        depth : int, optional
            The healpix depth to use as an approximation. Must be [2, 29].
            Default: 12
        radius : float, optional
            The radius to use for each image (in degrees). Only used if use_footprint
            is False. If None, the radius from the survey values will be used.
        use_footprint : bool, optional
            Whether to use the detector footprint to build the MOC. If True, the
            footprint will be used to compute the MOC regions for each pointing.
            If False, a simple cone with the given radius will be used.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the constructor.

        Returns
        -------
        ApproximateMOCSampler
            The created ApproximateMOCSampler object.
        """
        moc = obstable.build_moc(
            radius=radius,
            use_footprint=use_footprint,
            duplicate_threshold=10.0 / 3600.0,  # 10 arcsec
            max_depth=depth,
        )
        return cls(moc, depth=depth, **kwargs)

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Return the given values.

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : dict, optional
            Additional function arguments.

        Returns
        -------
        (ra, dec) : tuple of floats or np.ndarray
            If a single sample is generated, returns a tuple of floats. Otherwise,
            returns a tuple of np.ndarrays.
        """
        rng = rng_info if rng_info is not None else self._rng

        # Choose a starting pixel ID for each sample. Then randomly traverse
        # down the healpix tree by moving to one of the children pixels until
        # we reach level=29 (approximately 4.5 * 10^18 possible locations).
        pixel_ids = rng.choice(self.healpix_list, size=graph_state.num_samples).astype(np.uint64)
        start_pixel_ids29 = np.left_shift(pixel_ids, 2 * (29 - self.depth))
        offset_range = np.uint64(1) << np.uint64(2 * (29 - self.depth))
        pixel_ids29 = start_pixel_ids29 + rng.integers(
            offset_range, size=graph_state.num_samples, dtype=np.uint64
        )

        # Convert back the healpix centers to RA and dec.
        coords = healpix_to_skycoord(pixel_ids29, depth=29)
        ra = coords.ra.deg
        dec = coords.dec.deg

        # If we are generating a single sample, return floats.
        if graph_state.num_samples == 1:
            ra = ra[0]
            dec = dec[0]

        # Set the outputs and return the results. This takes the place of
        # function node's _save_results() function because we know the outputs.
        graph_state.set(self.node_string, "ra", ra)
        graph_state.set(self.node_string, "dec", dec)
        return (ra, dec)
