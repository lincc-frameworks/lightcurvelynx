"""The Passband and PassbandGroup objects store information about the filters used
to convert flux densities (as a function of wavelength) to bandfluxes. They also provide
methods for loading and manipulating passband data.
"""

import logging
from pathlib import Path
from typing import Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate
from astropy.io.votable import parse
from citation_compass import cite_function

from lightcurvelynx import _LIGHTCURVELYNX_BASE_DATA_DIR
from lightcurvelynx.consts import lsst_filter_plot_colors
from lightcurvelynx.utils.data_download import download_data_file_if_needed

logger = logging.getLogger(__name__)


class PassbandGroup:
    """A group of passbands.

    The given passbands can come from a single survey or multiple surveys. As such, a given filter may appear
    in multiple passbands in a passband group. For example we might generate data from a combination of Rubin
    and DECCAM and want to use both their r filters. Thus the primary mapping is done by the passband's full
    name “{SURVEY}_{FILTER}”. Lookups by filter name are permitted in cases where the filter only occurs once
    in the group and thus the desired passband is unambiguous.

    Attributes
    ----------
    passbands : dict of Passband
        A dictionary of Passband objects, where the keys are the full_names of the passbands (eg, "LSST_u").
    table_dir : str
        The path to the directory containing the passband tables.
    waves : np.ndarray
        The union of all wavelengths in the passbands.
    _in_band_wave_indices : dict
        A dictionary mapping the passband name (eg, "LSST_u") to the indices of that specific
        passband's wavelengths in the full waves list.
    _filter_to_name : dict
        A dictionary mapping the filter name to a list of matching passband full names.
    """

    def __init__(self, given_passbands, filters=None, **kwargs):
        """Construct a PassbandGroup object.

        Parameters
        ----------
        given_passbands : Passband, list of Passband, or dict
            A list of all the passbands to include in this group. These can either be
            Passband objects or dictionaries of passband parameters with the following keys:
            - required:
                - survey : str - The name of the survey to which the passband belongs, e.g., "LSST".
                - filter_name : str - The name of the filter, e.g., "u".
            - one of the following:
                - table_data : np.ndarray - The transmission table data as a (N, 2) array where
                    the first column is wavelengths and the second column is transmission values.
                - table_path : str or Path - The path of the file from which to load the passband.
                - table_url : str - The URL from which to download the passband file.
            - optional:
                - delta_wave : float
                - trim_quantile : float
                - units : str (either 'nm' or 'A')
        filters : list, optional
            A list of filters to include in this PassbandGroup. If None, includes all filters.
            Otherwise drops filters that do not occur and throws an error if a filter is missing.
            Used for loading a subset of the filters.
        **kwargs
            Additional keyword arguments to pass to the Passband constructor.
        """
        self.passbands = {}
        self._in_band_wave_indices = {}
        self._filter_to_name = {}

        # If we are given a single Passband object or a single dictionary, wrap it in a list.
        if isinstance(given_passbands, Passband):
            given_passbands = [given_passbands]
        elif isinstance(given_passbands, dict):
            given_passbands = [given_passbands]
        elif given_passbands is None or len(given_passbands) == 0:
            raise ValueError("No passbands provided to PassbandGroup.")

        for pb in given_passbands:
            if isinstance(pb, Passband):
                # If we are given a Passband object, add it directly.
                self.passbands[pb.full_name] = pb
            elif isinstance(pb, dict):
                # If we are given a dictionary of parameters, create a Passband from it.
                params = pb.copy()

                # Add any missing parameters from kwargs.
                for key, value in kwargs.items():
                    if key not in params:
                        params[key] = value

                passband = Passband.from_file(**params)
                self.passbands[passband.full_name] = passband
            else:
                raise TypeError(f"Expected a Passband object or a dictionary of parameters. Got {type(pb)}.")

        # Prune any filters that are not on the given list and check for any missing filters.
        # We match on either the full name or the filter name.
        if filters is not None:
            filters = set(filters)
            filters_remaining = filters.copy()
            all_bands = list(self.passbands.keys())

            for pb_name in all_bands:
                pb_obj = self.passbands[pb_name]
                if pb_name in filters:
                    filters_remaining.discard(pb_name)
                elif pb_obj.filter_name in filters:
                    filters_remaining.discard(pb_obj.filter_name)
                else:
                    del self.passbands[pb_name]

            if len(filters_remaining) != 0:
                raise ValueError(f"The following filters were not found: {filters_remaining}")

        # Compute the unique points and bounds for the group.
        self._update_internal_data()

    def __str__(self) -> str:
        """Return a string representation of the PassbandGroup."""
        return f"PassbandGroup containing {len(self.passbands)} passbands: {', '.join(self.passbands.keys())}"

    def __len__(self) -> int:
        return len(self.passbands)

    def __getitem__(self, key):
        """Return the passband corresponding to a full name or filter name."""
        if key in self.passbands:
            return self.passbands[key]
        elif key in self._filter_to_name:
            # If we are looking up the passband by filter name, we check
            # that the filter only appears in a single Passband object.
            pb_list = self._filter_to_name[key]
            if len(pb_list) > 1:
                raise KeyError(
                    f"Filter {key} corresponds to multiple passbands: {pb_list}.\n"
                    "Lookup the passband by full name."
                )
            return self.passbands[pb_list[0]]
        else:
            raise KeyError(f"Unknown passband {key}")

    def __contains__(self, key):
        if key in self.passbands:
            return True
        elif key in self._filter_to_name:
            return True
        return False

    def __iter__(self):
        for pb in self.passbands.values():
            yield pb

    @property
    def filters(self) -> list:
        """Return a list of filter names in the passband group."""
        return list(self._filter_to_name.keys())

    @classmethod
    def from_preprocessed_file(cls, file_path: Union[str, Path], **kwargs):
        """Load a PassbandGroup from a single file containing multiple preprocessed passbands.

        The file should be a CSV file with columns: survey, filter_name, wavelength, transmission.
        The wavelengths should be in Angstroms.

        Parameters
        ----------
        file_path : str or Path
            The path to the file containing the passband data.
        **kwargs
            Additional keyword arguments to pass to the Passband constructor.

        Returns
        -------
        PassbandGroup
            A PassbandGroup object containing the passbands from the file.
        """
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"{file_path} is not a valid file.")

        # Load the table using pandas and check that it has the required columns.
        df = pd.read_csv(file_path)
        required_columns = {"survey", "filter_name", "wavelength", "transmission"}
        if not required_columns.issubset(df.columns):
            missing_cols = required_columns - set(df.columns)
            raise ValueError(f"File {file_path} is missing required columns: {missing_cols}")

        pb_list = []
        for (survey, filter_name), group in df.groupby(["survey", "filter_name"]):
            table_values = group[["wavelength", "transmission"]].to_numpy()
            pb = Passband(
                table_values,
                survey,
                filter_name,
                delta_wave=None,  # Preserve original grid
                trim_quantile=None,  # Do not trim
                units="A",  # Units are Angstroms
            )

            # Because we are loading from a pre-processed table (normalized system responses instead
            # of unnormalized system throughputs), we need to retain the original table.
            pb.normalized_system_response[:, 1] = table_values[:, 1]
            pb_list.append(pb)

        # Create a PassbandGroup from the DataFrame.
        return cls(pb_list, **kwargs)

    @classmethod
    def from_dir(
        cls,
        dir_path: Union[str, Path],
        *,
        filters: list | None = None,
        delta_wave: float | None = 5.0,
        trim_quantile: float | None = 1e-3,
        units: Literal["nm", "A"] | None = "A",
    ):
        """Load the passbands from a directory where the directorty name corresponds
        to the survey and the file names correspond to the filters:
        path_to_survey_dir/survey_name/filter_name.dat

        Parameters
        ----------
        dir_path : str or Path
            The path to the passband files including the survey directory.
        filters : list, optional
            A list of filters to include in this PassbandGroup. If None, includes all filters.
            Otherwise drops filters that do not occur and throws an error if a filter is missing.
            Used for loading a subset of the filters.
        delta_wave : float or None, optional
            The grid step of the wave grid, in angstroms.
            It is typically used to downsample transmission using linear interpolation.
            Default is 5 angstroms. If None the original grid is used.
        trim_quantile : float or None, optional
            The quantile to trim the transmission table by. For example, if trim_quantile is 1e-3, the
            transmission table will be trimmed to include only the central 99.8% of the area under the
            transmission curve.
        units : Literal['nm','A'], optional
            Denotes whether the wavelength units of the table are nanometers ('nm') or Angstroms ('A').
            By default 'A'. Does not affect the output units of the class, only the interpretation of the
            provided passband table.
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise ValueError(f"{dir_path} is not a valid directory.")

        # Iterate through the files in the directory.
        all_params = []
        for entry in dir_path.iterdir():
            if entry.is_file():
                filter_name = entry.stem

                # If a list of filters is provided, skip the ones that are not in the list.
                if filters is None or filter_name in filters:
                    params = {
                        "survey": dir_path.name,
                        "filter_name": filter_name,
                        "table_path": dir_path / entry,
                        "delta_wave": delta_wave,
                        "trim_quantile": trim_quantile,
                        "units": units,
                    }
                    all_params.append(params)

        return PassbandGroup(given_passbands=all_params, filters=filters)

    @classmethod
    def from_preset(cls, preset: str, *, table_dir=None, filters=None, **kwargs) -> None:
        """Create a passband group from a pre-defined set of passbands.

        Parameters
        ----------
        preset : str
            The name of the pre-defined set of passbands to load.
        table_dir : str, optional
            The path to the directory in which to store cached passband tables. If the passband
            exists in this directory, it will be loaded from there; otherwise it will be downloaded
            and saved in that directory.
            The full path to the tables will be {table_dir}/{survey}/{filter_name}.dat.
        filters : list, optional
            A list of filters to include in this PassbandGroup. If None, includes all filters.
            Otherwise drops filters that do not occur and throws an error if a filter is missing.
            Used for loading a subset of the filters.
        **kwargs
            Additional keyword arguments to pass to the Passband constructor.
        """
        logger.info(f"Loading passbands from preset {preset}")

        # If we do not have the base table directory, use the default.
        table_dir = (
            Path(table_dir) if table_dir is not None else Path(_LIGHTCURVELYNX_BASE_DATA_DIR, "passbands")
        )

        preset = preset.lower()
        if preset == "lsst":
            passbands = PassbandGroup._lsst_load_preset(table_dir, **kwargs)
        elif preset == "ztf":
            passbands = []
            for filter_name in ["g", "r", "i"]:
                pb = Passband.from_sncosmo("ZTF", filter_name, f"ztf{filter_name}", **kwargs)
                passbands.append(pb)
        elif preset == "roman":
            passbands = PassbandGroup._roman_load_preset(table_dir, **kwargs)
        else:
            raise ValueError(f"Unknown passband preset: {preset}")

        # Build the actual PassbandGroup object.
        return cls(given_passbands=passbands, filters=filters, **kwargs)

    @staticmethod
    @cite_function
    def _lsst_load_preset(table_dir, **kwargs):
        """Load the LSST passbands from the LSST preset.

        References
        ----------
        LSST Filters: https://github.com/lsst/throughputs

        Parameters
        ----------
        table_dir : str or Path
            The path to the directory in which to store cached passband tables. If the passband
            exists in this directory, it will be loaded from there; otherwise it will be downloaded
            and saved in that directory.
            The full path to the tables will be {table_dir}/LSST/{filter_name}.dat.
        **kwargs
            Additional keyword arguments to pass to the Passband constructor.

        Returns
        -------
        passbands : list of Passband
            A list of Passband objects for the LSST filters.
        """
        passbands = []

        # Check that units are what is expected for this preset.
        if "units" not in kwargs:
            kwargs["units"] = "nm"
        elif kwargs["units"] != "nm":
            raise ValueError(
                "LSST passbands are expected to be in nanometers (nm). Please set units='nm' in the kwargs."
            )

        for filter_name in ["u", "g", "r", "i", "z", "y"]:
            url = f"https://github.com/lsst/throughputs/blob/main/baseline/total_{filter_name}.dat?raw=true"
            pb = Passband.from_file(
                "LSST",
                filter_name,
                table_path=table_dir / "LSST" / f"{filter_name}.dat",
                table_url=url,
                **kwargs,
            )
            passbands.append(pb)
        return passbands

    @staticmethod
    @cite_function
    def _roman_load_preset(table_dir, **kwargs):
        """Load the Roman passbands from the Roman preset.

        References
        ----------
        Roman Filters: https://github.com/RomanSpaceTelescope/roman-technical-information

        Parameters
        ----------
        table_dir : str or Path
            The path to the directory in which to store cached passband tables. If the passband
            exists in this directory, it will be loaded from there; otherwise it will be downloaded
            and saved in that directory.
            The full path to the tables will be {table_dir}/Roman/{filter_name}.dat.
        **kwargs
            Additional keyword arguments to pass to the Passband constructor.

        Returns
        -------
        passbands : list of Passband
            A list of Passband objects for the Roman filters.
        """
        passbands = []

        # Check that units are what is expected for this preset.
        if kwargs.get("units", "micron") != "micron":
            raise ValueError(
                f"Roman passbands are expected to be in microns (micron), but got {kwargs.get('units')}"
            )

        force_download = kwargs.get("force_download", False)
        table_path = table_dir / "Roman" / "roman_wfi_filters.ecsv"
        table_url = (
            "https://github.com/RomanSpaceTelescope/roman-technical-information/blob/main/data/"
            "WideFieldInstrument/Imaging/EffectiveAreas/Roman_effarea_v8_SCA01_20240301.ecsv?raw=true"
        )

        # Download the table if it does not exist or if force_download is True.
        success = download_data_file_if_needed(table_path, table_url, force_download=force_download)
        if not success:
            raise RuntimeError(f"Failed to download Roman passband table from {table_url}.")

        # Load the table, convert the wavelengths from microns to Angstroms, create Passband objects.
        # The table contains effective area in m^2, but since we normalize the transmission tables,
        # we can just use this directly.
        table = pd.read_csv(table_path, comment="#", sep=r"[\s,]+", engine="python")
        waves = table["Wave"].values * 10_000  # Convert microns to Angstroms

        for filter_name in ["F062", "F087", "F106", "F129", "F146", "F158", "F184", "F213"]:
            table_values = np.vstack([waves, table[filter_name].values.astype(float)]).T
            pb = Passband(table_values, "Roman", filter_name, **kwargs)
            passbands.append(pb)
        return passbands

    @classmethod
    @cite_function
    def from_svo(
        cls,
        all_filters: list[str],
        *,
        delta_wave: float | None = 5.0,
        trim_quantile: float | None = 1e-3,
        table_dir: Union[str, Path] | None = None,
        force_download: bool = False,
        **kwargs,
    ):
        """Create a PassbandGroup object from the SVO Filter Profile Service given a list
        of full filter names in the form of "{OBSERVATORY}/{CAMERA}.{FILTER}".

        References
        ----------
        This research has made use of the SVO Filter Profile Service "Carlos Rodrigo",
        funded by MCIN/AEI/10.13039/501100011033/ through grant PID2023-146210NB-I00
        * Rodrigo, C., Cruz, P., Aguilar, J.F., et al. 2024; https://ui.adsabs.harvard.edu/abs/2024A%26A...689A..93R/abstract
        * Rodrigo, C., Solano, E., Bayo, A., 2012; https://ui.adsabs.harvard.edu/abs/2012ivoa.rept.1015R/abstract
        * Rodrigo, C., Solano, E., 2020; https://ui.adsabs.harvard.edu/abs/2020sea..confE.182R/abstract

        Parameters
        ----------
        all_filters : list[str]
            A list of full filter names to load from the SVO Filter Profile Service in the
            form of "{OBSERVATORY}/{CAMERA}.{FILTER}". This can include filters from multiple surveys.
        delta_wave : float or None, optional
            The grid step of the wave grid, in angstroms.
            It is typically used to downsample transmission using linear interpolation.
            Default is 5 angstroms. If None the original grid is used.
        trim_quantile : float or None, optional
            The quantile to trim the transmission table by. For example, if trim_quantile is 1e-3, the
            transmission table will be trimmed to include only the central 99.8% of the area under the
            transmission curve.
        table_dir : str, optional
            The path to the base directory in which to store cached passband tables. If the passband
            exists in this directory, it will be loaded from there; otherwise it will be downloaded
            and saved in that directory.
        force_download : bool, optional
            If True, the transmission table will be downloaded even if it already exists locally. Default is
            False.
        **kwargs
            Additional keyword arguments to pass to the Passband constructor.
        """
        passband_list = []
        for full_filter_name in all_filters:
            pb = Passband.from_svo(
                full_filter_name,
                delta_wave=delta_wave,
                trim_quantile=trim_quantile,
                table_dir=table_dir,
                force_download=force_download,
                **kwargs,
            )
            passband_list.append(pb)
        return cls(given_passbands=passband_list, **kwargs)

    def add_passband(self, passband) -> None:
        """Manually add a passband to the group.

        Parameters
        ----------
        passband : Passband
            The passband to add to the group.
        """
        self.passbands[passband.full_name] = passband
        self._update_internal_data()

    def _update_internal_data(self) -> None:
        """Update the cached internal data."""
        # Update the mapping of filter name to full_name.
        self._filter_to_name = {}
        for full_name, pb_obj in self.passbands.items():
            if pb_obj.filter_name in self._filter_to_name:
                logger.info("Multiple passband objects detected for filter {pb_obj.filter_name}")
                self._filter_to_name[pb_obj.filter_name].append(full_name)
            else:
                self._filter_to_name[pb_obj.filter_name] = [full_name]

        self._update_waves()

    def _update_waves(self, threshold=1e-5) -> None:
        """Update the group's wave attribute to be the union of all wavelengths in
        the passbands and update the group's _in_band_wave_indices attribute, which is
        the indices of the group's wave grid that are in the passband's wave grid.

        Eg, if a group's waves are [11, 12, 13, 14, 15] and a single band's are [13, 14],
        we get [2, 3].

        The indices are stored in the passband's _in_band_wave_indices attribute as either
        a tuple of two ints (lower, upper) or a 1D np.ndarray of ints.

        Parameters
        ----------
        threshold : float
            The threshold for merging two "close" wavelengths. This is used to
            avoid problems with numerical precision.
            Default: 1e-5
        """
        if len(self.passbands) == 0:
            self.waves = np.array([])
        else:
            # Compute the unique wavelengths (accounting for floating point error) by
            # sorting the union of all the waves, computing the gaps between each adjacent
            # pair (using 1e8 for before the first), and saving the points with a gap
            # larger than the threshold.
            all_waves = np.concatenate([passband.waves for passband in self.passbands.values()])
            sorted_waves = np.sort(all_waves)
            gap_sizes = np.insert(sorted_waves[1:] - sorted_waves[:-1], 0, 1e8)
            self.waves = sorted_waves[gap_sizes >= threshold]

        # Update the mapping of each passband's wavelengths to the corresponding indices in the
        # unioned list of all wavelengths.
        self._in_band_wave_indices = {}
        for name, passband in self.passbands.items():
            # We only want the fluxes that are in the passband's wavelength range
            # So, find the indices in the group's wave grid that are in the passband's wave grid
            lower, upper = passband.waves[0], passband.waves[-1]
            lower_index, upper_index = np.searchsorted(self.waves, [lower, upper])

            # Check that this is the right grid after all (check will fail if passbands overlap and passbands
            # do not happen to be on the same phase of the grid; eg, even if the step is 10, if the first
            # passband starts at 100 and the second at 105, the passbands won't share the same grid)
            if np.array_equal(self.waves[lower_index : upper_index + 1], passband.waves):
                indices = slice(lower_index, upper_index + 1)
            else:
                indices = np.searchsorted(self.waves, passband.waves)
            self._in_band_wave_indices[name] = indices

    def wave_bounds(self):
        """Get the minimum and maximum wavelength for this group.

        Returns
        -------
        min_wave : float
            The minimum wavelength.
        max_wave : float
            The maximum wavelength.
        """
        min_wave = np.min(self.waves)
        max_wave = np.max(self.waves)
        return min_wave, max_wave

    def mask_by_filter(self, filters):
        """Compute a mask for whether a given observations is of interest for
        for a given analysis. For example this could be used to remove unneeded
        observations from an ObsTable.

        Parameters
        ----------
        filters : list-like
            A length T array of filter names or full names.

        Returns
        -------
        mask : numpy.ndarray
            A length T array of Booleans indicating whether the filter is of interest.
        """
        filters = np.asarray(filters)
        full_name_mask = np.isin(filters, list(self.passbands.keys()))
        filter_name_mask = np.isin(filters, list(self._filter_to_name.keys()))
        return full_name_mask | filter_name_mask

    def process_transmission_tables(
        self, *, delta_wave: float | None = 5.0, trim_quantile: float | None = 1e-3
    ):
        """Process the transmission tables for all passbands in the group; recalculate group's wave
        attribute. This function is used to change the preprocessing of the transmission tables after
        the PassbandGroup has been initialized, such as using different delta_wave or trim_quantile
        values.

        Parameters
        ----------
        delta_wave : float or None, optional
            The grid step of the wave grid. Default is 5.0.
        trim_quantile : float or None, optional
            The quantile to trim the transmission table by. For example, if trim_quantile is 1e-3, the
            transmission table will be trimmed to include only the central 99.8% of rows.
        """
        for passband in self.passbands.values():
            passband.process_transmission_table(
                delta_wave=delta_wave,
                trim_quantile=trim_quantile,
            )

        self._update_internal_data()

    def fluxes_to_bandflux(self, flux_density_matrix: np.ndarray, filter: str) -> np.ndarray:
        """Calculate bandfluxes for a single passband in the group.

        Parameters
        ----------
        flux_density_matrix : np.ndarray
            A 2D array of flux densities where of shape T x W where the rows are times and
            columns are wavelengths.
        filter : str
            The name of the filter to evaluate.

        Returns
        -------
        bandflux : np.ndarray
            A length T array of bandfluxes for the given passband.
        """
        if filter not in self.passbands:
            if filter in self._filter_to_name:
                filter = self._filter_to_name[filter][0]
            else:
                raise ValueError(f"Filter {filter} not found in passband group.")
        passband = self.passbands[filter]

        # Evaluate the bandflux using only the wavelengths for this passband.
        wave_indices = self._in_band_wave_indices[filter]
        if wave_indices is None:  # pragma: no cover
            raise ValueError(
                f"Passband {filter} does not have _in_band_wave_indices set. "
                "This should have been calculated in PassbandGroup._update_internal_data."
            )
        single_band_fluxes = flux_density_matrix[:, wave_indices]
        bandflux = passband.fluxes_to_bandflux(single_band_fluxes)

        return bandflux

    def fluxes_to_bandfluxes(self, flux_density_matrix: np.ndarray) -> np.ndarray:
        """Calculate bandfluxes for all passbands in the group.

        Parameters
        ----------
        flux_density_matrix : np.ndarray
            A 2D array of flux densities where of shape T x W where the rows are times and
            columns are wavelengths.

        Returns
        -------
        dict of np.ndarray
            A dictionary of bandfluxes with passband full names as keys and np.ndarrays of
            bandfluxes as values.
        """
        if flux_density_matrix.size == 0 or len(self.waves) != len(flux_density_matrix[0]):
            flux_density_matrix_num_cols = 0 if flux_density_matrix.size == 0 else len(flux_density_matrix[0])
            raise ValueError(
                f"PassbandGroup mismatched grids: Flux density matrix has {flux_density_matrix_num_cols} "
                f"columns, which does not match transmission table's {len(self.waves)} rows. Check that the "
                f"flux density matrix was calculated on the same grid as the transmission tables, which can "
                f"be accessed via the Passband's or PassbandGroup's waves attribute."
            )

        # Compute the bandfluxes for each passband.
        bandfluxes = {}
        for full_name in self.passbands:
            bandfluxes[full_name] = self.fluxes_to_bandflux(flux_density_matrix, full_name)
        return bandfluxes

    def to_file(self, file_path: Union[str, Path], *, overwrite: bool = False) -> None:
        """Save the entire passband group to a single file."""
        if not overwrite and Path(file_path).exists():
            raise FileExistsError(f"File {file_path} already exists. Use overwrite=True to overwrite it.")

        # Create a single pandas table with the information for each passband.
        all_data = []
        for pb in self.passbands.values():
            df = pd.DataFrame(pb.normalized_system_response, columns=["wavelength", "transmission"])
            df["survey"] = pb.survey
            df["filter_name"] = pb.filter_name
            all_data.append(df)
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(file_path, index=False)

    def plot(self, *, ax=None, figure=None):
        """Plot the PassbandGroup on a single plot.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes or None, optional
            Axes, None by default.
        figure : matplotlib.pyplot.Figure or None
            Figure, None by default.
        """
        if ax is None:
            if figure is None:
                figure = plt.figure()
            ax = figure.add_axes([0, 0, 1, 1])

        # Plot each passband.
        for pb_obj in self.passbands.values():
            pb_obj.plot(ax=ax, plot_loaded=False)
        ax.legend()


class Passband:
    """A passband contains information about its transmission curve and calculates its normalization.

    Attributes
    ----------
    survey : str
        The survey to which the passband belongs: eg, "LSST".
    filter_name : str
        The name of the passband's filter: eg, "u".
    full_name : str
        The full name of the passband. This is the survey and filter concatenated: eg, "LSST_u".
    waves : np.ndarray
        The wavelengths of the transmission table in Angstroms. To be used when evaluating models
        to generate fluxes that will be passed to fluxes_to_bandflux.
    delta_wave : float or None
        The grid step of the wave grid, in angstroms, if the table is a uniform grid. The value is None
        if the grid is not uniform.
    transmission_table : np.ndarray
        A 2D array of wavelengths and transmissions. This is the system throughput table loaded
        from the file, and is neither interpolated nor normalized.
    normalized_system_response : np.ndarray
        A 2D array where the first col is wavelengths (Angstrom) and the second col is transmission values.
        This table is both interpolated to the _wave_grid and normalized to calculate phi_b(λ).
    """

    def __init__(
        self,
        table_values: np.array,
        survey: str,
        filter_name: str,
        *,
        delta_wave: float | None = 5.0,
        trim_quantile: float | None = 1e-3,
        units: Literal["nm", "A"] | None = "A",
    ):
        """Construct a Passband object.

        Parameters
        ----------
        table_values : np.ndarray, optional
            A 2D array of wavelengths (in the given units) and transmissions.
        survey : str
            The survey to which the passband belongs: eg, "LSST".
        filter_name : str
            The filter_name of the passband: eg, "u".
        delta_wave : float or None, optional
            The grid step of the wave grid, in angstroms.
            It is typically used to downsample transmission using linear interpolation.
            Default is 5 angstroms. If None the original grid is used.
        trim_quantile : float or None, optional
            The quantile to trim the transmission table by. For example, if trim_quantile is 1e-3, the
            transmission table will be trimmed to include only the central 99.8% of the area under the
            transmission curve.
        units : Literal['nm','A'], optional
            Denotes whether the wavelength units of the table are nanometers ('nm') or Angstroms ('A').
            By default 'A'. Does not affect the output units of the class, only the interpretation of the
            provided passband table.
        """
        self.survey = survey
        self.filter_name = filter_name
        self.full_name = f"{survey}_{filter_name}"
        self.delta_wave = delta_wave

        # Perform validation of the transmission table.
        if table_values.shape[1] != 2:
            raise ValueError("Passband requires an input table with exactly two columns.")
        diffs = np.diff(table_values[:, 0])
        if np.any(diffs < 0.0):
            raise ValueError("Wavelengths in transmission table must be strictly increasing.")
        if np.any(diffs == 0.0):
            logger.warning("Duplicate wavelengths found in transmission table; averaging values.")
            dup_inds = np.where(diffs == 0.0)
            table_values[dup_inds, 1] = 0.5 * (table_values[dup_inds, 1] + table_values[dup_inds + 1, 1])
            table_values = np.delete(table_values, dup_inds + 1, axis=0)
        self.transmission_table = np.copy(table_values)

        # Ensure the wavelengths are in Angstroms.
        if units == "nm":
            # Multiply the first column (wavelength) by 10.0 to convert to Angstroms
            self.transmission_table[:, 0] *= 10.0
        elif units != "A":
            raise ValueError(f"Unknown Passband units {units}")

        # Preprocess the passband.
        self.process_transmission_table(delta_wave=delta_wave, trim_quantile=trim_quantile)

    def __str__(self) -> str:
        """Return a string representation of the Passband."""
        return f"Passband: {self.full_name}"

    def __eq__(self, other) -> bool:
        """Determine if two passbands have equal values for the processed tables."""
        # Check that they are using the same wavelengths.
        if len(self.waves) != len(other.waves):
            return False
        if not np.allclose(self.waves, other.waves):
            return False

        # Check that they have the (approximately) same transmission tables.
        if self.normalized_system_response.shape != other.normalized_system_response.shape:
            return False
        if not np.allclose(self.normalized_system_response, other.normalized_system_response):
            return False
        return True

    @classmethod
    def from_file(
        cls,
        survey: str,
        filter_name: str,
        *,
        delta_wave: float | None = 5.0,
        trim_quantile: float | None = 1e-3,
        table_path: Union[str, Path] | None = None,
        table_url: str | None = None,
        units: Literal["nm", "A"] | None = "A",
        force_download: bool = False,
    ):
        """Construct a Passband object from a file, downloading it if needed.

        Parameters
        ----------
        survey : str
            The survey to which the passband belongs: eg, "LSST".
        filter_name : str
            The filter_name of the passband: eg, "u".
        delta_wave : float or None, optional
            The grid step of the wave grid, in angstroms.
            It is typically used to downsample transmission using linear interpolation.
            Default is 5 angstroms. If None the original grid is used.
        trim_quantile : float or None, optional
            The quantile to trim the transmission table by. For example, if trim_quantile is 1e-3, the
            transmission table will be trimmed to include only the central 99.8% of the area under the
            transmission curve.
        table_path : str, optional
            The path to the transmission table file. If None, the table path will be set to a default path;
            if no file exists at this location, the file will be downloaded from table_url.
        table_url : str, optional
            The URL to download the transmission table file. If None, the table URL will be set to
            a default URL based on the survey and filter_name. Default is None.
        units : Literal['nm','A'], optional
            Denotes whether the wavelength units of the table are nanometers ('nm') or Angstroms ('A').
            By default 'A'. Does not affect the output units of the class, only the interpretation of the
            provided passband table.
        force_download : bool, optional
            If True, the transmission table will be downloaded even if it already exists locally. Default is
            False.
        """
        if table_path is None:
            # If no path is given, use the default.
            table_path = Path(
                _LIGHTCURVELYNX_BASE_DATA_DIR,
                "passbands",
                survey,
                f"{filter_name}.dat",
            )
        else:
            table_path = Path(table_path)

        # Download the table if it does not exist or if force_download is True.
        success = download_data_file_if_needed(table_path, table_url, force_download=force_download)
        if not success:
            raise RuntimeError(f"Failed to download passband table from {table_url}.")

        # Load the table and create the passband.
        loaded_table = Passband.load_transmission_table(table_path, wave_units=units)
        return Passband(
            loaded_table,
            survey,
            filter_name,
            delta_wave=delta_wave,
            trim_quantile=trim_quantile,
            units="A",  # All loaded tables are pre-converted to Angstroms
        )

    @classmethod
    @cite_function("https://sncosmo.readthedocs.io/en/stable/api/sncosmo.Bandpass.html")
    def from_sncosmo(cls, survey: str, filter_name: str, bandpass=None, **kwargs):
        """Create a Passband object from an sncosmo.Bandpass object.

        Parameters
        ----------
        survey : str
            The survey to which the passband belongs: eg, "LSST".
        filter_name : str
            The filter_name of the passband: eg, "u".
        bandpass : sncosmo.Bandpass or str
            The bandpass object from which to create the Passband object.
        **kwargs
            Additional keyword arguments (unused)

        Reference
        ---------
        snocosmo.Bandpass:
        https://sncosmo.readthedocs.io/en/stable/api/sncosmo.Bandpass.html
        """
        if bandpass is None:
            raise ValueError("bandpass must be provided as object or string")
        elif isinstance(bandpass, str):
            # Only import sncosmo if we need to.
            try:
                from sncosmo import get_bandpass
            except ImportError as err:  # pragma: no cover
                raise ImportError(
                    "sncosmo package is not installed be default. You can install it with "
                    "`pip install sncosmo` or `conda install conda-forge::sncosmo`."
                ) from err
            bandpass = get_bandpass(bandpass)

        table = np.column_stack([bandpass.wave, bandpass.trans])
        return Passband(
            table,
            survey,
            filter_name,
            trim_quantile=None,  # Trimming is done in sncosmo
            units="A",  # All sncosmo bandpasses are in Angstroms
        )

    @classmethod
    @cite_function
    def from_svo(
        cls,
        full_filter_name,
        *,
        delta_wave: float | None = 5.0,
        trim_quantile: float | None = 1e-3,
        table_dir: Union[str, Path] | None = None,
        force_download: bool = False,
        **kwargs,
    ):
        """Create a Passband object from the SVO Filter Profile Service.

        References
        ----------
        This research has made use of the SVO Filter Profile Service "Carlos Rodrigo",
        funded by MCIN/AEI/10.13039/501100011033/ through grant PID2023-146210NB-I00
        * Rodrigo, C., Cruz, P., Aguilar, J.F., et al. 2024; https://ui.adsabs.harvard.edu/abs/2024A%26A...689A..93R/abstract
        * Rodrigo, C., Solano, E., Bayo, A., 2012; https://ui.adsabs.harvard.edu/abs/2012ivoa.rept.1015R/abstract
        * Rodrigo, C., Solano, E., 2020; https://ui.adsabs.harvard.edu/abs/2020sea..confE.182R/abstract

        Parameters
        ----------
        full_filter_name : str
            The full name of the survey and filter in the SVO database in the form of
            "{OBSERVATORY}/{CAMERA}.{FILTER}", e.g., "SLOAN/SDSS.u".
        delta_wave : float or None, optional
            The grid step of the wave grid, in angstroms.
            It is typically used to downsample transmission using linear interpolation.
            Default is 5 angstroms. If None the original grid is used.
        trim_quantile : float or None, optional
            The quantile to trim the transmission table by. For example, if trim_quantile is 1e-3, the
            transmission table will be trimmed to include only the central 99.8% of the area under the
            transmission curve.
        table_dir : str, optional
            The path to the base directory in which to store cached passband tables. If the passband
            exists in this directory, it will be loaded from there; otherwise it will be downloaded
            and saved in that directory.
        force_download : bool, optional
            If True, the transmission table will be downloaded even if it already exists locally. Default is
            False.
        **kwargs
            Additional keyword arguments to pass to the Passband constructor.
        """
        # Parse the filter name to get the survey and filter and use it to construct the file
        # path and URL to the SVO database.
        survey, filter = full_filter_name.split(".")

        if table_dir is None:
            table_dir = Path(_LIGHTCURVELYNX_BASE_DATA_DIR, "passbands", survey)
        else:
            table_dir = Path(table_dir) / survey
        table_path = Path(table_dir, f"{filter}.xml")
        table_url = f"https://svo2.cab.inta-csic.es/svo/theory/fps3/fps.php?ID={full_filter_name}"

        return cls.from_file(
            survey=survey,
            filter_name=filter,
            delta_wave=delta_wave,
            trim_quantile=trim_quantile,
            table_path=table_path,
            table_url=table_url,
            force_download=force_download,
            **kwargs,
        )

    @staticmethod
    def load_transmission_table(
        table_path: Union[str, Path],
        *,
        wave_units: Literal["nm", "A"] = "A",
        **kwargs,
    ) -> np.ndarray:
        """Load a transmission table from a file in either VOTable of ASCII format.

        ASCII tables must have 2 columns: wavelengths and transmissions; wavelengths must be
        strictly increasing.

        Parameters
        ----------
        table_path : str or Path
            The path to the transmission table file.
        wave_units : Literal['nm','A'], optional
            Denotes whether the wavelength units of the table are nanometers ('nm') or Angstroms ('A').
            By default 'A'. Does not affect the output units of the class, only the interpretation of the
            provided passband table.
        **kwargs : dict
            Additional keyword arguments to pass to the reader method.

        Returns
        -------
        np.ndarray
            A 2D array of wavelengths (in Angstroms) and transmissions.
        """
        logger.info(f"Loading passband from file: {table_path}")

        table_path = Path(table_path)
        if not table_path.exists():
            raise FileNotFoundError(f"Transmission table not found at {table_path}")

        # Check if the file is in a VOTable format.
        if table_path.suffix in [".xml", ".vot", ".votable"]:
            table = parse(table_path).get_first_table()

            # Check that we have the correct data.
            if len(table.fields) != 2:
                raise ValueError("VOTable must have exactly 2 columns.")
            if not table.fields[0].name.lower().startswith("wave"):
                raise ValueError("VOTable first column must be named 'wavelength'.")

            # Transform the table from rows of (wavelength, transmission) tuples to a 2D array.
            loaded_table = np.zeros((len(table.array), 2), dtype=float)
            loaded_table[:, 0] = table.array[table.fields[0].name]
            loaded_table[:, 1] = table.array[table.fields[1].name]

            # Read the wavelength units from the VOTable.
            wave_unit_name = table.fields[0].unit.name.lower()
            if wave_unit_name in ["nm", "nanometers", "namo", "nanometer"]:
                wave_units = "nm"
            elif wave_unit_name in ["a", "aa", "angstrom", "angstroms"]:
                wave_units = "A"
            else:
                raise ValueError(f"Unsupported wavelength unit in VOTable: {wave_unit_name}")
        elif table_path.suffix in [".csv", ".ecsv", ".txt", ".dat"]:
            # Add default delimiter if not provided
            if (table_path.suffix == ".csv" or table_path.suffix == ".ecsv") and "delimiter" not in kwargs:
                kwargs["delimiter"] = ","

            # Load the table.
            loaded_table = np.loadtxt(table_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format for transmission table: {table_path.suffix}")

        # Check that the table has the correct shape
        if loaded_table.size == 0 or loaded_table.shape[1] != 2:
            raise ValueError("Transmission table must have exactly 2 columns.")

        # If the table is given in nanometers, convert to Angstroms (by multiplying by 10.0).
        if wave_units == "nm":
            loaded_table[:, 0] *= 10.0
        elif wave_units != "A":
            raise ValueError(f"Unknown wavelength units {wave_units}. Expected 'nm' or 'A'.")

        # Check that wavelengths are strictly increasing. If there are duplicates then
        # we average the values.
        diffs = np.diff(loaded_table[:, 0])
        if np.any(diffs < 0.0):
            raise ValueError("Wavelengths in transmission table must be increasing.")
        if np.any(diffs == 0.0):
            logger.warning("Duplicate wavelengths found in transmission table; averaging values.")
            dup_inds = np.where(diffs == 0.0)
            loaded_table[dup_inds, 1] = 0.5 * (loaded_table[dup_inds, 1] + loaded_table[dup_inds + 1, 1])
            loaded_table = np.delete(loaded_table, dup_inds + 1, axis=0)

        return loaded_table

    def process_transmission_table(
        self,
        *,
        delta_wave: float | None = 5.0,
        trim_quantile: float | None = 1e-3,
    ):
        """Process the transmission table, transforming it to the desired wave grid and
        and computing a normalized system response from the throughput table.

        Parameters
        ----------
        delta_wave : Optional[float] = 5.0
            The grid step in Angstroms of the wave grid. Default is 5.0 Angstroms.
        trim_quantile : Optional[float] = 1e-3
            The quantile to trim the transmission table by. For example, if trim_quantile is 1e-3, the
            transmission table will be trimmed to include only the central 99.8% of rows.
        """
        interpolated_table = self.interpolate_transmission_table(self.transmission_table, delta_wave)
        trimmed_table = self.trim_transmission_by_quantile(interpolated_table, trim_quantile)
        self.normalized_system_response = self.compute_system_response_table(trimmed_table)

        self.waves = self.normalized_system_response[:, 0]
        self.delta_wave = delta_wave

    @staticmethod
    def interpolate_transmission_table(table: np.ndarray, delta_wave: float | None) -> np.ndarray:
        """Interpolate the transmission table to a new wave grid.

        Parameters
        ----------
        table : np.ndarray
            A 2D array of wavelengths (in Angstroms) and transmissions.
        delta_wave : float or None
            The grid step in Angstroms of the wave grid.

        Returns
        -------
        np.ndarray
            The 2D interpolated array of wavelengths (in Angstroms) and transmissions.
        """
        # Don't interpolate if delta_wave is None or the table is already on the desired grid
        if delta_wave is None:
            return table
        if np.allclose(np.diff(table[:, 0]), delta_wave):
            return table

        # Regrid wavelengths to the new wave grid
        wavelengths = table[:, 0]
        lower_bound, upper_bound = wavelengths[0], wavelengths[-1]
        new_wavelengths = np.linspace(
            lower_bound, upper_bound, int((upper_bound - lower_bound) / delta_wave) + 1
        )

        # Interpolate the transmission table to the new wave grid
        spline = scipy.interpolate.InterpolatedUnivariateSpline(table[:, 0], table[:, 1], ext="raise", k=1)
        interpolated_transmissions = spline(new_wavelengths)
        return np.column_stack((new_wavelengths, interpolated_transmissions))

    @staticmethod
    def trim_transmission_by_quantile(table: np.ndarray, trim_quantile: float | None) -> np.ndarray:
        """Trim the transmission table so that it only includes the central (100 - 2*trim_quartile)% of rows.

        E.g., if trim_quantile is 1e-3, the transmission table will be trimmed to include only the central
        99.8% of rows.

        Parameters
        ----------
        table : np.ndarray
            A 2D array of wavelengths (in Angstroms) and transmissions.
        trim_quantile : float
            The quantile to trim the transmission table by. For example, if trim_quantile is 1e-3, the
            transmission table will be trimmed to include only the central 99.8% of rows. Must be greater than
            or equal to 0 and less than 0.5.

        Returns
        -------
        np.ndarray
            A 2D array of wavelengths (in Angstroms) and transmissions.
        """
        if trim_quantile is None or trim_quantile == 0.0:
            return table
        if trim_quantile < 0 or trim_quantile >= 0.5:
            raise ValueError(f"Trim quantile must be between 0 and 0.5; got {trim_quantile}.")

        # Separate wavelengths and transmissions
        wavelengths = table[:, 0]
        transmissions = table[:, 1]

        # Calculate the cumulative sum of the transmission values (area under the curve)
        cumulative_area = scipy.integrate.cumulative_trapezoid(transmissions, x=wavelengths)

        # Normalize cumulative area to range from 0 to 1
        cumulative_area /= cumulative_area[-1]

        # Find indices where the cumulative area exceeds the trim quantiles
        lower_bound = max(np.searchsorted(cumulative_area, trim_quantile, side="right") - 1, 0)
        upper_bound = np.searchsorted(cumulative_area, 1 - trim_quantile)

        # Trim the table to the desired range
        trimmed_table = table[lower_bound : upper_bound + 1]

        return trimmed_table

    @staticmethod
    def compute_system_response_table(transmission_table: np.ndarray) -> np.ndarray:
        """Calculate the value of phi_b for all wavelengths in a transmission table.

        This is eq. 8 from "On the Choice of LSST Flux Units" (Ivezić et al.):

        φ_b(λ) = S_b(λ)λ⁻¹ / ∫ S_b(λ)λ⁻¹ dλ

        where S_b(λ) is the system response of the passband. Note we use transmission table as our S_b(λ).

        Parameters
        ----------
        transmission_table : np.ndarray
            A 2D array of wavelengths (in Angstroms) and throughput values.

        Returns
        -------
        np.ndarray
            A 2D array of wavelengths (in Angstroms) and normalized system response values.

        Raises
        ------
        ValueError
            If the transmission table is the wrong size or the calculated denominator is zero.
        """
        if transmission_table.size == 0:
            raise ValueError("Transmission table is empty; cannot normalize.")
        elif transmission_table.shape[0] == 1:
            raise ValueError(f"Cannot normalize transmission table with only one row: {transmission_table}.")
        elif transmission_table.ndim != 2 or transmission_table.shape[1] != 2:
            raise ValueError("Transmission table must be 2D array with exactly 2 columns.")

        wavelengths_angstrom = transmission_table[:, 0]
        transmissions = transmission_table[:, 1]
        # Calculate the numerators and denominator
        numerators = transmissions / wavelengths_angstrom
        denominator = scipy.integrate.trapezoid(numerators, x=wavelengths_angstrom)

        if denominator == 0:
            raise ValueError("Denominator is zero; cannot normalize transmission table.")

        # Calculate phi_b for each wavelength
        normalized_system_response = numerators / denominator
        return np.column_stack((wavelengths_angstrom, normalized_system_response))

    def fluxes_to_bandflux(
        self,
        flux_density_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate the bandflux for a given set of flux densities.

        This is eq. 7 from "On the Choice of LSST Flux Units" (Ivezić et al.):

        F_b = ∫ f(λ)φ_b(λ) dλ

        where f(λ) is the flux density of an object at the top of the atmosphere, and φ_b(λ) is the
        normalized system response for given band b."

        Parameters
        ----------
        flux_density_matrix : np.ndarray
            A 2D or 3D array of flux densities. If the array is 2D it contains a single sample where
            the rows are the T times and columns are M wavelengths in Angstroms. If the array is 3D
            it contains S samples and the values are indexed as (sample_num, time, wavelength).

        Returns
        -------
        bandfluxes : np.ndarray
            A 1D or 2D array. If the flux_density_matrix contains a single sample (2D input) then
            the function returns a 1D length T array where each element is the bandflux
            at the corresponding time. Otherwise the function returns a size S x T array where
            each entry corresponds to the value for a given sample at a given time.
        """
        if flux_density_matrix.size == 0:
            raise ValueError("Empty flux density matrix used.")
        if len(flux_density_matrix.shape) == 2:
            w_axis = 1
            flux_density_matrix_num_cols = flux_density_matrix.shape[1]
        elif len(flux_density_matrix.shape) == 3:
            w_axis = 2
            flux_density_matrix_num_cols = flux_density_matrix.shape[2]
        else:
            raise ValueError("Invalid flux density matrix. Must be 2 or 3-dimensional.")

        # Check the number of wavelengths match.
        if len(self.waves) != flux_density_matrix_num_cols:
            raise ValueError(
                f"Passband mismatched grids: Flux density matrix has {flux_density_matrix_num_cols} "
                f"columns, which does not match the {len(self.waves)} rows in band {self.full_name}'s "
                f"transmission table. Check that the flux density matrix was calculated on the same grid as "
                f"the transmission tables, which can be accessed via the Passband's or PassbandGroup's waves "
                f"attribute."
            )

        # Calculate the bandflux as ∫ f(λ)φ_b(λ) dλ,
        # where f(λ) is the flux density and φ_b(λ) is the normalized system response
        integrand = flux_density_matrix * self.normalized_system_response[:, 1]
        if self.delta_wave is not None:
            # If the grid is equal spaced, we can use a faster method of computing a rectangular
            # integration and removing half the first and last values (to make it trapezoidal).
            first_val = np.take(integrand, 0, axis=w_axis)
            last_val = np.take(integrand, -1, axis=w_axis)
            bandfluxes = (np.sum(integrand, axis=w_axis) - 0.5 * (first_val + last_val)) * self.delta_wave
        else:
            # Do the full integration.
            bandfluxes = scipy.integrate.trapezoid(integrand, x=self.waves, axis=w_axis)
        return bandfluxes

    def plot(self, *, ax=None, figure=None, color=None, plot_loaded=False):
        """Plot the passband.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes or None, optional
            Axes, None by default.
        figure : matplotlib.pyplot.Figure or None
            Figure, None by default.
        color : str or None, optional
            The color of the curve.
        plot_loaded : bool
            Also plot the loaded table as a dashed line. Used for debugging.
        """
        if ax is None:
            if figure is None:
                figure = plt.figure()
            ax = figure.add_axes([0, 0, 1, 1])

        # If the color is provided, we use that. Otherwise we try
        # the LSST filter colors (or default to black).
        if color is None:
            color = lsst_filter_plot_colors.get(self.filter_name, "black")

        ax.plot(
            self.normalized_system_response[:, 0],  # X values are the wavelength
            self.normalized_system_response[:, 1],  # Y values are the transmission values.
            color=color,
            label=self.full_name,
        )
        if plot_loaded:
            ax.plot(
                self.transmission_table[:, 0],  # X values are the wavelength
                self.transmission_table[:, 1],  # Y values are the transmission values.
                color=color,
                linestyle="--",
            )

        ax.set_xlabel(r"Wavelength, $\AA$")
        ax.set_ylabel("Transmission Value")
        ax.set_ylim(0, None)
