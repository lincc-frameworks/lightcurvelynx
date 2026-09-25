"""SpectrographNoiseModels are used to simulate the noise on a per-observation, per-spectral-bin
basis.  They may use information from the Spectrograph object and/or ObsTable.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import interp1d

from lightcurvelynx.astro_utils.mag_flux import flux2mag
from lightcurvelynx.astro_utils.spectrograph import Spectrograph
from lightcurvelynx.obstable.spectrograph_table import SpectrographObsTable
from lightcurvelynx.utils.io_utils import read_snana_spectrograph_data


class SpectrographNoiseModel(ABC):
    """An abstract base class noise model for simulating spectrograph measurements.

    Noise is applied by computing `flux_err`, which is the standard deviation of Gaussian noise
    to apply to each measurement (observation and bin). Subclasses must implement the `compute_flux_error`
    method to compute the noise parameters.

    Attributes
    ----------
    spectrograph : Spectrograph
        The spectrograph object containing the instrument parameters.
    """

    # A list of column names that must be present in the ObsTable for this noise model to work.
    _required_values = []

    def __init__(self, *, spectrograph=None):
        """Create a SpectrographNoiseModel.

        Parameters
        ----------
        spectrograph : Spectrograph, optional
            The spectrograph object containing the instrument parameters.
        """
        self.spectrograph = spectrograph

    @property
    def required_values(self):
        """List of column names that must be present in the ObsTable for this noise model to work."""
        return self._required_values

    @abstractmethod
    def compute_flux_error(self, measurements, **kwargs):
        """Compute the flux error for the given measurements and observation parameters.

        Parameters
        ----------
        measurements : matrix of float
            A T x B matrix of flux measurements in energy units (e.g. erg/s/cm²), where
            T is the number of observations and B is the number of spectral bins.
        **kwargs
            Additional parameters for the noise model.

        Returns
        -------
        flux_err : numpy.ndarray
            The standard deviation of the flux measurement error (in erg/s/cm²)
        """
        raise NotImplementedError("Subclasses must implement this method.")  # pragma: no cover

    def apply_noise(
        self,
        measurements,
        *,
        obs_table=None,
        indices=None,
        rng=None,
        **kwargs,
    ):
        """Compute the noise parameters for given observations in an ObsTable and
        apply noise to the input measurements.

        Parameters
        ----------
        measurements : matrix of float
            A T x B matrix of flux measurements in energy units (e.g. erg/s/cm²), where
            T is the number of observations and B is the number of spectral bins.
        obs_table : ObsTable, optional
            Table containing the observation parameters, including all
            parameters needed to compute the noise.
        indices : array_like of int, optional
            Indices of the observations in the ObsTable to which noise should be applied.
            If provided, the length of `indices` must match the number of rows in `measurements`.
        rng : np.random.Generator, optional
            The random number generator to use for applying noise. If None,
            a default generator will be used.
        **kwargs
            Additional parameters for the noise model.

        Returns
        -------
        measurements : numpy.ndarray
            The updated T x B matrix of flux measurements after applying noise, in the same
            units as the input measurements.
        flux_err : numpy.ndarray
            The T x B matrix of flux measurement error used for applying noise, in the
            same units as the input measurements (erg/s/cm²).
        """
        # Define the random number generator if not provided.
        if rng is None:  # pragma: no cover
            rng = np.random.default_rng()

        # Compute the standard deviation of the noise and make sure it is a numpy array.
        flux_err = self.compute_flux_error(
            measurements,
            obs_table=obs_table,
            indices=indices,
            rng=rng,
            **kwargs,
        )
        flux_err = np.asarray(flux_err)

        # Generate the actual noisy bandflux measurements.
        noisy_measurements = rng.normal(loc=measurements, scale=flux_err)
        return noisy_measurements, flux_err

    def check_compatibility(self, obs_table, fail_on_incompatible=False):
        """Check if the noise model is compatible with the given ObsTable.

        Parameters
        ----------
        obs_table : ObsTable
            The observation table to check for compatibility.
        fail_on_incompatible : bool, optional
            If True, raise a ValueError if the noise model is not compatible with the ObsTable.
            If False, simply return False in that case. Default is False.

        Returns
        -------
        bool
            True if the noise model is compatible with the ObsTable, False otherwise.
        """
        missing_columns = [col for col in self._required_values if col not in obs_table]
        if missing_columns:
            if fail_on_incompatible:
                raise ValueError(
                    f"Noise model {self.__class__.__name__} is not compatible with the given ObsTable. "
                    f"Missing required columns: {missing_columns}"
                )
            return False

        # Check if the required columns have valid data for each row.
        for col in self._required_values:
            values = obs_table.get_value_per_row(col)
            if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
                if fail_on_incompatible:
                    raise ValueError(f"Found invalid values in column '{col}'")
                return False

        return True


class ConstantSpectrographNoiseModel(SpectrographNoiseModel):
    """A noise model that simulates photon noise for spectrograph measurements
    sampled from a normal distribution with a constant standard deviation (for all bins).
    This class is primarily meant for testing purposes.

    Attributes
    ----------
    noise_level : float
        The (constant) standard deviation of the noise to apply to the spectrograph
        measurements, in the same units as the input measurements.
    """

    def __init__(self, noise_level, *, spectrograph=None):
        """Create a ConstantSpectrographNoiseModel.

        Parameters
        ----------
        noise_level : float
            The (constant) standard deviation of the noise to apply to the
            spectrograph flux measurements, in the same units as the input measurements.
        spectrograph : Spectrograph, optional
            The spectrograph object containing the instrument parameters.
        """
        super().__init__(spectrograph=spectrograph)
        if noise_level < 0:
            raise ValueError("Noise level must be non-negative.")
        self.noise_level = noise_level

    def compute_flux_error(self, measurements, **kwargs):
        """Compute the flux error for the given measurements and observation parameters.

        Parameters
        ----------
        measurements : matrix of float
            A T x B matrix of flux measurements in energy units (e.g. nJy), where
            T is the number of observations and B is the number of spectral bins.
        **kwargs
            Additional parameters for the noise model.

        Returns
        -------
        flux_err : numpy.ndarray
            The standard deviation of the flux measurement error (in nJy)
        """
        return np.full_like(measurements, self.noise_level, dtype=float)

class SNANANoiseModel(SpectrographNoiseModel):
    """A noise model that implements an SNANA-like Poisson noise from SNANA spectrograph files.

    Attributes
    ----------
    spectrograph : Spectrograph
        The spectrograph object containing the instrument parameters.
    """

    def __init__(self, snana_file, instrument=None):
        """Create an SNANANoiseModel from an SNANA spectrograph file. It creates it's own spectrograph 
        object from the file.

        Parameters
        ----------
        snana_file : str
            The path to the SNANA spectrograph file.
        instrument : str, optional
            The name of the instrument corresponding to the SNANA file.
        """
        data = read_snana_spectrograph_data(snana_file)
        
        spectrograph = Spectrograph( # merge main in and use the helper function from PR #974
            waves_min=data["waves_min"],
            waves_max=data["waves_max"],
            wavelength_resolution=data["waves_sigma"],
            instrument=instrument
        )
        super().__init__(spectrograph=spectrograph)

        self.magref = data["magref"]  # (2,)
        self.texpose_grid = data["texpose"]  # (T,)
        self.zp_grid, self.sqsigsky_grid = self._solve(self.magref, data["snr"])  # each (W, T)

        # interpolate the zero point and sky noise grids for arbitrary exposure times
        # TODO: implement SNANA's ALLOW_TEXPTRAP
        self._zp_interp = interp1d(
                    np.log10(self.texpose_grid),
                    self.zp_grid,
                    axis=1,
                    bounds_error=False,
                    fill_value=(self.zp_grid[:, 0], self.zp_grid[:, -1]),
        )
        self._sqsigsky_interp = interp1d(
            self.texpose_grid,
            self.sqsigsky_grid,
            axis=1,
            bounds_error=False,
            fill_value=(self.sqsigsky_grid[:, 0], self.sqsigsky_grid[:, -1]),
        )

    def _solve(self, magref, snr):
        """Solve for the per-bin, per-texpose ZP and SQSIGSKY grid.

        Direct port of `solve_spectrograph` (`sntools_spectrograph.c:647-782`),
        vectorized over (bin, texpose) instead of SNANA's explicit `l`/`t`
        loops.

        Parameters
        ----------
        magref : np.ndarray
            The two reference magnitudes, shape (2,).
        snr : np.ndarray
            SNR at each bin, each magref, each texpose grid point,
            shape (W, 2, T).

        Returns
        -------
        zp : np.ndarray
            Shape (W, T).
        sqsigsky : np.ndarray
            Shape (W, T).
        """
        magref = np.asarray(magref, dtype=float)
        snr = np.asarray(snr, dtype=float)
        if np.any(snr <= 0):
            raise ValueError("All SNR values must be positive to solve for ZP/SQSIGSKY.")

        # following sntools_spectrograph
        powmag = 10.0 ** (-0.4 * magref)
        top = powmag[0] - powmag[1]
        bot = (powmag[0] / snr[:, 0]) ** 2 - (powmag[1] / snr[:, 1]) ** 2
        if top <= 0 or np.any(bot <= 0):
            raise ValueError(
                "Cannot solve for ZP. Check the spectrograph file."
            )
        zp = 2.5 * np.log10(top / bot)

        flux = 10.0 ** (-0.4 * (magref[:, None, None] - zp))  # (2, W, T)
        sqsigsky = (flux[0] / snr[:, 0]) ** 2 - flux[0]  # (W, T)
        if not (np.all(np.isfinite(zp)) and np.all(np.isfinite(sqsigsky))):
            raise ValueError("Solved ZP or SQSIGSKY is not finite. Check the spectrograph file.")

        # SNANA sanity check
        check = flux / np.sqrt(sqsigsky + flux)  # (2, W, T)
        if not np.allclose(snr.transpose(1, 0, 2), check, rtol=1e-3):
            raise ValueError("Solved ZP/SQSIGSKY cannot reproduce the input SNR values.")

        return zp, sqsigsky

    def compute_flux_error(self, measurements, *, true_flux, obs_table, **kwargs):
        """Compute the flux error for the smeared measurements.

        Parameters
        ----------
        measurements : array_like of float
            The smeared flux, shape (T_obs, W) --
            `spectrograph.evaluate(seds, smear=True)`.
        true_flux : array_like of float
            The *un-smeared* flux, shape (T_obs, W) --
            `spectrograph.evaluate(seds, smear=False)`.
            SNANA's SNR_TRUE is driven by the true flux but applied to the smeared ones.
        obs_table : ObsTable
            Must have an `exptime` column (`_required_values`).
            TODO: does obs_table have to be the desired observations or just include the subset desired?
        **kwargs
            Ignored -- absorbs `rng` and anything else `apply_noise`
            forwards.

        Returns
        -------
        flux_err : np.ndarray
            Shape (T_obs, W), same units as `measurements` (nJy).
        """
        measurements = np.asarray(measurements, dtype=float)
        true_flux = np.asarray(true_flux, dtype=float)
        exptime = np.asarray(obs_table.get_value_per_row("exptime"), dtype=float)

        # get interpolated ZP and SQSIGSKY for the given exposure times.
        zp_obs = self._zp_interp(np.log10(exptime)).T  # note ZP interpolated on log10(Texpose)
        sqsigsky_obs = self._sqsigsky_interp(exptime).T  # (T_obs, W)

        # get spectrograph snr
        # TODO: solve what are the incoming units? if Flam, then need a different flux2mag (currently assumes nJy)
        genmag = flux2mag(np.where(true_flux > 0, true_flux, np.nan))  # (T_obs, W)
        flux_pe = 10.0 ** (-0.4 * (genmag - zp_obs))  # p.e., sntools_spectrograph.c:1566-1567
        fluxerr = np.sqrt(sqsigsky_obs + flux_pe)
        snr_true = flux_pe / fluxerr

        # TODO: implement DO_SCALE_SNR - scales by a polynomial
        
        invalid = ~(snr_true > 1e-18) # catches on nans?
        return np.where(invalid, 0.0, measurements / np.where(invalid, 1.0, snr_true))
