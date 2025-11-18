"""Functions for extrapolating flux past the end of a model's range of valid
phases or wavelengths using flux = f(time, wavelengths).
"""

import abc

import numpy as np


class FluxExtrapolationModel(abc.ABC):
    """The base class for the flux extrapolation methods."""

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def _extrapolate(self, last_value, last_flux, query_values):
        """Evaluate the extrapolation given the last valid points(s) and a list of new
        query points.

        Note
        ----
        This function does not care which axis is being extrapolated. The returned values are
        always len(query_values) x len(last_flux) and may need to be transposed by the calling
        function.

        Parameters
        ----------
        last_value : float
            The last valid value along the extrapolation axis a which the flux was predicted
            (e.g., wavelength in AA or time in days).
        last_flux : numpy.ndarray
            A length N array of the flux values at the last valid time or wavelength (in nJy).
        query_values : numpy.ndarray
            A length M array of values along the extrapolation axis (times in days or wavelengths
            in AA) at which to extrapolate.

        Returns
        -------
        flux : numpy.ndarray
            A N x M matrix of extrapolated values. Where M is the number of query points and
            N is the number of flux values at the last valid point.
        """
        raise NotImplementedError("Subclasses must implement this method.")  # pragma: no cover

    def extrapolate_time(self, last_time, last_flux, query_times):
        """Extrapolate along the time axis.

        Parameters
        ----------
        last_time : float
            The last valid time (in days) at which the flux was predicted.
        last_flux : numpy.ndarray
            A length W array of the last valid flux values at each wavelength
            at the last valid time (in nJy).
        query_times : numpy.ndarray
            A length T array of the query times (in days) at which to extrapolate.

        Returns
        -------
        flux : numpy.ndarray
            A T x W matrix of extrapolated values.
        """
        return self._extrapolate(last_time, last_flux, query_times).T

    def extrapolate_wavelength(self, last_wave, last_flux, query_waves):
        """Extrapolate along the wavelength axis.

        Parameters
        ----------
        last_wave : float
            The last valid wavelength (in AA) at which the flux was predicted.
        last_flux : numpy.ndarray
            A length T array of the last valid flux values at each time
            at the last valid wavelength (in nJy).
        query_waves : numpy.ndarray
            A length W array of the query wavelengths (in AA) at which to extrapolate.

        Returns
        -------
        flux : numpy.ndarray
            A T x W matrix of extrapolated values.
        """
        # We transpose the result to turn the W x T matrix into a T x W matrix.
        return self._extrapolate(last_wave, last_flux, query_waves)


class ZeroPadding(FluxExtrapolationModel):
    """Extrapolate by zero padding the results."""

    def __init__(self):
        super().__init__()

    def _extrapolate(self, last_value, last_flux, query_values):
        """Evaluate the extrapolation given the last valid points(s) and a list of new
        query points.

        Note
        ----
        This function does not care which axis is being extrapolated. The returned values are
        always len(query_values) x len(last_flux) and may need to be transposed by the calling
        function.

        Parameters
        ----------
        last_value : float
            The last valid value along the extrapolation axis a which the flux was predicted
            (e.g., wavelength in AA or time in days).
        last_flux : numpy.ndarray
            A length N array of the flux values at the last valid time or wavelength (in nJy).
        query_values : numpy.ndarray
            A length M array of values along the extrapolation axis (times in days or wavelengths
            in AA) at which to extrapolate.

        Returns
        -------
        flux : numpy.ndarray
            A N x M matrix of extrapolated values. Where M is the number of query points and
            N is the number of flux values at the last valid point.
        """
        N_len = len(last_flux)
        M_len = len(query_values)
        return np.zeros((N_len, M_len))


class ConstantPadding(FluxExtrapolationModel):
    """Extrapolate using a constant value in nJy.

    Attributes
    ----------
    value : float
        The value (in nJy) to use for the extrapolation.
    """

    def __init__(self, value=0.0):
        super().__init__()

        if value < 0:
            raise ValueError("Extrapolation value must be positive.")
        self.value = value

    def _extrapolate(self, last_value, last_flux, query_values):
        """Evaluate the extrapolation given the last valid points(s) and a list of new
        query points.

        Note
        ----
        This function does not care which axis is being extrapolated. The returned values are
        always len(query_values) x len(last_flux) and may need to be transposed by the calling
        function.

        Parameters
        ----------
        last_value : float
            The last valid value along the extrapolation axis a which the flux was predicted
            (e.g., wavelength in AA or time in days).
        last_flux : numpy.ndarray
            A length N array of the flux values at the last valid time or wavelength (in nJy).
        query_values : numpy.ndarray
            A length M array of values along the extrapolation axis (times in days or wavelengths
            in AA) at which to extrapolate.

        Returns
        -------
        flux : numpy.ndarray
            A N x M matrix of extrapolated values. Where M is the number of query points and
            N is the number of flux values at the last valid point.
        """
        N_len = len(last_flux)
        M_len = len(query_values)
        return np.full((N_len, M_len), self.value)


class LastValue(FluxExtrapolationModel):
    """Extrapolate using the last valid value along the desired axis."""

    def __init__(self):
        super().__init__()

    def _extrapolate(self, last_value, last_flux, query_values):
        """Evaluate the extrapolation given the last valid points(s) and a list of new
        query points.

        Note
        ----
        This function does not care which axis is being extrapolated. The returned values are
        always len(query_values) x len(last_flux) and may need to be transposed by the calling
        function.

        Parameters
        ----------
        last_value : float
            The last valid value along the extrapolation axis a which the flux was predicted
            (e.g., wavelength in AA or time in days).
        last_flux : numpy.ndarray
            A length N array of the flux values at the last valid time or wavelength (in nJy).
        query_values : numpy.ndarray
            A length M array of values along the extrapolation axis (times in days or wavelengths
            in AA) at which to extrapolate.

        Returns
        -------
        flux : numpy.ndarray
            A N x M matrix of extrapolated values. Where M is the number of query points and
            N is the number of flux values at the last valid point.
        """
        return np.tile(last_flux[:, np.newaxis], (1, len(query_values)))


class LinearDecay(FluxExtrapolationModel):
    """Linear decay of the flux using the last valid point(s) down to zero.

    Attributes
    ----------
    decay_width : float
        The width of the decay region in Angstroms. The flux is
        linearly decreased to zero over this range.
    """

    def __init__(self, decay_width=100.0):
        super().__init__()

        if decay_width <= 0:
            raise ValueError("decay_width must be positive.")
        self.decay_width = decay_width

    def _extrapolate(self, last_value, last_flux, query_values):
        """Evaluate the extrapolation given the last valid points(s) and a list of new
        query points.

        Note
        ----
        This function does not care which axis is being extrapolated. The returned values are
        always len(query_values) x len(last_flux) and may need to be transposed by the calling
        function.

        Parameters
        ----------
        last_value : float
            The last valid value along the extrapolation axis a which the flux was predicted
            (e.g., wavelength in AA or time in days).
        last_flux : numpy.ndarray
            A length N array of the flux values at the last valid time or wavelength (in nJy).
        query_values : numpy.ndarray
            A length M array of values along the extrapolation axis (times in days or wavelengths
            in AA) at which to extrapolate.

        Returns
        -------
        flux : numpy.ndarray
            A N x M matrix of extrapolated values. Where M is the number of query points and
            N is the number of flux values at the last valid point.
        """
        last_flux = np.asarray(last_flux)
        query_values = np.asarray(query_values)
        dist = np.abs(query_values - last_value)

        multiplier = np.clip(1.0 - (dist / self.decay_width), 0.0, 1.0)
        flux = last_flux[:, np.newaxis] * multiplier[np.newaxis, :]
        return flux


class ExponentialDecay(FluxExtrapolationModel):
    """Exponential decay of the flux using the last valid point(s) down to zero.

    f(t, λ) = f(t, λ_last) * exp(- rate * \\|λ - λ_last\\|)

    Attributes
    ----------
    rate : float
        The decay rate in the exponential function.
    """

    def __init__(self, rate):
        super().__init__()

        if rate < 0:
            raise ValueError("Decay rate must be zero or positive.")
        self.rate = rate

    def _extrapolate(self, last_value, last_flux, query_values):
        """Evaluate the extrapolation given the last valid points(s) and a list of new
        query points.

        Note
        ----
        This function does not care which axis is being extrapolated. The returned values are
        always len(query_values) x len(last_flux) and may need to be transposed by the calling
        function.

        Parameters
        ----------
        last_value : float
            The last valid value along the extrapolation axis a which the flux was predicted
            (e.g., wavelength in AA or time in days).
        last_flux : numpy.ndarray
            A length N array of the flux values at the last valid time or wavelength (in nJy).
        query_values : numpy.ndarray
            A length M array of values along the extrapolation axis (times in days or wavelengths
            in AA) at which to extrapolate.

        Returns
        -------
        flux : numpy.ndarray
            A N x M matrix of extrapolated values. Where M is the number of query points and
            N is the number of flux values at the last valid point.
        """
        last_flux = np.asarray(last_flux)
        query_values = np.asarray(query_values)
        dist = np.abs(query_values - last_value)

        multiplier = np.exp(-self.rate * dist)
        flux = last_flux[:, np.newaxis] * multiplier[np.newaxis, :]
        return flux
