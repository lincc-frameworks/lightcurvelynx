import numpy as np
import pytest
from lightcurvelynx.astro_utils.sed import SED
from lightcurvelynx.utils.io_utils import write_numpy_data


def test_create_sed() -> None:
    """Test that we can create and sample a StaticSEDModel object with a single SED."""
    sed = np.array(
        [
            [100.0, 200.0, 300.0, 400.0],  # Wavelengths
            [10.0, 20.0, 20.0, 10.0],  # fluxes
        ]
    )
    model = SED(sed[0], sed[1])
    assert model.minwave() == 100.0
    assert model.maxwave() == 400.0

    # We can interpolate the SED at given wavelengths.
    wavelengths = np.array([50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0])
    values = model.evaluate(wavelengths)
    assert values.shape == (9,)

    expected = np.array([0.0, 10.0, 15.0, 20.0, 20.0, 20.0, 15.0, 10.0, 0.0])
    np.testing.assert_array_equal(values, expected)


def test_sed_fail() -> None:
    """Test that we correctly fail on bad SEDs."""
    with pytest.raises(ValueError):
        _ = SED([1.0, 2.0], [1.0])  # Different lengths
    with pytest.raises(ValueError):
        _ = SED([1.0], [1.0])  # Too few points
    with pytest.raises(ValueError):
        _ = SED([2.0, 1.0], [1.0, 2.0])  # Unsorted wavelengths


def test_sed_to_from_file(tmp_path) -> None:
    """Test that we can create a StaticSEDModel object from a file."""
    test_sed = np.array(
        [
            [100.0, 200.0, 300.0, 400.0],  # Wavelengths
            [10.0, 20.0, 20.0, 10.0],  # fluxes
        ]
    )
    wavelengths = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0])
    expected = np.array([10.0, 15.0, 20.0, 20.0, 20.0, 15.0, 10.0])
    sed_org = SED(test_sed[0], test_sed[1])

    for fmt in ["npy", "npz", "txt", "csv"]:
        file_path = tmp_path / f"test_sed_file.{fmt}"
        assert not file_path.exists()

        sed_org.to_file(file_path)
        assert file_path.exists()

        model = SED.from_file(file_path)
        values = model.evaluate(wavelengths)
        assert np.allclose(values, expected)

    # Try reading an invalid array shape.
    test_sed_invalid = np.array(
        [
            [100.0, 200.0, 300.0, 400.0],  # Wavelengths
            [10.0, 20.0, 20.0, 10.0],  # fluxes
            [0.0, 0.0, 0.0, 0.0],  # Other row
        ]
    )
    file_path_invalid = tmp_path / "test_invalid_sed.csv"
    write_numpy_data(file_path_invalid, test_sed_invalid.T)

    with pytest.raises(ValueError):
        _ = SED.from_file(file_path_invalid)


class DummySynphotModel:
    """A fake synphot model used for testing.

    Attributes
    ----------
    waveset : numpy.ndarray
        The wavelengths at which the SED is defined (in angstroms)
    fluxset : numpy.ndarray
        The flux at each given wavelength (in nJy.)
    """

    def __init__(self, waveset, fluxset):
        self.waveset = waveset
        self.fluxset = fluxset
        self.z = 0.0  # Redshift

    def __call__(self, waves, **kwargs):
        """Return the flux for the given wavelengths as interpolated PHOTLAM.

        Parameters
        ----------
        waves : numpy.ndarray
            The wavelengths at which to evaluate the SED (in angstroms).
        **kwargs : dict
            Additional keyword arguments (ignored).

        Returns
        -------
        numpy.ndarray
            The interpolated flux values at the given wavelengths (in PHOTLAM).
        """
        # Return a dummy SED for the given wavelengths
        return np.interp(waves, self.waveset, self.fluxset, left=0.0, right=0.0)


def test_sed_from_synphot() -> None:
    """Test that we can create a SED from a synphot model."""
    # Create a dummy model with 4 samples of SEDs [10.0, 20.0, 30.0, 40.0] in nJy.
    # Since synphot uses PHOTLAM, we preconvert and provide in that unit.
    sp_model = DummySynphotModel(
        waveset=np.array([1000.0, 2000.0, 3000.0, 4000.0]),
        fluxset=np.array([1.50919018e-08, 1.50919018e-08, 1.50919018e-08, 1.50919018e-08]),
    )
    model = SED.from_synphot(sp_model)

    wavelengths = np.array([500.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0])
    expected = np.array([0.0, 10.0, 15.0, 20.0, 30.0, 0.0])
    fluxes = model.evaluate(wavelengths)
    assert fluxes.shape == (len(wavelengths),)
    np.testing.assert_allclose(fluxes, expected, rtol=1e-5)
