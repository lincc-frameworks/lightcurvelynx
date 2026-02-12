import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix
from lightcurvelynx.obstable.argus_obstable import ArgusHealpixObsTable


def _ra_dec_to_healpix(ra, dec, nside=32):
    """Convert RA and Dec to HEALPix indices."""
    hpx = HEALPix(nside=nside, order="nested", frame="icrs")
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    return hpx.skycoord_to_healpix(coords)


def test_create_argus_obstable():
    """Create a minimal ArgusHealpixObsTable object and perform basic queries."""
    ra = np.array([15.0, 30.0, 15.0, 0.0, 15.0, 30.0])
    dec = np.array([10.0, -5.0, 10.0, 5.0, 10.0, -5.0])
    healpix = _ra_dec_to_healpix(ra, dec)
    values = {
        "epoch": 59000.0 + np.arange(len(ra)),
        "ra": ra,
        "dec": dec,
        "zp": np.ones_like(ra),
        "nside": np.full_like(ra, 32),
    }
    pdf = pd.DataFrame(values, index=healpix)

    # We fail if we are missing a healpix column or index.
    with pytest.raises(ValueError):
        _ = ArgusHealpixObsTable(pdf)

    pdf.index.name = "healpix"
    ops_data = ArgusHealpixObsTable(pdf)
    assert len(ops_data) == 6

    # We have all the attributes set at their default values.
    assert ops_data.survey_values["pixel_scale"] == 1.0
    assert ops_data.survey_values["radius"] == 52.0
    assert ops_data.survey_values["read_noise"] == 1.4
    assert ops_data.survey_values["survey_name"] == "Argus"

    # Check that we can extract the time bounds.
    t_min, t_max = ops_data.time_bounds()
    assert t_min == 59000.0
    assert t_max == 59005.0

    # We can query which columns the LSSTObsTable has by new or old name.
    assert "ra" in ops_data
    assert "dec" in ops_data
    assert "time" in ops_data
    assert "zp" in ops_data
    assert "healpix" in ops_data  # We move the index to a column

    # We can access columns directly as though it was a table.
    assert np.allclose(ops_data["ra"], values["ra"])
    assert np.allclose(ops_data["dec"], values["dec"])
    assert np.allclose(ops_data["time"], values["epoch"])

    # Check that we can determine which times a point is seen.
    assert np.array_equal(ops_data.range_search(15.0, 10.0), [0, 2, 4])
    assert np.array_equal(ops_data.range_search(30.0, -5.0), [1, 5])
    assert np.array_equal(ops_data.range_search(0.0, 5.0), [3])

    # Far away points are not seen.
    assert np.array_equal(ops_data.range_search(100.0, 80.0), [])
    assert np.array_equal(ops_data.range_search(15.0, -0.0), [])

    # Very close points fall in the same bucket, since they fall within an observed healpix pixel.
    assert np.array_equal(ops_data.range_search(15.00001, 9.99999), [0, 2, 4])


def test_create_argus_obstable_alternate():
    """Create a ArgusHealpixObsTable object with healpix as a column instead of index."""
    ra = np.array([15.0, 30.0, 15.0, 0.0, 15.0, 30.0])
    dec = np.array([10.0, -5.0, 10.0, 5.0, 10.0, -5.0])
    healpix = _ra_dec_to_healpix(ra, dec)
    values = {
        "epoch": 59000.0 + np.arange(len(ra)),
        "ra": ra,
        "dec": dec,
        "zp": np.ones_like(ra),
        "healpix": healpix,  # We move the index to a column
    }
    pdf = pd.DataFrame(values)

    # We fail without nside information.
    with pytest.raises(ValueError):
        _ = ArgusHealpixObsTable(pdf)

    table = ArgusHealpixObsTable(pdf, nside=32)
    assert len(table) == 6

    # We fail if we have inconsistent nside information.
    pdf["nside"] = np.array([32, 32, 16, 32, 32, 32])
    with pytest.raises(ValueError):
        _ = ArgusHealpixObsTable(pdf)


def test_argus_obstable_footprint():
    """We cannot use a footprint for the argus obstable since it is already healpix."""
    ra = np.array([15.0, 30.0, 15.0, 0.0, 15.0, 30.0])
    dec = np.array([10.0, -5.0, 10.0, 5.0, 10.0, -5.0])
    healpix = _ra_dec_to_healpix(ra, dec)
    values = {
        "epoch": 59000.0 + np.arange(len(ra)),
        "ra": ra,
        "dec": dec,
        "zp": np.ones_like(ra),
        "healpix": healpix,  # We move the index to a column
    }
    pdf = pd.DataFrame(values)
    table = ArgusHealpixObsTable(pdf, nside=32)
    assert not table.uses_footprint()

    # We fail if we even call set_detector_footprint.
    with pytest.raises(NotImplementedError):
        table.set_detector_footprint(None)
