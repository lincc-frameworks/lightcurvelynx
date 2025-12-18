import numpy as np
import pytest
from citation_compass import find_in_citations
from lightcurvelynx.models.salt2_jax import SALT2JaxModel
from lightcurvelynx.models.sncosmo_models import SncosmoWrapperModel
from sncosmo.models import SALT2Source


def test_salt2_model_parity(test_data_dir):
    """Test loading a SALT2 object from a file and test we get the same
    results as the sncosmo version.
    """
    dir_name = test_data_dir / "truncated-salt2-h17"
    td_model = SALT2JaxModel(x0=0.4, x1=0.3, c=1.1, t0=10.0, model_dir=dir_name)

    # We need to overwrite the source parameter to correspond to
    # the truncated directory data.
    sn_model = SncosmoWrapperModel("SALT2", x0=0.4, x1=0.3, c=1.1, t0=10.0)
    sn_model.source = SALT2Source(modeldir=dir_name)

    # Test compared to values computed via sncosmo's implementation that
    # fall within the range of the truncated grid. We multiple by 1e12
    # for comparison precision purposes.
    times = np.arange(-1.0, 15.0, 0.01) + 10.0
    waves = np.arange(3800.0, 4200.0, 0.5)

    # Allow LightCurveLynx to return both sets of results in f_nu.
    flux_td = td_model.evaluate_sed(times, waves)
    flux_sn = sn_model.evaluate_sed(times, waves)
    assert np.allclose(flux_td, flux_sn)


def test_salt2_model_no_t0(test_data_dir):
    """Test that the model fails if t0 is not provided, but redshift is."""
    dir_name = test_data_dir / "truncated-salt2-h17"
    td_model = SALT2JaxModel(x0=0.4, x1=0.3, c=1.1, redshift=0.1, model_dir=dir_name)

    with pytest.raises(ValueError):
        _ = td_model.evaluate_sed(np.arange(-1.0, 15.0, 0.01), np.arange(3800.0, 4200.0, 0.5))

    # The same models works if t0 is provided.
    td_model = SALT2JaxModel(x0=0.4, x1=0.3, c=1.1, redshift=0.1, t0=0.0, model_dir=dir_name)
    _ = td_model.evaluate_sed(np.arange(-1.0, 15.0, 0.01), np.arange(3800.0, 4200.0, 0.5))


def test_salt2_no_model(test_data_dir):
    """Test that we fail if using the wrong model directory."""
    dir_name = test_data_dir / "no_such_salt2_model_dir"
    with pytest.raises(FileNotFoundError):
        _ = SALT2JaxModel(x0=0.5, x1=0.2, c=1.0, t0=0.0, model_dir=dir_name)


def test_salt2_citation():
    """Test the citations for the SALT2 model."""
    salt_citations = find_in_citations("SALT2JaxModel")
    for citation in salt_citations:
        assert "Guy J., 2007" in citation
        assert "sncosmo" in citation
