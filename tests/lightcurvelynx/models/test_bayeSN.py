import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
from citation_compass import find_in_citations
from lightcurvelynx.models.bayesn import BayesnModel


def test_bayesn_matchness(test_data_dir):
    """Test loading a bayesnModel object from a file and test we get the same
    results as the bayesian model
    """
    # We get the bayesian_flux from running the following M20 model code provided
    # by bayesian model
    # from bayesn.bayesn_model import SEDmodel
    # model = SEDmodel(load_model='M20_model',
    #                  filter_yaml='../data/passbands/LSST/filters.yaml')
    # ts =  np.linspace(-20, 50, 71) # Set of phases at which to generate spectra
    # for each object
    # ts = np.array([0,10])
    # N = 1 # Number of SNe to generate spectra for.
    # If you specify parameter values, the number of values passed
    # needs to match N e.g. if you want spectra for 10 objects,
    # specify 10 theta values (unless you want to use the same value
    # for all of them, in which case you can just pass one single value)
    # sim = model.simulate_spectrum(ts, N, dl=1000 ,del_M=0, z=0.001, mu=0,
    #                               ebv_mw=0, theta=1, AV=1, RV=3, eps=0)
    # l, spec, params = sim

    bayesian_flux = np.array(
        [
            [
                3.28274773e07,
                2.74288959e08,
                2.70277520e08,
                4.07166239e08,
                2.86161691e08,
                2.43041990e08,
                2.41895044e08,
                1.63319541e08,
                1.56384906e08,
                1.46499586e08,
                1.45467824e08,
                1.13228160e08,
                9.41477182e07,
                7.14146377e07,
                9.50177663e07,
                9.53728123e07,
            ],
            [
                4.24470160e06,
                1.04358573e08,
                1.53981951e08,
                2.60752039e08,
                1.62446512e08,
                1.50724949e08,
                1.76743433e08,
                1.01958387e08,
                5.71958998e07,
                3.92839096e07,
                3.79761607e07,
                3.22384974e07,
                3.60499142e07,
                6.20347574e07,
                7.25112817e07,
                7.51344153e07,
            ],
        ]
    )
    times = np.array([0, 10])
    wavelengths = np.array(
        [
            3003.0,
            4004.0,
            5005.0,
            6006.0,
            7007.0,
            8008.0,
            9009.0,
            10010.0,
            11011.0,
            12012.0,
            13013.0,
            14014.0,
            15015.0,
            16016.0,
            17017.0,
            18018.0,
        ]
    )

    redshift = 0.001
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    distmod = cosmo.distmod(redshift).value
    m_abs = -19.5
    model = BayesnModel(
        theta=1,
        Av=1,
        Rv=3,
        t0=0,
        ra=0,
        dec=0,
        redshift=redshift,
        _M20_model_path=test_data_dir / "BAYESN.M20",
        hsiao_model_path=test_data_dir / "hsiao.h5",
        Amplitude=np.power(10.0, -0.4 * (distmod + m_abs)),
    )
    assert model.minphase() == pytest.approx(-20.0)
    assert model.maxphase() == pytest.approx(85.0)

    flux_density = model.evaluate_sed(times, wavelengths)
    assert np.allclose(bayesian_flux, flux_density, rtol=0.1)


def test_bayesn_no_model(test_data_dir):
    """Test that we fail if using the wrong model directory."""
    dir_name = test_data_dir / "no_such_salt2_model_dir"
    with pytest.raises(FileNotFoundError):
        redshift = 0.001
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        distmod = cosmo.distmod(redshift).value
        m_abs = -19.5
        _ = BayesnModel(
            theta=1,
            Av=1,
            Rv=3,
            t0=0,
            ra=0,
            dec=0,
            redshift=redshift,
            hsiao_model_path=test_data_dir / "hsiao.h5",
            Amplitude=np.power(10.0, -0.4 * (distmod + m_abs)),
            _M20_model_path=dir_name,
        )


def test_bayesn_citation():
    """Test the citations for the bayesian model."""
    bayesn_citations = find_in_citations("bayesnModel")
    for citation in bayesn_citations:
        assert "Mandel S., 2020" in citation
        assert "Hsiao E. Y., 2009" in citation
