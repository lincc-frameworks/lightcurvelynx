Noise Models
========================================================================================

LightCurveLynx uses noise models to simulate atmospheric and sensor noise at the bandflux level. The noise model object's ``apply_noise`` function takes in the observer frame bandflux and an optional `ObsTable` (which can provide additional per-observation information about the noise). It then computes updated bandflux measurements after applying the noise and an array of flux error values.

.. code-block:: python

    flux, fluxerr = noise_model.apply_noise(flux_perfect, obs_table=obs_table)

LightCurveLynx allows the user to select which noise model to use for the simulation, by including ``FluxNoiseModel`` object within the ``SurveyInfo`` object passed to the simulator. As described below, different noise models can be applied by using different subclasses of the ``FluxNoiseModel`` base class.

If no noise model is provided when creating the ``SurveyInfo`` object, the default noise model for that ``ObsTable`` subclass is used.


ConstantFluxNoiseModel
--------------------------------------------------------------------------------

The ``ConstantFluxNoiseModel`` is a simple noise model that adds Gaussian noise to the bandflux measurements. The standard deviation of the noise is constant for all observations and is specified by the user when creating the noise model object.


PoissonFluxNoiseModel
--------------------------------------------------------------------------------

The ``PoissonFluxNoiseModel`` is a noise model that adds Gaussian noise to the bandflux measurements based on Poisson photon statistics. It requires additional information from the ``ObsTable`` to compute the noise:

* ``dark_current``: Mean dark current (electrons per pixel per second).
* ``exptime``: The total exposure time for the observation (seconds).
* ``nexposure``: The number of exposures (optional, default is 1).
* ``psf_footprint``: Point spread function effective area (pixel^2).
* ``read_noise``: Standard deviation of the readout electrons per pixel per exposure.
* ``sky_bg_e``: Sky background (electrons / pixel^2).
* ``zp``: The photometric zero point (nJy / electron).
* ``zp_err_mag``: The uncertainty in the photometric zero point in magnitudes (optional, default is 0.0).

Each of these can either be provided as columns in the ``ObsTable`` or single values for the survey.

The noise is drawn from a Gaussian distribution with a standard deviation equal to the square root of the total variance:

.. code-block:: python

    source_variance = bandflux / zp
    sky_variance = sky_bg_e * psf_footprint
    readout_variance = read_noise**2 * psf_footprint * nexposure
    dark_variance = dark_current * exptime * psf_footprint
    zp_variance = (bandflux * zp_err_mag * np.log(10.0) / 2.5 / zp) ** 2
    total_variance = source_variance + sky_variance + readout_variance + dark_variance + zp_variance

    flux_err = np.sqrt(total_variance) * zp

This noise model is the default for most surveys.


GivenNoiseModel
--------------------------------------------------------------------------------

The ``GivenNoiseModel`` is a noise model that uses a Gaussian distribution with the noise's standard deviation provided as a given column in the ``ObsTable``. This model does not compute any additional noise based on the bandflux measurements or other observation parameters.


Learned Noise Models
--------------------------------------------------------------------------------

Users can also train machine learning models to learn the noise characteristics of their observations and use these learned models as noise models in LightCurveLynx. The ``PZFlowNoiseModel`` provides one example of this approach, where a normalizing flow model is trained to learn the noise distribution from the data. The learned model can then be used to generate noise samples for new observations.
