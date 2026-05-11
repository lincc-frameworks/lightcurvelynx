Notebooks
========================================================================================

Getting Started
-----------------------------------------------------------------------------------------

We recommend that new users start with the following notebooks to get a basic understanding
of the LightCurveLynx package and how to use it.

.. toctree::
    :maxdepth: 1

    Introduction <notebooks/introduction>
    Building Simple Models <notebooks/technical_overview>
    Sampling Model Parameters <notebooks/sampling>


Technical Details
-----------------------------------------------------------------------------------------

The following notebooks provide more in-depth technical deep dives into specific 
features and models of the LightCurveLynx package.

.. toctree::
    :maxdepth: 1

    Using Rubin OpSims in Simulations <notebooks/opsim_notebook>
    Using Rubin CCD Visit Table in Simulations <notebooks/pre_executed/rubin_ccdvisit>
    Sampling Object Positions <notebooks/sampling_positions>
    Passband Demo <notebooks/passband-demo>
    Simulating Spectra <notebooks/spectrograph_demo>
    Debugging Models <notebooks/debugging>
    Combining Models (including Hosts/Sources) <notebooks/host_source_models>
    Extrapolation in Time and Wavelength <notebooks/extrapolation>
    Advanced Sampling Techniques <notebooks/advanced_sampling>
    Simulating Multiple Surveys <notebooks/multiple_surveys>
    Detector Footprints <notebooks/detector_footprint>
    Parallel Runs <notebooks/parallel_runs>
    Setting Saturation Limits <notebooks/saturation>
    Using Learned PZFlow Noise Models <notebooks/pre_executed/pzflow_noise_models>


Citations
-----------------------------------------------------------------------------------------

The following note provides an overview of how to use the `citation_compass` package to
track citations in your LightCurveLynx simulations.

.. toctree::
    :maxdepth: 1

    Citations <notebooks/citations>


Example Simulations
-----------------------------------------------------------------------------------------

The following notebooks provide example simulations using the LightCurveLynx package (listed
in roughly alphabetical order of simulation package or simulation type).

.. toctree::
    :maxdepth: 1

    AGN Damped Random Walk Model <notebooks/pre_executed/agn>
    Basic SNIa Simulation <notebooks/simple_snia>
    EzTaoX <notebooks/pre_executed/eztaox_example>
    Lightcurve Template Model Demo <notebooks/lightcurve_source_demo>
    Microlensing Effect Example <notebooks/pre_executed/microlensing>
    PLAsTiCC SNIa <notebooks/pre_executed/plasticc_snia>
    PyLIMA Micro-Lensing Model <notebooks/pre_executed/pylima_example>
    PZFlow Source Demo <notebooks/pre_executed/using_pzflow>
    Redback Models <notebooks/pre_executed/redback_example>
    Resampling LCLIB <notebooks/pre_executed/lclib_example>
    SNANA Models <notebooks/pre_executed/snana_example>
    Synphot-based Models <notebooks/pre_executed/synphot_example>


Other Surveys
-----------------------------------------------------------------------------------------

The following notebooks provide example simulations using non-Rubin surveys.

.. toctree::
    :maxdepth: 1

    Argus Survey (prototype development)<notebooks/pre_executed/argus_example>
    SkyMapper Survey <notebooks/pre_executed/skymapper_example>


Extending LightCurveLynx
-----------------------------------------------------------------------------------------

The following notebooks provide examples of how you can add your own models, surveys, and effects
to the LightCurveLynx package.

.. toctree::
    :maxdepth: 1

    Adding New Model Types <notebooks/adding_models>
    Adding New Effect Types <notebooks/adding_effects>
    Creating Time Varying Effects <notebooks/time_varying_effects>
    Adding Custom Surveys <notebooks/custom_survey>
    Creating Custom Function Nodes <notebooks/function_nodes>


The following notebooks provide more in-depth examples of how to wrap external packages
for use in LightCurveLynx.

.. toctree::
    :maxdepth: 1

    Wrapping Bagle Models <notebooks/pre_executed/wrapping_bagle>
    Wrapping Redback Models <notebooks/pre_executed/wrapping_redback>
    Wrapping VBMicrolensing <notebooks/pre_executed/wrapping_vbmicrolensing>


Running LightCurveLynx in a Jupyter Notebook
-----------------------------------------------------------------------------------------

For instructions on how to add the LightCurveLynx kernel to your Jupyter Notebook environment, see the
:doc:`index page <index>`.
