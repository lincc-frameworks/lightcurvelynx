Frequently Asked Questions
========================================================================================

Where Does LightCurveLynx Store Downloaded Data?
-------------------------------------------------------------------------------

LightCurveLynx downloads data files (e.g., OpSim tables, passbands, dust maps) on demand
and caches them so they do not need to be re-downloaded on subsequent runs. The cache
location is resolved in the following priority order:

1. The ``LIGHTCURVELYNX_DATA_DIR`` environment variable, if set.
2. ``$XDG_CACHE_HOME/lightcurvelynx``, if ``XDG_CACHE_HOME`` is set.
3. ``~/.cache/lightcurvelynx`` (default).

To redirect the cache to a custom location — for example on an HPC system where the
home directory is read-only or quota-limited — set the environment variable before
running your code:

.. code-block:: bash

    export LIGHTCURVELYNX_DATA_DIR=/path/to/your/cache

or inline in Python before importing LightCurveLynx:

.. code-block:: python

    import os
    os.environ["LIGHTCURVELYNX_DATA_DIR"] = "/path/to/your/cache"


How Can I See More Detailed Output During Simulation?
-------------------------------------------------------------------------------

You can enable more detailed logging output by configuring the logging level for LightCurveLynx.
For example, to see info-level messages, you can add the following code snippet before running
your simulation:

.. code-block:: python

    import logging
    logging.basicConfig(level=logging.INFO)

You can also use `logging.DEBUG` to see even more detailed debug-level messages.


I Am Seeing Empty Light Curves for Some Entries. Why Is This Happening?
-------------------------------------------------------------------------------

The most common reason that LightCurveLynx produces rows with empty light curves is
that the object's sampled position (RA, dec) lies outside the area covered by the survey.
This can happen even if you use one of the predefined samplers, such as the `ApproximateMOCSampler`,
because all of the samplers are approximate.

You can find the empty rows in your results using:

.. code-block:: python

    idx = results.lightcurve.isna()

and check whether the row was observed using the `ObsTable.get_observations()` function
with the corresponding RA and dec.


Can I Use a Random Seed to Get Reproducible Results?
-------------------------------------------------------------------------------

Yes. LightCurveLynx allows the user to control randomness by passing in a predefined
random number generator via the `rng` parameter in the ``simulate_lightcurves()`` function.
If the user provides a random number generator with a fixed seed, then the results of the
simulation will be reproducible.


Can I Use a Random Seed with Parallelization?
-------------------------------------------------------------------------------

Yes. See the :doc:`parallelization notebook <notebooks/parallel_runs>` for details.


How Do I Use an External Simulation Package?
-------------------------------------------------------------------------------

LightCurveLynx is designed to be modular and extensible, allowing users to wrap external
simulation packages for use within the LightCurveLynx framework. How you wrap the package will
depend largely on the specifics of the package you are trying to wrap. We have provided
a few demo notebooks to illustrate various approachs, including:

  * :doc:`Wrapping Bagle Models <notebooks/pre_executed/wrapping_bagle>`
  * :doc:`Wrapping Redback Models <notebooks/pre_executed/wrapping_redback>`
  * :doc:`Wrapping VBMicrolensing <notebooks/pre_executed/wrapping_vbmicrolensing>`

The LightCurveLynx team is also happy to help you wrap your favorite package. Please
reach out if you have questions or need assistance.


Can I Combine Multiple Surveys in a Single Simulation?
-------------------------------------------------------------------------------

Yes. LightCurveLynx allows you to combine multiple surveys in a single simulation by passing
in multiple observation tables and corresponding passband groups. See the 
:doc:`multiple surveys demo notebook <notebooks/multiple_surveys>` for an example of how to do this.


Can I Rerun a Simulation with the Same Parameters?
-------------------------------------------------------------------------------

Yes. There are two approaches to doing this. 

First, if you want to produce exactly the same results, you can provide a random number generator with a fixed seed to the ``simulate_lightcurves()`` function. This will ensure that the same random numbers are used in the simulation for both parameter sampling and noise generation, resulting in identical outputs.

Second, LightCurveLynx allows you to rerun a new simulation with the same model parameters by using the ``GraphState`` object. The ``GraphState`` object captures the state of the simulation graph and allows you to rerun the simulation with the same parameters. This approach can be used to rerun a simulation with different survey information (or the same survey information and different noise realizations). This is particularly useful if you want to compare the results you get from different surveys on the exact same set of objects.

You can capture the state of the previous simulation from the "params" column in its results table:

.. code-block:: python

    previous_state = GraphState.from_list(results["params"].values)

Then you pass this ``previous_state`` to the ``simulate_lightcurves()`` function to rerun the simulation with the same parameters:

.. code-block:: python

    results2 = simulate_lightcurves(
        model,
        previous_state.num_samples,
        new_survey_info,
        graph_state=previous_state,
    )

You can see the :doc:`multiple surveys demo notebook <notebooks/multiple_surveys>` for an example of how to do this.

It is possible to change the values within the ``GraphState`` object before passing it to the ``simulate_lightcurves()`` function. For example, you might want to change the objects' ``t0`` values to correspond to the new survey's time range. However, care should be taken when changing the values within the ``GraphState`` object. If other parameters depend on the values you change, they will **not** be updated automatically and you can end up with inconsistent results.


Can I Simulate Spectra?
--------------------------------------------------------------------------------

Yes with some caveats. LightCurveLynx has built-in support for simulating spectra, but it is currently
in an early stage of development and does not yet add noise to the measurements. In addition, spectra
simulation is **only** compaible with models that generate data on the spectral level (not bandflux-only
models).  For more detail see :doc:`the spectrograph demo notebook <notebooks/spectrograph_demo>`.


Can I Generate Points from a Catalog?
--------------------------------------------------------------------------------

Yes. LightCurveLynx allows you to generate light curves for objects in a catalog containing 
positions using the``CatalogRADECSampler`` object This sampler takes in a table
of information with at least "ra" and "dec" columns. The helper function 
``from_hats()`` is provided to load directly from a [HATS](https://www.ivoa.net/documents/Notes/HATS/)
catalog.

See the :doc:`sampling positions demo notebook <notebooks/sampling_positions>` notebook
for a detailed description of how to sample (RA, dec) positions.


Why does my light curve have multiple points at the same time?
--------------------------------------------------------------------------------

While this can legitimately happen if multiple surveys have the exact same MJD for an observations, this is more likely an artifact of per-CCD level information. If the survey data is provided at the CCD-level (such as with Rubin's DP1 CCD visit table) **and** no detector footprint is set, the code will estimate a circular footprint per-CCD. Points that lie near the edge of one CCD may also be picked up by another CCD. This can often be solved by setting the detector footprint for each CCD. See the :doc:`ccd-level obstable <notebooks/ccd_obstables>` for more information.
