Frequently Asked Questions
========================================================================================


Can I Use a Random Seed to Get Reproducible Results?
-------------------------------------------------------------------------------

Yes. LightCurveLynx allows the user to control randomness by passing in a predefined
random number generator via the `rng` parameter in the ``simulate_lightcurves()`` function.
If the user provides a random number generator with a fixed seed, then the results of the
simulation will be reproducible.


Can I Use a Random Seed with Parallelization?
-------------------------------------------------------------------------------

Yes. See the :doc:`parallelization notebook <notebooks/parallelization>` for details.
