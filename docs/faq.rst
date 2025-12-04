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
