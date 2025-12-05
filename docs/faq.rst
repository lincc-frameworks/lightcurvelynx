Frequently Asked Questions
========================================================================================

How Can I See More Detailed Output During Simulation?
-------------------------------------------------------------------------------

You can enable more detailed logging output by configuring the logging level for LightCurveLynx.
For example, to see info-level messages, you can add the following code snippet before running
your simulation:

.. code-block:: python

    import logging
    logging.basicConfig(level=logging.INFO)

You can also use `logging.DEBUG` to see even more detailed debug-level messages.


Can I Use a Random Seed to Get Reproducible Results?
-------------------------------------------------------------------------------

Yes. LightCurveLynx allows the user to control randomness by passing in a predefined
random number generator via the `rng` parameter in the ``simulate_lightcurves()`` function.
If the user provides a random number generator with a fixed seed, then the results of the
simulation will be reproducible.


Can I Use a Random Seed with Parallelization?
-------------------------------------------------------------------------------

Yes. See the :doc:`parallelization notebook <notebooks/parallelization>` for details.
