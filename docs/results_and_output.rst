Results and Output
========================================================================================

Results
-------------------------------------------------------------------------------

Results of the simulation are returned in a `nested-pandas <https://github.com/lincc-frameworks/nested-pandas>`_ 
NestedDataFrame. Each row corresponds to a single simulated object with columns for attributes such as 
``ra``, ``dec``, and ``t0`` (if provided). You can access these columns as you would with a normal Pandas DataFrame.

.. code-block:: python

    results = simulate_lightcurves(...)
    print("The first result:", results.iloc[0])
    print(f"The first object's location: ({results['ra'].iloc[0]}, {results['dec'].iloc[0]})")


The ``lightcurve`` column stores a nested frame for each object with the corresponding time series information,
including time (MJD), flux, and flux error.  When accessing the light curve at a specific row the result is a
Pandas DataFrame with one row for each observation.

.. code-block:: python

    lightcurve = results["lightcurve"].iloc[0]
    print("The first object's light curve:")
    print(lightcurve)


Each row also contains a copy of the parameters use to simulate that object (in the ``params`` column), allowing the
user to easily lookup the object's information. The parameters are stored as a dictionary using a structure
based on the ``GraphState`` object. Each key consists to a combination of the node name and the parameter name
(separated by a dot). For example, the parameter ``c`` from the node ``salt2`` would be stored under the key
``salt2.c``.

.. code-block:: python

    salt2_c_value = results["params"].iloc[0]["salt2.c"]

The parameter values will either be scalars or arrays depending on the number of samples generated.


Plotting Results
-------------------------------------------------------------------------------

LightCurveLynx includes a variety of plotting functions to visualize the results. For example, the
``plot_lightcurves()`` function can be used to plot the light curve for a single object.

.. code-block:: python

    from lightcurvelynx.utils.plotting import plot_lightcurves
    plot_lightcurves(
        lightcurve["flux"],
        lightcurve["mjd"],
        fluxerrs=lightcurve["fluxerr"],
        filters=lightcurve["filter"],
    )

The function takes arrays for flux and time, as well as optional arrays for flux errors and filters.
The resulting plot will show the light curve with error bars and different colors for each filter.

For example a plot of one of the light curves from the SIMSED.TDE-MOSFIT data set
(`zenodo link <https://zenodo.org/records/2612896>`_) is shown below.

.. figure:: _static/plotted_lightcurve.png
   :class: no-scaled-link
   :scale: 80 %
   :align: center
   :alt: An example light curve plot using data from the SIMSED.TDE-MOSFIT data set.

The function also provides the ability to plot the underlying (noise-free) light curve.


Saving Results
-------------------------------------------------------------------------------

After simulating a population of objects, users may want to save the results for later analysis.
LightCurveLynx returns the results as a NestedDataFrame using the
`nested-pandas <https://nested-pandas.readthedocs.io/en/latest/>`__ package.
This allows users to easily save all the results in a single file using the
``to_parquet()`` function.

.. code-block:: python

    results.to_parquet("simulated_lightcurves.parquet")

In addition, each individual light curve is stored as a (nested) Pandas DataFrame. Users can
access and save light curves individually using the standard Pandas functions such as
``to_csv()`` or ``to_parquet()``.

.. code-block:: python

    results["lightcurve"].iloc[0].to_parquet("lightcurve_0.parquet")


Exporting to HATS format
-------------------------------------------------------------------------------

The results can also be exported to the `HATS format <https://github.com/astronomy-commons/hats>`_ 
using the ``write_results_as_hats()`` utility function. HATS data is stored as a set of files in
a directory, so the function takes a directory path as input.

.. code-block:: python

    from lightcurvelynx.utils.io_utils import write_results_as_hats
    write_results_as_hats(dir_path, results, overwrite=True)
