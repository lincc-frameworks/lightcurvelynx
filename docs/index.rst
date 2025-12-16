
.. lightcurvelynx documentation main file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
LightCurveLynx - A fast and nimble package for realistic time-domain light curve simulations
========================================================================================

**NOTE:** This project was recently renamed from TDAstro to LightCurveLynx. See the
details for updating your code below.

Time-Domain Forward-Modeling for the Rubin Era
-------------------------------------------------------------------------------

Realistic light curve simulations are essential to many time-domain problems. 
Simulations are needed to evaluate observing strategy, characterize biases, 
and test pipelines. LightCurveLynx aims to provide a flexible, scalable, and user-friendly
time-domain simulation software with realistic effects and survey strategies.

.. figure:: _static/lightcurvelynx-intro.png
   :class: no-scaled-link
   :scale: 80 %
   :align: center
   :alt: LightCurveLynx simulation components


The main simulation components in LightCurveLynx include:

* A statistical simulation step where the hyperparameters of the model are drawn
  from one or more prior distributions. This can include existing packages such as
  `pzflow <https://jfcrenshaw.github.io/pzflow/>`_ or custom distributions.
* A model that defines the properties of the time-domain light source, which can
  also include a host-galaxy model, and is used to generate the noise-free light curves.
* ``ObsTable`` contains the survey information such as survey strategy and observing
  conditions. It is used to specify the observing times and bands.
* A set of predefined effects, such as dust extinction and detector noise, are applied to
  the noise-free light curves to produce realistic light curves.
* The ``PassbandGroup`` contains the filter information of the telescope and is used
  to calculate the fluxes in each band.

LightCurveLynx can generate numerous random realizations of the parameters for the physical model 
(``SEDModel`` or ``BandfluxModel``), and then apply the effects to these realizations. The ``ObsTable``
component is used to produce realistic light curves using the above components.  See the
:doc:`simulations <simulations>` page for a more detailed description of the process.

In addition to built-in and user-defined models, LightCurveLynx is designed to build off the
many excellent existing modeling packages. Wrappers are provided for popular packages including:

* `bayesn <https://github.com/bayesn/bayesn>`_ - A package for hierarchical modeling of a Type Ia supernova (:doc:`example <notebooks/pre_executed/Bayesian>`).
* `redback <https://github.com/nikhil-sarin/redback>`_ - A package for simulating and fitting a range of cosmological phenomena (:doc:`example <notebooks/pre_executed/redback_example>`).
* `sncosmo <https://sncosmo.readthedocs.io/en/stable/>`_ - A package for simulating supernovae.
* `synphot <https://synphot.readthedocs.io/en/latest/>`_ - A package for simulating photometric observations (:doc:`example <notebooks/pre_executed/synphot_example>`).

LightCurveLynx also allows users to load and sample pre-generated light curves, such as the
``LCLIB`` (:doc:`example <notebooks/pre_executed/lclib_example>`) and
``SIMSED`` (:doc:`example <notebooks/pre_executed/snana_example>`) formats used by SNANA.

For an overview of the package, we recommend starting with the notebooks in the "Getting Started"
section of the :doc:`notebooks page <notebooks>`. The :doc:`glossary <glossary>` provides definitions of
key terms, such as *GraphState*, *Node*, *Parameter*, *ParameterizedNode*, *BasePhysicalModel*,
*BandfluxModel*, and *SEDModel*.

The `full source code <https://github.com/lincc-frameworks/lightcurvelynx>`_ is available on GitHub.


Getting Started
-------------------------------------------------------------------------------

You can install LightCurveLynx from PyPI with pip or from conda-forge with conda. We recommend using a dedicated environment.

.. tab-set::
   :sync-group: packagemanager

   .. tab-item:: pip
      :sync: pip

      .. code-block:: bash

         # Create a virtual environment
         python3 -m venv ~/envs/lightcurvelynx
         # Activate it
         source ~/envs/lightcurvelynx/bin/activate
         # Install from PyPI
         python -m pip install lightcurvelynx

   .. tab-item:: conda
      :sync: conda

      .. code-block:: bash

         # Create a virtual environment
         conda create -p lightcurvelynx python=3.12
         # Activate it
         conda activate lightcurvelynx
         # Install from conda-forge channel
         conda install conda-forge::lightcurvelynx

Since LightCurveLynx relies on a large number of existing packages, not all of the packages
are installed in the default configuration. For example the microlensing (`VBMicrolensing`),
pzflow (`pzflow`), and sncosmo (`sncosmo`) packages are not included by default. If you try to
import a module that is not installed, LightCurveLynx will raise an `ImportError` with information on which
packages you need to install. You will need to install these manually. You can also install most
optional dependencies with:

.. tab-set::
   :sync-group: packagemanager

   .. tab-item:: pip
      :sync: pip

      .. code-block:: bash

         python -m pip install 'lightcurvelynx[all]'

   .. tab-item:: conda
      :sync: conda

      .. code-block:: bash

         conda install conda-forge::lightcurvelynx conda-forge::pzflow conda-forge::sncosmo

See our selection of :doc:`tutorial notebooks <notebooks>` for usage examples.
We recommend starting with the :doc:`introduction notebook <notebooks/introduction>`
to get a high level overview.

If you are interested in installing from `source <https://github.com/lincc-frameworks/lightcurvelynx>`_,
or contributing to the package, see the :doc:`contribution guide <contributing>`.

Running LightCurveLynx in a Jupyter Notebook
-------------------------------------------------------------------------------

Note that to run the LightCurveLynx within a notebook, your notebook kernel will need to have LightCurveLynx installed.
If you are using JupyterLab, you can install the kernel with the following commands
(using the Terminal program):

.. code-block:: bash

   >> python3 -m pip install ipykernel
   >> python3 -m ipykernel install --user --name=lightcurvelynx

You will need to restart your notebook server to see the new kernel.

Updating from TDAstro
-------------------------------------------------------------------------------

This package was recently renamed from TDAstro to LightCurveLynx. A few changes will be needed to
update your code:

   * If you use the package as a dependency (e.g. in pyproject.toml), update your requirements to use `lightcurvelynx` instead of `tdastro`.
   * Update your import statements from `import tdastro` to `import lightcurvelynx`.
   * If you have cloned the repository, update your remote URL to `https://github.com/lincc-frameworks/lightcurvelynx`.
   * If you have installed from PyPI or conda-forge, uninstall `tdastro` and install `lightcurvelynx` instead.

If you run into any problems or have any questions, please reach out to the team. We are happy to help!


Acknowledgements
-------------------------------------------------------------------------------

This project is supported by Schmidt Sciences.


.. toctree::
   :hidden:

   Home page <self>
   Simulations <simulations>
   Custom Models and Effects <custom_models>
   Results and Output <results_and_output>
   Notebooks <notebooks>
   API Reference <autoapi/index>
   Glossary <glossary>
   Contribution Guide <contributing>
   Citations <citations>
   Frequently Asked Questions <faq>
