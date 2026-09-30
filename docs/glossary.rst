Glossary and Index
========================================================================================

This page provides a list of topics with definitions and links to corresponding documentation that mention that topic. For a single list of all notebooks (organized by category), please refer to :doc:`notebooks page <notebooks>`.

**AdditiveMultiObjectModel** - A model type that represents the combination of multiple objects, such as a host galaxy and a supernova. See also :doc:`Simulating Combinations of Objects Notebook <notebooks/host_source_models>`.

**ApproximateMOCSampler** - A node that approximates the survey area using a Multi-Order Coverage Map (MOC) and generates (RA, Dec) samples uniformly from that region. See also :doc:`Sampling Positions Notebook <notebooks/sampling_positions>`.

**Bandflux**: A flux density averaged over passband transmission, in nJy. It is given by this equation (we always assume "photon counter" detector): :math:`\frac{\int_0^\infty F_\nu(\lambda) T(\lambda) \mathrm{d}\lambda/\lambda}{\int_0^\infty T(\lambda) \mathrm{d}\lambda/\lambda}`, where :math:`\lambda` is the wavelength, :math:`F_\nu(\lambda)` is the source flux density, and :math:`T(\lambda)` is the passband transmission.

**BandfluxModel**: A ``BandfluxModel`` is a subclass of the ``BasePhysicalModel`` class that represents broadband photometry in specific bands (band fluxes), such as the g-band or r-band (instead of a full spectral energy distribution). See also :doc:`Adding New Models Notebook <notebooks/adding_models>`, :doc:`LightcurveTemplateModel Notebook <notebooks/lightcurve_source_demo>`, or :doc:`Custom Models and Effects <custom_models>`.

**BasePhysicalModel**: ``BasePhysicalModel`` is a superclass for all nodes that represent physical phenomena that produce flux. ``BasePhysicalModel`` itself is a subclass of ``ParameterizedNode``. See also :doc:`Introduction Notebook <notebooks/introduction>` and :doc:`Adding New Models Notebook <notebooks/adding_models>`.

**BasicMathNode** - A parameterized node that performs basic mathematical operations on its input parameters. See also :doc:`Sampling Parameters Notebook <notebooks/sampling>`.

**BinarySampler** - A parameterized node that samples binary outcomes (e.g., true/false or 0/1) based on input probabilities. See also :doc:`Advanced Sampling Notebook <notebooks/advanced_sampling>`.

**CCD-Level Observation Tables** - Observation tables that contain viewing information at the CCD level. See also :doc:`CCD-Level Observation Tables Notebook <notebooks/ccd_obstables>`.

**CatalogRADECSampler** - A node that samples (RA, Dec) positions from a catalog. See also :doc:`Sampling Positions Notebook <notebooks/sampling_positions>`.

**Dependency Graph (Parameters)** - A graph that represents the dependencies between parameters in the simulation. See also :doc:`Debugging Notebook <notebooks/debugging>`.

**DetectorFootprint** - An object that represents the footprint of a detector on the sky. See also :doc:`DetectorFootprint Notebook <notebooks/detector_footprint>` and :doc:`CCD-Level Observation Tables Notebook <notebooks/ccd_obstables>`.

**Dustmaps** - A map that provides information about the distribution of dust in the Milky Way for extinction corrections. See also :doc:`Dustmaps Notebook <notebooks/adding_effects>`.

**EffectModel**: An ``EffectModel`` applies some transformation to the flux density of an object. Example effects include extinction due to dust or white noise. See also :doc:`Introduction Notebook <notebooks/introduction>`, :doc:`Effects Notebook <notebooks/adding_effects>`, :doc:`Custom Models and Effects <custom_models>`, and :doc:`Time Varying Effects Notebook <notebooks/time_varying_effects>`.

**Extrapolation (Time and Wavelength)** - The computation used for times and/or wavelengths outside the models' defined ranges. See also :doc:`Extrapolation in Time and Wavelength Notebook <notebooks/extrapolation>`.

**Filter**: A filter corresponds to the physical filter used on a telescope to limit the wavelengths of light that hit the detector. Filters are represented by ``Passband`` objects, and sets of filters are represented by ``PassbandGroup`` objects. See also :doc:`Passband Demo Notebook <notebooks/passband-demo>`.

**Function Nodes** - The parameterized nodes that compute functions of other parameters. See also :doc:`Function Nodes Notebook <notebooks/function_nodes>`

**GraphState**: The ``GraphState`` object is an internal bookkeeping object that tracks the values of parameters during the simulation. It is implemented as nested dictionaries where the outer dictionary maps the name of the generating node to a dictionary of that node's parameters. The inner dictionary maps a parameter name to its values. See also :doc:`Introduction Notebook <notebooks/introduction>`, :doc:`Sampling Notebook <notebooks/sampling>`, and :doc:`Debugging Notebook <notebooks/debugging>`.

**GivenValueList** - A node that returns the values from a given list in the order in which they are given. See also :doc:`Advanced Sampling Notebook <notebooks/advanced_sampling>`.

**GivenValueSampler** - A node that returns a random value from a list with replacement. See also :doc:`Advanced Sampling Notebook <notebooks/advanced_sampling>`.

**GivenValueSelector** - A node that takes a single input parameter, an index, and uses it to look up the corresponding value in a given list. See also :doc:`Advanced Sampling Notebook <notebooks/advanced_sampling>`.

**HATS** - A catalog data format that can be used to store LightCurveLynx results. See also `HATS Page <https://docs.lsdb.io/en/latest/data-access/hats.html>`_, :doc:`Frequently Asked Questions <faq>`, :doc:`Results and Output <results_and_output>`.

**LightcurveTemplateModel** - A model that uses a predefined light curve template to generate fluxes. See also :doc:`LightcurveTemplateModel Notebook <notebooks/lightcurve_source_demo>`.

**LocationFreeObsTable** - An observation table that does not perform spatial filtering. See also :doc:`LocationFreeObsTable Notebook <notebooks/location_free_obstable>`.

**MilkyWayCoordSampler** - A parameterized node used to sample positions in the Milky Way in (RA, Dec, distance). See also :doc:`Sampling Positions Notebook <notebooks/sampling_positions>`.

**MOC**: A Multi-Order Coverage map (MOC) is a data structure for efficiently representing a spatial region on a sphere. See the `MOC IVOA note <https://www.ivoa.net/documents/MOC/>`_ for more details.

**Model**: The term model refers to any physical phenomenon that produces flux. All models are implemented as subclasses of the ``SEDModel`` or ``BandfluxModel`` classes. Also called a *physical model*.

**Model Parameterization** - For information on how to set the parameters of a model, see :doc:`Introduction Notebook <notebooks/introduction>`, :doc:`Building Simple Models Notebook <notebooks/technical_overview>`, and :doc:`Adding New Models Notebook <notebooks/adding_models>`.

**Multiple Surveys** - For information on how to simulate from multiple surveys, see :doc:`Frequently Asked Questions <faq>`, :doc:`Simulations <simulations>`, :doc:`Multiple Surveys Notebook <notebooks/multiple_surveys>`

**MultiLightcurveTemplateModel** - A model that uses multiple predefined light curve templates to generate fluxes. See also :doc:`MultiLightcurveTemplateModel Notebook <notebooks/lightcurve_source_demo>`.

**Node**: Nodes are the Python objects within the simulation that generate or use parameters. A node might represent a physical object that we are simulating, such as a Type Ia supernova with input parameters x0, x1, and c, or it might represent the statistical distributions for parameters, such as a Gaussian distribution for sampling an object's redshift (z). It is easiest to think of nodes as machines for generating portions of the simulation data. Nodes are implemented as subclasses of the ``ParameterizedNode`` class.

**node_label**: The node label is a unique identifier for each ``ParameterizedNode`` that allows the simulation (and the user) to track which Python object (and thus which part of the simulation) is using a particular parameter value. All parameters are indexed by a combination of node label and parameter name so that multiple ``ParameterizedNode`` objects can use the same parameter name without inadvertently overwriting each other's values. If a node label is not specified, LightCurveLynx automatically assigns one.

**noise model**: A noise model is a subclass of the ``FluxNoiseModel`` class that simulates the noise in the observations. Noise models are applied to the bandflux measurements to simulate the effects of atmospheric and detector noise. See also :doc:`Noise Models <noise_models>`, :doc:`Introduction Notebook <notebooks/introduction>`, and :doc:`PZFlow Learned Noise Model Notebook <notebooks/pre_executed/pzflow_noise_models>`

**NumpyRandomFunc** - A parameterized node used to sample values with numpy's random library. See also: :doc:`Building Simple Models Notebook <notebooks/technical_overview>` and :doc:`Sampling Parameters Notebook <notebooks/sampling>`.

**Observer Frame**: The reference frame of the observer. Observations in the observer frame account for effects such as redshift.

**ObsTable**: The ``ObsTable`` represents the set of data about the individual observations being simulated, including where the telescope is pointing (RA, Dec) and conditions affecting the detector noise. See also: :doc:`Survey Data <survey_data>` and :doc:`Introduction Notebook <notebooks/introduction>`.

**OpSim**: A specific version of the ``ObsTable`` that stores data for the Rubin Observatory's LSST `Operations Simulator <https://www.lsst.org/scientists/simulations/opsim>`_ outputs. See also :doc:`Survey Data <survey_data>` and :doc:`OpSim Notebook <notebooks/opsim_notebook>`.

**Parameter**: A parameter within the model corresponds to a variable in the mathematical equations for simulating flux. Each run of the simulation effectively samples the parameters in the model and uses them to compute the flux for a given model. Parameters may themselves be the result of computations performed with other parameters. For example we may choose an object's brightness parameter from a Gaussian distribution that is parameterized by mean and standard deviation. Parameters are generated by nodes (``ParameterizedNode`` objects) and stored in the ``GraphState`` objects.

**ParameterizedNode**: ParameterizedNodes are Python objects that are subclasses of the ``ParameterizedNode`` class and produce or use model parameters during the simulation. They contain built-in mechanisms for accepting samples of input parameters and producing dependent samples of output parameters. See also: *node*.

**Parallelization** - For information on how to perform parallel computation in LightCurveLynx, see :doc:`Parallel Computation Notebook <notebooks/parallel_runs>`.

**Passband**: A ``Passband`` object stores the information needed to transform the observed flux density over multiple wavelengths into a single band flux for a given filter. See also :doc:`Passband Demo Notebook <notebooks/passband-demo>`.

**PassbandGroup**: A ``PassbandGroup`` object implements a collection of ``Passband`` objects, providing convenient helper functions for loading and processing multiple passbands. Generally, users will use a single ``PassbandGroup`` corresponding to the filters on the instrument being simulated. See also :doc:`Passband Demo Notebook <notebooks/passband-demo>`.

**Periodic Light Curves** - Light curves that repeat over time. See also: :doc:`LightcurveTemplateModel Notebook <notebooks/lightcurve_source_demo>`

**Physical model**: A physical model is an astronomical phenomenon that is modeled using a subclass of ``BasePhysicalModel`` (usually also subclasses of ``SEDModel`` or ``BandfluxModel``) that represents a physical phenomenon that produces flux. See also: :doc:`Custom Models and Effects <custom_models>` and :doc:`Adding New Models Notebook <notebooks/adding_models>`.

**Randomness** - For information on controlling randomness, see :doc:`Randomness <randomness>`, :doc:`Frequently Asked Questions <faq>`, :doc:`Introduction Notebook <notebooks/introduction>`, and, for parallel computation, :doc:`Parallel Computation Notebook <notebooks/parallel_runs>`.

**Rest Frame**: The reference frame of the astronomical phenomenon being simulated. Observations in the rest frame do not account for effects like dust extinction or redshift, because they are local to the phenomenon.

**Saturation Thresholds** - The threshold at which a detector becomes saturated. For more information, see :doc:`Saturation Notebook <notebooks/saturation>`.

**Saving Results** - For information on saving results from a simulation, see :doc:`Results and Output <results_and_output>`, :doc:`Introduction Notebook <notebooks/introduction>`, and, for parallel computation, :doc:`Parallel Computation Notebook <notebooks/parallel_runs>`.

**SED**: Spectral energy distribution. In `LightCurveLynx`, this term refers to the distribution of flux density over wavelengths, :math:`F_\nu(\lambda)`. Flux density is in nJy, and wavelength is in angstroms.

**SEDModel**: ``SEDModel`` is a subclass of the ``BasePhysicalModel`` class that specifically represents flux density of a physical source as a function of time and wavelength.

**Spectrograph**: A spectrograph is an instrument that measures the flux density of a source as a function of wavelength. See also :doc:`Multiple Surveys Notebook <notebooks/multiple_surveys>` and :doc:`Spectrograph Demo Notebook <notebooks/spectrograph_demo>`.

**SurveyInfo**: The ``SurveyInfo`` object contains all the information about the survey(s) being simulated, including the observation information (``ObsTable``), passbands (``PassbandGroup``), and noise model for each survey. See also :doc:`Introduction Notebook <notebooks/introduction>` and :doc:`Survey Data <survey_data>`. For spectrograph simulations, see :doc:`Spectrograph Demo Notebook <notebooks/spectrograph_demo>`; for multiple surveys, see :doc:`Multiple Surveys Notebook <notebooks/multiple_surveys>`.

**TableSampler** - A sampler that draws samples from a table of precomputed values. See also :doc:`Advanced Sampling Notebook <notebooks/advanced_sampling>`.

**Time-Varying Effects** - For information on creating time-varying effects in simulations, see :doc:`Time Varying Effects Notebook <notebooks/time_varying_effects>`.

**UniformRADEC** - A sampler that draws uniform samples in right ascension and declination. See also :doc:`Sampling Positions Notebook <notebooks/sampling_positions>`.

**Wavelength**: Photon wavelength, in angstroms.
