Survey Data
========================================================================================

In order to generate a simulation, the simulator needs information on whether the telescope is pointing and the noise characteristics at the time. As you can imagine, the available data and its format varies significantly from survey to survey (and even between data releases of the same survey). `LightCurveLynx` uses an object-based model to encapsulate the survey information and provide a common interface.


ObsTable
-------------------------------------------------------------------------------

The ``ObsTable`` parent class is top level interface for storing and manipulating survey data. At a minimum, it requires the pointing information (RA, dec, and time for image). In addition it can store a wide range of per-visit information (e.g. airmass, exposure time, filter, etc.) and or survey-specific constants (e.g. pixel scale and saturation magnitudes).

In addition to a common API, ``ObsTable`` includes common processing that is used across many different types of surveys, including:

    * Column name mapping (for users trying to load data from custom tables),
    * Filtering rows by index or mask (``filter_rows``),
    * Addition and querying of pixel-level detector footprints (e.g., ``set_detector_footprint``),
    * Construction and use of spatial data structures for fast search including built-in cone search (``range_search``)
    * Application of saturation thresholds,
    * Plotting the survey's footprint on the sky (``plot_footprint``),
    * Construction of a [multi-order coverage map (MOC)](https://www.ivoa.net/documents/MOC/) of the survey (``build_moc``), 
    * Resampling the table (``make_resampled_table``), and
    * Writing the observation information to disk.

Users do not work with the ``ObsTable`` class directly, but instead use one of the child classes that are designed for specific surveys. We discuss some of these below.


LSST Data (LSSTObsTable)
-------------------------------------------------------------------------------

The ``LSSTObsTable`` class stores actual pointing and noise information from the Rubin Observatory data releases. It includes functions to read from the currently defined data formats, including:

    * [DP1](https://sdm-schemas.lsst.io/dp1.html#CcdVisit) and [DP2+](https://sdm-schemas.lsst.io/lsstcam.html#CcdVisit) as CCDVisit tables, using the ``from_ccdvisit_table`` method, and
    * [Science Validation](https://survey-strategy.lsst.io/progress/sv_status/sv_20250930.html) as the released DB file, using the ``from_sv_visits_table`` method.

The survey-specific constants are set to [published values](https://lsstcam.lsst.io/index.html).

See the (:doc:`Rubin CCDVisit Notebook <notebooks/pre_executed/rubin_ccdvisit>`) for an example of how to load Rubin data using the ``LSSTObsTable`` class.


OpSim Data (OpSim)
-------------------------------------------------------------------------------

The ``OpSim`` class stores pointing and noise information from the Rubin Observatory's simulated survey data (OpSim).  The survey-specific constants are set to [the values defined in the simulator](https://smtn-002.lsst.io/v/OPSIM-1171/index.html).

See the (:doc:`Rubin OpSim Notebook <notebooks/opsim_notebook>`) for an example of how to load simulated Rubin data using the ``OpSim`` class. This notebook also provides an overview of the functionality of the ``ObsTable`` class.


Roman Simulated Data (RomanObsTable)
-------------------------------------------------------------------------------
The ``RomanObsTable`` class stores pointing and noise information from the [Nancy Grace Roman Space Telescope](https://roman.gsfc.nasa.gov)'s simulated survey data. The survey-specific constants are set to the values defined in the [Roman documentation](https://roman-docs.stsci.edu/) and the [Roman Github repository](https://github.com/RomanSpaceTelescope). 

Special preprocessing is needed to handle the spectra observations (spectra simulation within LightCurveLynx is in early testing). Please contact the LightCurveLynx team if you need help with this.


ZTF Data (ZTFObsTable)
-------------------------------------------------------------------------------
The ``ZTFObsTable`` class stores pointing and noise information from the [Zwicky Transient Facility (ZTF)](https://www.ztf.caltech.edu/) visit information.


Argus Data (ArgusObsTable)
-------------------------------------------------------------------------------
The ``ArgusObsTable`` class stores pointing and noise information from simulations of the upcoming [Argus Array](https://www.argusarray.org/). This class is still being validated and may change. Please contact the LightCurveLynx team if you want to use this class or have suggestions for improvement.
