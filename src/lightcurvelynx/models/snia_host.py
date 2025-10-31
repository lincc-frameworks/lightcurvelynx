from lightcurvelynx.models.physical_model import SEDModel


class SNIaHost(SEDModel):
    """A SN Ia host galaxy model with a hostmass parameter, more to be added.

    Parameterized values include:
      * dec - The object's declination in degrees. [from BasePhysicalModel]
      * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
      * hostmass - The hostmass value in units of log10(M/M_solar).
      * ra - The object's right ascension in degrees. [from BasePhysicalModel]
      * redshift - The object's redshift. [from BasePhysicalModel]
      * t0 - The t0 of the zero phase, date. [from BasePhysicalModel]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter(
            "hostmass",
            description="The hostmass value in units of log10(M/M_solar).",
            **kwargs,
        )
