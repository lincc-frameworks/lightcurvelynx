"""A place holder with a deprecation warning for the old ExtinctionEffect model."""

from lightcurvelynx.effects.effect_model import EffectModel


class ExtinctionEffect(EffectModel):
    """A place holder with a deprecation warning for the old ExtinctionEffect model."""

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "The ExtinctionEffect has been deprecated and removed. Please use the "
            "DustExtinctionEffect from lightcurvelynx.effects.dust_extinction instead."
        )
