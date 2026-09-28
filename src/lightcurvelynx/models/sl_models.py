import numpy as np
from lightcurvelynx.math_nodes.basic_math_node import BasicMathNode
from lightcurvelynx.models.multi_object_model import MultiObjectModel
from lightcurvelynx.models.physical_model import SEDModel, BasePhysicalModel

class LensGalaxy(SEDModel):
    def __init__(self, mass, ellipticity, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter("mass", mass)
        self.add_parameter("ellipticity", ellipticity)
        self.mass = mass
        self.ellipticity = ellipticity

class LensPhysics():
    def __init__(self, lens_galaxy, unlensed_source, **kwargs):
        super().__init__(**kwargs)
        self.lens_galaxy = lens_galaxy
        self.unlensed_source = unlensed_source
        self.compute_lensed_parameters()

    def compute_lensed_parameters(self):
        # This function computes the lensed source properties (time_delay, ra_offset, dec_offset, magnification) based on the lens galaxy and unlensed source properties. 
        self.nsources = 2  # Placeholder for actual number of lensed sources based on lensing configuration
        self.time_delays = self.lens_galaxy.mass * np.array([0.1,0.12])  # Placeholder for actual time delay calculation
        self.ra_offsets = self.lens_galaxy.ellipticity * 0.01 * np.array([1,-1])  # Placeholder for actual RA offset calculation
        self.dec_offsets = self.lens_galaxy.ellipticity * 0.01 * np.array([1,-1])  # Placeholder for actual Dec offset calculation
        self.magnifications = 1.0 + self.lens_galaxy.mass * 0.01 * np.array([0.2,0.5])  # Placeholder for actual magnification

class MultiLensedSource(MultiObjectModel):
    def __init__(self, lens_physics=None, unlensed_source=None, **kwargs):
        super().__init__(objects=[], **kwargs)
        self.lens_physics = lens_physics
        self.unlensed_source = unlensed_source
        for i in range(self.lens_physics.nsources):
            time_delay = self.lens_physics.time_delays[i]
            magnification = self.lens_physics.magnifications[i]
            ra_offset = self.lens_physics.ra_offsets[i]
            dec_offset = self.lens_physics.dec_offsets[i]
            self.objects.append(LensedSource(t0 = BasicMathNode("t0 + time_delay",t0=self.unlensed_source.t0, time_delay=time_delay),
                                ra = BasicMathNode("ra + ra_offset",ra=self.unlensed_source.ra, ra_offset=ra_offset),
                                dec = BasicMathNode("dec + dec_offset",dec=self.unlensed_source.dec, dec_offset=dec_offset),
                                x0 = BasicMathNode("x0 * magnification",x0=self.unlensed_source.x0, magnification=magnification),
                                x1 = self.unlensed_source.x1,
                                c = self.unlensed_source.c,
                                )
            )