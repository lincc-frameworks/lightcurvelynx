#!/usr/bin/env python

import numpy as np
from astropy.io import fits

from lightcurvelynx import _LIGHTCURVELYNX_TEST_DATA_DIR


def main(args=None):
    """Create fake, small dust maps for testing only."""
    print("Making fake SFD dust map data...")

    # Create a fake map directory
    fake_map_dir = _LIGHTCURVELYNX_TEST_DATA_DIR / "dustmaps" / "sfdmap2"
    if not fake_map_dir.exists():
        fake_map_dir.mkdir(parents=True)

    # Northern Hemisphere Dust
    hdu = fits.PrimaryHDU(np.zeros((2, 2)))
    hdu.header["SIMPLE"] = True
    hdu.header["BITPIX"] = -32
    hdu.header["NAXIS"] = 2
    hdu.header["NAXIS1"] = 2
    hdu.header["NAXIS2"] = 2
    hdu.header["OBJECT"] = "E(B-V)  "
    hdu.header["BUNIT"] = "mag     "
    hdu.header["CRPIX1"] = 0.5
    hdu.header["CRVAL1"] = 90.0
    hdu.header["CTYPE1"] = "GLON-ZEA"
    hdu.header["CRPIX2"] = 0.5
    hdu.header["CRVAL2"] = 90.0
    hdu.header["CTYPE2"] = "GLAT-ZEA"
    hdu.header["CD1_1"] = -0.0395646818624
    hdu.header["CD1_2"] = 0.0
    hdu.header["CD2_1"] = 0.0
    hdu.header["CD2_2"] = 0.0395646818624
    hdu.header["LONPOLE"] = 180
    hdu.header["LAM_NSGP"] = 1
    hdu.header["LAM_SCAL"] = 2
    hdu.writeto(fake_map_dir / "SFD_dust_4096_ngp.fits", overwrite=True)
    print(f"Writing fake SFD dust map data to {fake_map_dir / 'SFD_dust_4096_ngp.fits'}")

    # Southern Hemisphere Dust
    hdu = fits.PrimaryHDU(100.0 * np.ones((2, 2)))
    hdu.header["SIMPLE"] = True
    hdu.header["BITPIX"] = -32
    hdu.header["NAXIS"] = 2
    hdu.header["NAXIS1"] = 2
    hdu.header["NAXIS2"] = 2
    hdu.header["OBJECT"] = "E(B-V)  "
    hdu.header["BUNIT"] = "mag     "
    hdu.header["CRPIX1"] = 0.5
    hdu.header["CRVAL1"] = -90.0
    hdu.header["CTYPE1"] = "GLON-ZEA"
    hdu.header["CRPIX2"] = 0.5
    hdu.header["CRVAL2"] = -90.0
    hdu.header["CTYPE2"] = "GLAT-ZEA"
    hdu.header["CD1_1"] = 0.0395646818624
    hdu.header["CD1_2"] = 0.0
    hdu.header["CD2_1"] = 0.0
    hdu.header["CD2_2"] = -0.0395646818624
    hdu.header["LONPOLE"] = 180
    hdu.header["LAM_NSGP"] = -1
    hdu.header["LAM_SCAL"] = 2
    hdu.writeto(fake_map_dir / "SFD_dust_4096_sgp.fits", overwrite=True)
    print(f"Writing fake SFD dust map data to {fake_map_dir / 'SFD_dust_4096_sgp.fits'}")


if __name__ == "__main__":
    main()
