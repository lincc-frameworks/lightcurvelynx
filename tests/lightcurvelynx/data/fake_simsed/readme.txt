This directory contains SIMSED data for resting. These files are subsets of the SIMSED.SNIa-91bg data at: https://zenodo.org/records/2612896:
 - fake_simsed1.sed is a subsampled version of 91BG_ST6_C3.SED.gz
 - fake_simsed2.sed is a subsampled version of 91BG_ST1_C4.SED.gz

The data was generated with:

    import numpy as np

    data = np.loadtxt(filename)
    time_mask = (data[:, 0] >= -10) & (data[:, 0] <= 30) & (data[:, 0] % 2 == 0)
    wave_mask = (data[:, 1] >= 3000) & (data[:, 1] <= 8000) & (data[:, 1] % 100 == 0)
    data = data[time_mask & wave_mask]
    np.savetxt(new_filename, data, fmt="%.6e")

The parameters in this SED.INFO (including FLUX_SCALE) are NOT the same as the original data. Specifically the scale has been changes to not be 1.0.

Attribution:
R. Kessler et al., 2019 (PLASTICC models)
https://ui.adsabs.harvard.edu/abs/2019PASP..131i4501K