import argparse

import numpy as np
import pandas as pd
from pzflow import Flow


def main():
    """Generate the fake OpSim data and save it to a file.

    To generate an updated small_db file use:
        python tests/lightcurvelynx/utils/make_fake_opsim.py tests/lightcurvelynx/data/opsim_small.db
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, help="Output filename")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="The number of samples used to generate the flow.",
    )
    args = parser.parse_args()

    # Use a basic noise model that only considers source variance and readout
    # noise (assuming 1 exposure and psf_footprint = 1).
    zp = np.random.normal(loc=25.0, scale=0.5, size=args.num_samples)
    bandflux = np.random.normal(loc=500.0, scale=100.0, size=args.num_samples)
    source_variance = bandflux / zp
    readout_noise = np.random.normal(loc=5.0, scale=1.0, size=args.num_samples)
    readout_variance = readout_noise**2  # Assume 1 exposure and psf_footprint = 1.
    flux_err = np.sqrt(source_variance + readout_variance) * zp

    # Train a flow to learn the noise model that maps from the input parameters
    training_data = pd.DataFrame(
        {
            "bandflux": bandflux,
            "zp": zp,
            "readout_noise": readout_noise,
            "flux_err": flux_err,
        }
    )
    flow = Flow(
        data_columns=["flux_err"],
        conditional_columns=["bandflux", "zp", "readout_noise"],
    )
    _ = flow.train(training_data, verbose=True)

    # Save the flow file.
    flow.save(args.output)


if __name__ == "__main__":
    main()
